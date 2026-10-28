"""Folder auto-annotation: model candidates, human review and YOLO / COCO export.

No Qt here. Model boxes are candidates, never human ground truth: only images a
human explicitly accepted (or accepted after editing) are exported, and every
export manifest records that the labels are model candidates reviewed by a human.
"""
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from i18n import message

PROJECT = Path(__file__).resolve().parent
ANNOTATION_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
STATES = ("pending", "accepted", "edited", "rejected")
EXPORTED_STATES = frozenset({"accepted", "edited"})
MANIFEST_NAME = "auto_annotation_manifest.json"
SESSION_SCHEMA = "ee6008-auto-annotation-session/1"
SESSION_CACHE_DIRNAME = "annotation_sessions"
LABEL_PROVENANCE = "model candidates reviewed by a human"
GROUND_TRUTH_NOTE = (
    "Labels start from model candidates and were accepted or edited by a human reviewer. "
    "They are not independent human ground truth: do not use them to evaluate the model "
    "that proposed them, and do not mix them into the human-GT evaluation datasets.")
CODE_FILES = ("auto_annotation.py", "auto_annotation_ui.py", "engines.py", "main.py",
              "prompt_modes.py", "resource_safety.py")
MAX_BOXES = 1000
_SHA_CACHE = {}


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_sha256(model_path):
    """SHA-256 of a local checkpoint; empty for hub IDs such as Grounding DINO."""
    path = Path(str(model_path))
    if not path.is_absolute():
        path = PROJECT / path
    if not str(model_path) or not path.is_file():
        return ""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha256_file(path)
    return _SHA_CACHE[key]


def code_fingerprint():
    return {name: sha256_file(PROJECT / name) for name in CODE_FILES if (PROJECT / name).is_file()}


def scan_folder(folder, recursive=False):
    """Sorted JPG/JPEG/PNG/BMP files; earlier export folders are skipped."""
    root = Path(os.path.abspath(folder))
    if not root.is_dir():
        raise ValueError(message("图片文件夹不存在：{path}", path=str(folder)))
    found = []
    for directory, subdirectories, files in os.walk(root):
        if MANIFEST_NAME in files:
            subdirectories[:] = []
            continue
        subdirectories.sort()
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() in ANNOTATION_SUFFIXES and path.is_file():
                found.append(path)
        if not recursive:
            break
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def read_image_with_sha(path):
    """Decode non-ASCII paths and hash exactly the bytes that were decoded."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None, ""
    if not data.size:
        return None, ""
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    return image, hashlib.sha256(data.tobytes()).hexdigest()


def clip_box(xyxy, width, height):
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    return (min(max(x1, 0.), width), min(max(y1, 0.), height),
            min(max(x2, 0.), width), min(max(y2, 0.), height))


def normalize_box(box, width=None, height=None):
    label = str(box.get("label", "")).strip()
    if not label or len(label) > 120 or any(ord(char) < 32 for char in label):
        raise ValueError(message("{what}不能为空、过长或包含控制字符", what=message("类别")))
    xyxy = [float(value) for value in box["xyxy"]]
    if len(xyxy) != 4 or not all(math.isfinite(value) for value in xyxy):
        raise ValueError("框坐标必须是四个有限数值")
    if width is not None and height is not None:
        xyxy = list(clip_box(xyxy, width, height))
    score = box.get("score")
    score = None if score is None else float(score)
    return {"label": label, "xyxy": xyxy, "score": score,
            "source": str(box.get("source") or ("human" if score is None else "original"))}


def _same_box(first, second):
    return first["label"] == second["label"] and all(
        abs(a - b) < 1e-3 for a, b in zip(first["xyxy"], second["xyxy"]))


def box_origin(box, candidates):
    """human: drawn by the reviewer; model: an untouched candidate; model_edited: moved/relabelled."""
    if box.get("score") is None:
        return "human"
    return "model" if any(_same_box(box, item) for item in candidates) else "model_edited"


def yolo_line(class_id, xyxy, width, height):
    """``cls cx cy w h`` normalised to [0, 1] with 6 decimals, after clipping to the image."""
    if width <= 0 or height <= 0:
        raise ValueError("图片宽高必须大于零")
    x1, y1, x2, y2 = clip_box(xyxy, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    values = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
    return f"{int(class_id)} " + " ".join(f"{min(1., max(0., value)):.6f}" for value in values)


def coco_bbox(xyxy, width, height):
    x1, y1, x2, y2 = clip_box(xyxy, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]


class ImageEntry:
    """One image: immutable model candidates plus the human's current review."""

    def __init__(self, path, relative):
        self.path, self.relative = Path(path), relative
        self.width = self.height = None
        self.sha256 = ""
        self.candidates, self.boxes = [], []
        self.state = "pending"
        self.detected = False
        self.error = None
        self.run_index = None
        self.review_method = ""
        self.bulk_min_score = None

    @property
    def modified(self):
        return len(self.boxes) != len(self.candidates) or not all(
            _same_box(a, b) for a, b in zip(self.boxes, self.candidates))

    @property
    def exported(self):
        return self.state in EXPORTED_STATES


class AnnotationSession:
    """Review state for one folder. Re-running detection never overwrites human work."""

    def __init__(self, folder, recursive=False, paths=None):
        self.folder = Path(os.path.abspath(folder))
        self.recursive = bool(recursive)
        paths = scan_folder(self.folder, recursive) if paths is None else [Path(os.path.abspath(p)) for p in paths]
        self.entries = [ImageEntry(path, path.relative_to(self.folder).as_posix()) for path in paths]
        self.classes, self.runs = [], []
        self.created_at = now_iso()
        self.dirty = False

    def __len__(self):
        return len(self.entries)

    # Classes -------------------------------------------------------------
    def set_classes(self, names):
        self.classes = list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))

    def ensure_class(self, name):
        name = str(name).strip()
        if name and name not in self.classes:
            self.classes.append(name)

    def rename_class(self, old, new):
        """Rename in every reviewed box and the class list; candidates stay as predicted.

        Only images that had a box renamed change state. A bulk-accepted image
        stays a bulk acceptance: a class rename is not a person looking at it.
        """
        new = normalize_box({"label": new, "xyxy": [0, 0, 1, 1]})["label"]
        changed = 0
        for entry in self.entries:
            renamed = 0
            for box in entry.boxes:
                if box["label"] == old:
                    box["label"] = new
                    renamed += 1
            changed += renamed
            if renamed and entry.exported and entry.review_method != "bulk_min_score":
                entry.state = "edited" if entry.modified else "accepted"
        self.classes = list(dict.fromkeys(new if name == old else name for name in self.classes))
        self.dirty = self.dirty or changed > 0
        return changed

    def labels_in_use(self):
        return list(dict.fromkeys(box["label"] for entry in self.entries for box in entry.boxes))

    # Detection runs ------------------------------------------------------
    def begin_run(self, metadata):
        self.runs.append(dict(deepcopy(metadata), started_at=now_iso(), finished_at=None, processed=0))
        return len(self.runs) - 1

    def finish_run(self, run_index, **values):
        self.runs[run_index].update(values, finished_at=now_iso())

    def targets(self, config):
        """Pending, untouched images that lack candidates for this exact configuration.

        ``config=None`` (configuration not known yet) returns every pending, untouched image.
        """
        result = []
        for index, entry in enumerate(self.entries):
            if entry.state != "pending" or entry.modified:
                continue
            same = config is not None and entry.detected and entry.run_index is not None and \
                self.runs[entry.run_index].get("config") == config
            if entry.error or not same:
                result.append((index, entry.path))
        return result

    def set_image_info(self, index, width, height, sha256):
        entry = self.entries[index]
        if entry.sha256 and sha256 and entry.sha256 != sha256:
            raise ValueError(message("源图片在审核期间被修改：{name}", name=entry.relative))
        entry.width, entry.height, entry.sha256 = int(width), int(height), sha256 or entry.sha256

    def set_candidates(self, index, boxes, width, height, sha256, run_index=None):
        """Store a detection result; True if it was applied.

        A result for an image the person reviewed or changed after the job was
        queued is discarded: their boxes, the candidates they saw and the run
        those came from stay as they are (counted in the run record).
        """
        entry = self.entries[index]
        if len(boxes) > MAX_BOXES:
            raise ValueError("单张图最多保存 1000 个框")
        self.set_image_info(index, width, height, sha256)
        candidates = [normalize_box(box, width, height) for box in boxes]
        run = None if run_index is None else self.runs[run_index]
        if run is not None and run.get("finished_at") is None:
            run["processed"] = run.get("processed", 0) + 1
        if entry.state != "pending" or entry.modified:
            if run is not None:
                run["results_discarded_after_review"] = run.get("results_discarded_after_review", 0) + 1
            return False
        entry.candidates, entry.boxes = candidates, deepcopy(candidates)
        entry.detected, entry.error, entry.run_index = True, None, run_index
        return True

    def set_error(self, index, error):
        self.entries[index].error = str(error)

    # Review --------------------------------------------------------------
    def update_boxes(self, index, boxes):
        entry = self.entries[index]
        if entry.width is None:
            raise ValueError("请先打开这张图片再修改或接受")
        if len(boxes) > MAX_BOXES:
            raise ValueError("单张图最多保存 1000 个框")
        entry.boxes = [normalize_box(box, entry.width, entry.height) for box in boxes]
        if entry.exported:
            entry.state = "edited" if entry.modified else "accepted"
            entry.review_method, entry.bulk_min_score = "per_image", None
        elif entry.state == "rejected":
            entry.state = "pending"
        self.dirty = True

    def accept(self, index):
        entry = self.entries[index]
        if entry.width is None:
            raise ValueError("请先打开这张图片再修改或接受")
        entry.state = "edited" if entry.modified else "accepted"
        entry.review_method, entry.bulk_min_score = "per_image", None
        self.dirty = True
        return entry.state

    def reject(self, index):
        entry = self.entries[index]
        entry.state, entry.review_method, entry.bulk_min_score = "rejected", "per_image", None
        self.dirty = True

    def reset(self, index):
        entry = self.entries[index]
        entry.boxes = deepcopy(entry.candidates)
        entry.state, entry.review_method, entry.bulk_min_score = "pending", "", None
        self.dirty = True

    def bulk_accept(self, min_score, dry_run=False):
        """Accept untouched pending images that keep >=1 candidate at or above min_score.

        Images with no remaining candidate stay pending: an empty label file is a
        claim that nothing is present, which needs a person to look at the image.
        ``dry_run`` only counts the images that would be accepted.
        """
        min_score = float(min_score)
        if not 0 <= min_score <= 1:
            raise ValueError("置信度和 IoU 必须在 0 到 1 之间")
        accepted = 0
        for entry in self.entries:
            if entry.state != "pending" or not entry.detected or entry.error or entry.modified:
                continue
            kept = [deepcopy(box) for box in entry.candidates if box["score"] is not None and box["score"] >= min_score]
            if not kept:
                continue
            accepted += 1
            if dry_run:
                continue
            entry.boxes = kept
            entry.state, entry.review_method, entry.bulk_min_score = "accepted", "bulk_min_score", min_score
        self.dirty = self.dirty or (accepted > 0 and not dry_run)
        return accepted

    def counts(self):
        values = {state: 0 for state in STATES}
        values.update(total=len(self.entries), detected=0, failed=0, boxes=0)
        for entry in self.entries:
            values[entry.state] += 1
            values["detected"] += entry.detected
            values["failed"] += entry.error is not None
            if entry.exported:
                values["boxes"] += len(entry.boxes)
        values["exportable"] = values["accepted"] + values["edited"]
        return values

    def export_boxes(self, entry, min_score=0.0):
        """(boxes, counts) to write for one exported image.

        ``min_score`` only filters images accepted in bulk by score, which nobody
        opened; boxes a person accepted or edited are exported as reviewed, since
        dropping them would turn objects into false negatives. Boxes with no area
        inside the image are dropped here, so the plan, the manifest and the
        writers all count the same boxes.
        """
        bulk = entry.review_method == "bulk_min_score"
        boxes, counts = [], {"filtered_by_min_score": 0, "zero_area": 0}
        for box in entry.boxes:
            if bulk and box["score"] is not None and box["score"] < min_score:
                counts["filtered_by_min_score"] += 1
                continue
            x1, y1, x2, y2 = clip_box(box["xyxy"], entry.width, entry.height)
            if x2 <= x1 or y2 <= y1:
                counts["zero_area"] += 1
                continue
            boxes.append(box)
        return boxes, counts

    def export_classes(self, plan):
        names = list(self.classes)
        for _, boxes, _ in plan:
            for box in boxes:
                if box["label"] not in names:
                    names.append(box["label"])
        return names

    # Session persistence (C2: auto-save / resume) ------------------------
    def to_dict(self):
        """Everything needed to resume review: candidates, boxes, states, class order and runs.

        Model identity and run options already live in ``self.runs`` (each
        entry recorded by ``begin_run``); they are not duplicated here.
        """
        entries = []
        for entry in self.entries:
            try:
                stat = entry.path.stat()
                size, mtime_ns = stat.st_size, stat.st_mtime_ns
            except OSError:
                size, mtime_ns = None, None
            entries.append({
                "relative": entry.relative, "size": size, "mtime_ns": mtime_ns,
                "sha256": entry.sha256, "width": entry.width, "height": entry.height,
                "candidates": deepcopy(entry.candidates), "boxes": deepcopy(entry.boxes),
                "state": entry.state, "detected": entry.detected, "error": entry.error,
                "run_index": entry.run_index, "review_method": entry.review_method,
                "bulk_min_score": entry.bulk_min_score,
            })
        return {
            "schema": SESSION_SCHEMA, "saved_at": now_iso(), "folder": str(self.folder),
            "recursive": self.recursive, "created_at": self.created_at,
            "classes": list(self.classes), "runs": deepcopy(self.runs), "entries": entries,
        }

    @classmethod
    def from_dict(cls, data, folder, recursive=False):
        """Rebuild from a saved session, restoring only images that have not changed.

        A fresh scan decides which images exist now; a saved record is only
        applied when the file's size and mtime still match what was recorded,
        so an image that changed on disk is never resumed with stale candidates
        (it is left at "undetected / pending", like a brand-new scan would see
        it). A malformed per-image record is treated the same way, rather than
        guessing at its content. ``restored_count`` / ``changed_count`` on the
        returned session report what happened, for the caller to display.
        """
        if not isinstance(data, dict) or data.get("schema") != SESSION_SCHEMA:
            raise ValueError(message("标注进度文件版本不兼容或已损坏"))
        if not {"entries", "runs", "classes"}.issubset(data):
            raise ValueError(message("标注进度文件缺少必要字段"))
        if not all(isinstance(data.get(key) or [], list) for key in ("entries", "runs", "classes")):
            raise ValueError(message("标注进度文件版本不兼容或已损坏"))
        session = cls(folder, recursive=recursive)
        if isinstance(data.get("created_at"), str) and data["created_at"]:
            session.created_at = data["created_at"]
        session.set_classes(name for name in data.get("classes") or [] if isinstance(name, str))
        session.runs = [dict(run) for run in data.get("runs") or [] if isinstance(run, dict)]
        saved = {raw.get("relative"): raw for raw in data.get("entries") or []
                if isinstance(raw, dict) and isinstance(raw.get("relative"), str)}
        restored = changed = 0
        for entry in session.entries:
            raw = saved.get(entry.relative)
            if raw is None:
                continue
            try:
                stat = entry.path.stat()
            except OSError:
                continue
            if raw.get("size") != stat.st_size or raw.get("mtime_ns") != stat.st_mtime_ns:
                changed += 1
                continue
            try:
                width, height = raw.get("width"), raw.get("height")
                saved_candidates, saved_boxes = raw.get("candidates") or [], raw.get("boxes") or []
                if not all(isinstance(items, list) and all(isinstance(box, dict) for box in items)
                           for items in (saved_candidates, saved_boxes)):
                    raise ValueError("boxes must be a list of objects")
                candidates = [normalize_box(box) for box in saved_candidates]
                boxes = [normalize_box(box) for box in saved_boxes]
                state = raw.get("state") if raw.get("state") in STATES else "pending"
                run_index = raw.get("run_index")
                run_index = run_index if isinstance(run_index, int) and 0 <= run_index < len(session.runs) else None
                bulk_min_score = raw.get("bulk_min_score")
                bulk_min_score = float(bulk_min_score) if isinstance(bulk_min_score, (int, float)) else None
            except (TypeError, ValueError, KeyError, AttributeError):
                changed += 1
                continue
            entry.width = int(width) if isinstance(width, (int, float)) else None
            entry.height = int(height) if isinstance(height, (int, float)) else None
            entry.sha256 = str(raw.get("sha256") or "")
            entry.candidates, entry.boxes = candidates, boxes
            entry.state, entry.detected = state, bool(raw.get("detected"))
            entry.error = raw.get("error") if isinstance(raw.get("error"), str) else None
            entry.run_index, entry.review_method = run_index, str(raw.get("review_method") or "")
            entry.bulk_min_score = bulk_min_score
            restored += 1
        session.dirty = any(entry.state != "pending" for entry in session.entries)
        session.restored_count, session.changed_count = restored, changed
        return session


def session_cache_dir():
    """Where saved review sessions live; ``EE6008_ANNOTATION_CACHE_DIR`` redirects it (tests, shared drives)."""
    override = os.environ.get("EE6008_ANNOTATION_CACHE_DIR")
    directory = Path(override) if override else PROJECT / ".cache" / SESSION_CACHE_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def session_cache_path(folder):
    """A stable, collision-resistant cache path for one source folder.

    Kept outside the image folder itself (never written next to the user's
    photos) and under the project's git-ignored ``.cache/``, consistent with
    the GPU lock file; never committed.
    """
    key = hashlib.sha256(str(Path(os.path.abspath(folder))).encode("utf-8")).hexdigest()
    return session_cache_dir() / (key + ".json")


def write_session(session, path=None):
    """Atomically write *session* as JSON: a temp file, fsync, then one rename."""
    path = Path(path) if path is not None else session_cache_path(session.folder)
    data = session.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path


def read_session_file(path):
    """The saved session dict, or None when there is nothing saved yet."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(message("标注进度文件损坏，无法读取：{error}", error=str(exc))) from exc
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(message("标注进度文件损坏，无法读取：{error}", error=str(exc))) from exc


def delete_session_file(folder, path=None):
    """Remove the saved session for *folder*, if any; never raises."""
    path = Path(path) if path is not None else session_cache_path(folder)
    with suppress(OSError):
        path.unlink()


def _plan(session, target, min_score):
    min_score = float(min_score)
    if not 0 <= min_score <= 1:
        raise ValueError("置信度和 IoU 必须在 0 到 1 之间")
    target = Path(os.path.abspath(target))
    if target == session.folder or target.is_relative_to(session.folder):
        raise ValueError("导出目录不能位于源图片文件夹内，原始图片保持不变")
    if target.exists():
        raise FileExistsError(message("导出目录已存在，不覆盖旧导出：{path}", path=str(target)))
    plan, skipped = [], []
    for entry in session.entries:
        if not entry.exported:
            continue
        if entry.width is None or not entry.sha256:
            raise ValueError(message("图片尚未读取，不能导出：{name}", name=entry.relative))
        if sha256_file(entry.path) != entry.sha256:
            raise ValueError(message("源图片在审核期间被修改：{name}", name=entry.relative))
        boxes, counts = session.export_boxes(entry, min_score)
        if not boxes and entry.review_method == "bulk_min_score":
            # An empty label claims "no objects"; nobody looked at this image.
            skipped.append(entry.relative)
            continue
        plan.append((entry, boxes, counts))
    if not plan and skipped:
        raise ValueError(message("导出最低分数过滤后没有可导出的图片：{count} 张批量接受的图片没有达到该分数的框", count=len(skipped)))
    if not plan:
        raise ValueError("没有已接受或已修改的图片可导出；待审和已拒绝的图片不会导出")
    return target, plan, skipped


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _commit(staging, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, target)


def _staging_dir(target):
    staging = target.with_name(target.name + ".partial-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    return staging


def build_manifest(session, fmt, min_score, plan, skipped, classes, files):
    counts = session.counts()
    used_runs = sorted({entry.run_index for entry, _, _ in plan if entry.run_index is not None})
    images = []
    for entry, boxes, dropped in plan:
        origins = {"model": 0, "model_edited": 0, "human": 0}
        for box in boxes:
            origins[box_origin(box, entry.candidates)] += 1
        images.append({"file": entry.relative, "source_sha256": entry.sha256,
                       "width": entry.width, "height": entry.height, "state": entry.state,
                       "review_method": entry.review_method, "bulk_min_score": entry.bulk_min_score,
                       "run_index": entry.run_index, "candidate_count": len(entry.candidates),
                       "exported_boxes": len(boxes), "filtered_by_min_score": dropped["filtered_by_min_score"],
                       "zero_area_dropped": dropped["zero_area"], "box_origins": origins})
    return {
        "schema": "ee6008-auto-annotation/1",
        "format": fmt,
        "created_at": now_iso(),
        "label_provenance": LABEL_PROVENANCE,
        "label_provenance_zh": "模型候选框经人工审核",
        "human_ground_truth": False,
        "ground_truth_note": GROUND_TRUTH_NOTE,
        "review": {
            "exported_images": len(plan),
            "accepted_per_image": sum(e.state == "accepted" and e.review_method == "per_image" for e, _, _ in plan),
            "accepted_in_bulk_by_score": sum(e.review_method == "bulk_min_score" for e, _, _ in plan),
            "edited": sum(e.state == "edited" for e, _, _ in plan),
            "bulk_accepted_not_exported_no_box_at_min_score": len(skipped),
            "rejected_not_exported": counts["rejected"],
            "pending_not_exported": counts["pending"],
            "bulk_note": "Bulk-accepted images were accepted by score threshold without opening each image.",
        },
        "export_filter": {"min_score": float(min_score),
                          "applies_to": "boxes in images accepted in bulk by score, which nobody opened; images a person "
                                        "accepted or edited are exported exactly as reviewed",
                          "bulk_images_without_box_at_min_score": skipped,
                          "empty_after_filter_note": "A bulk-accepted image left with no box is not exported: an "
                                                     "empty label would claim it has no objects, which nobody checked."},
        "source_folder": str(session.folder),
        "recursive_scan": session.recursive,
        "session_created_at": session.created_at,
        "classes": classes,
        "detection_runs": session.runs,
        "runs_used_by_exported_images": used_runs,
        "code_sha256": code_fingerprint(),
        "files": files,
        "images": images,
        "license_note": "Exported image copies keep the licences of the source images; this tool grants none.",
    }


def _copy_image(entry, root):
    destination = root / "images" / entry.relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(entry.path, destination)
    if sha256_file(destination) != entry.sha256:
        raise ValueError(message("源图片在审核期间被修改：{name}", name=entry.relative))
    return destination


def export_yolo(session, target, min_score=0.0):
    """images/ copies + labels/*.txt, classes.txt, images.txt and data.yaml."""
    target, plan, skipped = _plan(session, target, min_score)
    labels = {}
    for entry, _, _ in plan:
        # Ultralytics finds a label by replacing the LAST "/images/" of the image
        # path with "/labels/", so images/images/x.jpg would be read from
        # images/labels/x.txt, a file this export never writes (silent background).
        if "images" in Path(entry.relative).parts[:-1]:
            raise ValueError(message("YOLO 导出不支持名为 images 的子文件夹：{name}。Ultralytics 会到别处找它的标签；请直接选择该 images 文件夹，或改用 COCO 导出", name=entry.relative))
        # a.jpg and a.png would both need labels/a.txt (also for Ultralytics).
        key = str(Path(entry.relative).with_suffix(".txt")).casefold()
        if key in labels:
            raise ValueError(message("YOLO 标签文件名冲突：{first} 与 {second} 会写入同一个 .txt", first=labels[key], second=entry.relative))
        labels[key] = entry.relative
    classes = session.export_classes(plan)
    ids = {name: index for index, name in enumerate(classes)}
    staging = _staging_dir(target)
    try:
        listed, boxes_written = [], 0
        for entry, boxes, _ in plan:
            _copy_image(entry, staging)
            lines = [line for line in (yolo_line(ids[box["label"]], box["xyxy"], entry.width, entry.height)
                                       for box in boxes) if line]
            boxes_written += len(lines)
            label_path = (staging / "labels" / entry.relative).with_suffix(".txt")
            _write_text(label_path, "".join(line + "\n" for line in lines))
            listed.append("./images/" + entry.relative)
        _write_text(staging / "classes.txt", "".join(name + "\n" for name in classes))
        _write_text(staging / "images.txt", "".join(path + "\n" for path in listed))
        names = "".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}\n" for index, name in enumerate(classes))
        _write_text(staging / "data.yaml", (
            "# Auto-annotation export: model candidates reviewed by a human (not independent ground truth).\n"
            "# train and val list the same images only to satisfy the Ultralytics format; this is NOT a validation split.\n"
            "train: images.txt\nval: images.txt\n"
            f"nc: {len(classes)}\nnames:\n{names}"))
        files = {"images": "images/", "labels": "labels/", "classes": "classes.txt",
                 "image_list": "images.txt", "data_yaml": "data.yaml"}
        _write_json(staging / MANIFEST_NAME, build_manifest(session, "yolo", min_score, plan, skipped, classes, files))
        _commit(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    session.dirty = False
    return {"path": target, "images": len(plan), "boxes": boxes_written, "classes": classes, "skipped": len(skipped)}


def export_coco(session, target, min_score=0.0):
    """images/ copies + annotations.json (category IDs from 1) + manifest."""
    target, plan, skipped = _plan(session, target, min_score)
    classes = session.export_classes(plan)
    ids = {name: index + 1 for index, name in enumerate(classes)}
    staging = _staging_dir(target)
    try:
        created = now_iso()
        images, annotations = [], []
        for image_id, (entry, boxes, _) in enumerate(plan, 1):
            _copy_image(entry, staging)
            images.append({"id": image_id, "file_name": entry.relative, "width": entry.width,
                           "height": entry.height, "license": 0, "date_captured": ""})
            for box in boxes:
                bbox = coco_bbox(box["xyxy"], entry.width, entry.height)
                if bbox is None:
                    continue
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": ids[box["label"]],
                    "bbox": bbox, "area": round(bbox[2] * bbox[3], 3), "iscrowd": 0, "segmentation": [],
                    "attributes": {"origin": box_origin(box, entry.candidates), "model_score": box["score"]}})
        coco = {
            "info": {"description": "EE6008 auto-annotation export: " + LABEL_PROVENANCE,
                     "version": "1.0", "year": datetime.now().year, "contributor": "EE6008 object-detection app",
                     "date_created": created, "url": ""},
            "licenses": [{"id": 0, "name": "Source image licence (unchanged; see source)", "url": ""}],
            "images": images, "annotations": annotations,
            "categories": [{"id": ids[name], "name": name, "supercategory": "object"} for name in classes],
        }
        _write_json(staging / "annotations.json", coco)
        files = {"images": "images/", "annotations": "annotations.json"}
        _write_json(staging / MANIFEST_NAME, build_manifest(session, "coco", min_score, plan, skipped, classes, files))
        _commit(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    session.dirty = False
    return {"path": target, "images": len(plan), "boxes": len(annotations), "classes": classes, "skipped": len(skipped)}


def detect_candidates(engine, frame, options, enhancer=None):
    """One image through the app's shared, guarded detection path.

    No tracker: ByteTrack numbers (person1, ...) are not classes. Engines run
    their own resource preflight; failures propagate without retry or device switch.
    """
    from feedback import prediction_boxes
    engine.imgsz = options["imgsz"]
    low_light = enhancer if options.get("low_light") else None
    if low_light is not None:
        low_light.reset()  # independent images: no temporal smoothing between files
    _, detections, status = engine.process_frame(
        frame, options["conf"], options["iou"], list(options["class_names"]), None,
        enhancer=low_light, enhancement_strength=options.get("enhancement_strength", 0.65),
        supplement_dark=options.get("supplement_dark", True), tiled=options.get("tiled", False),
        sliced=options.get("sliced", False), slice_size=options.get("slice_size", 512),
        slice_overlap=options.get("slice_overlap", 0.2), max_slices=options.get("max_slices", 9),
        tracker=None, max_det=options["max_det"])
    return prediction_boxes(detections, frame.shape[1], frame.shape[0]), status
