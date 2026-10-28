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
import uuid
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
        """Rename in every reviewed box and the class list; candidates stay as predicted."""
        new = normalize_box({"label": new, "xyxy": [0, 0, 1, 1]})["label"]
        changed = 0
        for entry in self.entries:
            for box in entry.boxes:
                if box["label"] == old:
                    box["label"] = new
                    changed += 1
            if entry.exported:
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
        """Pending, untouched images that lack candidates for this exact configuration."""
        result = []
        for index, entry in enumerate(self.entries):
            if entry.state != "pending" or entry.modified:
                continue
            same = entry.detected and entry.run_index is not None and \
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
        entry = self.entries[index]
        if len(boxes) > MAX_BOXES:
            raise ValueError("单张图最多保存 1000 个框")
        untouched = entry.state == "pending" and not entry.modified
        self.set_image_info(index, width, height, sha256)
        entry.candidates = [normalize_box(box, width, height) for box in boxes]
        entry.detected, entry.error, entry.run_index = True, None, run_index
        if run_index is not None:
            self.runs[run_index]["processed"] = self.runs[run_index].get("processed", 0) + 1
        if untouched:
            entry.boxes = deepcopy(entry.candidates)
        return untouched

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
        """Edited images are exported as reviewed; the score filter only trims
        untouched model boxes in images accepted without edits."""
        boxes = []
        for box in entry.boxes:
            if entry.state == "accepted" and box_origin(box, entry.candidates) == "model" \
                    and box["score"] is not None and box["score"] < min_score:
                continue
            boxes.append(box)
        return boxes

    def export_classes(self, plan):
        names = list(self.classes)
        for _, boxes in plan:
            for box in boxes:
                if box["label"] not in names:
                    names.append(box["label"])
        return names


def _plan(session, target, min_score):
    min_score = float(min_score)
    if not 0 <= min_score <= 1:
        raise ValueError("置信度和 IoU 必须在 0 到 1 之间")
    target = Path(os.path.abspath(target))
    if target == session.folder or target.is_relative_to(session.folder):
        raise ValueError("导出目录不能位于源图片文件夹内，原始图片保持不变")
    if target.exists():
        raise FileExistsError(message("导出目录已存在，不覆盖旧导出：{path}", path=str(target)))
    plan = []
    for entry in session.entries:
        if not entry.exported:
            continue
        if entry.width is None or not entry.sha256:
            raise ValueError(message("图片尚未读取，不能导出：{name}", name=entry.relative))
        if sha256_file(entry.path) != entry.sha256:
            raise ValueError(message("源图片在审核期间被修改：{name}", name=entry.relative))
        plan.append((entry, session.export_boxes(entry, min_score)))
    if not plan:
        raise ValueError("没有已接受或已修改的图片可导出；待审和已拒绝的图片不会导出")
    return target, plan


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


def build_manifest(session, fmt, min_score, plan, classes, files):
    counts = session.counts()
    used_runs = sorted({entry.run_index for entry, _ in plan if entry.run_index is not None})
    images = []
    for entry, boxes in plan:
        origins = {"model": 0, "model_edited": 0, "human": 0}
        for box in boxes:
            origins[box_origin(box, entry.candidates)] += 1
        images.append({"file": entry.relative, "source_sha256": entry.sha256,
                       "width": entry.width, "height": entry.height, "state": entry.state,
                       "review_method": entry.review_method, "bulk_min_score": entry.bulk_min_score,
                       "run_index": entry.run_index, "candidate_count": len(entry.candidates),
                       "exported_boxes": len(boxes), "box_origins": origins})
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
            "accepted_per_image": sum(e.state == "accepted" and e.review_method == "per_image" for e, _ in plan),
            "accepted_in_bulk_by_score": sum(e.review_method == "bulk_min_score" for e, _ in plan),
            "edited": sum(e.state == "edited" for e, _ in plan),
            "rejected_not_exported": counts["rejected"],
            "pending_not_exported": counts["pending"],
            "bulk_note": "Bulk-accepted images were accepted by score threshold without opening each image.",
        },
        "export_filter": {"min_score": float(min_score),
                          "applies_to": "untouched model boxes in images accepted without edits; edited images are exported exactly as reviewed"},
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
    target, plan = _plan(session, target, min_score)
    classes = session.export_classes(plan)
    ids = {name: index for index, name in enumerate(classes)}
    staging = _staging_dir(target)
    try:
        listed, boxes_written = [], 0
        for entry, boxes in plan:
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
        _write_json(staging / MANIFEST_NAME, build_manifest(session, "yolo", min_score, plan, classes, files))
        _commit(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    session.dirty = False
    return {"path": target, "images": len(plan), "boxes": boxes_written, "classes": classes}


def export_coco(session, target, min_score=0.0):
    """images/ copies + annotations.json (category IDs from 1) + manifest."""
    target, plan = _plan(session, target, min_score)
    classes = session.export_classes(plan)
    ids = {name: index + 1 for index, name in enumerate(classes)}
    staging = _staging_dir(target)
    try:
        created = now_iso()
        images, annotations = [], []
        for image_id, (entry, boxes) in enumerate(plan, 1):
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
        _write_json(staging / MANIFEST_NAME, build_manifest(session, "coco", min_score, plan, classes, files))
        _commit(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    session.dirty = False
    return {"path": target, "images": len(plan), "boxes": len(annotations), "classes": classes}


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
