"""Local, append-only human corrections and versioned training datasets.

No training or model import occurs here. Predictions are suggestions, not GT.
"""
import hashlib
import json
import math
import os
import re
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from i18n import message
from model_families import MODEL_FAMILIES, record_family

PROJECT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT / "feedback_data"
SPLITS = ("train", "val", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_text(value, what="名称"):
    value = str(value).strip()
    if not value or len(value) > 120 or any(ord(c) < 32 for c in value):
        raise ValueError(message("{what}不能为空、过长或包含控制字符", what=message(what)))
    return value


def checked_classes(classes):
    classes = list(dict.fromkeys(checked_text(x, "类别") for x in classes))
    if not 1 <= len(classes) <= 80:
        raise ValueError("请填写 1 到 80 个类别，英文名称更适合文字提示训练")
    return classes


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 \
            or image.shape[2] != 3 or min(image.shape[:2]) < 2 or image.shape[0] * image.shape[1] > 16_000_000:
        raise ValueError("需要不超过 1600 万像素的 uint8 BGR 原图")


def image_id(image):
    validate_image(image)
    digest = hashlib.sha256(str(image.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image)))
    return digest.hexdigest()


def checked_boxes(boxes, width, height):
    if len(boxes) > 1000:
        raise ValueError("单张图最多保存 1000 个框")
    result = []
    for item in boxes:
        label = checked_text(item["label"], "类别")
        xyxy = [float(x) for x in item["xyxy"]]
        if len(xyxy) != 4 or not all(math.isfinite(x) for x in xyxy):
            raise ValueError("框坐标必须是四个有限数值")
        x1, y1, x2, y2 = xyxy
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("框必须完整位于原图内，且宽高大于零")
        result.append({"label": label, "xyxy": xyxy})
    return result


def source_group(kind, source, session=""):
    # One camera session, video or image directory cannot straddle splits.
    identity = session if kind == "camera" else os.path.normcase(str(Path(str(source)).resolve() if kind == "video" else Path(str(source)).resolve().parent))
    return kind + ":" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def prediction_boxes(detections, width, height):
    result = []
    for item in detections:
        x1, y1, x2, y2 = item.xyxy
        box = [max(0., min(width, float(x1))), max(0., min(height, float(y1))),
               max(0., min(width, float(x2))), max(0., min(height, float(y2)))]
        if all(math.isfinite(x) for x in item.xyxy) and box[0] < box[2] and box[1] < box[3]:
            result.append({"label": item.label, "xyxy": box, "score": float(item.score), "source": item.source})
    return result


def safe_path(root, relative):
    root = Path(root).absolute()
    path = root / relative
    def is_link(item):
        try:
            return item.is_symlink() or bool(getattr(item.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        except FileNotFoundError:
            return False
    if is_link(root):
        raise ValueError("数据目录不能是链接")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("路径越出了数据目录")
    for part in (root, *path.relative_to(root).parents, path):
        candidate = part if part.is_absolute() else root / part
        if is_link(candidate):
            raise ValueError("数据目录中不能包含链接")
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class FeedbackStore:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).absolute()

    def records(self, family=None):
        directory = safe_path(self.root, "records")
        if not directory.exists():
            return []
        latest = {}
        for path in sorted(directory.glob("*.json")):
            safe_path(self.root, path.relative_to(self.root))
            value = json.loads(path.read_text(encoding="utf-8"))
            if not re.fullmatch(r"[0-9a-f]{64}", value.get("image_id", "")):
                raise ValueError("样本编号损坏")
            if value.get("split") not in SPLITS:
                raise ValueError("样本用途损坏")
            value["record_path"] = str(path)
            value["model_family"] = record_family(value)
            latest[(value["model_family"], value["image_id"])] = value
        return [row for row in latest.values() if family is None or row["model_family"] == family]

    def load_image(self, record):
        path = safe_path(self.root, "images/" + record["image_id"] + ".png")
        if sha256_file(path) != record["image_sha256"]:
            raise ValueError("样本原图被修改，停止读取")
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image_id(image) != record["image_id"]:
            raise ValueError("样本内容校验失败")
        return image

    def save(self, image, boxes, *, classes, scene, split, reviewed, provenance, candidates=()):
        validate_image(image)
        classes = checked_classes(classes)
        scene = checked_text(scene, "场景组")
        if split not in SPLITS or not isinstance(reviewed, bool):
            raise ValueError("请选择正确用途和人工审核状态")
        boxes = checked_boxes(boxes, image.shape[1], image.shape[0])
        if any(box["label"] not in classes for box in boxes):
            raise ValueError("框的类别不在本次审核类别中；请补充类别列表后重新检查整图")
        origin = checked_text(provenance.get("origin_group", ""), "来源组")
        source = Path(str(provenance.get("source", ""))).resolve()
        protected = (PROJECT / "datasets/mini_eval", PROJECT / "datasets/mini_development_quick")
        if split != "test" and any(source.is_relative_to(path) for path in protected):
            raise ValueError("已有开发诊断图与视觉参考不能加入训练/验证，请使用新的个人场景照片")
        identity = image_id(image)
        existing = self.records()
        for record in existing:
            if (record["image_id"] == identity or record["scene"] == scene or record["provenance"]["origin_group"] == origin) and record["split"] != split:
                raise ValueError("同一原图、视频/来源组或场景不能跨训练、验证、测试用途；请另拍独立场景")
        return self._write_record(image, boxes, classes, scene, split,
                                  "reviewed" if reviewed else "draft", provenance, candidates)

    def _write_record(self, image, boxes, classes, scene, split, status, provenance, candidates, purpose="strict"):
        identity = image_id(image)
        image_path = safe_path(self.root, "images/" + identity + ".png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise OSError("原图 PNG 编码失败")
            temporary = image_path.with_name(identity + "." + uuid.uuid4().hex + ".tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(encoded.tobytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, image_path)
            finally:
                temporary.unlink(missing_ok=True)
        stored_image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
        if image_id(stored_image) != identity:
            raise ValueError("已保存原图被修改，拒绝建立新修订")
        revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + uuid.uuid4().hex[:8]
        record = {"schema": 2, "revision": revision, "image_id": identity,
                  "model_family": record_family({"provenance": provenance}),
                  "image_sha256": sha256_file(image_path), "width": image.shape[1], "height": image.shape[0],
                  "classes": classes, "scene": scene, "split": split,
                  "status": status, "purpose": purpose, "boxes": boxes,
                  "candidates": list(candidates), "provenance": provenance,
                  "notice": "演示纠错：点击更新采用当前框，不是独立评价真值。" if purpose == "correction_demo" else
                            "候选框不是人工真值；reviewed 是用户对整张图片和列出的全部类别明确确认。"}
        path = safe_path(self.root, "records/" + revision + ".json")
        atomic_json(path, record)
        return record

    def save_demo(self, image, boxes, *, classes, provenance, candidates=(), accepted=True):
        """Separate ledger: explicit same-image fitting, never relabel strict data."""
        validate_image(image)
        classes = checked_classes(classes)
        boxes = checked_boxes(boxes, image.shape[1], image.shape[0])
        if any(box["label"] not in classes for box in boxes):
            raise ValueError("框的类别不在本次审核类别中；请补充类别列表后重新检查整图")
        provenance = dict(provenance)
        if accepted and not record_family({"provenance": provenance}):
            raise ValueError("请选择支持纠错的 YOLOE 基础型号")
        provenance.setdefault("origin_group", "demo:" + image_id(image)[:24])
        scene = "demo:" + image_id(image)[:24]
        return FeedbackStore(safe_path(self.root, "demo"))._write_record(
            image, boxes, classes, scene, "train", "demo_accepted" if accepted else "demo_draft",
            provenance, candidates, purpose="correction_demo")

    def demo_records(self, family=None):
        return FeedbackStore(safe_path(self.root, "demo")).records(family)

    def export_demo(self, family):
        """Replay accepted demo edits + strict training samples, no holdout claims."""
        if family not in MODEL_FAMILIES:
            raise ValueError("请选择支持纠错的 YOLOE 基础型号")
        pairs = {row["image_id"]: (row, self) for row in self.records(family)
                 if row["status"] == "reviewed" and row["split"] == "train"}
        demo_store = FeedbackStore(safe_path(self.root, "demo"))
        for row in demo_store.records(family):
            # A draft supersedes this image's older accepted revision, not the
            # weights already committed. Only another Update click accepts it.
            pairs.pop(row["image_id"], None)
            if row["status"] == "demo_accepted":
                pairs[row["image_id"]] = (row, demo_store)
        if not pairs:
            raise ValueError("请先打开图片并修改框，再点击一键更新。")
        pairs = list(pairs.values())
        classes = checked_classes(list(dict.fromkeys(name for row, _ in pairs for name in row["classes"])))
        for row, store in pairs:
            store.load_image(row)
        return self._export_records(classes, pairs, family=family, per_sample_scope=True,
                                    mode="correction_demo")

    def export_cumulative(self, family):
        if family not in MODEL_FAMILIES:
            raise ValueError("请选择支持纠错的 YOLOE 基础型号")
        rows = [row for row in self.records(family) if row["status"] == "reviewed"]
        if not rows:
            raise ValueError("至少需要两个独立场景：一组已人工审核的训练图和一组验证图。草稿不会导出。")
        classes = list(dict.fromkeys(name for row in rows for name in row["classes"]))
        return self.export(classes, family=family, per_sample_scope=True)

    def export(self, classes, *, family=None, per_sample_scope=False):
        classes = checked_classes(classes)
        all_records = self.records(family)
        records = [row for row in all_records if row["status"] == "reviewed"]
        if not records or not all(any(row["split"] == split for row in records) for split in ("train", "val")):
            raise ValueError("至少需要两个独立场景：一组已人工审核的训练图和一组验证图。草稿不会导出。")
        for row in records:
            if not per_sample_scope and not set(classes).issubset(row["classes"]):
                raise ValueError("部分图片尚未审核本次全部类别；先重新检查并保存，避免漏标被当作背景")
            self.load_image(row)
        groups = {}
        for row in records:
            for group in ("scene:" + row["scene"], "origin:" + row["provenance"]["origin_group"]):
                if groups.setdefault(group, row["split"]) != row["split"]:
                    raise ValueError("样本组跨用途，拒绝导出")
        return self._export_records(classes, [(row, self) for row in records], family=family,
                                    per_sample_scope=per_sample_scope, groups=groups,
                                    excluded_drafts=len(all_records)-len(records))

    def _export_records(self, classes, pairs, *, family, per_sample_scope, groups=None,
                        excluded_drafts=0, mode="strict"):
        # Export has its own RAM/disk preflight; inference's budget is not used.
        from resource_safety import available_ram_mb
        if available_ram_mb() < 1536:
            raise RuntimeError("导出停止：系统可用内存不足 1536 MiB")
        existing = self.root
        while not existing.exists():
            existing = existing.parent
        estimated = sum(safe_path(store.root, "images/" + row["image_id"] + ".png").stat().st_size for row, store in pairs)
        if shutil.disk_usage(existing).free < estimated + 1024**3:
            raise RuntimeError("导出停止：磁盘空间不足，需要保留至少 1 GiB")
        records = [row for row, _ in pairs]
        version = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        output = safe_path(self.root, "datasets/" + version)
        output.mkdir(parents=True, exist_ok=False)
        files = {}
        for row, store in pairs:
            relative = f"images/{row['split']}/{row['image_id']}.png"
            target = safe_path(output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(store.root, "images/" + row["image_id"] + ".png"), target)
            boxes = checked_boxes(row["boxes"], row["width"], row["height"])
            lines = []
            for box in boxes:
                if box["label"] not in classes:
                    continue
                x1, y1, x2, y2 = box["xyxy"]
                values = ((x1+x2)/2/row["width"], (y1+y2)/2/row["height"], (x2-x1)/row["width"], (y2-y1)/row["height"])
                lines.append(str(classes.index(box["label"])) + " " + " ".join(f"{x:.10f}" for x in values))
            label_relative = f"labels/{row['split']}/{row['image_id']}.txt"
            label_path = safe_path(output, label_relative)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            for name in (relative, label_relative):
                files[name] = sha256_file(output / name)
        config = {"path": str(output), "train": "images/train", "names": classes}
        if mode == "strict":
            config["val"] = "images/val"
        if any(row["split"] == "test" for row in records):
            config["test"] = "images/test"
        # JSON is valid YAML and does not require an extra dependency.
        atomic_json(output / "data.yaml", config)
        files["data.yaml"] = sha256_file(output / "data.yaml")
        atomic_json(output / "manifest.json", {"schema": 3 if mode == "correction_demo" else 2 if per_sample_scope else 1, "version": version, "classes": classes, "files": files,
            "training_mode": mode, "independent_validation": mode != "correction_demo",
            "model_family": family, "per_sample_review_scope": bool(per_sample_scope),
            "samples": records, "excluded_drafts": excluded_drafts, "groups": groups or {},
            "code_sha256": sha256_file(Path(__file__)),
            "notice": "同图训练演示；只用训练损失选候选，没有独立验证，不作为泛化评价。" if mode == "correction_demo" else
                      "仅人工审核的纠错记录；不把模型预测当作 GT；测试组不参与训练或选模型。"})
        (output/"manifest.sha256").write_text(sha256_file(output/"manifest.json"), encoding="ascii")
        return output

    def model_versions(self):
        """Compatibility API: at most one visible modified model per family."""
        from modified_models import ModifiedModels
        return ModifiedModels(self.root).listed()
