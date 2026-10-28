"""Sequential COCO evaluation of text/visual prompts and SAHI slicing.

Importing this module or requesting --help does not import torch, load models,
or access a GPU. Test ground truth is used only for evaluation/illustrations;
visual prompts come exclusively from the separate reference images.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODES = ("text", "text_sahi", "visual", "visual_sahi")
METRIC_NAMES = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
                "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")
DEFAULT_TEXT_CONF = 0.05
DEFAULT_VISUAL_CONF = 0.001
DEFAULT_GUI_CONF = 0.40
GUI_MATCH_IOU = 0.5
NOTICE = (
    "Small human-annotated COCO subset (the default 24-image set is a diagnostic set), not a COCO "
    "leaderboard result or a camera-domain validation. "
    "Visual references and evaluation image IDs are disjoint. Visual inference uses one reference "
    "class per pass; its per-image latency sums all class passes. Detection timings include "
    "preprocessing, forward, postprocessing, SAHI fusion and result transfer, excluding image "
    "decoding, prompt encoding, warmup, loading and drawing. Prompt encoding is reported separately. "
    "COCO uses its standard maxDets=[1,10,100]; the same per-image application cap applies to every "
    "mode, after combining all visual classes. AP prediction thresholds: text modes use --conf "
    "(default 0.05, the earlier protocol); visual modes use --visual-conf (default 0.001), because "
    "single-reference SAVPE scores sit about an order of magnitude below text scores and a 0.05 cut "
    "truncates most of their precision-recall curve. Precision / recall / F1 count only boxes scoring "
    "at least --gui-conf (default 0.40, the GUI default) and match them to human boxes of the same "
    "class at IoU 0.50 with COCO's greedy matching (crowd regions ignored)."
)


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("必须是 0 到 1 之间的有限数值")
    return number


def parse_modes(value):
    modes = [item.strip() for item in value.split(",")]
    if not modes or len(set(modes)) != len(modes) or any(item not in MODES for item in modes):
        raise argparse.ArgumentTypeError("模式必须是 text,text_sahi,visual,visual_sahi 中不重复的逗号分隔列表")
    return modes


def build_parser():
    parser = argparse.ArgumentParser(description="小型人工标注集上的文本/视觉提示和 SAHI 对照；一次只加载一个模型。")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/mini_eval"))
    parser.add_argument("--annotations", type=Path,
                        help="COCO 人工标注 JSON，默认 <dataset>/annotations.json；例如 240 图子集的 instances_subset.json")
    parser.add_argument("--images", type=Path, help="评测图片目录，默认 <dataset>/images")
    parser.add_argument("--references", type=Path,
                        help="视觉参考 references.json，参考图路径相对它所在的目录；默认 <dataset>/references.json。"
                             "只跑文本模式时可以没有")
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--device", type=str.upper, choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--modes", type=parse_modes, default=list(MODES))
    parser.add_argument("--output", type=Path, default=Path("work/mini_evaluation"))
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--conf", type=probability, default=DEFAULT_TEXT_CONF,
                        help="文本模式的预测阈值（原协议 0.05）")
    parser.add_argument("--visual-conf", type=probability, default=DEFAULT_VISUAL_CONF,
                        help="视觉模式的预测阈值；SAVPE 单参考的分数比文本低一个数量级，0.05 会截掉大部分真阳性")
    parser.add_argument("--gui-conf", type=probability, default=DEFAULT_GUI_CONF,
                        help="界面阈值：另算这个分数以上的精确率 / 召回率 / F1（IoU 0.50）；不能低于预测阈值")
    parser.add_argument("--iou", type=probability, default=0.5)
    parser.add_argument("--max-det", type=positive_integer, default=100)
    parser.add_argument("--slice-size", type=positive_integer, default=320)
    parser.add_argument("--slice-overlap", type=probability, default=0.2)
    parser.add_argument("--max-slices", type=positive_integer, default=9)
    parser.add_argument("--limit", type=positive_integer, help="仅评测前 N 张图片；报告会注明实际数量")
    parser.add_argument("--warmup", type=positive_integer, default=1)
    parser.add_argument("--samples", type=positive_integer, default=3, help="每个模式保存几张真实框/预测框并排示例")
    return parser


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"数据文件必须位于数据集目录内：{relative}")
    if not candidate.is_file():
        raise ValueError(f"数据文件不存在：{candidate}")
    return candidate


def image_path(root, item):
    name = item["file_name"]
    return local_path(root, name if name.startswith("images/") else "images/" + name)


def dataset_image_path(dataset, item):
    """Image path inside the dataset's image folder (default <dataset>/images)."""
    name = item["file_name"]
    return local_path(dataset["image_dir"], name[len("images/"):] if name.startswith("images/") else name)


def reference_path(dataset, reference):
    """Reference image path, relative to the folder holding references.json."""
    return local_path(dataset["reference_root"], reference["image_file"])


def existing_file(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"{label}不存在：{path}")
    return path


def load_dataset(root, limit=None, annotations=None, images=None, references=None, require_references=True):
    """Load human COCO GT plus visual references; defaults keep the <dataset>/ layout and fingerprint.

    ``annotations`` / ``images`` / ``references`` point at another human-annotated COCO
    source (for example the held-out 240-image subset). Without a references file only
    text modes can run (``require_references=False``); classes then follow category ID order.
    """
    root = Path(root).resolve()
    custom_source = annotations is not None or images is not None
    annotation_file = existing_file(annotations, "标注文件") if annotations is not None \
        else local_path(root, "annotations.json")
    image_dir = Path(images).resolve() if images is not None else root / "images"
    if not image_dir.is_dir():
        raise ValueError(f"图片目录不存在：{image_dir}")
    # A custom --annotations/--images source has its own (possibly absent) references; it must
    # never silently fall back to the default --dataset's references.json (wrong classes).
    if references is not None:
        reference_file = Path(references).resolve()
    elif custom_source:
        reference_file = None
    else:
        reference_file = root / "references.json"
    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = annotations.get("images", [])
    categories = annotations.get("categories", [])
    category_names = {int(item["id"]): item["name"] for item in categories}
    if reference_file is not None and reference_file.is_file():
        reference_root = reference_file.parent
        if references is None:
            reference_file = local_path(root, "references.json")
        reference_data = json.loads(reference_file.read_text(encoding="utf-8"))
        classes = reference_data.get("classes", [])
        references = sorted(reference_data.get("references", []), key=lambda item: item["class_id"])
    elif require_references or references is not None:
        hint = str(reference_file) if reference_file is not None else "--references"
        raise ValueError(f"视觉模式需要独立参考文件 references.json（{hint}）")
    else:
        reference_file, reference_root, references = None, None, []
        classes = [category_names[key] for key in sorted(category_names)]
    ids = [int(item["id"]) for item in images]
    if not images or len(set(ids)) != len(ids):
        raise ValueError("评测图片必须非空且 ID 唯一")
    if reference_file is not None and (not classes or len(classes) != len(references)
                                       or len(set(classes)) != len(classes)):
        raise ValueError("每个目标类别必须有且只有一个独立视觉参考")
    if not classes or len(set(classes)) != len(classes) or len(category_names) != len(classes) \
            or set(category_names.values()) != set(classes):
        raise ValueError("参考类别和 COCO 评价类别必须一致")
    used_reference_ids = set()
    for index, reference in enumerate(references):
        category = int(reference["category_id"])
        if int(reference["class_id"]) != index or reference["class_name"] != classes[index] \
                or category_names.get(category) != classes[index]:
            raise ValueError("视觉参考的 class_id/class_name/category_id 映射不一致")
        original_id = int(reference["original_image_id"])
        if original_id in ids or original_id in used_reference_ids:
            raise ValueError("视觉参考图必须与所有评测图片完全分离，不能从测试真值取提示")
        used_reference_ids.add(original_id)
        local_path(reference_root, reference["image_file"])
        box = reference["bbox_xyxy"]
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box) \
                or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("视觉参考框无效")
    layout = {"image_dir": image_dir, "reference_root": reference_root}
    for item in images:
        dataset_image_path(layout, item)
    selected = images[:limit] if limit is not None else images
    # Default keys are unchanged, so the default dataset keeps its earlier fingerprint.
    hashes = {"annotations.json" if annotation_file == root / "annotations.json" else annotation_file.name:
              sha256_file(annotation_file)}
    if reference_file is not None:
        hashes["references.json" if reference_file == root / "references.json" else reference_file.name] = \
            sha256_file(reference_file)
    if not custom_source and (root / "manifest.json").is_file():
        hashes["manifest.json"] = sha256_file(root / "manifest.json")
    for item in selected:
        path = dataset_image_path(layout, item)
        hashes["images/" + path.relative_to(image_dir).as_posix()] = sha256_file(path)
    for reference in references:
        hashes[reference["image_file"]] = sha256_file(local_path(reference_root, reference["image_file"]))
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    category_ids = [int(item["category_id"]) for item in references] if references \
        else sorted(category_names)
    return {"root": root, "annotations": annotations, "images": selected,
            "all_image_count": len(images), "classes": classes, "references": references,
            "category_ids": category_ids, "image_dir": image_dir, "reference_root": reference_root,
            "sources": {"annotations": str(annotation_file), "images": str(image_dir),
                        "references": None if reference_file is None else str(reference_file)},
            "sha256": fingerprint, "file_hashes": hashes}


def decode_image(path):
    import cv2
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise ValueError(f"图片解码失败：{path}")
    return frame


def precision_recall_f1(tp, fp, positives):
    """Counts -> precision, recall and F1 = 2TP / (2TP + FP + FN); None when undefined."""
    return {"tp": int(tp), "fp": int(fp), "gt": int(positives),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positives if positives else None,
            "f1": 2 * tp / (tp + fp + positives) if tp + fp + positives else None}


def threshold_counts(evaluator, threshold, iou=GUI_MATCH_IOU):
    """Per-category TP / FP / GT at one score threshold from an evaluated COCOeval.

    Uses COCO's own greedy matching (area "all", maxDets 100, crowd and ignored boxes
    excluded). Matching runs in score order, so keeping only boxes >= threshold after
    matching gives the same result as matching only those boxes.
    """
    import numpy as np
    params = evaluator.params
    level = int(np.flatnonzero(np.isclose(params.iouThrs, iou))[0])
    area = list(params.areaRngLbl).index("all")
    count_a, count_i = len(params.areaRng), len(params.imgIds)
    counts = {}
    for k, category in enumerate(params.catIds):
        tp = fp = positives = 0
        for i in range(count_i):
            entry = evaluator.evalImgs[k * count_a * count_i + area * count_i + i]
            if entry is None:
                continue
            positives += int(np.count_nonzero(np.asarray(entry["gtIgnore"]) == 0))
            if len(entry["dtScores"]):
                kept = np.asarray(entry["dtScores"], dtype=float) >= threshold
                matched = np.asarray(entry["dtMatches"])[level] > 0
                ignored = np.asarray(entry["dtIgnore"])[level].astype(bool)
                tp += int(np.count_nonzero(kept & matched & ~ignored))
                fp += int(np.count_nonzero(kept & ~matched & ~ignored))
        counts[int(category)] = (tp, fp, positives)
    return counts


def coco_metrics(annotations, predictions, image_ids, gui_threshold=None):
    """Standard bbox COCO AP; preserve original area and iscrowd metadata.

    With ``gui_threshold`` also report precision / recall / F1 of the boxes scoring at
    least that value (IoU 0.50, same class), overall and per class.
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = annotations
        truth.createIndex()
        if predictions:
            detected = truth.loadRes([dict(record) for record in predictions])
        else:
            # COCO.loadRes([]) indexes anns[0]; an explicit empty result is valid.
            detected = COCO()
            detected.dataset = {"images": annotations["images"],
                                "categories": annotations["categories"], "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = sorted(int(item["id"]) for item in annotations["categories"])
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    def valid_mean(values):
        selected = values[values > -1]
        return float(selected.mean()) if selected.size else None

    metrics = {name: float(value) if value >= 0 else None
               for name, value in zip(METRIC_NAMES, evaluator.stats)}
    class_results = {}
    names = {int(item["id"]): item["name"] for item in annotations["categories"]}
    at_50 = np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))
    at_75 = np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))
    for index, category_id in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        class_results[names[category_id]] = {
            "category_id": int(category_id), "AP": valid_mean(precision),
            "AP50": valid_mean(precision[at_50]), "AP75": valid_mean(precision[at_75]),
            "APsmall": valid_mean(evaluator.eval["precision"][:, :, index, 1, -1]),
            "AR100": valid_mean(evaluator.eval["recall"][:, index, 0, -1]),
        }
    result = {"metrics": metrics, "per_class": class_results, "coco_max_dets": [1, 10, 100]}
    if gui_threshold is not None:
        counts = threshold_counts(evaluator, gui_threshold)
        for category_id, values in counts.items():
            class_results[names[category_id]]["gui"] = precision_recall_f1(*values)
        totals = [sum(values[index] for values in counts.values()) for index in range(3)]
        result["gui_threshold"] = {"score_threshold": float(gui_threshold), "match_iou": GUI_MATCH_IOU,
                                   **precision_recall_f1(*totals)}
    return result


def mode_conf(args, mode):
    """AP prediction threshold of one mode: visual modes use --visual-conf, text modes --conf."""
    return args.visual_conf if mode.startswith("visual") else args.conf


def infer(engine, frame, args, sliced, classes, conf=None):
    common = {"conf": args.conf if conf is None else conf, "iou": args.iou,
              "class_names": classes, "max_det": args.max_det}
    if sliced:
        return engine.detect_sliced(frame, **common, slice_size=args.slice_size,
                                    slice_overlap=args.slice_overlap, max_slices=args.max_slices)
    return engine.detect(frame, source="original", **common)


def detection_records(detections, image_id, category_ids, visual_category=None):
    records = []
    for item in detections:
        x1, y1, x2, y2 = (float(value) for value in item.xyxy)
        score = float(item.score)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)) \
                or x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
            raise ValueError("模型返回了非法坐标或置信度")
        if visual_category is None:
            class_id = int(item.class_id)
            if not 0 <= class_id < len(category_ids):
                raise ValueError("模型类别 ID 与指定文本类别映射不一致")
            category = category_ids[class_id]
        else:
            category = visual_category
        records.append({"image_id": int(image_id), "category_id": int(category),
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return records


def cap_predictions(predictions, image_ids, max_det):
    by_image = {int(image_id): [] for image_id in image_ids}
    for record in predictions:
        by_image[int(record["image_id"])].append(record)
    return [item for image_id in image_ids
            for item in sorted(by_image[int(image_id)], key=lambda row: row["score"], reverse=True)[:max_det]]


def run_mode(engine, dataset, args, mode, synchronize=lambda: None,
             clock=time.perf_counter, decoder=decode_image):
    """Stream one frame at a time; visual passes use reference class -> images."""
    visual = mode.startswith("visual")
    sliced = mode.endswith("_sahi")
    conf = mode_conf(args, mode)
    image_ids = [int(item["id"]) for item in dataset["images"]]
    latency = {image_id: 0.0 for image_id in image_ids}
    predictions, reference_times = [], []
    phases = dataset["references"] if visual else [None]
    for reference in phases:
        category = None
        if reference is not None:
            reference_frame = decoder(reference_path(dataset, reference))
            synchronize()
            start = clock()
            engine.set_visual_prompt(reference_frame, reference["bbox_xyxy"], reference["class_name"])
            synchronize()
            reference_times.append({"class_name": reference["class_name"],
                                    "encoding_ms": (clock() - start) * 1000})
            del reference_frame
            category = int(reference["category_id"])
        classes = [reference["class_name"]] if reference is not None else dataset["classes"]
        first = decoder(dataset_image_path(dataset, dataset["images"][0]))
        for _ in range(args.warmup):
            infer(engine, first, args, sliced, classes, conf)
        synchronize()
        del first
        for index, item in enumerate(dataset["images"]):
            frame = decoder(dataset_image_path(dataset, item))
            if frame.shape[:2] != (int(item["height"]), int(item["width"])):
                raise ValueError(f"图片尺寸与标注不一致：{item['file_name']}")
            synchronize()
            start = clock()
            detections = infer(engine, frame, args, sliced, classes, conf)
            synchronize()
            latency[int(item["id"])] += (clock() - start) * 1000
            predictions.extend(detection_records(detections, item["id"], dataset["category_ids"], category))
            del detections, frame
            print(f"{mode} | {classes} | {index + 1}/{len(dataset['images'])}", flush=True)
        # Same result as one cap at the end (stable score order), but low visual thresholds
        # no longer keep every class pass's boxes in memory.
        predictions = cap_predictions(predictions, image_ids, args.max_det)
    latencies = [latency[image_id] for image_id in image_ids]
    ordered = sorted(latencies)
    p95_index = (len(ordered) - 1) * 0.95
    p95 = ordered[math.floor(p95_index)] + (ordered[math.ceil(p95_index)] - ordered[math.floor(p95_index)]) \
        * (p95_index - math.floor(p95_index))
    return {"predictions": predictions, "prediction_conf": conf,
            "timings": [{"image_id": image_id, "latency_ms": latency[image_id]} for image_id in image_ids],
            "timing_summary": {"mean_ms": statistics.mean(latencies), "median_ms": statistics.median(latencies),
                               "p95_ms": p95, "class_passes_per_image": len(phases)},
            "reference_encodings": reference_times}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_csv(path, rows, fields):
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def annotate_samples(output, mode, dataset, predictions, sample_count):
    import cv2
    import numpy as np
    folder = output / "samples"
    folder.mkdir(exist_ok=True)
    category_names = {int(item["id"]): item["name"] for item in dataset["annotations"]["categories"]}
    paths = []
    for item in dataset["images"][:sample_count]:
        frame = decode_image(dataset_image_path(dataset, item))
        truth, predicted = frame.copy(), frame.copy()
        entries = [(truth, dataset["annotations"]["annotations"], (80, 220, 80), "GT (human)", False),
                   (predicted, predictions, (0, 160, 255), "Prediction score >= 0.25", True)]
        for image, records, color, title, is_prediction in entries:
            cv2.rectangle(image, (0, 0), (image.shape[1], 28), (25, 25, 25), -1)
            cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for record in records:
                if int(record["image_id"]) != int(item["id"]) or is_prediction and record["score"] < 0.25:
                    continue
                x, y, width, height = record["bbox"]
                first, last = (round(x), round(y)), (round(x + width), round(y + height))
                label = category_names[int(record["category_id"])]
                if is_prediction:
                    label += f" {record['score']:.2f}"
                cv2.rectangle(image, first, last, color, 2)
                cv2.putText(image, label, (first[0], max(38, first[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        target = folder / f"{mode}_{item['id']:012d}.jpg"
        ok, encoded = cv2.imencode(".jpg", np.hstack([truth, predicted]))
        if not ok:
            raise ValueError("示例图片编码失败")
        target.write_bytes(encoded.tobytes())
        paths.append(str(target.relative_to(output)))
    return paths


GUI_FIELDS = ("precision", "recall", "f1", "tp", "fp", "gt")


def write_summary(output, report):
    rows = []
    for mode, result in report["modes"].items():
        if result.get("status") != "complete":
            continue
        gui = result.get("gui_threshold", {})
        rows.append({"mode": mode, "prediction_conf": result.get("prediction_conf"), **result["metrics"],
                     **{f"{name}_gui": gui.get(name) for name in GUI_FIELDS},
                     **result["timing_summary"], **result["memory"]})
    fields = ["mode", "prediction_conf", *METRIC_NAMES, *(f"{name}_gui" for name in GUI_FIELDS),
              "mean_ms", "median_ms", "p95_ms", "class_passes_per_image", "peak_allocated_mb", "peak_reserved_mb"]
    write_csv(output / "comparison.csv", rows, fields)
    settings = report["settings"]
    gui = f"{settings.get('gui_conf', DEFAULT_GUI_CONF):.2f}"
    lines = ["# 小型标注集对照结果", "", f"状态：{report['status']}；评测图片：{report['image_count']} / {report['all_image_count']}。", "",
             f"| 模式 | 预测阈值 | AP | AP50 | AP75 | APsmall | AR100 | P@{gui} | R@{gui} | F1@{gui} "
             "| 平均 ms/图 | 峰值分配 MiB | 峰值保留 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["mode"], "—" if row["prediction_conf"] is None else f"{row['prediction_conf']:g}"] + \
            ["—" if row[name] is None else f"{row[name]:.3f}" for name in
             ("AP", "AP50", "AP75", "APsmall", "AR100", "precision_gui", "recall_gui", "f1_gui",
              "mean_ms", "peak_allocated_mb", "peak_reserved_mb")]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "AP 与 AR 的范围是 0–1；缺少对应尺寸/类别真值时显示 —。", "",
                  f"AP 的预测阈值：文本模式 {settings.get('conf', DEFAULT_TEXT_CONF):g}（原协议），视觉模式 "
                  f"{settings.get('visual_conf', DEFAULT_VISUAL_CONF):g}。单参考视觉提示的真阳性分数大多低于 0.05，"
                  "0.05 会截掉它们的 PR 曲线；两种模式的阈值不同，文本与视觉的 AP 不是同阈值对照（同阈值对照可加 --conf 0.001）。", "",
                  f"P / R / F1 只统计分数 ≥ {gui}（界面默认阈值）的框：同类、IoU ≥ 0.50、COCO 贪心匹配、crowd 区域忽略；"
                  "近似界面显示的结果，NMS IoU 和每图框数上限仍按本次参数。", "",
                  "视觉模式逐类推理，延迟是所有类别的总和；参考编码单独记录，不计入 ms/图。", "",
                  "示例左侧为人工真值，右侧仅显示置信度 ≥ 0.25 的框；评价使用所有达到该模式预测阈值的预测。", "",
                  NOTICE, "", "参数、版本、数据来源与哈希、各类别 AP 和 P/R/F1：见 report.json。原始预测：modes/<mode>/predictions.json。"])
    if report.get("error"):
        lines.extend(["", f"停止原因：{report['error']}"])
    temporary = output / "summary.md.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "summary.md")
    atomic_json(output / "report.json", report)


# The code this tool runs (AGENTS.md: record code, model, data, parameters and device).
SOURCE_FILES = ("evaluate_mini_dataset.py", "engines.py", "visual_engine.py", "sliced_inference.py",
                "resource_safety.py", "detection_utils.py", "detection_branch.py")


def load_average():
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (OSError, AttributeError):  # Windows has no load average
        return None


def code_fingerprint(root=None):
    """git HEAD, uncommitted tracked files, SHA-256 of the evaluation code, and the host load."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    record = {"git_head": None, "git_dirty": None, "dirty_files": None, "source_sha256": {},
              "loadavg_start": load_average()}
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=20)
        if head.returncode == 0:
            record["git_head"] = head.stdout.strip()
        if status.returncode == 0:
            record["dirty_files"] = sorted(line[3:] for line in status.stdout.splitlines() if line.strip())
            record["git_dirty"] = bool(record["dirty_files"])
    except (OSError, subprocess.SubprocessError):
        pass  # not a git checkout (e.g. a release zip): the source hashes below still identify the code
    for name in SOURCE_FILES:
        if (root / name).is_file():
            record["source_sha256"][name] = sha256_file(root / name)
    return record


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "ultralytics", "sahi", "pycocotools", "numpy", "opencv-python"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def preflight_dataset(args, dataset):
    """Validate every selected shape and slice plan before checkpoint loading."""
    from resource_safety import preflight_inference
    if not 32 <= args.imgsz <= 960 or not 128 <= args.slice_size <= 960 \
            or not 0 <= args.slice_overlap <= 0.3 or not 1 <= args.max_slices <= 9 \
            or not 1 <= args.max_det <= 1000:
        raise ValueError("安全模式要求 imgsz 32–960、slice-size 128–960、overlap 0–0.3、max-slices 1–9、max-det 1–1000。")
    highest = max(mode_conf(args, mode) for mode in args.modes)
    if args.gui_conf < highest:
        raise ValueError(f"界面阈值 --gui-conf {args.gui_conf:g} 不能低于预测阈值 {highest:g}，否则 P/R 会漏掉中间的框。")
    sliced = any(mode.endswith("_sahi") for mode in args.modes)
    plans = []
    for item in dataset["images"]:
        shape = (int(item["height"]), int(item["width"]), 3)
        preflight_inference("cpu", args.imgsz, shape)
        if sliced:
            from sliced_inference import validate_slice_plan
            boxes = validate_slice_plan(shape, args.slice_size, args.slice_overlap, args.max_slices)
            plans.append({"image_id": int(item["id"]), "slice_count": len(boxes)})
    return plans


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.dataset, args.limit, args.annotations, args.images, args.references,
                           require_references=any(mode.startswith("visual") for mode in args.modes))
    try:
        slice_plans = preflight_dataset(args, dataset)
    except (ValueError, RuntimeError) as exc:
        print(f"加载模型前已停止：{exc}", file=sys.stderr)
        return 1
    # Check evaluator availability before allocating a checkpoint.
    from pycocotools.coco import COCO  # noqa: F401
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {"status": "running", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "settings": settings, "versions": versions(), "code": code_fingerprint(), "notice": NOTICE,
              "image_count": len(dataset["images"]), "all_image_count": dataset["all_image_count"],
              "image_ids": [int(item["id"]) for item in dataset["images"]],
              "prediction_backend": "BoxOnlySegmentationPredictor; no per-object masks",
              "resource_policy": {"pytorch_gpu_budget_mib": 3072, "min_available_ram_mib": 1536,
                                  "min_free_gpu_inference_mib": 1024, "max_slices": 9,
                                  "max_imgsz": 960, "max_det": 1000},
              "thresholds": {"text_conf": args.conf, "visual_conf": args.visual_conf,
                             "gui_conf": args.gui_conf, "gui_match_iou": GUI_MATCH_IOU},
              "data_sources": dataset["sources"],
              "dataset_sha256": dataset["sha256"], "file_hashes": dataset["file_hashes"],
              "model_sha256": sha256_file(args.model), "references": dataset["references"],
              "slice_plans": slice_plans, "modes": {}}
    write_summary(output, report)
    engine = None
    handle_cuda_oom = lambda exc: False
    try:
        from engines import OpenVocabEngine
        from visual_engine import VisualPromptEngine
        from resource_safety import handle_cuda_oom
        for mode in args.modes:
            report["modes"][mode] = {"status": "running"}
            write_summary(output, report)
            engine = VisualPromptEngine() if mode.startswith("visual") else OpenVocabEngine()
            engine.imgsz = args.imgsz
            loaded = engine.load_model(args.model, args.device)
            if loaded != "SUCCESS":
                raise RuntimeError(f"模型加载未完成：{loaded}")
            if args.device == "CUDA":
                import torch
                synchronize = lambda: torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
            else:
                synchronize = lambda: None
            result = run_mode(engine, dataset, args, mode, synchronize)
            memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}
            if args.device == "CUDA":
                memory = {"peak_allocated_mb": torch.cuda.max_memory_allocated(0) / 1024 ** 2,
                          "peak_reserved_mb": torch.cuda.max_memory_reserved(0) / 1024 ** 2}
            evaluation = coco_metrics(dataset["annotations"], result["predictions"], report["image_ids"],
                                      gui_threshold=args.gui_conf)
            mode_folder = output / "modes" / mode
            mode_folder.mkdir(parents=True, exist_ok=True)
            atomic_json(mode_folder / "predictions.json", result["predictions"])
            write_csv(mode_folder / "timings.csv", result["timings"], ["image_id", "latency_ms"])
            summary = {"status": "complete", "prediction_conf": result["prediction_conf"], **evaluation,
                       "timing_summary": result["timing_summary"],
                       "reference_encodings": result["reference_encodings"], "memory": memory,
                       "predictions_file": str((mode_folder / "predictions.json").relative_to(output)),
                       "sample_images": annotate_samples(output, mode, dataset, result["predictions"], args.samples)}
            atomic_json(mode_folder / "metrics.json", summary)
            report["modes"][mode] = summary
            # Release before loading the next mode, including the GPU lock.
            engine.unload()
            engine = None
            write_summary(output, report)
        report["status"] = "complete"
        report["code"]["loadavg_end"] = load_average()
        write_summary(output, report)
        print(f"完成：{output / 'summary.md'}", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "stopped"
        report["error"] = f"{type(exc).__name__}: {exc}"
        for result in report["modes"].values():
            if result["status"] == "running":
                result.update(status="stopped", error=report["error"])
        write_summary(output, report)
        handle_cuda_oom(exc)
        print(f"评测已停止，完成的模式已保存：{exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.unload()


if __name__ == "__main__":
    raise SystemExit(main())
