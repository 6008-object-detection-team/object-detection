"""Quantitative synthetic low-light evaluation of the enhancement paths.

Human COCO boxes score five conditions per darkness level: original reference,
dark without enhancement, dark + legacy CLAHE, dark + adaptive enhancement fed
to the detector, and the application's dual path (detect the dark frame, then
supplement from the enhanced frame through ``OpenVocabEngine.process_frame``).

Importing this module or requesting --help does not import torch, load a
model or access a GPU. Synthetic darkening is a controlled proxy, not real
night footage; no generated image is part of the repository.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluate_mini_dataset import (  # noqa: E402
    METRIC_NAMES, atomic_json, coco_metrics, decode_image, detection_records,
    probability, positive_integer, sha256_file, versions,
)

SCHEMA = "lowlight-eval/1"
DARK_CONDITIONS = ("dark", "dark_clahe", "dark_adaptive", "dark_dual")
ENHANCEMENTS = ("dark_clahe", "dark_adaptive", "dark_dual")
LABELS = {"original": "Original (reference)", "dark": "Dark, no enhancement",
          "dark_clahe": "Dark + legacy CLAHE", "dark_adaptive": "Dark + adaptive, detect enhanced",
          "dark_dual": "Dark + app dual path"}
# (enhanced, base) pairs; every level is compared with its own unenhanced dark frames.
PAIRS = (("dark_clahe", "dark"), ("dark_adaptive", "dark"), ("dark_dual", "dark"),
         ("dark_dual", "dark_adaptive"), ("dark", "original"))
BOOTSTRAP_METRICS = ("AP", "AP50", "APsmall", "AR100", "recall_gui", "precision_gui")
AREAS = ("all", "small", "medium", "large")
GAMMA = 2.2
# Photo-electrons at white for the reference exposure: shot-noise variance 1e-3 * signal,
# the log-midpoint of the range sampled by Brooks et al. (CVPR 2019). Their read/shot
# relation at that point gives a read-noise sigma of about 1e-3 in linear units.
FULL_WELL = 1000.0
READ_NOISE = 0.001
DEFAULT_EXPOSURES = (0.25, 0.12, 0.06, 0.02)
DEFAULT_SOURCES = (
    ("mini_eval", "datasets/mini_eval/annotations.json", "datasets/mini_eval/images"),
    ("probe_train", "datasets/mini_train_probe/annotations/train.json", "datasets/mini_train_probe/images/train"),
    ("probe_val", "datasets/mini_train_probe/annotations/val.json", "datasets/mini_train_probe/images/val"),
)
CODE_FILES = ("tools/evaluate_lowlight.py", "evaluate_mini_dataset.py", "enhancement.py", "engines.py",
              "detection_utils.py", "validate_improvements.py", "box_predictor.py", "resource_safety.py")
NOTICE = (
    "Synthetic low light: sRGB is linearised with gamma 2.2, exposure is scaled, Poisson shot and "
    "Gaussian read noise are added and the result is re-encoded to 8 bits. This is a controlled "
    "proxy without a real ISP, motion blur, artificial light sources or real sensor noise; it is "
    "not night footage. All images are COCO val2017 with human boxes; the 48+12 training-probe "
    "images trained only separate probe models, never the zero-shot YOLOE weights evaluated here. "
    "Prompts and thresholds were fixed before scoring. Bootstrap intervals resample images, are "
    "percentile intervals without multiple-comparison correction, and cannot add independent scenes."
)


def exposure_list(value):
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("曝光系数必须是逗号分隔的数字") from exc
    if not values or any(not math.isfinite(item) or not 0 < item < 1 for item in values) \
            or values != sorted(values, reverse=True) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("曝光系数必须在 (0, 1) 内、互不相同且从亮到暗排列")
    return values


def source_spec(value):
    name, separator, paths = value.partition("=")
    annotations, comma, images = paths.partition(",")
    if not separator or not comma or not name or not annotations or not images:
        raise argparse.ArgumentTypeError("数据源格式：名称=标注.json,图片目录")
    return name, annotations, images


def build_parser():
    parser = argparse.ArgumentParser(
        description="合成暗光下的人工 GT 定量评价：原图、暗图、旧 CLAHE、自适应增强、应用双路径。一次只加载一个模型。")
    parser.add_argument("--source", type=source_spec, action="append",
                        help="名称=COCO标注.json,图片目录；可重复。默认 mini_eval + mini_train_probe train/val")
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--device", type=str.upper, choices=("CPU", "CUDA"), default="CPU",
                        help="显式选择设备；失败时停止，不自动切换")
    parser.add_argument("--output", type=Path, default=Path("work/lowlight_eval"),
                        help="结果目录；必须不存在或为空，不覆盖旧实验")
    parser.add_argument("--exposures", type=exposure_list, default=list(DEFAULT_EXPOSURES))
    parser.add_argument("--full-well", type=float, default=FULL_WELL, help="原曝光白电平的光电子数")
    parser.add_argument("--read-noise", type=float, default=READ_NOISE, help="线性域读出噪声标准差")
    parser.add_argument("--seed", type=int, default=6008, help="暗光合成种子；每图每级派生独立随机流")
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--conf", type=probability, default=0.05, help="AP 评价阈值")
    parser.add_argument("--gui-conf", type=probability, default=0.40, help="界面默认阈值，用于召回 / 精确率")
    parser.add_argument("--iou", type=probability, default=0.45, help="NMS 与补检合并 IoU（界面默认）")
    parser.add_argument("--max-det", type=positive_integer, default=300)
    parser.add_argument("--strength", type=probability, default=0.65, help="自适应增强强度（界面默认）")
    parser.add_argument("--bootstrap-samples", type=positive_integer, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261028)
    parser.add_argument("--timing-repeats", type=positive_integer, default=3)
    parser.add_argument("--limit", type=positive_integer, help="只评测前 N 张图片（冒烟检查）")
    parser.add_argument("--example-image-id", type=int, help="示例拼图图片；默认按许可证与人工框数固定选择")
    parser.add_argument("--save-images", action="store_true", help="把合成暗图与增强图另存为 PNG（仅输出目录）")
    return parser


# ---------------------------------------------------------------- synthesis

def synthesize_low_light(image, exposure, seed, full_well=FULL_WELL, read_noise=READ_NOISE, gamma=GAMMA):
    """Darken an 8-bit BGR frame as a shorter exposure of the same scene.

    The sRGB-like values are linearised with ``gamma``, scaled by ``exposure``,
    receive Poisson shot noise (``full_well`` electrons at white for the original
    exposure) plus Gaussian read noise, and are re-encoded and quantised to uint8.
    ``seed`` may be an int or a sequence such as (seed, level, image_id).
    ``full_well=None`` and ``read_noise=0`` give the noise-free exposure change.
    """
    import numpy as np
    if image is None or not getattr(image, "size", 0) or image.dtype != np.uint8 \
            or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("暗光合成需要非空 uint8 BGR 图像")
    if not math.isfinite(exposure) or not 0 < exposure <= 1:
        raise ValueError("曝光系数必须在 (0, 1] 内")
    if full_well is not None and (not math.isfinite(full_well) or full_well <= 0):
        raise ValueError("full_well 必须为正数或 None")
    if not math.isfinite(read_noise) or read_noise < 0 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("读出噪声不能为负，gamma 必须为正")
    rng = np.random.default_rng(seed)
    signal = np.power(image.astype(np.float64) / 255.0, gamma) * exposure
    if full_well is not None:
        signal = rng.poisson(signal * full_well) / full_well
    if read_noise:
        signal = signal + rng.normal(0.0, read_noise, signal.shape)
    encoded = np.power(np.clip(signal, 0.0, 1.0), 1.0 / gamma)
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def image_seed(seed, level_index, image_id):
    """Order-independent stream: identical noise for an image regardless of --limit."""
    return [int(seed), int(level_index), int(image_id)]


def mean_luma(image):
    import cv2
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0)


def legacy_clahe(image):
    """The superseded LAB CLAHE baseline, imported unchanged from validate_improvements.py."""
    from validate_improvements import old_clahe
    return old_clahe(image)


# ---------------------------------------------------------------- conditions

def run_dark_conditions(engine, dark, classes, conf, gui_conf, iou, max_det, strength, enhancer_factory=None):
    """Detections for one dark frame; dual path uses the app's own process_frame merge.

    Returns ``(ap, gui, info)``: AP-threshold detections, GUI-threshold detections
    and enhancement measurements. A fresh enhancer per still image matches a scene
    cut, so no exposure smoothing leaks between unrelated images.
    """
    if enhancer_factory is None:
        from enhancement import AdaptiveLowLightEngine as enhancer_factory
    common = {"iou": iou, "class_names": classes, "max_det": max_det}
    ap = {"dark": engine.detect(dark, conf, source="original", **common)}
    clahe = legacy_clahe(dark)
    ap["dark_clahe"] = engine.detect(clahe, conf, source="original", **common)
    enhanced = enhancer_factory().process(dark, strength)
    ap["dark_adaptive"] = engine.detect(enhanced.image, conf, source="original", **common)
    _, ap["dark_dual"], _ = engine.process_frame(
        dark, conf=conf, enhancer=enhancer_factory(), enhancement_strength=strength,
        supplement_dark=True, **common)
    # A higher threshold keeps exactly the top of the same NMS output for single
    # passes; the dual merge depends on which base boxes survive, so rerun it.
    gui = {name: [item for item in ap[name] if item.score >= gui_conf]
           for name in ("dark", "dark_clahe", "dark_adaptive")}
    _, gui["dark_dual"], _ = engine.process_frame(
        dark, conf=gui_conf, enhancer=enhancer_factory(), enhancement_strength=strength,
        supplement_dark=True, **common)
    info = {"luma": {"dark": mean_luma(dark), "dark_clahe": mean_luma(clahe),
                     "dark_adaptive": mean_luma(enhanced.image)},
            "applied": bool(enhanced.applied), "brightness": float(enhanced.brightness),
            "gamma": float(enhanced.gamma)}
    return ap, gui, info, {"dark_clahe": clahe, "dark_adaptive": enhanced.image}


# ---------------------------------------------------------------- dataset

def load_sources(specs, root=PROJECT, limit=None):
    """Merge COCO files that share one category list; image IDs must be unique."""
    merged_images, merged_annotations, categories, licenses = [], [], None, []
    paths, source_of, summary, hashes = {}, {}, {}, {}
    for name, annotation_file, image_dir in specs:
        annotation_path = (Path(root) / annotation_file).resolve()
        image_root = (Path(root) / image_dir).resolve()
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        current = sorted(((int(item["id"]), item["name"]) for item in data["categories"]))
        if categories is None:
            categories, licenses = current, data.get("licenses", [])
        elif current != categories:
            raise ValueError(f"数据源 {name} 的类别与其他数据源不同，不能合并评价")
        hashes[f"{name}/annotations"] = sha256_file(annotation_path)
        counts = {"images": 0, "annotations": 0, "crowd": 0}
        for item in data["images"]:
            image_id = int(item["id"])
            if image_id in paths:
                raise ValueError(f"图片 ID {image_id} 在多个数据源中重复")
            path = (image_root / item["file_name"]).resolve()
            if not path.is_relative_to(image_root) or not path.is_file():
                raise ValueError(f"图片不存在或不在数据目录内：{path}")
            paths[image_id], source_of[image_id] = path, name
            merged_images.append(item)
            counts["images"] += 1
        for item in data["annotations"]:
            merged_annotations.append(item)
            counts["annotations"] += 1
            counts["crowd"] += int(bool(item.get("iscrowd", 0)))
        summary[name] = {"annotations_file": str(annotation_path), "images_dir": str(image_root), **counts}
    if not merged_images:
        raise ValueError("评测图片为空")
    annotation_ids = [int(item["id"]) for item in merged_annotations]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("人工标注 ID 重复")
    selected = merged_images[:limit] if limit else merged_images
    keep = {int(item["id"]) for item in selected}
    annotations = {"info": {"description": "merged human COCO ground truth for low-light evaluation"},
                   "licenses": licenses, "images": selected,
                   "annotations": [item for item in merged_annotations if int(item["image_id"]) in keep],
                   "categories": [{"id": cid, "name": cname} for cid, cname in categories]}
    for item in selected:
        hashes[f"{source_of[int(item['id'])]}/{item['file_name']}"] = sha256_file(paths[int(item["id"])])
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"annotations": annotations, "paths": paths, "source_of": source_of, "sources": summary,
            "classes": [cname for _, cname in categories], "category_ids": [cid for cid, _ in categories],
            "sha256": fingerprint, "file_hashes": hashes, "all_image_count": len(merged_images)}


def area_counts(annotations):
    counts = {"small": 0, "medium": 0, "large": 0, "crowd": 0}
    for item in annotations["annotations"]:
        if item.get("iscrowd", 0):
            counts["crowd"] += 1
        else:
            area = float(item["area"])
            counts["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1
    return counts


def default_example_image(annotations):
    """Fixed before scoring: most human boxes among CC BY / BY-SA images, then lowest ID."""
    boxes = {}
    for item in annotations["annotations"]:
        if not item.get("iscrowd", 0):
            boxes[int(item["image_id"])] = boxes.get(int(item["image_id"]), 0) + 1
    ranked = sorted(annotations["images"], key=lambda item: (
        item.get("license") not in (4, 5), -boxes.get(int(item["id"]), 0), int(item["id"])))
    return int(ranked[0]["id"])


# ---------------------------------------------------------------- evaluation

def evaluation_tables(annotations, predictions, image_ids):
    """Run pycocotools matching once and keep per-image results for weighted accumulation.

    ``weighted_metrics`` with unit weights reproduces COCOeval.accumulate/summarize
    at maxDets=100; integer weights are equivalent to duplicating resampled images.
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
            detected = COCO()
            detected.dataset = {"images": annotations["images"], "categories": annotations["categories"],
                                "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = [int(value) for value in image_ids]
        evaluator.params.catIds = sorted(int(item["id"]) for item in annotations["categories"])
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
    params = evaluator.params
    if list(params.areaRngLbl) != list(AREAS):
        raise RuntimeError("pycocotools 面积分组顺序与预期不同")
    count_t, count_a, count_i = len(params.iouThrs), len(params.areaRng), len(params.imgIds)
    tables = {}
    for k, category in enumerate(params.catIds):
        for a, area in enumerate(AREAS):
            scores, images, ids, tp, fp = [], [], [], [], []
            positives = np.zeros(count_i)
            for i in range(count_i):
                entry = evaluator.evalImgs[k * count_a * count_i + a * count_i + i]
                if entry is None:
                    continue
                positives[i] = np.count_nonzero(np.asarray(entry["gtIgnore"]) == 0)
                if len(entry["dtScores"]):
                    matched = np.asarray(entry["dtMatches"]) > 0
                    ignored = np.asarray(entry["dtIgnore"]).astype(bool)
                    tp.append(matched & ~ignored)
                    fp.append(~matched & ~ignored)
                    scores.append(np.asarray(entry["dtScores"], dtype=float))
                    images.append(np.full(len(entry["dtScores"]), i))
                    ids.append(np.asarray(entry["dtIds"]))
            if scores:
                scores = np.concatenate(scores)
                order = np.argsort(-scores, kind="mergesort")
                table = {"scores": scores[order], "image": np.concatenate(images)[order],
                         "ids": np.concatenate(ids)[order],
                         "tp": np.concatenate(tp, axis=1)[:, order], "fp": np.concatenate(fp, axis=1)[:, order]}
            else:
                table = {"scores": np.zeros(0), "image": np.zeros(0, int), "ids": np.zeros(0, int),
                         "tp": np.zeros((count_t, 0), bool), "fp": np.zeros((count_t, 0), bool)}
            table["positives"] = positives
            tables[(int(category), area)] = table
    names = {int(item["id"]): item["name"] for item in annotations["categories"]}
    required = {area: [category for category in params.catIds
                       if tables[(int(category), area)]["positives"].sum() > 0] for area in AREAS}
    return {"image_ids": list(params.imgIds), "category_ids": [int(value) for value in params.catIds],
            "names": names, "iou_thresholds": np.asarray(params.iouThrs),
            "recall_thresholds": np.asarray(params.recThrs), "tables": tables, "required": required}


def _accumulate(table, weights, recall_thresholds):
    import numpy as np
    positives = float(np.dot(weights, table["positives"]))
    if positives <= 0:
        return None, None
    count_t = table["tp"].shape[0]
    if not table["scores"].size:
        return np.zeros((count_t, recall_thresholds.size)), np.zeros(count_t)
    detection_weights = weights[table["image"]]
    tp = np.cumsum(table["tp"] * detection_weights, axis=1)
    fp = np.cumsum(table["fp"] * detection_weights, axis=1)
    recall = tp / positives
    precision = tp / (tp + fp + np.spacing(1))
    precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
    interpolated = np.zeros((count_t, recall_thresholds.size))
    for t in range(count_t):
        indices = np.searchsorted(recall[t], recall_thresholds, side="left")
        valid = indices < recall.shape[1]
        interpolated[t, valid] = precision[t, indices[valid]]
    return interpolated, recall[:, -1]


def weighted_metrics(evaluation, weights=None, areas=AREAS, gui_threshold=None):
    """COCO AP/AR at maxDets=100 with per-image multiplicities (None = each image once).

    An aggregate is None when a category that has positives in the full set has
    none in this draw, as in the team's earlier bootstrap.
    """
    import numpy as np
    count_i = len(evaluation["image_ids"])
    weights = np.ones(count_i) if weights is None else np.asarray(weights, dtype=float)
    thresholds = evaluation["iou_thresholds"]
    at_50 = int(np.flatnonzero(np.isclose(thresholds, 0.5))[0])
    at_75 = int(np.flatnonzero(np.isclose(thresholds, 0.75))[0])
    result, per_class = {}, {}
    for area in areas:
        precisions, recalls, missing = [], [], False
        for category in evaluation["category_ids"]:
            precision, recall = _accumulate(evaluation["tables"][(category, area)], weights,
                                            evaluation["recall_thresholds"])
            if precision is None:
                missing = missing or category in evaluation["required"][area]
                continue
            precisions.append(precision)
            recalls.append(recall)
            if area == "all":
                per_class[evaluation["names"][category]] = float(precision.mean())
        valid = bool(precisions) and not missing
        suffix = "" if area == "all" else area
        result["AP" + suffix] = float(np.mean([p.mean() for p in precisions])) if valid else None
        if area == "all":
            result["AP50"] = float(np.mean([p[at_50].mean() for p in precisions])) if valid else None
            result["AP75"] = float(np.mean([p[at_75].mean() for p in precisions])) if valid else None
            result["AR100"] = float(np.mean([r.mean() for r in recalls])) if valid else None
    result["per_class_AP"] = per_class
    if gui_threshold is not None:
        result.update(threshold_counts(evaluation, gui_threshold, weights))
    return result


def threshold_counts(evaluation, threshold, weights=None):
    """Recall / precision at IoU 0.50 for boxes scoring at least ``threshold``."""
    import numpy as np
    count_i = len(evaluation["image_ids"])
    weights = np.ones(count_i) if weights is None else np.asarray(weights, dtype=float)
    at_50 = int(np.flatnonzero(np.isclose(evaluation["iou_thresholds"], 0.5))[0])
    tp = fp = positives = 0.0
    for category in evaluation["category_ids"]:
        table = evaluation["tables"][(category, "all")]
        positives += float(np.dot(weights, table["positives"]))
        kept = table["scores"] >= threshold
        detection_weights = weights[table["image"]] * kept
        tp += float(np.dot(table["tp"][at_50], detection_weights))
        fp += float(np.dot(table["fp"][at_50], detection_weights))
    return {"recall_gui": tp / positives if positives else None,
            "precision_gui": tp / (tp + fp) if tp + fp else None,
            "tp_gui": tp, "fp_gui": fp, "gt_gui": positives}


def source_breakdown(evaluation, records, threshold):
    """TP/FP at IoU 0.50 split by detection source (original vs enhanced supplement)."""
    import numpy as np
    at_50 = int(np.flatnonzero(np.isclose(evaluation["iou_thresholds"], 0.5))[0])
    counts = {}
    for category in evaluation["category_ids"]:
        table = evaluation["tables"][(category, "all")]
        for column, (identifier, score) in enumerate(zip(table["ids"], table["scores"])):
            if score < threshold:
                continue
            source = records[int(identifier) - 1].get("source", "original")
            entry = counts.setdefault(source, {"boxes": 0, "tp": 0, "fp": 0, "ignored": 0})
            entry["boxes"] += 1
            if table["tp"][at_50, column]:
                entry["tp"] += 1
            elif table["fp"][at_50, column]:
                entry["fp"] += 1
            else:
                entry["ignored"] += 1
    return counts


def percentile_interval(values, total, confidence=0.95):
    """Same summary as tools/analyze_saved_predictions.interval, for missing-aware draws."""
    import numpy as np
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    result = {"requested_samples": total, "valid_samples": int(valid.size), "mean": None,
              "percentile_interval": None, "probability_positive": None}
    if valid.size:
        tail = (1 - confidence) / 2
        result.update(mean=float(valid.mean()),
                      percentile_interval=[float(value) for value in np.quantile(valid, [tail, 1 - tail])],
                      probability_positive=float((valid > 0).mean()))
    return result


def paired_bootstrap(evaluations, gui_evaluations, pairs, samples, seed, gui_threshold,
                     metrics=BOOTSTRAP_METRICS, confidence=0.95, progress=None):
    """Paired image bootstrap: one multiset of images per draw, shared by every key."""
    import numpy as np
    keys = list(evaluations)
    count_i = len(evaluations[keys[0]]["image_ids"])
    if any(evaluation["image_ids"] != evaluations[keys[0]]["image_ids"]
           for evaluation in [*evaluations.values(), *gui_evaluations.values()]):
        raise ValueError("配对 bootstrap 需要所有条件使用同一组图片")
    rng = np.random.default_rng(seed)
    values = {key: {metric: [] for metric in metrics} for key in keys}
    for sample in range(samples):
        weights = np.bincount(rng.integers(0, count_i, count_i), minlength=count_i)
        for key in keys:
            point = weighted_metrics(evaluations[key], weights, areas=("all", "small"))
            point.update(threshold_counts(gui_evaluations[key], gui_threshold, weights))
            for metric in metrics:
                values[key][metric].append(point[metric])
        if progress and ((sample + 1) % 200 == 0 or sample + 1 == samples):
            progress(sample + 1)
    point = {key: {**weighted_metrics(evaluations[key], areas=("all", "small")),
                   **threshold_counts(gui_evaluations[key], gui_threshold)} for key in keys}
    absolute = {key: {metric: percentile_interval(values[key][metric], samples, confidence)
                      for metric in metrics} for key in keys}
    differences = {}
    for left, right in pairs:
        result = {}
        for metric in metrics:
            draws = [a - b if a is not None and b is not None else None
                     for a, b in zip(values[left][metric], values[right][metric])]
            entry = percentile_interval(draws, samples, confidence)
            a, b = point[left][metric], point[right][metric]
            entry["point_difference"] = a - b if a is not None and b is not None else None
            result[metric] = entry
        differences[f"{left}_minus_{right}"] = {"left": left, "right": right, **result}
    return {"samples": samples, "seed": seed, "confidence": confidence, "sampling_unit": "image",
            "paired_across_all_conditions": True, "interval": "percentile",
            "missing_positive_policy": "None when a category with positives in the full set has none in a draw",
            "absolute": absolute, "differences": differences}


# ---------------------------------------------------------------- timing

def summarize_times(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low, high = math.floor(position), math.ceil(position)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"n": len(values), "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": p95, "min_ms": ordered[0], "max_ms": ordered[-1]}


def time_enhancements(frames, repeats, strength, enhancer_factory=None, clock=time.perf_counter):
    """Back-to-back CPU cost of CLAHE vs adaptive enhancement on identical dark frames.

    Each repeat is a full pass; the method order alternates per frame and per
    repeat so neither method always runs first. Host load is recorded around
    every repeat; nothing here is a model-speed claim.
    """
    import cv2
    if enhancer_factory is None:
        from enhancement import AdaptiveLowLightEngine as enhancer_factory
    methods = {"dark_clahe": legacy_clahe,
               "dark_adaptive": lambda frame: enhancer_factory().process(frame, strength)}
    for method in methods.values():
        method(frames[0])  # warm OpenCV kernels; excluded
    runs = []
    for repeat in range(repeats):
        load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
        times = {name: [] for name in methods}
        for index, frame in enumerate(frames):
            order = list(methods) if (index + repeat) % 2 == 0 else list(reversed(methods))
            for name in order:
                start = clock()
                methods[name](frame)
                times[name].append((clock() - start) * 1000)
        load_after = os.getloadavg() if hasattr(os, "getloadavg") else None
        runs.append({"repeat": repeat, "loadavg_before": load_before, "loadavg_after": load_after,
                     **{name: summarize_times(values) for name, values in times.items()}})
    summary = {}
    for name in methods:
        medians = [run[name]["median_ms"] for run in runs]
        means = [run[name]["mean_ms"] for run in runs]
        summary[name] = {"mean_of_means_ms": statistics.mean(means), "median_of_medians_ms": statistics.median(medians),
                         "p95_of_p95_ms": statistics.median(run[name]["p95_ms"] for run in runs),
                         "median_spread_ms": [min(medians), max(medians)]}
    return {"frames_per_repeat": len(frames), "repeats": repeats, "cv2_threads": cv2.getNumThreads(),
            "runs": runs, "summary": summary,
            "notice": "CPU wall time of the enhancement call only, interleaved back-to-back in one process; "
                      "excludes detection, decoding and synthesis."}


# ---------------------------------------------------------------- outputs

def condition_key(level, condition):
    return condition if level is None else f"{level}/{condition}"


def validate_report(report):
    """Schema check used by tests and before the final write."""
    required = {"schema": str, "status": str, "created_at": str, "notice": str, "settings": dict,
                "fingerprints": dict, "versions": dict, "host": dict, "dataset": dict, "synthesis": dict,
                "results": dict, "enhancement": dict, "dual_path": dict, "bootstrap": dict,
                "timing": dict, "outputs": dict}
    for key, kind in required.items():
        if not isinstance(report.get(key), kind):
            raise ValueError(f"report.json 缺少或类型错误：{key}")
    if report["schema"] != SCHEMA:
        raise ValueError("report.json schema 版本不符")
    levels = [item["name"] for item in report["synthesis"]["levels"]]
    keys = ["original"] + [condition_key(level, condition) for level in levels for condition in DARK_CONDITIONS]
    for key in keys:
        entry = report["results"].get(key)
        if not isinstance(entry, dict) or set(METRIC_NAMES) - set(entry.get("metrics", {})) \
                or not {"recall_gui", "precision_gui", "tp_gui", "fp_gui", "gt_gui"} <= set(entry.get("gui", {})):
            raise ValueError(f"缺少条件结果：{key}")
    for level in levels:
        for left, right in PAIRS:
            name = f"{condition_key(level, left)}_minus_{condition_key(level, right) if right != 'original' else 'original'}"
            difference = report["bootstrap"]["differences"].get(name)
            if not isinstance(difference, dict) or "AP" not in difference:
                raise ValueError(f"缺少 bootstrap 差值：{name}")
            interval = difference["AP"]["percentile_interval"]
            if interval is not None and (len(interval) != 2 or interval[0] > interval[1]):
                raise ValueError(f"bootstrap 区间无效：{name}")
        if level not in report["enhancement"] or level not in report["dual_path"]:
            raise ValueError(f"缺少增强或双路径统计：{level}")
    return True


def fmt(value, digits=3, signed=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def write_summary(path, report):
    levels = report["synthesis"]["levels"]
    results, differences = report["results"], report["bootstrap"].get("differences", {})
    dataset = report["dataset"]
    lines = ["# 合成暗光定量评价", "",
             f"状态：{report['status']}；{dataset['image_count']} 张 COCO val2017 图片，"
             f"{dataset['annotation_count']} 个人工框（非 crowd：小 {dataset['areas']['small']} / "
             f"中 {dataset['areas']['medium']} / 大 {dataset['areas']['large']}；crowd {dataset['areas']['crowd']}）。", "",
             f"YOLOE 文本提示 `{', '.join(report['dataset']['classes'])}`，imgsz {report['settings']['imgsz']}，"
             f"AP 阈值 {report['settings']['conf']}，界面阈值 {report['settings']['gui_conf']}，"
             f"NMS IoU {report['settings']['iou']}，max_det {report['settings']['max_det']}，设备 {report['device']}。", "",
             "| 级别 | 曝光 | 条件 | AP | AP50 | APsmall | AR100 | Recall@界面 | Precision@界面 | ΔAP vs 暗图 [95% CI] |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    original = results.get("original")
    if original:
        lines.append("| 原图 | 1.00 | " + LABELS["original"] + " | " + " | ".join(
            fmt(original["metrics"][name]) for name in ("AP", "AP50", "APsmall", "AR100"))
            + f" | {fmt(original['gui']['recall_gui'])} | {fmt(original['gui']['precision_gui'])} | — |")
    for level in levels:
        for condition in DARK_CONDITIONS:
            entry = results.get(condition_key(level["name"], condition))
            if not entry:
                continue
            ci = "—"
            if condition != "dark":
                diff = differences.get(f"{level['name']}/{condition}_minus_{level['name']}/dark", {}).get("AP")
                if diff and diff["percentile_interval"]:
                    low, high = diff["percentile_interval"]
                    ci = f"{fmt(diff['point_difference'], signed=True)} [{fmt(low, signed=True)}, {fmt(high, signed=True)}]"
            lines.append(f"| {level['name']} | {level['exposure']:.2f} | {LABELS[condition]} | " + " | ".join(
                fmt(entry["metrics"][name]) for name in ("AP", "AP50", "APsmall", "AR100"))
                + f" | {fmt(entry['gui']['recall_gui'])} | {fmt(entry['gui']['precision_gui'])} | {ci} |")
    lines.extend(["", "## 暗图亮度与双路径", "",
                  "| 级别 | 暗图平均亮度 | CLAHE 后 | 自适应后 | 自适应生效图片 | 双路径第二次检测 | 界面阈值下 +E 框（TP / FP） |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for level in levels:
        enhancement = report["enhancement"].get(level["name"])
        dual = report["dual_path"].get(level["name"])
        if not enhancement or not dual:
            continue
        added = dual["gui_sources"].get("enhanced", {"boxes": 0, "tp": 0, "fp": 0})
        lines.append(f"| {level['name']} | {fmt(enhancement['mean_luma']['dark'])} | "
                     f"{fmt(enhancement['mean_luma']['dark_clahe'])} | {fmt(enhancement['mean_luma']['dark_adaptive'])} | "
                     f"{enhancement['applied_images']}/{enhancement['images']} | {dual['second_pass_images']} | "
                     f"{added['boxes']}（{added['tp']} / {added['fp']}） |")
    timing = report["timing"].get("summary", {})
    if timing:
        lines.extend(["", "## 增强耗时（CPU，仅增强调用）", "",
                      f"{report['timing']['repeats']} 次完整重复，每次 {report['timing']['frames_per_repeat']} 帧，"
                      f"OpenCV 线程 {report['timing']['cv2_threads']}。", "",
                      "| 方法 | 均值 ms | 中位数 ms | p95 ms | 各次中位数范围 ms |", "|---|---:|---:|---:|---|"])
        for name, value in timing.items():
            low, high = value["median_spread_ms"]
            lines.append(f"| {LABELS[name]} | {value['mean_of_means_ms']:.2f} | {value['median_of_medians_ms']:.2f} | "
                         f"{value['p95_of_p95_ms']:.2f} | {low:.2f}–{high:.2f} |")
    lines.extend(["", report["notice"], "",
                  "完整类别指标、mini_eval 子集指标、配对 bootstrap、主机负载与指纹见 report.json。"])
    if report.get("error"):
        lines.extend(["", f"停止原因：{report['error']}"])
    temporary = Path(path).with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def draw_plot(report, output):
    """AP and GUI recall versus darkness plus paired ΔAP intervals; static PNG."""
    try:
        import matplotlib
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib is not installed"}
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    levels = report["synthesis"]["levels"]
    results, differences = report["results"], report["bootstrap"]["differences"]
    # Reference categorical slots 1-4 in fixed order; colour follows the condition.
    colors = {"dark": "#2a78d6", "dark_clahe": "#eb6834", "dark_adaptive": "#1baf7a", "dark_dual": "#eda100"}
    # Marker shape and the dashed dual line keep identity readable without colour, and keep
    # the dark line visible where the dual path coincides with it.
    markers = {"dark": "o", "dark_clahe": "D", "dark_adaptive": "^", "dark_dual": "s"}
    styles = {"dark": "-", "dark_clahe": "-", "dark_adaptive": "-", "dark_dual": (0, (4, 2))}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    positions = np.arange(len(levels))
    ticks = [f"{level['name']}\nexposure x{level['exposure']:g}" for level in levels]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    panels = (("AP", "metrics", "COCO AP (IoU 0.50:0.95)"),
              ("recall_gui", "gui", f"Recall at score >= {report['settings']['gui_conf']:.2f}, IoU 0.50"))
    for axis, (metric, group, title) in zip((axes[0], axes[2]), panels):
        reference = results["original"][group][metric] or 0.0
        top = max([reference] + [results[condition_key(level["name"], condition)][group][metric] or 0.0
                                 for level in levels for condition in DARK_CONDITIONS])
        axis.axhline(reference, color=muted, linewidth=1, linestyle=(0, (1, 2)))
        axis.text(-0.3, reference, f" original, exposure x1: {reference:.3f}", va="bottom", fontsize=8, color=muted)
        for condition in DARK_CONDITIONS:
            values = [results[condition_key(level["name"], condition)][group][metric] for level in levels]
            axis.plot(positions, values, color=colors[condition], linewidth=2, linestyle=styles[condition],
                      marker=markers[condition], markersize=7 if condition == "dark" else 6,
                      markeredgecolor="white", markeredgewidth=1.5, label=LABELS[condition],
                      solid_capstyle="round", zorder=3 if condition == "dark_dual" else 2)
        axis.set_title(title, fontsize=10, color=ink, loc="left")
        axis.set_xticks(positions, ticks)
        axis.set_xlim(-0.35, len(levels) - 0.65)
        axis.set_ylim(0, max(0.05, top * 1.15))
    width = 0.22
    for offset, condition in zip((-width, 0, width), ENHANCEMENTS):
        points, low, high = [], [], []
        for level in levels:
            entry = differences[f"{level['name']}/{condition}_minus_{level['name']}/dark"]["AP"]
            interval = entry["percentile_interval"] or [entry["point_difference"]] * 2
            points.append(entry["point_difference"])
            low.append(entry["point_difference"] - interval[0])
            high.append(interval[1] - entry["point_difference"])
        axes[1].errorbar(positions + offset, points, yerr=[low, high], fmt=markers[condition], color=colors[condition],
                         markersize=7, markeredgecolor="white", markeredgewidth=1.5, elinewidth=2, capsize=0,
                         label=LABELS[condition])
    axes[1].axhline(0, color=muted, linewidth=1)
    axes[1].set_title("AP change vs dark, no enhancement\n(paired image bootstrap, 95% percentile interval)",
                      fontsize=10, color=ink, loc="left")
    axes[1].set_xticks(positions, ticks)
    for axis in axes:
        axis.grid(axis="y", color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["right", "top"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(grid)
        axis.tick_params(colors=muted, labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    dataset = report["dataset"]
    figure.suptitle(f"Synthetic low light, YOLOE text prompts: {dataset['image_count']} COCO val2017 images, "
                    f"{dataset['annotation_count']} human boxes (not real night footage)", fontsize=11, color=ink)
    try:
        figure.savefig(output, dpi=160, facecolor="#fcfcfb")
    except RuntimeError as exc:
        # An optional figure must not discard finished metrics. Seen when Qt has already
        # loaded a different system FreeType into the same process (mixed conda/pip stacks).
        print(f"图表未生成，指标不受影响：{exc}", file=sys.stderr)
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        plt.close(figure)
    return {"status": "created", "path": str(output)}


def match_flags(records, truth, iou_threshold=0.5):
    """Greedy same-class IoU matching for illustrations only (COCO order, crowd ignored)."""
    from detection_utils import box_iou
    gts = [item for item in truth if not item.get("iscrowd", 0)]
    used, flags = set(), []
    for record in sorted(records, key=lambda row: row["score"], reverse=True):
        x, y, w, h = record["bbox"]
        best, best_iou = None, iou_threshold
        for index, gt in enumerate(gts):
            if index in used or gt["category_id"] != record["category_id"]:
                continue
            gx, gy, gw, gh = gt["bbox"]
            overlap = box_iou((x, y, x + w, y + h), (gx, gy, gx + gw, gy + gh))
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            used.add(best)
        flags.append((record, best is not None))
    return flags


def draw_example_grid(path, panels, truth, names, threshold, width=320):
    """Rows = darkness levels, columns = conditions; boxes at the GUI threshold."""
    import cv2
    import numpy as np
    rows = []
    for row_label, cells in panels:
        tiles = []
        for title, image, records in cells:
            scale = width / image.shape[1]
            tile = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            for record, correct in match_flags([r for r in records if r["score"] >= threshold], truth):
                x, y, w, h = (value * scale for value in record["bbox"])
                color = (60, 200, 60) if correct else (40, 40, 230)
                thickness = 1 if record.get("source") == "enhanced" else 2
                cv2.rectangle(tile, (round(x), round(y)), (round(x + w), round(y + h)), color, thickness)
                label = names[int(record["category_id"])] + (" +E" if record.get("source") == "enhanced" else "")
                cv2.putText(tile, label, (round(x) + 2, max(12, round(y) - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, color, 1, cv2.LINE_AA)
            header = np.full((22, width, 3), 32, np.uint8)
            text = title if title.startswith("original") else f"{row_label} | {title}"
            cv2.putText(header, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(np.vstack([header, tile]))
        height = max(tile.shape[0] for tile in tiles)
        tiles = [np.vstack([tile, np.full((height - tile.shape[0], width, 3), 32, np.uint8)]) for tile in tiles]
        rows.append(np.hstack(tiles))
    footer = np.full((24, rows[0].shape[1], 3), 32, np.uint8)
    cv2.putText(footer, f"Boxes with score >= {threshold:.2f}: green = matches a human box (IoU >= 0.5), "
                        "red = no match; thin +E = dual-path supplement from the enhanced frame",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", np.vstack([*rows, footer]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("示例拼图编码失败")
    Path(path).write_bytes(encoded.tobytes())


def example_attribution(annotations, image_id):
    """Licence and source of the illustrated COCO image, for figure captions."""
    image = next(item for item in annotations["images"] if int(item["id"]) == image_id)
    licences = {int(item["id"]): item for item in annotations.get("licenses", [])}
    licence = licences.get(int(image.get("license", -1)), {})
    return {"id": image_id, "file_name": image.get("file_name"), "license_id": image.get("license"),
            "license_name": licence.get("name"), "license_url": licence.get("url"),
            "coco_url": image.get("coco_url"), "flickr_url": image.get("flickr_url")}


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                text=True, timeout=20, check=True).stdout.strip()
        # The repository tracks some __pycache__ bytecode; running Python rewrites it.
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                                ":(exclude)*.pyc"], cwd=PROJECT, capture_output=True, text=True,
                               timeout=20, check=True).stdout.strip()
        return {"commit": commit, "tracked_source_changes": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_source_changes": None}


def default_engine_factory(args):
    from engines import OpenVocabEngine
    engine = OpenVocabEngine()
    engine.imgsz = args.imgsz
    status = engine.load_model(args.model, args.device)
    if status != "SUCCESS":
        raise RuntimeError(f"模型加载未完成：{status}")
    return engine


def device_description(device):
    if device != "CUDA":
        return {"device": "CPU", "name": platform.processor() or platform.machine()}
    import torch
    return {"device": "CUDA", "name": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


# ---------------------------------------------------------------- main

def main(argv=None, engine_factory=None, enhancer_factory=None):
    import numpy as np
    args = build_parser().parse_args(argv)
    if not 32 <= args.imgsz <= 960 or not 1 <= args.max_det <= 1000 or args.gui_conf < args.conf:
        print("参数超出安全范围：imgsz 32–960、max-det 1–1000，且界面阈值不能低于 AP 阈值。", file=sys.stderr)
        return 1
    output = args.output if args.output.is_absolute() else PROJECT / args.output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        # Never overwrite an earlier report or leave its prediction files next to a new one.
        print(f"输出目录非空，不覆盖旧实验：{output}", file=sys.stderr)
        return 1
    specs = args.source or list(DEFAULT_SOURCES)
    dataset = load_sources(specs, PROJECT, args.limit)
    annotations = dataset["annotations"]
    image_ids = [int(item["id"]) for item in annotations["images"]]
    from resource_safety import preflight_inference
    for item in annotations["images"]:
        preflight_inference("cpu", args.imgsz, (int(item["height"]), int(item["width"]), 3))
    from pycocotools.coco import COCO  # noqa: F401  # evaluator available before any model loads
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    levels = [{"name": f"L{index + 1}", "exposure": exposure,
               "display_scale": exposure ** (1 / GAMMA)} for index, exposure in enumerate(args.exposures)]
    example_id = args.example_image_id if args.example_image_id is not None else default_example_image(annotations)
    if example_id not in image_ids:
        raise ValueError("示例图片不在评测集中")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["source"] = [list(item) for item in specs]
    model_path = args.model if args.model.is_absolute() else PROJECT / args.model
    report = {
        "schema": SCHEMA, "status": "running",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "notice": NOTICE,
        "settings": settings, "device": args.device, "versions": versions(),
        "fingerprints": {"git": git_state(), "code_sha256": {name: sha256_file(PROJECT / name) for name in CODE_FILES},
                         "model": str(args.model),
                         "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
                         "dataset_sha256": dataset["sha256"], "file_hashes": dataset["file_hashes"]},
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                 "loadavg_start": os.getloadavg() if hasattr(os, "getloadavg") else None},
        "dataset": {"sources": dataset["sources"], "image_count": len(image_ids),
                    "all_image_count": dataset["all_image_count"], "annotation_count": len(annotations["annotations"]),
                    "areas": area_counts(annotations), "classes": dataset["classes"],
                    "category_ids": dataset["category_ids"], "image_ids": image_ids,
                    "evaluation_use": "COCO val2017 human boxes; YOLOE weights are zero-shot here, so the "
                                      "training-probe split is unseen by the evaluated model."},
        "synthesis": {"levels": levels, "gamma": GAMMA, "full_well_electrons": args.full_well,
                      "read_noise_sigma": args.read_noise, "seed": args.seed,
                      "seed_derivation": "numpy default_rng([seed, level_index, image_id])",
                      "quantisation": "8-bit after gamma re-encoding", "dark_image_sha256": {}},
        "results": {}, "enhancement": {}, "dual_path": {}, "bootstrap": {}, "timing": {},
        "subsets": {}, "consistency": {}, "outputs": {},
    }
    atomic_json(output / "report.json", report)
    engine = None
    try:
        # 1. Enhancement cost, before the detector exists (CPU only).
        timing_frames = []
        for item in annotations["images"]:
            frame = decode_image(dataset["paths"][int(item["id"])])
            for index, level in enumerate(levels):
                timing_frames.append(synthesize_low_light(
                    frame, level["exposure"], image_seed(args.seed, index, item["id"]), args.full_well, args.read_noise))
        report["timing"] = time_enhancements(timing_frames, args.timing_repeats, args.strength, enhancer_factory)
        del timing_frames
        atomic_json(output / "report.json", report)

        # 2. Detection, one model, one image at a time.
        engine = engine_factory(args) if engine_factory else default_engine_factory(args)
        report["device_detail"] = device_description(args.device) if engine_factory is None else {"device": "mock"}
        classes, common = dataset["classes"], {"iou": args.iou, "max_det": args.max_det}
        first = decode_image(dataset["paths"][image_ids[0]])
        engine.detect(first, args.conf, class_names=classes, **common)  # prompt/backend warm-up; excluded
        del first
        records = {"original": []}
        gui_records = {"original": []}
        for level in levels:
            for condition in DARK_CONDITIONS:
                records[condition_key(level["name"], condition)] = []
                gui_records[condition_key(level["name"], condition)] = []
        per_image, mismatched, example = [], 0, {}
        for position, item in enumerate(annotations["images"]):
            image_id = int(item["id"])
            frame = decode_image(dataset["paths"][image_id])
            if frame.shape[:2] != (int(item["height"]), int(item["width"])):
                raise ValueError(f"图片尺寸与标注不一致：{item['file_name']}")
            detections = engine.detect(frame, args.conf, class_names=classes, source="original", **common)
            records["original"] += detection_records(detections, image_id, dataset["category_ids"])
            gui_records["original"] += detection_records(
                [d for d in detections if d.score >= args.gui_conf], image_id, dataset["category_ids"])
            if image_id == example_id:
                example["original"] = (frame.copy(), records["original"][-len(detections):] if detections else [])
            row = {"image_id": image_id, "source": dataset["source_of"][image_id], "original_luma": mean_luma(frame)}
            for index, level in enumerate(levels):
                dark = synthesize_low_light(frame, level["exposure"], image_seed(args.seed, index, image_id),
                                            args.full_well, args.read_noise)
                report["synthesis"]["dark_image_sha256"][f"{level['name']}/{image_id}"] = \
                    hashlib.sha256(dark.tobytes()).hexdigest()
                ap, gui, info, enhanced_images = run_dark_conditions(
                    engine, dark, classes, args.conf, args.gui_conf, args.iou, args.max_det, args.strength,
                    enhancer_factory)
                base_gui = sorted((d.xyxy, d.class_id) for d in gui["dark"])
                dual_base = sorted((d.xyxy, d.class_id) for d in gui["dark_dual"] if d.source == "original")
                if len(base_gui) != len(dual_base) or any(
                        a[1] != b[1] or max(abs(x - y) for x, y in zip(a[0], b[0])) > 0.5
                        for a, b in zip(base_gui, dual_base)):
                    mismatched += 1
                for condition in DARK_CONDITIONS:
                    key = condition_key(level["name"], condition)
                    for target, items in ((records[key], ap[condition]), (gui_records[key], gui[condition])):
                        rows = detection_records(items, image_id, dataset["category_ids"])
                        for record, detection in zip(rows, items):
                            record["source"] = detection.source
                        target.extend(rows)
                    if image_id == example_id:
                        shown = {"dark": dark, "dark_dual": enhanced_images["dark_adaptive"], **enhanced_images}[condition]
                        example[key] = (shown.copy(), [r for r in gui_records[key] if r["image_id"] == image_id])
                row.update({f"{level['name']}_{name}_luma": value for name, value in info["luma"].items()})
                row.update({f"{level['name']}_applied": info["applied"], f"{level['name']}_brightness": info["brightness"],
                            f"{level['name']}_gamma": info["gamma"]})
                if args.save_images:
                    folder = output / "images" / level["name"]
                    folder.mkdir(parents=True, exist_ok=True)
                    import cv2
                    for name, image in (("dark", dark), *enhanced_images.items()):
                        cv2.imwrite(str(folder / f"{image_id:012d}_{name}.png"), image)
            per_image.append(row)
            print(f"lowlight | {position + 1}/{len(image_ids)} | image {image_id}", flush=True)
        engine.unload()
        engine = None
        report["consistency"] = {
            "gui_threshold_filter_vs_dual_base_mismatched_frames": mismatched,
            "meaning": "Frames whose dual-path original-exposure boxes at the GUI threshold differ (>0.5 px or "
                       "class) from the AP-threshold dark pass filtered to that threshold; 0 confirms filtering "
                       "equals rerunning single passes at the higher threshold."}

        # 3. Scoring.
        for key in records:
            atomic_json(output / "predictions" / f"{key.replace('/', '_')}.json", records[key])
            atomic_json(output / "predictions" / f"{key.replace('/', '_')}_gui.json", gui_records[key])
        evaluations, gui_evaluations = {}, {}
        mini_ids = [image_id for image_id in image_ids if dataset["source_of"][image_id] == "mini_eval"]
        for key in records:
            standard = coco_metrics(annotations, records[key], image_ids)
            evaluations[key] = evaluation_tables(annotations, records[key], image_ids)
            gui_evaluations[key] = evaluation_tables(annotations, gui_records[key], image_ids)
            fast = weighted_metrics(evaluations[key])
            for name, value in (("AP", fast["AP"]), ("AP50", fast["AP50"]), ("AP75", fast["AP75"]),
                                ("APsmall", fast["APsmall"]), ("APmedium", fast["APmedium"]),
                                ("APlarge", fast["APlarge"]), ("AR100", fast["AR100"])):
                reference = standard["metrics"][name]
                if (value is None) != (reference is None) or (value is not None and abs(value - reference) > 1e-9):
                    raise RuntimeError(f"加权 COCO 实现与 pycocotools 不一致：{key} {name} {value} != {reference}")
            report["results"][key] = {
                "metrics": standard["metrics"], "per_class": standard["per_class"],
                "gui": threshold_counts(gui_evaluations[key], args.gui_conf),
                "predictions": len(records[key]), "gui_predictions": len(gui_records[key]),
                "sources": source_breakdown(gui_evaluations[key], gui_records[key], args.gui_conf)}
            if mini_ids and len(mini_ids) != len(image_ids):
                report["subsets"].setdefault("mini_eval", {"image_count": len(mini_ids)})[key] = \
                    coco_metrics(annotations, records[key], mini_ids)["metrics"]
        report["fast_evaluator_matches_pycocotools"] = True
        for level in levels:
            name = level["name"]
            report["enhancement"][name] = {
                "images": len(per_image),
                "applied_images": sum(bool(row[f"{name}_applied"]) for row in per_image),
                "mean_luma": {"original": statistics.mean(row["original_luma"] for row in per_image),
                              **{condition: statistics.mean(row[f"{name}_{condition}_luma"] for row in per_image)
                                 for condition in ("dark", "dark_clahe", "dark_adaptive")}},
                "mean_gamma_applied": statistics.mean([row[f"{name}_gamma"] for row in per_image
                                                       if row[f"{name}_applied"]] or [1.0])}
            dual_key = condition_key(name, "dark_dual")
            report["dual_path"][name] = {
                "second_pass_images": report["enhancement"][name]["applied_images"],
                "ap_sources": source_breakdown(evaluations[dual_key], records[dual_key], args.conf),
                "gui_sources": report["results"][dual_key]["sources"],
                "note": "The supplement pass uses max(conf, 0.45); enhanced boxes overlapping any kept "
                        "original-exposure box are dropped, so at the AP threshold most are absorbed by "
                        "low-score dark boxes."}
        with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)

        pairs = []
        for level in levels:
            for left, right in PAIRS:
                pairs.append((condition_key(level["name"], left),
                              condition_key(level["name"], right) if right != "original" else "original"))
        started = time.perf_counter()
        report["bootstrap"] = paired_bootstrap(
            evaluations, gui_evaluations, pairs, args.bootstrap_samples, args.bootstrap_seed, args.gui_conf,
            progress=lambda done: print(f"bootstrap {done}/{args.bootstrap_samples}; "
                                        f"{time.perf_counter() - started:.1f}s", flush=True))
        report["bootstrap"]["elapsed_seconds"] = time.perf_counter() - started

        # 4. Figures and final report.
        report["host"]["loadavg_end"] = os.getloadavg() if hasattr(os, "getloadavg") else None
        report["status"] = "complete"
        report["outputs"] = {"report": "report.json", "summary": "summary.md", "per_image": "per_image.csv",
                             "predictions": "predictions/", "plot": "lowlight_ap.png",
                             "example_grid": f"example_{example_id:012d}.jpg", "example_image_id": example_id,
                             "example_image": example_attribution(annotations, example_id),
                             "images": "images/" if args.save_images else None}
        report["outputs"]["plot_status"] = draw_plot(report, output / "lowlight_ap.png")
        names = {int(item["id"]): item["name"] for item in annotations["categories"]}
        truth = [item for item in annotations["annotations"] if int(item["image_id"]) == example_id]
        panels = []
        for level in levels:
            cells = [("original, exposure x1", *example["original"])]
            cells += [(LABELS[condition].replace("Dark + ", "+ ").replace(", detect enhanced", ""),
                       *example[condition_key(level["name"], condition)]) for condition in DARK_CONDITIONS]
            panels.append((f"{level['name']} x{level['exposure']:g}", cells))
        draw_example_grid(output / report["outputs"]["example_grid"], panels, truth, names, args.gui_conf)
        validate_report(report)
        atomic_json(output / "report.json", report)
        write_summary(output / "summary.md", report)
        print(f"完成：{output / 'summary.md'}", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "stopped"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(output / "report.json", report)
        if engine_factory is None:
            from resource_safety import handle_cuda_oom
            handle_cuda_oom(exc)
        traceback.print_exc()
        print(f"评测已停止，已保存的部分结果见 report.json：{exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.unload()


if __name__ == "__main__":
    raise SystemExit(main())
