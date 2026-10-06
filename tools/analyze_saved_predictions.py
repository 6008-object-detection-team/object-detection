"""CPU-only diagnostics and paired image bootstrap of saved COCO detections.

No detector, torch, image decoder, or GPU is imported. This uses the existing
human annotations; the error taxonomy is an explicit heuristic, not TIDE.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODES = ("text", "text_sahi", "visual", "visual_sahi")
PAIRS = (("text_sahi", "text"), ("visual_sahi", "visual"))
ERROR_TYPES = ("duplicate", "class_confusion", "localization",
               "class_and_localization", "background")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(left, right, crowd=False):
    """COCO xywh boxes; crowd overlap divides intersection by detection area."""
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * \
        max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    denominator = lw * lh if crowd else lw * lh + rw * rh - intersection
    return intersection / denominator if denominator > 0 else 0.0


def validate_inputs(annotations, predictions_by_mode):
    ids = [int(item["id"]) for item in annotations["images"]]
    ann_ids = [int(item["id"]) for item in annotations["annotations"]]
    if not ids or len(set(ids)) != len(ids) or len(set(ann_ids)) != len(ann_ids):
        raise ValueError("Image and annotation IDs must be unique; images cannot be empty")
    categories = {int(item["id"]) for item in annotations["categories"]}
    for row in annotations["annotations"]:
        if row["image_id"] not in ids or row["category_id"] not in categories:
            raise ValueError("Unknown GT image or category")
    for mode, predictions in predictions_by_mode.items():
        for row in predictions:
            if row["image_id"] not in ids or row["category_id"] not in categories:
                raise ValueError(f"Unknown prediction image or category: {mode}")
            values = [*row["bbox"], row["score"]]
            if len(row["bbox"]) != 4 or not all(math.isfinite(float(x)) for x in values) \
                    or row["bbox"][2] <= 0 or row["bbox"][3] <= 0 \
                    or not 0 <= row["score"] <= 1:
                raise ValueError(f"Invalid prediction: {mode}")


def diagnose(annotations, predictions, confidence=0.25, match_iou=0.5,
             localization_iou=0.1):
    """Score-ordered, one-to-one, same-class matching at the chosen threshold.

    Unmatched predictions receive the first matching rule: duplicate same-class
    IoU>=0.5, wrong-class IoU>=0.5, same-class IoU in [0.1,0.5), wrong-class
    IoU in [0.1,0.5), background. Crowd/ignore overlaps are excluded after TP
    matching. They are not false positives or false negatives.
    """
    names = {int(item["id"]): item["name"] for item in annotations["categories"]}
    ground_truth = defaultdict(list)
    for row in annotations["annotations"]:
        ground_truth[int(row["image_id"])].append(row)
    sorted_predictions = sorted(
        [(index, row) for index, row in enumerate(predictions) if row["score"] >= confidence],
        key=lambda item: (-item[1]["score"], item[0]),
    )
    matched = set()
    records = []
    for prediction_index, row in sorted_predictions:
        truth = ground_truth[int(row["image_id"])]
        valid = [item for item in truth if not item.get("iscrowd", 0) and not item.get("ignore", 0)]
        same = [(box_iou(row["bbox"], item["bbox"]), item)
                for item in valid if item["category_id"] == row["category_id"]]
        other = [(box_iou(row["bbox"], item["bbox"]), item)
                 for item in valid if item["category_id"] != row["category_id"]]
        candidates = [(overlap, item) for overlap, item in same
                      if overlap >= match_iou and item["id"] not in matched]
        result = {"prediction_index": prediction_index, "image_id": row["image_id"],
                  "category_id": row["category_id"], "score": row["score"],
                  "bbox": row["bbox"], "matched_annotation_id": None}
        if candidates:
            overlap, target = max(candidates, key=lambda item: (item[0], -item[1]["id"]))
            matched.add(target["id"])
            result.update(kind="tp", matched_annotation_id=target["id"], iou=overlap)
        else:
            ignored = [item for item in truth if item.get("iscrowd", 0) or item.get("ignore", 0)]
            ignore_overlap = max((box_iou(row["bbox"], item["bbox"], bool(item.get("iscrowd", 0)))
                                  for item in ignored if item["category_id"] == row["category_id"]),
                                 default=0.0)
            best_same = max((value for value, _ in same), default=0.0)
            best_other = max((value for value, _ in other), default=0.0)
            if ignore_overlap >= match_iou:
                kind = "ignored"
            elif best_same >= match_iou:
                kind = "duplicate"
            elif best_other >= match_iou:
                kind = "class_confusion"
            elif best_same >= localization_iou:
                kind = "localization"
            elif best_other >= localization_iou:
                kind = "class_and_localization"
            else:
                kind = "background"
            result.update(kind=kind, best_same_class_iou=best_same, best_other_class_iou=best_other)
        records.append(result)

    classes = {}
    for category, name in names.items():
        gt = [item for item in annotations["annotations"] if item["category_id"] == category
              and not item.get("iscrowd", 0) and not item.get("ignore", 0)]
        counts = Counter(item["kind"] for item in records if item["category_id"] == category)
        tp, fp = counts["tp"], sum(counts[kind] for kind in ERROR_TYPES)
        classes[name] = {"category_id": category, "gt_count": len(gt), "tp": tp, "fp": fp,
                         "fn": len(gt) - tp, "ignored": counts["ignored"],
                         "precision": tp / (tp + fp) if tp + fp else None,
                         "recall": tp / len(gt) if gt else None,
                         "fp_types": {kind: counts[kind] for kind in ERROR_TYPES}}
    return {"confidence": confidence, "match_iou": match_iou,
            "localization_iou": localization_iou, "per_class": classes,
            "matched_annotation_ids": sorted(matched), "predictions": records}


def positive_counts(annotations):
    counts = {int(item["id"]): {"all": 0, "small": 0}
              for item in annotations["categories"]}
    for row in annotations["annotations"]:
        if row.get("iscrowd", 0) or row.get("ignore", 0):
            continue
        counts[int(row["category_id"])]["all"] += 1
        if row.get("area", row["bbox"][2] * row["bbox"][3]) <= 32 ** 2:
            counts[int(row["category_id"])]["small"] += 1
    return counts


def confidence_reliability(diagnostic, bins=10):
    """Describe score reliability under this exact IoU matching rule.

    This is a histogram ECE for binary TP/FP labels, not standard multivariate
    detection ECE. Matching depends on image context, duplicate priority,
    score ordering and available labels. Nothing is fitted or recalibrated.
    """
    if bins < 1:
        raise ValueError("bins must be positive")

    def summarize(records):
        buckets = [[] for _ in range(bins)]
        for row in records:
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("scores must be finite and between zero and one")
            buckets[min(int(score * bins), bins - 1)].append(row)
        rows, weighted_error = [], 0.0
        for index, bucket in enumerate(buckets):
            count = len(bucket)
            mean_score = sum(row["score"] for row in bucket) / count if count else None
            tp_rate = sum(row["kind"] == "tp" for row in bucket) / count if count else None
            gap = abs(tp_rate - mean_score) if count else None
            if count:
                weighted_error += count * gap
            rows.append({"bin": index, "lower_inclusive": index / bins,
                         "upper": (index + 1) / bins, "upper_inclusive": index == bins - 1,
                         "count": count, "tp_count": sum(row["kind"] == "tp" for row in bucket),
                         "mean_score": mean_score, "empirical_tp_rate": tp_rate,
                         "absolute_gap": gap})
        return {"prediction_count": len(records),
                "ece": weighted_error / len(records) if records else None, "bins": rows}

    records = [row for row in diagnostic["predictions"] if row["kind"] != "ignored"]
    return {"confidence_floor": diagnostic["confidence"], "match_iou": diagnostic["match_iou"],
            "definition": "10 equal score bins; count-weighted absolute mean-score vs binary TP-rate gap",
            "limitations": "Rule-specific descriptive ECE; no fitted calibration, not multivariate D-ECE; incomplete category labels and score-order matching affect TP/FP",
            "overall": summarize(records),
            "per_class": {name: summarize([row for row in records if row["category_id"] == values["category_id"]])
                          for name, values in diagnostic["per_class"].items()}}


def coco_ap(annotations, predictions, required_categories=None):
    """Standard COCO precision, preserving area/crowd. Missing positives => null.

    Global bootstrap metrics require every originally valid category to remain
    present in that area range. This avoids changing the category composition
    of the aggregate when a resample drops a rare class.
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    counts = positive_counts(annotations)
    categories = sorted(counts)
    required_categories = required_categories or {
        area: [category for category in categories if counts[category][area] > 0]
        for area in ("all", "small")
    }
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(annotations)
        truth.createIndex()
        if predictions:
            detected = truth.loadRes(copy.deepcopy(predictions))
        else:
            detected = COCO()
            detected.dataset = {"images": annotations["images"],
                                "categories": annotations["categories"], "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = [int(item["id"]) for item in annotations["images"]]
        evaluator.params.catIds = categories
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()

    def valid_mean(values):
        values = values[values > -1]
        return float(values.mean()) if values.size else None

    precision = evaluator.eval["precision"]
    per_class = {}
    for index, category in enumerate(categories):
        per_class[category] = {"AP": valid_mean(precision[:, :, index, 0, -1]),
                               "APsmall": valid_mean(precision[:, :, index, 1, -1])}
    metrics = {}
    for metric, area in (("AP", "all"), ("APsmall", "small")):
        required = required_categories[area]
        metrics[metric] = (sum(per_class[category][metric] for category in required) / len(required)
                           if required and all(counts[category][area] > 0 for category in required)
                           else None)
    return {"metrics": metrics, "per_class": per_class, "positive_counts": counts}


def resample_images(annotations, predictions_by_mode, sampled_ids):
    """Duplicate a sampled image and its GT/predictions under fresh IDs.

    Passing repeated original IDs directly to COCOeval is incorrect: its ID
    lookup/set handling would collapse duplicate observations.
    """
    images = {int(item["id"]): item for item in annotations["images"]}
    truth_by_image = defaultdict(list)
    predictions_by_image = {mode: defaultdict(list) for mode in predictions_by_mode}
    for row in annotations["annotations"]:
        truth_by_image[int(row["image_id"])].append(row)
    for mode, predictions in predictions_by_mode.items():
        for row in predictions:
            predictions_by_image[mode][int(row["image_id"])].append(row)
    sampled = {key: copy.deepcopy(value) for key, value in annotations.items()
               if key not in ("images", "annotations")}
    sampled.update(images=[], annotations=[])
    detected = {mode: [] for mode in predictions_by_mode}
    new_annotation_id = 1
    for new_id, original_id in enumerate(sampled_ids, start=1):
        image = copy.deepcopy(images[int(original_id)])
        image.update(id=new_id, source_image_id=int(original_id))
        sampled["images"].append(image)
        for row in truth_by_image[int(original_id)]:
            copied = copy.deepcopy(row)
            copied.update(id=new_annotation_id, image_id=new_id, source_annotation_id=row["id"])
            sampled["annotations"].append(copied)
            new_annotation_id += 1
        for mode in predictions_by_mode:
            for row in predictions_by_image[mode][int(original_id)]:
                copied = copy.deepcopy(row)
                copied["image_id"] = new_id
                detected[mode].append(copied)
    return sampled, detected


def interval(values, total, confidence=0.95):
    import numpy as np
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    result = {"requested_samples": total, "valid_samples": int(valid.size),
              "valid_fraction": float(valid.size / total), "mean": None,
              "percentile_interval": None, "probability_positive": None}
    if valid.size:
        tail = (1 - confidence) / 2
        result.update(mean=float(valid.mean()),
                      percentile_interval=[float(value) for value in np.quantile(valid, [tail, 1 - tail])],
                      probability_positive=float((valid > 0).mean()))
    return result


def paired_bootstrap(annotations, predictions_by_mode, samples=200, seed=20261004,
                     confidence=0.95, progress=None):
    import numpy as np
    image_ids = [int(item["id"]) for item in annotations["images"]]
    baseline_counts = positive_counts(annotations)
    required = {area: [category for category in sorted(baseline_counts)
                       if baseline_counts[category][area] > 0] for area in ("all", "small")}
    point = {mode: coco_ap(annotations, predictions, required)
             for mode, predictions in predictions_by_mode.items()}
    rng = np.random.default_rng(seed)
    draws = []
    for sample_index in range(samples):
        chosen = rng.choice(image_ids, size=len(image_ids), replace=True).tolist()
        sampled, detected = resample_images(annotations, predictions_by_mode, chosen)
        metrics = {mode: coco_ap(sampled, predictions, required)
                   for mode, predictions in detected.items()}
        draws.append({"sample": sample_index, "source_image_ids": chosen, "modes": metrics})
        if progress and ((sample_index + 1) % 20 == 0 or sample_index + 1 == samples):
            progress(draws)
    mode_results = {}
    for mode in predictions_by_mode:
        mode_results[mode] = {metric: interval(
            [draw["modes"][mode]["metrics"][metric] for draw in draws], samples, confidence)
            for metric in ("AP", "APsmall")}
        mode_results[mode]["per_class"] = {
            category: {metric: interval([draw["modes"][mode]["per_class"][category][metric]
                                        for draw in draws], samples, confidence)
                       for metric in ("AP", "APsmall")}
            for category in sorted(baseline_counts)
        }
    differences = {}
    for enhanced, base in PAIRS:
        if enhanced not in predictions_by_mode or base not in predictions_by_mode:
            continue
        key = f"{enhanced}_minus_{base}"
        differences[key] = {}
        for metric in ("AP", "APsmall"):
            values = []
            for draw in draws:
                left = draw["modes"][enhanced]["metrics"][metric]
                right = draw["modes"][base]["metrics"][metric]
                values.append(left - right if left is not None and right is not None else None)
            result = interval(values, samples, confidence)
            left, right = point[enhanced]["metrics"][metric], point[base]["metrics"][metric]
            result["point_difference"] = left - right if left is not None and right is not None else None
            differences[key][metric] = result
        differences[key]["per_class"] = {}
        for category in sorted(baseline_counts):
            differences[key]["per_class"][category] = {}
            for metric in ("AP", "APsmall"):
                values = []
                for draw in draws:
                    left = draw["modes"][enhanced]["per_class"][category][metric]
                    right = draw["modes"][base]["per_class"][category][metric]
                    values.append(left - right if left is not None and right is not None else None)
                result = interval(values, samples, confidence)
                left, right = point[enhanced]["per_class"][category][metric], point[base]["per_class"][category][metric]
                result["point_difference"] = left - right if left is not None and right is not None else None
                differences[key]["per_class"][category][metric] = result
    return {"seed": seed, "samples": samples, "confidence": confidence,
            "sampling_unit": "image", "paired_across_all_modes": True,
            "required_categories_by_area": required,
            "missing_positive_policy": "null; aggregate requires every original valid category in that area",
            "point": point, "modes": mode_results, "differences": differences, "draws": draws}


def pixel_audit(annotations, predictions_by_mode, diagnostics, image_size=640):
    images = {int(item["id"]): item for item in annotations["images"]}
    names = {int(item["id"]): item["name"] for item in annotations["categories"]}
    matched = {key: set(value["matched_annotation_ids"]) for key, value in diagnostics.items()}
    predictions_by_image = {mode: defaultdict(list) for mode in predictions_by_mode}
    for mode, predictions in predictions_by_mode.items():
        for row in predictions:
            predictions_by_image[mode][int(row["image_id"])].append(row)
    records = []
    for row in annotations["annotations"]:
        image = images[int(row["image_id"])]
        ratio = min(image_size / image["width"], image_size / image["height"])
        width, height = row["bbox"][2:]
        effective_width, effective_height = width * ratio, height * ratio
        record = {"annotation_id": row["id"], "image_id": row["image_id"],
                  "file_name": image["file_name"], "class_name": names[row["category_id"]],
                  "category_id": row["category_id"], "iscrowd": row.get("iscrowd", 0),
                  "original_image_width": image["width"], "original_image_height": image["height"],
                  "original_bbox_width": width, "original_bbox_height": height,
                  "coco_annotation_area": row.get("area", width * height),
                  "coco_small": row.get("area", width * height) <= 32 ** 2,
                  "letterbox_scale": ratio, "letterbox_bbox_width": effective_width,
                  "letterbox_bbox_height": effective_height,
                  "letterbox_min_side": min(effective_width, effective_height),
                  "letterbox_bbox_area": effective_width * effective_height,
                  "letterbox_min_side_p3_cells": min(effective_width, effective_height) / 8,
                  "letterbox_min_side_p2_cells": min(effective_width, effective_height) / 4}
        for mode in predictions_by_mode:
            candidates = [(box_iou(row["bbox"], item["bbox"]), item["score"])
                          for item in predictions_by_image[mode][int(row["image_id"])]
                          if item["category_id"] == row["category_id"]]
            best = max(candidates, default=(0.0, None))
            record[f"{mode}_best_same_class_iou"] = best[0]
            record[f"{mode}_best_iou_prediction_score"] = best[1]
            for threshold in (0.05, 0.25):
                record[f"{mode}_matched_conf_{threshold:.2f}"] = row["id"] in matched[f"{mode}@{threshold:.2f}"]
        records.append(record)
    return records


def reliability_csv_records(reliability):
    return [{"mode": mode, "class_name": name, **bucket}
            for mode, result in reliability.items()
            for name, values in [("all", result["overall"]), *result["per_class"].items()]
            for bucket in values["bins"]]


def csv_records(path, records):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [])
        writer.writeheader()
        writer.writerows(records)


def number(value):
    return "无正例" if value is None else f"{value:.3f}"


def write_report(path, result):
    rows = ["# 已保存预测的 CPU 分析", "",
            f"完成时间：{result['created_at']}；耗时 {result['elapsed_seconds']:.1f} 秒。",
            "本次只读取已有预测与人工标注；未加载 Torch、模型或访问 GPU。", "",
            "## 规则诊断（置信度 0.25，IoU 0.5）", "",
            "| 模式 | 类别 | TP | FP | FN | 重复框 | 串类 | 定位偏差 | 串类且定位偏差 | 背景 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for name, row in result["diagnostics"][f"{mode}@0.25"]["per_class"].items():
            errors = row["fp_types"]
            rows.append(f"| {mode} | {name} | {row['tp']} | {row['fp']} | {row['fn']} | "
                        + " | ".join(str(errors[kind]) for kind in ERROR_TYPES) + " |")
    rows += ["", "匹配按置信度降序、同类 IoU≥0.5、一对一进行。未匹配预测依次归入：同类重叠重复框；"
             "异类 IoU≥0.5 串类；同类 IoU∈[0.1,0.5) 定位偏差；异类 IoU∈[0.1,0.5)；背景。"
             "有重叠的复杂错误会按该优先级归类。这是明确规则的辅助诊断，并非 TIDE 原工具或其误差对 AP 的贡献。"
             "低置信度完整诊断另存 diagnostics.json。此数据集只有三个已标注类别；背景列只表示没有满足规则的这三个类别 GT。",
             "", "## 配对按图 Bootstrap", "",
             f"种子 {result['bootstrap']['seed']}，{result['bootstrap']['samples']} 次，"
             "每次有放回抽取 24 张图片，四组使用相同图片序列，重复图片和其标注均重新编号。"
             "标准 COCO IoU 0.50:0.95、maxDets=100，保留原始标注面积。", "",
             "| 对照（前者减后者） | 指标 | 原始差值 | 95% 百分位区间 | 有效抽样 |",
             "|---|---|---:|---|---:|"]
    for pair, metrics in result["bootstrap"]["differences"].items():
        for metric in ("AP", "APsmall"):
            entry = metrics[metric]
            interval_text = ("无正例" if entry["percentile_interval"] is None else
                             "[" + ", ".join(number(value) for value in entry["percentile_interval"]) + "]")
            rows.append(f"| {pair} | {metric} | {number(entry['point_difference'])} | {interval_text} | "
                        f"{entry['valid_samples']}/{entry['requested_samples']} |")
    rows += ["", "某类别/面积档在重采样中没有正例时，该 AP 记为 null，不记为 0。"
             "整体 AP 要求原测试集有正例的所有类别仍有正例，避免把缺失类别后改变组成的均值混进同一统计；"
             "区间因此只使用有效抽样，并公开有效比例。这些是条件百分位区间，200 次仅适合初步诊断，"
             "不等同于模型泛化的严格显著性证明。相邻视频帧应按场景或片段抽样，不能当独立图片。",
             "小手机只有 4 个 GT，不能据此下稳定提升结论；扩大独立场景测试集比增加抽样次数更重要。", "",
             "## 像素审计", "",
             "ground_truth_pixels.csv 逐个列出 71 个 GT 的原框尺寸、原图尺寸、LetterBox 640 的缩放比例/框尺寸/"
             "最短边、最短边对应的 stride 8 与 stride 4 网格数量，以及四组在 0.05 和 0.25 下是否一对一匹配。"
             "该尺寸审计不表示每个目标能检出的硬阈值，也没有运行 P2 模型。", ""]
    for name, row in result["pixel_summary"].items():
        rows.append(f"- {name}：{row['gt_count']} 个 GT，其中 COCO small {row['small_count']} 个；"
                    f"640 输入最短边 <8 像素 {row['min_side_below_8']} 个，<16 像素 {row['min_side_below_16']} 个。")
    rows += ["", "原图最长边均不超过 640；缩放可以放大已有像素，但不会恢复采集时缺失的细节。"
             "此处有效像素是几何框在模型输入中的大小，COCO small 仍按原始人工标注 area 判断。", "",
             "## 置信度可靠性（描述性）", "",
             "复用 IoU 0.5、一对一匹配后的 TP/FP，保留 score≥0.05，排除 ignored。按 [0,0.1)、"
             "[0.1,0.2)…[0.9,1.0] 分成十个等宽桶，记录样本数、平均 score、经验 TP 比例；"
             "空桶为 null。ECE 是各桶绝对差的样本数加权平均。", "",
             "| 模式 | 类别 | 框数 | 规则 ECE |", "|---|---|---:|---:|"]
    for mode, reliability in result.get("confidence_reliability", {}).items():
        for name, row in [("all", reliability["overall"]), *reliability["per_class"].items()]:
            rows.append(f"| {mode} | {name} | {row['prediction_count']} | {number(row['ece'])} |")
    rows += ["", "这是当前阈值和匹配规则下的描述性可靠性分析，不是标准多变量 D-ECE；"
             "它受重复框匹配优先级、类别标注范围和置信度筛选影响，不能直接解读成真实场景中的概率误差。"
             "本次没有拟合校准器、修改分数或改变任何已有 AP。71 个 GT 不足以同时可靠地拟合和验证校准器。"
             "各桶详情见 confidence_reliability.json / .csv。", "",
             "## 可复现输入", "", "文件 SHA-256、参数、每次抽样 ID、原始差值及全部类别区间均记录在 report.json / bootstrap_draws.json。"]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("datasets/mini_eval/annotations.json"))
    parser.add_argument("--predictions-root", type=Path, default=Path("work/mini_evaluation/modes"))
    parser.add_argument("--output", type=Path, default=Path("work/additional_experiments/offline_analysis"))
    parser.add_argument("--samples", type=positive_int, default=200)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    prediction_paths = {mode: args.predictions_root / mode / "predictions.json" for mode in MODES}
    predictions_by_mode = {mode: json.loads(path.read_text(encoding="utf-8"))
                           for mode, path in prediction_paths.items()}
    validate_inputs(annotations, predictions_by_mode)
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics = {f"{mode}@{threshold:.2f}": diagnose(annotations, predictions, threshold)
                   for mode, predictions in predictions_by_mode.items() for threshold in (0.05, 0.25)}
    atomic_json(args.output / "diagnostics.json", diagnostics)
    reliability = {mode: confidence_reliability(diagnostics[f"{mode}@0.05"])
                   for mode in predictions_by_mode}
    atomic_json(args.output / "confidence_reliability.json", reliability)
    csv_records(args.output / "confidence_reliability.csv", reliability_csv_records(reliability))
    pixels = pixel_audit(annotations, predictions_by_mode, diagnostics, args.imgsz)
    csv_records(args.output / "ground_truth_pixels.csv", pixels)

    def progress(draws):
        atomic_json(args.output / "bootstrap_draws.partial.json", {"seed": args.seed, "samples": draws})
        print(f"CPU bootstrap {len(draws)}/{args.samples}; elapsed {time.perf_counter() - started:.1f}s", flush=True)

    bootstrap = paired_bootstrap(annotations, predictions_by_mode, args.samples, args.seed, progress=progress)
    atomic_json(args.output / "bootstrap_draws.json", {"seed": args.seed, "samples": bootstrap.pop("draws")})
    pixel_summary = {}
    for category in annotations["categories"]:
        selected = [row for row in pixels if row["category_id"] == category["id"]]
        pixel_summary[category["name"]] = {"gt_count": len(selected),
            "small_count": sum(row["coco_small"] for row in selected),
            "min_side_below_8": sum(row["letterbox_min_side"] < 8 for row in selected),
            "min_side_below_16": sum(row["letterbox_min_side"] < 16 for row in selected)}
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
              "elapsed_seconds": time.perf_counter() - started,
              "image_count": len(annotations["images"]), "annotation_count": len(annotations["annotations"]),
              "settings": {"samples": args.samples, "seed": args.seed, "imgsz": args.imgsz,
                           "diagnostic_confidences": [0.05, 0.25], "iou": 0.5},
              "input_sha256": {str(args.annotations): sha256(args.annotations),
                               **{str(path): sha256(path) for path in prediction_paths.values()}},
              "diagnostics": {key: {k: v for k, v in value.items() if k != "predictions"}
                              for key, value in diagnostics.items()},
              "pixel_summary": pixel_summary, "confidence_reliability": reliability,
              "bootstrap": bootstrap}
    atomic_json(args.output / "report.json", result)
    write_report(args.output / "summary.md", result)
    (args.output / "bootstrap_draws.partial.json").unlink(missing_ok=True)
    print(f"CPU analysis complete: {args.output.resolve()}; {result['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
