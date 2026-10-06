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
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODES = ("text", "text_sahi", "visual", "visual_sahi")
METRIC_NAMES = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
                "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")
NOTICE = (
    "24-image diagnostic subset, not a COCO leaderboard result or a camera-domain validation. "
    "Visual references and evaluation image IDs are disjoint. Visual inference uses one reference "
    "class per pass; its per-image latency sums all class passes. Detection timings include "
    "preprocessing, forward, postprocessing, SAHI fusion and result transfer, excluding image "
    "decoding, prompt encoding, warmup, loading and drawing. Prompt encoding is reported separately. "
    "COCO uses its standard maxDets=[1,10,100]; the same per-image application cap and threshold "
    "apply to every mode, after combining all visual classes."
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
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--device", type=str.upper, choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--modes", type=parse_modes, default=list(MODES))
    parser.add_argument("--output", type=Path, default=Path("work/mini_evaluation"))
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--conf", type=probability, default=0.05)
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


def load_dataset(root, limit=None):
    root = Path(root).resolve()
    annotations = json.loads(local_path(root, "annotations.json").read_text(encoding="utf-8"))
    reference_data = json.loads(local_path(root, "references.json").read_text(encoding="utf-8"))
    images = annotations.get("images", [])
    categories = annotations.get("categories", [])
    classes = reference_data.get("classes", [])
    references = sorted(reference_data.get("references", []), key=lambda item: item["class_id"])
    ids = [int(item["id"]) for item in images]
    if not images or len(set(ids)) != len(ids):
        raise ValueError("评测图片必须非空且 ID 唯一")
    if not classes or len(classes) != len(references) or len(set(classes)) != len(classes):
        raise ValueError("每个目标类别必须有且只有一个独立视觉参考")
    category_names = {int(item["id"]): item["name"] for item in categories}
    if set(category_names.values()) != set(classes):
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
        local_path(root, reference["image_file"])
        box = reference["bbox_xyxy"]
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box) \
                or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("视觉参考框无效")
    for item in images:
        image_path(root, item)
    selected = images[:limit] if limit is not None else images
    hashes = {"annotations.json": sha256_file(root / "annotations.json"),
              "references.json": sha256_file(root / "references.json")}
    if (root / "manifest.json").is_file():
        hashes["manifest.json"] = sha256_file(root / "manifest.json")
    for item in selected:
        path = image_path(root, item)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    for reference in references:
        hashes[reference["image_file"]] = sha256_file(local_path(root, reference["image_file"]))
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"root": root, "annotations": annotations, "images": selected,
            "all_image_count": len(images), "classes": classes, "references": references,
            "category_ids": [int(item["category_id"]) for item in references],
            "sha256": fingerprint, "file_hashes": hashes}


def decode_image(path):
    import cv2
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise ValueError(f"图片解码失败：{path}")
    return frame


def coco_metrics(annotations, predictions, image_ids):
    """Standard bbox COCO AP; preserve original area and iscrowd metadata."""
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
    return {"metrics": metrics, "per_class": class_results, "coco_max_dets": [1, 10, 100]}


def infer(engine, frame, args, sliced, classes):
    common = {"conf": args.conf, "iou": args.iou, "class_names": classes, "max_det": args.max_det}
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
    image_ids = [int(item["id"]) for item in dataset["images"]]
    latency = {image_id: 0.0 for image_id in image_ids}
    predictions, reference_times = [], []
    phases = dataset["references"] if visual else [None]
    for reference in phases:
        category = None
        if reference is not None:
            reference_frame = decoder(local_path(dataset["root"], reference["image_file"]))
            synchronize()
            start = clock()
            engine.set_visual_prompt(reference_frame, reference["bbox_xyxy"], reference["class_name"])
            synchronize()
            reference_times.append({"class_name": reference["class_name"],
                                    "encoding_ms": (clock() - start) * 1000})
            del reference_frame
            category = int(reference["category_id"])
        classes = [reference["class_name"]] if reference is not None else dataset["classes"]
        first = decoder(image_path(dataset["root"], dataset["images"][0]))
        for _ in range(args.warmup):
            infer(engine, first, args, sliced, classes)
        synchronize()
        del first
        for index, item in enumerate(dataset["images"]):
            frame = decoder(image_path(dataset["root"], item))
            if frame.shape[:2] != (int(item["height"]), int(item["width"])):
                raise ValueError(f"图片尺寸与标注不一致：{item['file_name']}")
            synchronize()
            start = clock()
            detections = infer(engine, frame, args, sliced, classes)
            synchronize()
            latency[int(item["id"])] += (clock() - start) * 1000
            predictions.extend(detection_records(detections, item["id"], dataset["category_ids"], category))
            del detections, frame
            print(f"{mode} | {classes} | {index + 1}/{len(dataset['images'])}", flush=True)
    predictions = cap_predictions(predictions, image_ids, args.max_det)
    latencies = [latency[image_id] for image_id in image_ids]
    ordered = sorted(latencies)
    p95_index = (len(ordered) - 1) * 0.95
    p95 = ordered[math.floor(p95_index)] + (ordered[math.ceil(p95_index)] - ordered[math.floor(p95_index)]) \
        * (p95_index - math.floor(p95_index))
    return {"predictions": predictions,
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
        frame = decode_image(image_path(dataset["root"], item))
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


def write_summary(output, report):
    rows = []
    for mode, result in report["modes"].items():
        if result.get("status") != "complete":
            continue
        rows.append({"mode": mode, **result["metrics"], **result["timing_summary"], **result["memory"]})
    fields = ["mode", *METRIC_NAMES, "mean_ms", "median_ms", "p95_ms", "class_passes_per_image",
              "peak_allocated_mb", "peak_reserved_mb"]
    write_csv(output / "comparison.csv", rows, fields)
    lines = ["# 小型标注集对照结果", "", f"状态：{report['status']}；评测图片：{report['image_count']} / {report['all_image_count']}。", "",
             "| 模式 | AP | AP50 | AP75 | APsmall | AR100 | 平均 ms/图 | 峰值分配 MiB | 峰值保留 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["mode"]] + ["—" if row[name] is None else f"{row[name]:.3f}" for name in
                  ("AP", "AP50", "AP75", "APsmall", "AR100", "mean_ms", "peak_allocated_mb", "peak_reserved_mb")]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "AP 与 AR 的范围是 0–1；缺少对应尺寸/类别真值时显示 —。", "",
                  "视觉模式逐类推理，延迟是所有类别的总和；参考编码单独记录，不计入 ms/图。", "",
                  "示例左侧为人工真值，右侧仅显示置信度 ≥ 0.25 的框；评价使用所有达到配置阈值的预测。", "",
                  NOTICE, "", "参数、版本、数据哈希与各类别 AP：见 report.json。原始预测：modes/<mode>/predictions.json。"])
    if report.get("error"):
        lines.extend(["", f"停止原因：{report['error']}"])
    temporary = output / "summary.md.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "summary.md")
    atomic_json(output / "report.json", report)


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
    dataset = load_dataset(args.dataset, args.limit)
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
              "settings": settings, "versions": versions(), "notice": NOTICE,
              "image_count": len(dataset["images"]), "all_image_count": dataset["all_image_count"],
              "image_ids": [int(item["id"]) for item in dataset["images"]],
              "prediction_backend": "BoxOnlySegmentationPredictor; no per-object masks",
              "resource_policy": {"pytorch_gpu_budget_mib": 3072, "min_available_ram_mib": 1536,
                                  "min_free_gpu_inference_mib": 1024, "max_slices": 9,
                                  "max_imgsz": 960, "max_det": 1000},
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
            evaluation = coco_metrics(dataset["annotations"], result["predictions"], report["image_ids"])
            mode_folder = output / "modes" / mode
            mode_folder.mkdir(parents=True, exist_ok=True)
            atomic_json(mode_folder / "predictions.json", result["predictions"])
            write_csv(mode_folder / "timings.csv", result["timings"], ["image_id", "latency_ms"])
            summary = {"status": "complete", **evaluation, "timing_summary": result["timing_summary"],
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
