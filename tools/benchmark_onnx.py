"""PyTorch vs ONNX Runtime on the human-annotated 24-image diagnostic set.

--help and --plan load no model. Each backend runs in its own fresh worker
process, one at a time, so its peak RSS is its own and no two models coexist.
Every repeat runs all backends back to back in a rotated order; host load is
recorded before and after each worker. A failed worker stops the benchmark
without retrying it or switching device. Evaluation uses the dataset's human
COCO boxes only; agreement compares predictions with predictions.
"""
import argparse
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

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluate_mini_dataset import (
    METRIC_NAMES, atomic_json, cap_predictions, coco_metrics, decode_image, detection_records,
    image_path, load_dataset, positive_integer, probability, sha256_file, versions, write_csv,
)

BACKENDS = {
    "pt_cpu_rect": {"runtime": "pytorch", "device": "CPU", "letterbox": "rect", "graph": "seg",
                    "note": "Current app path (OpenVocabEngine, box-only predictor, minimal rectangular letterbox)."},
    "pt_cpu_square": {"runtime": "pytorch", "device": "CPU", "letterbox": "square", "graph": "seg",
                      "note": "Same app path, but the static ONNX square letterbox; main PyTorch reference."},
    "pt_cpu_det_square": {"runtime": "pytorch", "device": "CPU", "letterbox": "square", "graph": "det",
                          "note": "PyTorch detection-branch-only conversion with the square letterbox."},
    "ort_cpu_seg": {"runtime": "onnxruntime", "device": "CPU", "letterbox": "square", "graph": "seg", "onnx": "seg",
                    "note": "ONNX full segmentation graph; mask prototypes computed, ignored."},
    "ort_cpu_det": {"runtime": "onnxruntime", "device": "CPU", "letterbox": "square", "graph": "det", "onnx": "det",
                    "note": "ONNX detection-branch-only graph."},
    "ort_cpu_det_rect": {"runtime": "onnxruntime", "device": "CPU", "letterbox": "rect", "graph": "det",
                         "onnx": "det_rect",
                         "note": "ONNX detection-branch-only graph with dynamic height/width (app-path rectangle)."},
    "pt_cuda_rect": {"runtime": "pytorch", "device": "CUDA", "letterbox": "rect", "graph": "seg",
                     "note": "Reference only: app path on CUDA under the 3 GiB budget and GPU lock."},
}
DEFAULT_BACKENDS = ["pt_cpu_rect", "pt_cpu_square", "pt_cpu_det_square", "ort_cpu_seg", "ort_cpu_det"]
ONNX_INPUTS = {"seg": ("seg", False), "det": ("det", False), "det_rect": ("det", True)}  # graph, dynamic H/W
REFERENCE = "pt_cpu_square"
NOTICE = (
    "24-image development diagnostic subset with human COCO boxes; not an unseen final test or a leaderboard. "
    "Latency = one detect() call (preprocessing, forward, decoding/NMS, result transfer); excludes image "
    "decoding, model loading, prompt encoding and warm-up. PyTorch's thread count is restored after Ultralytics "
    "predictor setup (which resets it to min(8, CPUs - 1)) and verified after timing. Each backend runs in a fresh process per repeat; "
    "repeats are back to back on one host in a rotated order with load averages recorded. Peak RSS is each "
    "worker's own lifetime high-water mark including loading. onnxruntime-gpu is not installed, so ONNX Runtime "
    "ran only on CPUExecutionProvider. Agreement compares two models' predictions, never ground truth."
)


def parse_backends(value):
    names = [part.strip() for part in value.split(",")]
    if not names or len(set(names)) != len(names) or any(name not in BACKENDS for name in names):
        raise argparse.ArgumentTypeError("后端必须是不重复的已知名称：" + ",".join(BACKENDS))
    return names


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch 与 ONNX Runtime 在 24 图人工标注诊断集上的精度、一致性、时延与内存对照。")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/mini_eval"))
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--onnx-seg", type=Path, help="tools/export_onnx.py 生成的 seg 变体 .onnx")
    parser.add_argument("--onnx-det", type=Path, help="tools/export_onnx.py 生成的 det 变体 .onnx")
    parser.add_argument("--onnx-det-rect", type=Path, help="--dynamic-hw 导出的 det 变体 .onnx（矩形 letterbox）")
    parser.add_argument("--backends", type=parse_backends, default=list(DEFAULT_BACKENDS))
    parser.add_argument("--output", type=Path, default=Path("work/onnx_benchmark"))
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--threads", type=positive_integer, default=2)
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--conf", type=probability, default=0.05)
    parser.add_argument("--iou", type=probability, default=0.5)
    parser.add_argument("--max-det", type=positive_integer, default=100)
    parser.add_argument("--warmup", type=positive_integer, default=1)
    parser.add_argument("--limit", type=positive_integer, help="只用前 N 张图做流程检查；报告标为 smoke")
    parser.add_argument("--app-conf", type=probability, default=0.25, help="一致性额外按界面常用阈值过滤后再比较")
    parser.add_argument("--match-iou", type=probability, default=0.9)
    parser.add_argument("--timeout", type=positive_integer, default=900, help="单个 worker 的秒数上限")
    parser.add_argument("--plan", action="store_true", help="只验证输入并输出计划，不加载模型")
    parser.add_argument("--worker", choices=tuple(BACKENDS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


# ---------------------------------------------------------------- statistics

def percentile(values, fraction):
    """Linear interpolation at (N-1)*fraction, as the existing timing summaries."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("没有计时样本")
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_summary(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("计时样本必须是非负有限数")
    return {"n": len(values), "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95), "min_ms": min(values), "max_ms": max(values),
            "sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0}


def repeat_summary(per_repeat):
    """Pooled statistics plus the spread of the per-repeat means/medians."""
    pooled = latency_summary([value for values in per_repeat for value in values])
    summaries = [latency_summary(values) for values in per_repeat]
    means = [item["mean_ms"] for item in summaries]
    medians = [item["median_ms"] for item in summaries]
    return {"pooled": pooled, "per_repeat": summaries, "repeat_means_ms": means, "repeat_medians_ms": medians,
            "repeat_mean_range_ms": [min(means), max(means)],
            "repeat_mean_sd_ms": statistics.stdev(means) if len(means) > 1 else 0.0}


# ---------------------------------------------------------------- agreement

def xywh_iou(first, second):
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    width = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    height = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_image(reference, candidate, iou_threshold):
    """Greedy by reference score: same class and IoU strictly above the threshold."""
    used, matches = set(), []
    for left in sorted(range(len(reference)), key=lambda index: reference[index]["score"], reverse=True):
        best, best_iou = None, iou_threshold
        for right, item in enumerate(candidate):
            if right in used or item["category_id"] != reference[left]["category_id"]:
                continue
            overlap = xywh_iou(reference[left]["bbox"], item["bbox"])
            if overlap > best_iou:
                best, best_iou = right, overlap
        if best is not None:
            used.add(best)
            matches.append({"iou": best_iou, "score_delta": candidate[best]["score"] - reference[left]["score"]})
    return matches


def agreement(reference, candidate, image_ids, iou_threshold=0.9, min_score=0.0):
    def grouped(records):
        groups = {int(image_id): [] for image_id in image_ids}
        for record in records:
            if record["score"] >= min_score:
                groups[int(record["image_id"])].append(record)
        return groups

    left, right = grouped(reference), grouped(candidate)
    matches = [match for image_id in left for match in match_image(left[image_id], right[image_id], iou_threshold)]
    reference_count = sum(map(len, left.values()))
    candidate_count = sum(map(len, right.values()))
    deltas = [abs(match["score_delta"]) for match in matches]
    ious = [match["iou"] for match in matches]
    return {
        "min_score": min_score, "iou_threshold": iou_threshold,
        "reference_count": reference_count, "candidate_count": candidate_count, "matched": len(matches),
        "unmatched_reference": reference_count - len(matches), "unmatched_candidate": candidate_count - len(matches),
        "reference_match_rate": len(matches) / reference_count if reference_count else None,
        "candidate_match_rate": len(matches) / candidate_count if candidate_count else None,
        "mean_iou": statistics.mean(ious) if ious else None, "min_iou": min(ious) if ious else None,
        "mean_abs_score_delta": statistics.mean(deltas) if deltas else None,
        "p95_abs_score_delta": percentile(deltas, 0.95) if deltas else None,
        "max_abs_score_delta": max(deltas) if deltas else None,
        "mean_signed_score_delta": statistics.mean(match["score_delta"] for match in matches) if matches else None,
    }


# ---------------------------------------------------------------- validation and fingerprints

def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def required_onnx(backends):
    return {key: any(BACKENDS[name].get("onnx") == key for name in backends) for key in ONNX_INPUTS}


def validate_inputs(args, dataset):
    """Everything checked before any worker loads a model."""
    from onnx_engine import read_manifest
    from resource_safety import preflight_inference
    if args.repeats < 3 and args.limit is None:
        raise ValueError("正式计时至少重复 3 次；少于 3 次只能配合 --limit 做流程检查")
    if args.match_iou <= 0 or args.max_det > 1000 or not 32 <= args.imgsz <= 960:
        raise ValueError("match-iou 必须大于 0，max-det ≤ 1000，imgsz 在 32–960")
    if REFERENCE not in args.backends:
        raise ValueError(f"必须包含参考后端 {REFERENCE}")
    model = project_path(args.model)
    if not model.is_file() or model.stat().st_size < 1024 * 1024:
        raise ValueError(f"找不到可加载的 PyTorch 权重：{model}")
    for item in dataset["images"]:
        preflight_inference("cpu", args.imgsz, (int(item["height"]), int(item["width"]), 3))
    manifests = {}
    for key, needed in required_onnx(args.backends).items():
        path, option = getattr(args, f"onnx_{key}"), "--onnx-" + key.replace("_", "-")
        if not needed:
            continue
        if path is None:
            raise ValueError(f"所选后端需要 {option}")
        manifest = read_manifest(project_path(path))
        graph, dynamic = ONNX_INPUTS[key]
        if manifest["variant"] != graph or manifest["export"].get("dynamic_hw", False) != dynamic:
            raise ValueError(f"{option} 的 manifest 变体/动态尺寸不符：{manifest['variant']}")
        if manifest["classes"] != list(dataset["classes"]):
            raise ValueError("ONNX 固定类别及顺序必须与数据集类别一致：" + ", ".join(manifest["classes"]))
        if manifest["export"]["imgsz"] != args.imgsz:
            raise ValueError("ONNX 输入尺寸与 --imgsz 不一致")
        if manifest["source_model"]["sha256"] != sha256_file(model):
            raise ValueError("ONNX 不是由同一份 PyTorch 权重导出的")
        manifests[key] = manifest
    return manifests


def host_info():
    cpu = None
    try:
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), None)
    except OSError:
        pass
    return {"cpu": cpu or platform.processor(), "logical_cpus": os.cpu_count(), "platform": platform.platform()}


def loadavg():
    return list(os.getloadavg()) if hasattr(os, "getloadavg") else None


def source_hashes():
    return {relative: sha256_file(PROJECT / relative) for relative in (
        "tools/benchmark_onnx.py", "tools/export_onnx.py", "onnx_engine.py", "engines.py", "box_predictor.py",
        "resource_safety.py", "evaluate_mini_dataset.py", "detection_utils.py")}


def model_files(args, manifests):
    from tools.export_onnx import TEXT_ENCODER, file_record
    model = project_path(args.model)
    files = {"pytorch_weights": file_record(model)}
    encoder = PROJECT / TEXT_ENCODER
    if encoder.is_file():
        files["text_encoder"] = {**file_record(encoder),
                                 "note": "Needed by PyTorch to encode text prompts; not needed by the fixed-vocabulary ONNX."}
    for graph, manifest in manifests.items():
        path = project_path(getattr(args, f"onnx_{graph}"))
        files[f"onnx_{graph}"] = {**file_record(path), "manifest_sha256": sha256_file(path.with_suffix(".manifest.json")),
                                  "opset": manifest["export"]["opset"]}
    return files


# ---------------------------------------------------------------- worker (one backend, one fresh process)

def rss_mib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def square_engine_class():
    from engines import OpenVocabEngine

    class SquareLetterboxEngine(OpenVocabEngine):
        """Unchanged app predictor with a fixed imgsz x imgsz letterbox (rect=False)."""

        def _runtime_prediction_options(self):
            return {**super()._runtime_prediction_options(), "rect": False}

    return SquareLetterboxEngine


def load_pytorch_detection_branch(engine, model_path, classes):
    """Seed text embeddings with the seg checkpoint, release it, then build the detect-only graph."""
    from ultralytics import YOLOE
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from tools.export_onnx import build_detection_model
    from resource_safety import prepare_device
    import gc
    engine.device = prepare_device("CPU")
    seed = YOLOE(str(model_path))
    start = time.perf_counter()
    embeddings = seed.get_text_pe(classes).detach().cpu()
    encoding_ms = (time.perf_counter() - start) * 1000
    state = {key: value.detach().float().clone() for key, value in seed.model.state_dict().items()}
    del seed
    gc.collect()
    engine.model, transfer = build_detection_model(model_path, classes, embeddings, state)
    del state
    gc.collect()
    engine._prediction_head_options = {"nms": None} if DEFAULT_CFG_DICT.get("nms", False) is None else {"end2end": False}
    engine._precision_uses_quantize = "quantize" in DEFAULT_CFG_DICT
    engine._select_head_options()
    engine.current_classes = list(classes)
    return encoding_ms, transfer


def pytorch_stage_ms(engine):
    try:
        speed = engine.model.predictor.results[0].speed
        return {"preprocess_ms": speed["preprocess"], "inference_ms": speed["inference"],
                "postprocess_ms": speed["postprocess"]}
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def run_worker(args):
    config = BACKENDS[args.worker]
    started = {"loadavg_before": loadavg(), "started_at": datetime.now(timezone.utc).isoformat()}
    from resource_safety import available_ram_mb, process_max_rss_mib
    started["available_ram_mib_before"] = available_ram_mb()
    import cv2
    cv2.setNumThreads(1)
    dataset = load_dataset(project_path(args.dataset), args.limit)
    classes = list(dataset["classes"])
    model_path = project_path(args.model)
    synchronize = lambda: None
    extra = {}
    threads = {"requested": args.threads}
    clock = time.perf_counter
    start = clock()
    if config["runtime"] == "onnxruntime":
        from onnx_engine import OnnxRuntimeEngine
        engine = OnnxRuntimeEngine()
        status = engine.load_model(project_path(getattr(args, f"onnx_{config['onnx']}")), "CPU", threads=args.threads)
        if status != "SUCCESS":
            raise RuntimeError(f"ONNX 模型加载未完成：{status}")
        load_ms, encoding_ms = (clock() - start) * 1000, 0.0
        extra["providers"] = engine.providers
        threads["onnxruntime_intra_op"] = args.threads
    else:
        import torch
        torch.set_num_threads(args.threads)
        if config["device"] == "CUDA":
            from engines import OpenVocabEngine
            engine = OpenVocabEngine()
        elif config["letterbox"] == "square":
            engine = square_engine_class()()
        else:
            from engines import OpenVocabEngine
            engine = OpenVocabEngine()
        engine.imgsz = args.imgsz
        if config["graph"] == "det":
            encoding_ms, extra["detection_transfer"] = load_pytorch_detection_branch(engine, model_path, classes)
            load_ms = (clock() - start) * 1000 - encoding_ms
        else:
            status = engine.load_model(model_path, config["device"])
            if status != "SUCCESS":
                raise RuntimeError(f"模型加载未完成（未重试、未切换设备）：{status}")
            load_ms = (clock() - start) * 1000
            if config["device"] == "CUDA":
                synchronize = lambda: torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                extra["gpu"] = torch.cuda.get_device_name(0)
            synchronize()
            start = clock()
            engine._configure_classes(classes)
            synchronize()
            encoding_ms = (clock() - start) * 1000
    rss_after_load = rss_mib()
    try:
        first = decode_image(image_path(dataset["root"], dataset["images"][0]))
        options = {"conf": args.conf, "iou": args.iou, "class_names": classes, "max_det": args.max_det}
        synchronize()
        start = clock()
        engine.detect(first, **options)  # Builds the predictor/session state.
        if config["runtime"] == "pytorch":
            # Ultralytics select_device("cpu") resets torch to min(8, CPUs - 1) threads
            # while setting up the predictor; restore the requested count before warm-up.
            threads["torch_after_predictor_setup"] = torch.get_num_threads()
            torch.set_num_threads(args.threads)
        for _ in range(args.warmup):
            engine.detect(first, **options)
        synchronize()
        warmup_ms = (clock() - start) * 1000
        del first
        latencies, stages, predictions = [], [], []
        for item in dataset["images"]:
            frame = decode_image(image_path(dataset["root"], item))
            if frame.shape[:2] != (int(item["height"]), int(item["width"])):
                raise ValueError(f"图片尺寸与人工标注不一致：{item['file_name']}")
            synchronize()
            start = clock()
            detections = engine.detect(frame, **options)
            synchronize()
            latencies.append((clock() - start) * 1000)
            stages.append(dict(engine.last_timing) if config["runtime"] == "onnxruntime" else pytorch_stage_ms(engine))
            predictions.extend(detection_records(detections, item["id"], dataset["category_ids"]))
            del frame, detections
        if config["runtime"] == "pytorch":
            threads["torch_after_timing"] = torch.get_num_threads()
            if threads["torch_after_timing"] != args.threads:
                raise RuntimeError(f"PyTorch 计时线程数为 {threads['torch_after_timing']}，不是请求的 {args.threads}")
        if config["device"] == "CUDA":
            extra["peak_allocated_mib"] = torch.cuda.max_memory_allocated(0) / 1024 ** 2
            extra["peak_reserved_mib"] = torch.cuda.max_memory_reserved(0) / 1024 ** 2
    finally:
        engine.unload()
    image_ids = [int(item["id"]) for item in dataset["images"]]
    result = {
        "status": "complete", "backend": args.worker, "config": config, "pid": os.getpid(), **started,
        "loadavg_after": loadavg(), "threads": threads,
        "torch_imported": "torch" in sys.modules, "model_load_ms": load_ms, "prompt_encoding_ms": encoding_ms,
        "warmup_ms": warmup_ms, "image_ids": image_ids, "latency_ms": latencies, "stage_ms": stages,
        "predictions": cap_predictions(predictions, image_ids, args.max_det),
        "rss_after_load_mib": rss_after_load, "process_max_rss_mib": process_max_rss_mib(), **extra,
    }
    atomic_json(args.worker_output, result)
    return 0


# ---------------------------------------------------------------- orchestration

def worker_command(args, backend, output):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", backend, "--worker-output", str(output),
               "--dataset", str(project_path(args.dataset)), "--model", str(project_path(args.model)),
               "--threads", str(args.threads), "--imgsz", str(args.imgsz), "--conf", str(args.conf),
               "--iou", str(args.iou), "--max-det", str(args.max_det), "--warmup", str(args.warmup)]
    for key in ONNX_INPUTS:
        path = getattr(args, f"onnx_{key}")
        if path is not None:
            command += ["--onnx-" + key.replace("_", "-"), str(project_path(path))]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    return command


def worker_environment(threads):
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads)
    env.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
    Path(env["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    env["YOLO_AUTOINSTALL"] = "False"
    return env


def run_order(backends, repeat):
    """Rotate per repeat so no backend always runs first or last."""
    shift = repeat % len(backends)
    return backends[shift:] + backends[:shift]


def summarise(report, dataset, args):
    image_ids = [int(item["id"]) for item in dataset["images"]]
    runs = report["runs"]
    backends = {}
    for name in args.backends:
        rows = sorted((run for run in runs if run["backend"] == name and run["status"] == "complete"),
                      key=lambda run: run["repeat"])
        if len(rows) != args.repeats:
            continue
        first = rows[0]["predictions"]
        evaluation = coco_metrics(dataset["annotations"], first, image_ids)
        stage_names = ("preprocess_ms", "inference_ms", "postprocess_ms")
        stages = [stage for row in rows for stage in row["stage_ms"] if stage]
        backends[name] = {
            "config": BACKENDS[name], **evaluation, "prediction_count": len(first),
            "predictions_identical_across_repeats": all(row["predictions"] == first for row in rows[1:]),
            "latency": repeat_summary([row["latency_ms"] for row in rows]),
            "stage_median_ms": {key: statistics.median(stage[key] for stage in stages) for key in stage_names}
            if stages else None,
            "model_load_ms": [row["model_load_ms"] for row in rows],
            "prompt_encoding_ms": [row["prompt_encoding_ms"] for row in rows],
            "warmup_ms": [row["warmup_ms"] for row in rows],
            "process_max_rss_mib": [row["process_max_rss_mib"] for row in rows],
            "rss_after_load_mib": [row["rss_after_load_mib"] for row in rows],
            "torch_imported": rows[0]["torch_imported"],
            "loadavg_1min_before": [row["loadavg_before"][0] for row in rows if row["loadavg_before"]],
            **{key: rows[0][key] for key in ("providers", "gpu", "detection_transfer") if key in rows[0]},
        }
    report["backends"] = backends
    if REFERENCE not in backends:
        return report
    reference = backends[REFERENCE]
    pairs = [(name, REFERENCE) for name in backends if name != REFERENCE]
    if "pt_cpu_rect" in backends:
        # Export plus square letterbox against today's app path; CUDA against the same CPU app path.
        pairs += [(name, "pt_cpu_rect") for name in backends
                  if name != "pt_cpu_rect" and (backends[name]["config"]["runtime"] == "onnxruntime"
                                                or backends[name]["config"]["device"] == "CUDA")]
    predictions = {name: next(run["predictions"] for run in runs if run["backend"] == name and run["repeat"] == 0)
                   for name in backends}
    report["agreement"] = [{"candidate": candidate, "reference": base, "thresholds": [
        agreement(predictions[base], predictions[candidate], image_ids, args.match_iou, threshold)
        for threshold in (args.conf, args.app_conf)]} for candidate, base in pairs]
    reference_median = reference["latency"]["pooled"]["median_ms"]
    reference_means = reference["latency"]["repeat_means_ms"]
    report["latency_vs_reference"] = {name: {
        "median_ratio": entry["latency"]["pooled"]["median_ms"] / reference_median,
        "mean_ratio": entry["latency"]["pooled"]["mean_ms"] / reference["latency"]["pooled"]["mean_ms"],
        "every_repeat_mean_below_reference_min": max(entry["latency"]["repeat_means_ms"]) < min(reference_means),
        "every_repeat_mean_above_reference_max": min(entry["latency"]["repeat_means_ms"]) > max(reference_means),
    } for name, entry in backends.items() if entry["config"]["device"] == "CPU"}
    return report


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def write_summary(output, report):
    lines = ["# PyTorch 与 ONNX Runtime 对照（24 图人工标注诊断集）", "",
             f"状态：{report['status']}；图片：{report['image_count']}；重复：{report['settings']['repeats']}；"
             f"线程：{report['settings']['threads']}；CPU：{report['host']['cpu']}。", "",
             "| 后端 | AP | AP50 | APsmall | 预测数 | 中位 ms | 平均 ms | p95 ms | 各次平均 ms | 峰值 RSS MiB | 导入 torch |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for name, entry in report.get("backends", {}).items():
        metrics, pooled = entry["metrics"], entry["latency"]["pooled"]
        means = " / ".join(fmt(value, 1) for value in entry["latency"]["repeat_means_ms"])
        lines.append("| " + " | ".join([name, fmt(metrics["AP"], 4), fmt(metrics["AP50"], 4), fmt(metrics["APsmall"], 4),
                                          str(entry["prediction_count"]), fmt(pooled["median_ms"], 1),
                                          fmt(pooled["mean_ms"], 1), fmt(pooled["p95_ms"], 1), means,
                                          fmt(max(entry["process_max_rss_mib"]), 0),
                                          "是" if entry["torch_imported"] else "否"]) + " |")
    loads = [run["orchestrator_loadavg_before"][0] for run in report.get("runs", []) if run.get("orchestrator_loadavg_before")]
    if loads:
        lines.extend(["", f"各 worker 启动前 1 分钟负载：{min(loads):.2f}–{max(loads):.2f}（{len(loads)} 次，逐次见 report.json）。"])
    if report.get("agreement"):
        lines.extend(["", f"预测一致性（同类且 IoU > {report['settings']['match_iou']}，按参考分数贪心匹配）：", "",
                      "| 候选 | 参考 | 分数下限 | 参考框 | 候选框 | 匹配 | 参考匹配率 | 平均 IoU | 平均 |Δscore| | 最大 |Δscore| |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in report["agreement"]:
            for item in row["thresholds"]:
                lines.append("| " + " | ".join([row["candidate"], row["reference"], fmt(item["min_score"], 2),
                                                  str(item["reference_count"]), str(item["candidate_count"]),
                                                  str(item["matched"]), fmt(item["reference_match_rate"], 4),
                                                  fmt(item["mean_iou"], 4), fmt(item["mean_abs_score_delta"], 5),
                                                  fmt(item["max_abs_score_delta"], 5)]) + " |")
    lines.extend(["", NOTICE, "", "逐次计时：timings.csv；逐后端原始预测：predictions/<后端>.json；完整参数、指纹与负载：report.json。"])
    if report.get("error"):
        lines.extend(["", f"停止原因：{report['error']}"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(output / "report.json", report)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args)
    dataset = load_dataset(project_path(args.dataset), args.limit)
    try:
        manifests = validate_inputs(args, dataset)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"加载模型前已停止：{exc}", file=sys.stderr)
        return 1
    settings = {key: str(project_path(value)) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ("worker", "worker_output", "plan")}
    plan = {"selected_backends": args.backends, "reference": REFERENCE,
            "order": [run_order(args.backends, repeat) for repeat in range(args.repeats)]}
    if args.plan:
        print(json.dumps({"status": "planned_not_executed", "settings": settings, **plan}, ensure_ascii=False, indent=2))
        return 0
    output = project_path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        print(f"输出目录非空，不覆盖旧实验：{output}", file=sys.stderr)
        return 1
    (output / "workers").mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "smoke" if args.limit else "diagnostic_24_images",
              "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "settings": settings, **plan,
              "notice": NOTICE, "image_count": len(dataset["images"]), "classes": list(dataset["classes"]),
              "image_ids": [int(item["id"]) for item in dataset["images"]],
              "dataset_sha256": dataset["sha256"], "dataset_files": dataset["file_hashes"],
              "model_files": model_files(args, manifests), "source_hashes": source_hashes(),
              "versions": {**versions(), **{name: _version(name) for name in ("onnx", "onnxslim", "onnxruntime")}},
              "onnxruntime_available_providers": _providers(), "host": host_info(), "runs": []}
    write_summary(output, report)
    environment = worker_environment(args.threads)
    try:
        for repeat in range(args.repeats):
            for index, backend in enumerate(run_order(args.backends, repeat)):
                target = output / "workers" / f"r{repeat}_{backend}.json"
                before = loadavg()
                start = time.perf_counter()
                completed = subprocess.run(worker_command(args, backend, target), cwd=PROJECT, env=environment,
                                           capture_output=True, text=True, timeout=args.timeout)
                wall = time.perf_counter() - start
                if completed.returncode != 0 or not target.is_file():
                    tail = (completed.stderr or completed.stdout or "")[-2000:]
                    raise RuntimeError(f"{backend} 第 {repeat + 1} 次失败（未重试）：{tail}")
                run = json.loads(target.read_text(encoding="utf-8"))
                run.update(repeat=repeat, order_index=index, wall_seconds=wall, orchestrator_loadavg_before=before,
                           orchestrator_loadavg_after=loadavg())
                report["runs"].append(run)
                pooled = latency_summary(run["latency_ms"])
                print(f"r{repeat + 1} {backend}: median {pooled['median_ms']:.1f} ms, load {before[0]:.2f}", flush=True)
                write_summary(output, report)
        report["status"] = "complete"
    except (RuntimeError, subprocess.TimeoutExpired, ValueError, OSError) as exc:
        report["status"] = "stopped"
        report["error"] = f"{type(exc).__name__}: {exc}"
    rows = [{"backend": run["backend"], "repeat": run["repeat"], "order_index": run["order_index"],
             "image_id": image_id, "latency_ms": latency}
            for run in report["runs"] for image_id, latency in zip(run["image_ids"], run["latency_ms"])]
    write_csv(output / "timings.csv", rows, ["backend", "repeat", "order_index", "image_id", "latency_ms"])
    if report["status"] == "complete":
        summarise(report, dataset, args)
        (output / "predictions").mkdir(exist_ok=True)
        for name in report["backends"]:
            run = next(run for run in report["runs"] if run["backend"] == name and run["repeat"] == 0)
            atomic_json(output / "predictions" / f"{name}.json", run["predictions"])
    for run in report["runs"]:
        run.pop("predictions", None)  # Kept once per backend under predictions/.
    write_summary(output, report)
    print(f"{report['status']}: {output / 'summary.md'}", flush=True)
    return 0 if report["status"] == "complete" else 1


def _version(name):
    import importlib.metadata
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _providers():
    try:
        import onnxruntime
        return onnxruntime.get_available_providers()
    except ImportError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
