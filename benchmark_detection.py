"""Independent confidence/max_det sweeps on identical, predecoded local inputs.

Only the detector's original-image detect() call is timed. Unlabelled data
cannot establish accuracy, recall or mAP. Model dependencies are imported
lazily so --help and the benchmark's control-variable tests work offline.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NOTICE = ("Unlabelled benchmark: latency, confidence and box counts are not accuracy, "
          "recall or mAP. max_det caps final returned boxes; it does not set the "
          "detector's learned query count. Timing includes detector preprocessing, "
          "forward pass, postprocessing and result transfer, excluding decoding, "
          "drawing, tracking, enhancement and display.")


@dataclass(frozen=True)
class SweepConfig:
    sweep: str
    confidence: float
    max_det: int


@dataclass
class InputFrame:
    input_id: str
    image: object
    metadata: dict


def probability(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是 0 到 1 的数值") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("必须是 0 到 1 的有限数值")
    return number


def positive_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def nonnegative_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是非负整数") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return number


def comma_values(converter):
    def parse(value):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise argparse.ArgumentTypeError("请输入逗号分隔的非空值")
        converted = [converter(part) for part in parts]
        if len(set(converted)) != len(converted):
            raise argparse.ArgumentTypeError("列表中不可有重复值")
        return converted
    return parse


def class_names(value):
    names = comma_values(str)(value)
    if any(not name.isascii() or not any(char.isalpha() for char in name) for name in names):
        raise argparse.ArgumentTypeError("请使用英文目标名称，例如 person,bottle,cell phone")
    return names


def build_parser():
    parser = argparse.ArgumentParser(description="独立测试置信度阈值和输出框数上限；无标注数据不计算 mAP。")
    parser.add_argument("--engine", choices=("yoloe", "dino"), required=True)
    parser.add_argument("--model", required=True, help="YOLOE 权重路径或 Grounding DINO 模型目录/ID")
    parser.add_argument("--source", type=Path, required=True, help="本地图片、图片目录或视频文件；不使用摄像头")
    parser.add_argument("--source-kind", choices=("auto", "image", "video"), default="auto")
    parser.add_argument("--classes", type=class_names, required=True, help="逗号分隔英文名称，例如 person,bottle,cell phone")
    parser.add_argument("--mode", choices=("confidence", "max-det", "both"), default="both")
    parser.add_argument("--conf-values", type=comma_values(probability), default=[0.1, 0.25, 0.4, 0.6])
    parser.add_argument("--max-det-values", type=comma_values(positive_integer), default=[10, 30, 100, 300])
    parser.add_argument("--fixed-conf", type=probability, default=0.4)
    parser.add_argument("--fixed-max-det", type=positive_integer, default=300)
    parser.add_argument("--iou", type=probability, default=0.45)
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=comma_values(nonnegative_integer), default=[0],
                        help="视频固定帧号，从 0 起，如 0,30,60；每个配置使用完全相同的帧")
    parser.add_argument("--warmup", type=positive_integer, default=3, help="每个配置/输入的预热次数（不计时）")
    parser.add_argument("--repeat", type=positive_integer, default=10, help="每个配置/输入的测量次数")
    parser.add_argument("--seed", type=int, default=0, help="配置执行顺序打乱的固定随机种子")
    parser.add_argument("--output", type=Path, help="新结果目录；默认 validation/experiments/UTC时间_引擎")
    parser.add_argument("--save-detections", action="store_true", help="在 JSON 中保存各配置/输入首次测量的原始检测框")
    return parser


def build_configurations(args):
    """Two independent 1-D experiments, never a coupled or Cartesian sweep."""
    configurations = []
    if args.mode in ("confidence", "both"):
        configurations.extend(SweepConfig("confidence", conf, args.fixed_max_det) for conf in args.conf_values)
    if args.mode in ("max-det", "both"):
        configurations.extend(SweepConfig("max-det", args.fixed_conf, limit) for limit in args.max_det_values)
    return configurations


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def load_inputs(source, source_kind, frame_indices):
    """Decode once, sequentially for video; fail if any requested frame is absent."""
    import cv2
    import numpy as np

    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"输入不存在：{source}")
    frames = []
    files_metadata = []

    def add_frame(frame, path, index, kind):
        if frame is None or frame.size == 0:
            raise ValueError(f"无法解码输入：{path}，帧号 {index}")
        input_id = f"input_{len(frames):04d}"
        height, width = frame.shape[:2]
        metadata = {"input_id": input_id, "path": str(path), "kind": kind,
                    "frame_index": index, "width": int(width), "height": int(height),
                    "decoded_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
        frames.append(InputFrame(input_id, frame, metadata))

    if source.is_dir():
        if source_kind == "video":
            raise ValueError("视频输入必须是单个文件；图片目录不接受 --source-kind video")
        paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError(f"目录中没有支持的图片：{source}")
        for path in paths:
            files_metadata.append(describe_file(path))
            add_frame(cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR), path, None, "image")
    elif source.is_file():
        files_metadata.append(describe_file(source))
        kind = ("image" if source.suffix.lower() in IMAGE_EXTENSIONS else "video") if source_kind == "auto" else source_kind
        if kind == "image":
            add_frame(cv2.imdecode(np.frombuffer(source.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR), source, None, kind)
        else:
            wanted = set(frame_indices)
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError(f"无法打开视频：{source}")
                for index in range(max(wanted) + 1):
                    ok, frame = capture.read()
                    if not ok:
                        missing = sorted(wanted - {item.metadata["frame_index"] for item in frames})
                        raise ValueError(f"视频在帧 {index} 结束/解码失败，无法取得要求的帧：{missing}")
                    if index in wanted:
                        add_frame(frame, source, index, kind)
            finally:
                capture.release()
    else:
        raise ValueError("输入必须是本地普通文件或图片目录")
    return frames, files_metadata


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples):
    latencies = [sample["latency_ms"] for sample in samples]
    counts = [sample["box_count"] for sample in samples]
    mean_ms = statistics.mean(latencies)
    return {"measurements": len(samples), "mean_ms": mean_ms,
            "median_ms": statistics.median(latencies), "p95_ms": percentile(latencies, 0.95),
            "min_ms": min(latencies), "max_ms": max(latencies),
            "std_ms": statistics.pstdev(latencies),
            "detection_calls_per_second": 1000 / mean_ms if mean_ms > 0 else None,
            "mean_box_count": statistics.mean(counts), "min_box_count": min(counts),
            "max_box_count": max(counts)}


def detection_record(item):
    return {"xyxy": [float(value) for value in item.xyxy], "score": float(item.score),
            "class_id": int(item.class_id), "label": item.label, "source": item.source}


def measure(engine, inputs, configurations, args, synchronize, clock=time.perf_counter):
    """Warm each exact setting/input; synchronize both sides of every timed call."""
    planned = list(enumerate(configurations))
    random.Random(args.seed).shuffle(planned)
    samples, summaries, predictions = [], [], []
    for execution_index, (config_index, config) in enumerate(planned):
        config_id = f"config_{config_index:03d}"
        base = {"config_id": config_id, "execution_index": execution_index,
                "sweep": config.sweep, "confidence": config.confidence, "max_det": config.max_det}
        config_samples = []
        for item in inputs:
            kwargs = {"conf": config.confidence, "iou": args.iou, "class_names": args.classes,
                      "source": "original", "max_det": config.max_det}
            for _ in range(args.warmup):
                engine.detect(item.image, **kwargs)
            synchronize()
            frame_samples = []
            for repetition in range(args.repeat):
                synchronize()
                start = clock()
                detections = engine.detect(item.image, **kwargs)
                synchronize()
                elapsed_ms = (clock() - start) * 1000
                scores = [float(detection.score) for detection in detections]
                counts = dict(Counter(detection.label for detection in detections))
                sample = {**base, "input_id": item.input_id, "repetition": repetition,
                          "latency_ms": elapsed_ms, "box_count": len(detections),
                          "counts_by_class": counts,
                          "mean_confidence": statistics.mean(scores) if scores else None,
                          "min_confidence": min(scores) if scores else None,
                          "max_confidence": max(scores) if scores else None}
                frame_samples.append(sample)
                if args.save_detections and repetition == 0:
                    predictions.append({**base, "input_id": item.input_id,
                                        "detections": [detection_record(detection) for detection in detections]})
            samples.extend(frame_samples)
            config_samples.extend(frame_samples)
            summaries.append({**base, "input_id": item.input_id, "scope": "input", **summarize(frame_samples)})
        overall = {**base, "input_id": "ALL", "scope": "all_inputs", **summarize(config_samples)}
        summaries.append(overall)
        print(f"{config.sweep}: conf={config.confidence:g}, max_det={config.max_det}; "
              f"median={overall['median_ms']:.2f} ms, p95={overall['p95_ms']:.2f} ms, "
              f"mean boxes={overall['mean_box_count']:.2f}", flush=True)
    return {"samples": samples, "summary": summaries, "first_predictions": predictions,
            "execution_order": [f"config_{index:03d}" for index, _ in planned]}


def software_metadata():
    versions = {}
    for name in ("torch", "torchvision", "ultralytics", "transformers", "numpy", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def initialize_engine(args):
    import torch
    from engines import GroundingDINOEngine, OpenVocabEngine

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("已指定 CUDA，但当前 PyTorch 无可用 CUDA；可改用 --device cpu")
    engine = OpenVocabEngine() if args.engine == "yoloe" else GroundingDINOEngine()
    engine.imgsz = args.imgsz
    status = engine.load_model(args.model, device="CUDA" if use_cuda else "CPU")
    if status != "SUCCESS":
        raise RuntimeError(f"模型加载失败：{status}")
    if args.engine == "dino":
        engine.text_threshold = 0.25
    device_metadata = {"requested": args.device, "actual": str(engine.device),
                       "cuda_available": bool(torch.cuda.is_available()), "torch_cuda": torch.version.cuda,
                       "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()) if use_cuda else None,
                       "torch_threads": torch.get_num_threads(),
                       "cuda_synchronization": use_cuda}
    if use_cuda:
        def synchronize():
            torch.cuda.synchronize(engine.device)
    else:
        def synchronize():
            pass
    return engine, synchronize, device_metadata


def write_report(output, report):
    """CSV carries settings in each row; JSON also holds fingerprints and environment."""
    output.mkdir(parents=True, exist_ok=False)
    settings = report["settings"]
    common = {"run_id": report["run_id"], "engine": settings["engine"], "model": settings["model"],
              "device": report["device"]["actual"], "imgsz": settings["imgsz"], "iou": settings["iou"],
              "text_threshold": settings["text_threshold"], "classes_json": json.dumps(settings["classes"]),
              "warmup": settings["warmup"], "repeat": settings["repeat"], "seed": settings["seed"],
              "source": settings["source"], "mode": settings["mode"],
              "fixed_conf": settings["fixed_conf"], "fixed_max_det": settings["fixed_max_det"],
              "conf_values_json": json.dumps(settings["conf_values"]),
              "max_det_values_json": json.dumps(settings["max_det_values"]),
              "settings_json": json.dumps(settings, ensure_ascii=False, sort_keys=True),
              "device_json": json.dumps(report["device"], ensure_ascii=False, sort_keys=True),
              "software_json": json.dumps(report.get("software", {}), ensure_ascii=False, sort_keys=True),
              "cuda_synchronization": report["device"]["cuda_synchronization"]}
    input_metadata = {item["input_id"]: item for item in report["inputs"]}
    for filename, key in (("samples.csv", "samples"), ("summary.csv", "summary")):
        rows = []
        for item in report[key]:
            metadata = input_metadata.get(item["input_id"], {})
            row = {**common, "input_path": metadata.get("path"), "input_kind": metadata.get("kind"),
                   "frame_index": metadata.get("frame_index"), "width": metadata.get("width"),
                   "height": metadata.get("height"), "decoded_bgr_sha256": metadata.get("decoded_bgr_sha256"),
                   **item}
            if "counts_by_class" in row:
                row["counts_by_class"] = json.dumps(row["counts_by_class"], sort_keys=True)
            rows.append(row)
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    start = datetime.now(timezone.utc)
    run_id = f"{start.strftime('%Y%m%dT%H%M%S_%fZ')}_{args.engine}"
    output = (args.output or Path(__file__).resolve().parent / "validation" / "experiments" / run_id).expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    if output.exists():
        parser.error(f"结果目录已存在，为避免覆盖请指定新目录：{output}")
    original_cwd = Path.cwd()
    try:
        inputs, source_files = load_inputs(args.source, args.source_kind, args.frames)
        configurations = build_configurations(args)
        # YOLOE's text encoder looks up bundled weights relative to project cwd.
        # Resolve user paths before changing cwd, so CLI paths retain normal meaning.
        candidate_model = Path(args.model).expanduser()
        if candidate_model.exists():
            args.model = str(candidate_model.resolve())
        elif args.engine == "yoloe":
            project_candidate = Path(__file__).resolve().parent / candidate_model
            if not project_candidate.is_file():
                raise ValueError(f"YOLOE 本地权重不存在：{candidate_model}；请先下载权重")
            args.model = str(project_candidate.resolve())
        os.chdir(Path(__file__).resolve().parent)
        engine, synchronize, device = initialize_engine(args)
        settings = vars(args).copy()
        settings["source"] = str(args.source)
        settings["output"] = str(output)
        settings["text_threshold"] = 0.25 if args.engine == "dino" else None
        settings["timed_operation"] = "engine.detect(original BGR frame)"
        settings["resize_policy"] = ({"method": "shortest_edge_then_longest_edge_cap",
                                       "shortest_edge": args.imgsz,
                                       "longest_edge": round(args.imgsz * 1333 / 800)}
                                      if args.engine == "dino" else
                                      {"method": "ultralytics_auto_letterbox", "max_edge": args.imgsz,
                                       "stride_padding": True})
        settings["tracking"] = settings["tiles"] = settings["enhancement"] = False
        model_path = Path(args.model)
        model_config = getattr(engine.model, "config", None)
        model_provenance = {"requested_model": args.model,
                            "name_or_path": getattr(model_config, "_name_or_path", None),
                            "commit_hash": getattr(model_config, "_commit_hash", None),
                            "revision": getattr(model_config, "revision", None)}
        if model_path.is_dir():
            extensions = {".json", ".safetensors", ".bin", ".txt", ".model", ".vocab", ".merges"}
            model_provenance["local_files"] = [describe_file(path) for path in sorted(model_path.iterdir())
                                               if path.is_file() and path.suffix in extensions]
        report = {"schema_version": 1, "run_id": run_id, "started_at_utc": start.isoformat(),
                  "notice": NOTICE, "ground_truth": None, "accuracy_metrics": None,
                  "settings": settings, "device": device, "software": software_metadata(),
                  "source_files": source_files, "inputs": [item.metadata for item in inputs],
                  "model_file": describe_file(model_path) if model_path.is_file() else None,
                  "model_provenance": model_provenance,
                  "planned_configurations": [asdict(config) for config in configurations]}
        report.update(measure(engine, inputs, configurations, args, synchronize))
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(output, report)
        print(f"\n已保存：{output}\n{NOTICE}")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"实验失败：{exc}\n")
    finally:
        os.chdir(original_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
