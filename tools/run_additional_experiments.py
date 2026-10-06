"""Serial, resumable small-object ablations with CPU as the default device.

Importing this module and --help do not load torch or a checkpoint. Existing
GPU jobs are never stopped. A resource failure stops the entire batch without
retrying on either device. Predictions are saved after every image/class pass.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluate_mini_dataset import (
    METRIC_NAMES, atomic_json, cap_predictions, coco_metrics, decode_image,
    detection_records, image_path, load_dataset, local_path, positive_integer,
    probability, sha256_file, versions, write_csv,
)

SCENARIOS = {
    "text_baseline": {},
    "text_class_nms": {"agnostic_nms": False},
    "text_one2one": {"head_mode": "one2one"},
    "prompt_smartphone": {"prompt_replace": [2, "smartphone"]},
    "prompt_mug": {"prompt_replace": [1, "mug"]},
    "prompt_water_bottle": {"prompt_replace": [0, "water bottle"]},
    "distractors": {"extra_prompts": ["remote control", "calculator"]},
    "input768": {"imgsz": 768},
    "input960": {"imgsz": 960},
    "dark0.6": {"perturbation": {"kind": "dark", "factor": 0.6}},
    "motion_blur9": {"perturbation": {"kind": "motion_blur", "kernel": 9}},
    "jpeg35": {"perturbation": {"kind": "jpeg", "quality": 35}},
    "visual_bbox": {"visual": True},
    "visual_loose_bbox": {"visual": True, "reference_variant": "loose", "box_scale": 1.25},
    "visual_human_mask": {"visual": True, "reference_variant": "human_mask"},
    "text_fp16": {"quantize": 16},
    "visual_fp16": {"visual": True, "quantize": 16},
    "prompt_cached": {"prompt_cache": True},
    "detect_only": {"detect_only": True},
    "teacher_26l": {"teacher": True},
}

NOTICE = (
    "Diagnostic ablations on the same small public test subset; comparisons do not constitute "
    "camera-domain validation or a held-out model-selection result. Synonyms can narrow class "
    "meaning; their outputs are mapped to the original category IDs. Extra distractor outputs "
    "are discarded before the global output cap. Visual inference performs one separate pass "
    "per class, with total per-image latency summed across all classes. Timings exclude image "
    "decoding, CPU perturbations, checkpoint writes, loading, prompt encoding and warmup; they "
    "include model preprocessing/forward/postprocessing/result transfer. Resumed timings may "
    "come from multiple sessions. CPU timings are affected by concurrent host workloads and "
    "must not be treated as head/device speed improvements without controlled repeated timing. "
    "FP16 scenarios are skipped on CPU. Teacher predictions are "
    "unreviewed annotation candidates, never substituted for human evaluation ground truth."
)


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parse_scenarios(value):
    names = [part.strip() for part in value.split(",")]
    if not names or len(names) != len(set(names)) or any(name not in SCENARIOS for name in names):
        raise argparse.ArgumentTypeError("请提供不重复的已知场景；使用 --list-scenarios 查看列表")
    # The larger teacher always follows release of every preceding small model.
    return [name for name in names if name != "teacher_26l"] + (["teacher_26l"] if "teacher_26l" in names else [])


def build_parser():
    parser = argparse.ArgumentParser(description="逐图保存的串行小目标消融；CPU 默认，不停止其他 GPU 作业。")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/mini_eval"))
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--teacher-model", type=Path, default=Path("yoloe-26l-seg.pt"))
    parser.add_argument("--device", type=str.upper, choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--output", type=Path, default=Path("work/additional_experiments/cpu_ablation"))
    parser.add_argument("--reference-masks", type=Path, default=Path("work/additional_experiments/reference_masks"))
    parser.add_argument("--scenarios", type=parse_scenarios, default=list(SCENARIOS))
    parser.add_argument("--conf", type=probability, default=0.05)
    parser.add_argument("--iou", type=probability, default=0.5)
    parser.add_argument("--max-det", type=positive_integer, default=100)
    parser.add_argument("--warmup", type=positive_integer, default=1)
    parser.add_argument("--threads", type=positive_integer, default=2)
    parser.add_argument("--limit", type=positive_integer)
    parser.add_argument("--resume", action="store_true", help="只恢复完全匹配数据、权重、代码和参数的检查点")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--plan", action="store_true", help="验证输入并输出实验配置，不加载模型")
    return parser


def scenario_config(name, dataset, args):
    spec = SCENARIOS[name]
    classes = list(dataset["classes"])
    if "prompt_replace" in spec:
        index, value = spec["prompt_replace"]
        classes[index] = value
    classes.extend(spec.get("extra_prompts", []))
    return {
        "name": name, "visual": spec.get("visual", False), "prompts": classes,
        "category_map": list(dataset["category_ids"]) + [None] * len(spec.get("extra_prompts", [])),
        "imgsz": spec.get("imgsz", 640), "conf": args.conf, "iou": args.iou,
        "max_det": args.max_det, "warmup": args.warmup,
        "inference_options": {"agnostic_nms": spec.get("agnostic_nms", True),
                              "head_mode": spec.get("head_mode", "one2many"),
                              "quantize": spec.get("quantize", 32)},
        "perturbation": spec.get("perturbation", {"kind": "none"}),
        "reference_variant": spec.get("reference_variant", "bbox"),
        "box_scale": spec.get("box_scale", 1.0),
        "prompt_cache": spec.get("prompt_cache", False),
        "detect_only": spec.get("detect_only", False), "teacher": spec.get("teacher", False),
    }


def perturb_image(frame, config):
    """CPU photometric changes preserve dimensions and all GT coordinates."""
    import cv2
    import numpy as np

    kind = config["kind"]
    if kind == "none":
        return frame
    if kind == "dark":
        transformed = np.clip(frame.astype(np.float32) * config["factor"], 0, 255).astype(np.uint8)
    elif kind == "motion_blur":
        size = int(config["kernel"])
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        transformed = cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    elif kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(config["quality"])])
        if not ok:
            raise ValueError("JPEG 扰动编码失败")
        transformed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"未知扰动：{kind}")
    if transformed is None or transformed.shape != frame.shape or transformed.dtype != frame.dtype:
        raise ValueError("扰动改变了图片尺寸或类型，停止评价")
    return transformed


def loose_reference_box(box, shape, scale=1.25):
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * (scale - 1) / 2, (y2 - y1) * (scale - 1) / 2
    return [max(0.0, x1 - dx), max(0.0, y1 - dy),
            min(float(shape[1]), x2 + dx), min(float(shape[0]), y2 + dy)]


def mapped_records(detections, image_id, category_map, visual_category=None):
    if visual_category is not None:
        return detection_records(detections, image_id, [], visual_category)
    records = []
    for item in detections:
        class_id = int(item.class_id)
        if class_id != item.class_id or not 0 <= class_id < len(category_map):
            raise ValueError("模型返回的类别 ID 与本次提示词映射不一致")
        if category_map[class_id] is None:
            continue
        records.extend(detection_records([item], image_id, category_map))
    return records


def mask_provenance(args, dataset):
    root = args.reference_masks.resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        return {"available": False, "reason": "尚无独立参考图的人类 mask 与来源记录"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = {int(item["class_id"]): item for item in metadata.get("references", [])}
    files = {"metadata.json": sha256_file(metadata_path)}
    records = {}
    for reference in dataset["references"]:
        index = int(reference["class_id"])
        record = entries.get(index)
        if record is None or int(record.get("original_image_id", -1)) != int(reference["original_image_id"]) \
                or int(record.get("original_annotation_id", -1)) != int(reference["original_annotation_id"]):
            raise ValueError("人工 mask 来源必须匹配独立参考图的原始人工标注")
        file = record.get("file", record.get("mask_file"))
        if not isinstance(file, str):
            raise ValueError("人工 mask 来源记录缺少 file")
        path = local_path(root, file)
        actual_hash = sha256_file(path)
        if record.get("sha256") is not None and record["sha256"] != actual_hash:
            raise ValueError("人工 mask 内容与来源记录哈希不一致")
        files[file] = actual_hash
        records[str(index)] = {**record, "file": file, "sha256": actual_hash}
    return {"available": True, "root": str(root), "files": files, "references": records}


def source_hashes():
    return {relative: sha256_file(PROJECT / relative) for relative in (
        "tools/run_additional_experiments.py", "evaluate_mini_dataset.py", "engines.py",
        "visual_engine.py", "box_predictor.py", "resource_safety.py", "detection_utils.py",
    )}


def manifest_for(args, dataset, masks):
    return {
        "schema": 1, "dataset_sha256": dataset["sha256"], "dataset_files": dataset["file_hashes"],
        "classes": list(dataset["classes"]),
        "image_ids": [int(item["id"]) for item in dataset["images"]],
        "model_path": str(args.model.resolve()), "model_sha256": sha256_file(args.model),
        "teacher_path": str(args.teacher_model.resolve()),
        "teacher_sha256": sha256_file(args.teacher_model) if args.teacher_model.is_file() else None,
        "device": args.device, "threads": args.threads, "conf": args.conf, "iou": args.iou,
        "max_det": args.max_det, "warmup": args.warmup, "source_hashes": source_hashes(),
        "reference_masks": masks, "versions": versions(),
    }


class CheckpointStore:
    """Atomically persist each unit; reject every incompatible resume."""

    def __init__(self, output, manifest, resume=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.fingerprint = digest_json(manifest)
        path = self.output / "manifest.json"
        if path.is_file():
            if not resume:
                raise ValueError("输出目录已有实验；请用 --resume 或新的 --output，避免覆盖")
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != self.fingerprint or saved.get("manifest") != manifest:
                raise ValueError("拒绝恢复：数据、权重、代码、设备或参数的哈希与检查点不一致")
        else:
            if any(self.output.iterdir()):
                raise ValueError("输出目录非空且没有有效实验 manifest，请使用新的 --output")
            atomic_json(path, {"fingerprint": self.fingerprint, "manifest": manifest})

    def scenario_fingerprint(self, config):
        return digest_json({"manifest": self.fingerprint, "scenario": config})

    def unit_path(self, name, phase, image_id):
        return self.output / "checkpoints" / name / phase / f"{int(image_id):012d}.json"

    def load_unit(self, config, phase, image_id):
        path = self.unit_path(config["name"], phase, image_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != self.scenario_fingerprint(config) or data.get("status") != "complete" \
                or data.get("image_id") != int(image_id) or data.get("phase") != phase:
            raise ValueError(f"检查点配置或身份不匹配：{path}")
        if not isinstance(data.get("predictions"), list) or not math.isfinite(data.get("latency_ms", float("nan"))) \
                or data["latency_ms"] < 0:
            raise ValueError(f"检查点结果损坏：{path}")
        if data.get("result_sha256") != digest_json({"predictions": data["predictions"],
                                                      "latency_ms": data["latency_ms"]}):
            raise ValueError(f"检查点预测结果哈希不一致：{path}")
        return data

    def save_unit(self, config, phase, image_id, predictions, latency_ms):
        data = {"fingerprint": self.scenario_fingerprint(config), "status": "complete",
                "image_id": int(image_id), "phase": phase, "predictions": predictions,
                "latency_ms": latency_ms, "saved_at": datetime.now(timezone.utc).isoformat(),
                "result_sha256": digest_json({"predictions": predictions, "latency_ms": latency_ms})}
        atomic_json(self.unit_path(config["name"], phase, image_id), data)
        return data

    def summary_path(self, name):
        return self.output / "modes" / name / "metrics.json"

    def load_summary(self, config):
        path = self.summary_path(config["name"])
        if not path.is_file():
            return None
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("fingerprint") != self.scenario_fingerprint(config):
            raise ValueError(f"完整场景结果的配置不匹配：{path}")
        if result.get("status") == "complete":
            predictions = local_path(self.output, result["predictions_file"])
            if result.get("predictions_sha256") != sha256_file(predictions):
                raise ValueError(f"完整场景预测文件哈希不一致：{predictions}")
        return result if result.get("status") == "complete" else None


def skip_reason(config, args, masks):
    if config["inference_options"]["quantize"] == 16 and args.device == "CPU":
        return "等待 CUDA 空闲后单独运行；CPU 不强行执行 FP16 对照"
    if config["reference_variant"] == "human_mask" and not masks["available"]:
        return masks["reason"]
    if config["teacher"] and not args.teacher_model.is_file():
        return "本地没有 26L 权重；没有自动下载或替换模型"
    if config["detect_only"] and args.model.name != "yoloe-26s-seg.pt":
        return "检测架构转换限定匹配的 yoloe-26s-seg.pt，避免错误形状的权重转移"
    return None


def load_engine(config, args):
    """Exactly one owned checkpoint, with the existing safety guards."""
    from engines import OpenVocabEngine
    from visual_engine import VisualPromptEngine

    engine = VisualPromptEngine() if config["visual"] else OpenVocabEngine()
    engine.imgsz = config["imgsz"]
    engine.set_inference_options(**config["inference_options"])
    try:
        if config["detect_only"]:
            from resource_safety import prepare_device
            target = prepare_device(args.device)
            engine.device = target
            from ultralytics import YOLOE
            from ultralytics.cfg import DEFAULT_CFG_DICT
            engine.model = YOLOE("yoloe-26s.yaml").load(str(args.model.resolve()))
            engine.model.to(target)
            if engine.model.task != "detect":
                raise ValueError("匹配 YAML 没有构建检测架构，停止转换实验")
            # .load() temporarily retains the source checkpoint in .ckpt. The
            # matching detection model has already copied its usable weights.
            engine.model.ckpt = None
            engine._prediction_head_options = {"nms": None} if DEFAULT_CFG_DICT.get("nms", False) is None \
                else {"end2end": False}
            engine._precision_uses_quantize = "quantize" in DEFAULT_CFG_DICT
            engine._select_head_options()
            engine._validate_head_available()
            import gc
            gc.collect()
        else:
            path = args.teacher_model if config["teacher"] else args.model
            loaded = engine.load_model(path, args.device)
            if loaded != "SUCCESS":
                raise RuntimeError(f"模型加载未完成：{loaded}")
        return engine
    except BaseException:
        engine.unload()
        raise


def profile_metadata(store, classes):
    return {"schema": 1, "model_sha256": store.manifest["model_sha256"],
            "dataset_sha256": store.manifest["dataset_sha256"], "classes": list(classes),
            "source_hashes": store.manifest["source_hashes"]}


def profile_folder(store, classes):
    canonical = store.manifest.get("classes", classes)
    folder = store.output / "prompt_cache"
    return folder if list(classes) == list(canonical) else folder / "variants" / digest_json(list(classes))[:16]


def profile_exists(store, classes):
    folder = profile_folder(store, classes)
    return (folder / "text_prompts.npz").is_file() and (folder / "metadata.json").is_file()


def load_profile(engine, store, classes):
    folder = profile_folder(store, classes)
    profile, metadata_path = folder / "text_prompts.npz", folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = profile_metadata(store, classes)
    if metadata.get("binding") != expected or metadata.get("npz_sha256") != sha256_file(profile):
        raise ValueError("提示嵌入 profile 的权重、类别、来源或文件哈希不匹配")
    engine.model.load_prompt_embeddings(profile)
    engine.current_classes = list(classes)
    return profile


def save_profile(engine, store, classes):
    folder = profile_folder(store, classes)
    folder.mkdir(parents=True, exist_ok=True)
    temporary, profile = folder / "text_prompts.pending.npz", folder / "text_prompts.npz"
    engine.model.save_prompt_embeddings(temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    temporary.replace(profile)
    atomic_json(folder / "metadata.json", {"binding": profile_metadata(store, classes),
                                           "npz_sha256": sha256_file(profile)})
    return profile


def cache_prompt(engine, store, classes, synchronize, clock):
    import numpy as np

    profile = profile_folder(store, classes) / "text_prompts.npz"
    encoding_ms = 0.0
    created = not profile_exists(store, classes)
    if created:
        synchronize()
        start = clock()
        engine.model.set_classes(classes)
        synchronize()
        encoding_ms = (clock() - start) * 1000
        engine.current_classes = list(classes)
        save_profile(engine, store, classes)
    with np.load(profile, allow_pickle=False) as payload:
        original_embedding = payload["embeddings"].copy()
    synchronize()
    start = clock()
    load_profile(engine, store, classes)
    synchronize()
    load_ms = (clock() - start) * 1000
    installed = engine.model.model.pe.detach().cpu().float().numpy()
    if installed.shape != original_embedding.shape or not np.array_equal(installed, original_embedding):
        raise ValueError("提示嵌入 NPZ 往返后内容不一致")
    return {"profile": str(profile.relative_to(store.output)), "created": created,
            "encoding_ms": encoding_ms, "profile_load_ms": load_ms,
            "embedding_exact_match": True, "npz_sha256": sha256_file(profile)}


def prepare_text_prompt(engine, store, classes, synchronize, clock, teacher=False):
    """Persist every text vocabulary once; subsequent ablations reuse its PE."""
    if not teacher and profile_exists(store, classes):
        synchronize()
        start = clock()
        profile = load_profile(engine, store, classes)
        synchronize()
        return {"phase": "text", "encoding_ms": 0.0, "profile_load_ms": (clock() - start) * 1000,
                "profile_reused": True, "profile": str(profile.relative_to(store.output))}
    synchronize()
    start = clock()
    engine.model.set_classes(classes)
    synchronize()
    encoding_ms = (clock() - start) * 1000
    engine.current_classes = list(classes)
    profile = save_profile(engine, store, classes) if not teacher else None
    return {"phase": "text", "encoding_ms": encoding_ms, "profile_reused": False,
            "profile": str(profile.relative_to(store.output)) if profile else None,
            "teacher_model": teacher}


def read_reference_mask(reference, masks, frame_shape):
    import cv2
    import numpy as np

    root = Path(masks["root"])
    entry = masks["references"][str(reference["class_id"])]
    path = local_path(root, entry["file"])
    mask = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != frame_shape[:2] or not (mask > 0).any():
        raise ValueError("人工参考 mask 必须非空且与独立参考原图尺寸完全一致")
    return (mask > 0).astype(np.uint8)


def timing_summary(timings, passes):
    latencies = [entry["latency_ms"] for entry in timings]
    ordered = sorted(latencies)
    position = (len(ordered) - 1) * 0.95
    p95 = ordered[math.floor(position)] + (ordered[math.ceil(position)] - ordered[math.floor(position)]) \
        * (position - math.floor(position))
    return {"mean_ms": statistics.mean(latencies), "median_ms": statistics.median(latencies),
            "p95_ms": p95, "class_passes_per_image": passes}


def run_scenario(config, dataset, args, store, masks, engine_factory=load_engine,
                 decoder=decode_image, clock=time.perf_counter):
    """Resume at image/class granularity; completed units never run again."""
    saved = store.load_summary(config)
    if saved is not None:
        print(f"{config['name']} | 完整结果已验证，直接恢复", flush=True)
        return saved
    image_ids = [int(item["id"]) for item in dataset["images"]]
    phases = [(f"class_{item['class_id']}", item) for item in dataset["references"]] if config["visual"] \
        else [("text", None)]
    completed = {(phase, int(item["id"])): store.load_unit(config, phase, item["id"])
                 for phase, _ in phases for item in dataset["images"]}
    needs_engine = any(unit is None for unit in completed.values())
    engine = None
    synchronize = lambda: None
    cold = {"model_load_ms": 0.0, "encodings": [], "warmup_ms": 0.0,
            "units_reused": sum(unit is not None for unit in completed.values()), "units_executed": 0}
    memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}
    cache_details = None
    try:
        if needs_engine:
            if config["detect_only"] and not profile_exists(store, config["prompts"]):
                # Seed an original-seg prompt profile first, release that model,
                # then build the matching detection architecture sequentially.
                seed = dict(config, detect_only=False)
                start = clock()
                engine = engine_factory(seed, args)
                seed_load = (clock() - start) * 1000
                if args.device == "CUDA":
                    import torch
                    seed_sync = lambda: torch.cuda.synchronize(0)
                else:
                    seed_sync = lambda: None
                cache_details = cache_prompt(engine, store, config["prompts"], seed_sync, clock)
                cache_details["seed_model_load_ms"] = seed_load
                engine.unload()
                engine = None
            start = clock()
            engine = engine_factory(config, args)
            cold["model_load_ms"] = (clock() - start) * 1000
            if args.device == "CUDA":
                import torch
                synchronize = lambda: torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
            if config["prompt_cache"]:
                cache_details = cache_prompt(engine, store, config["prompts"], synchronize, clock)
                cold["encodings"].append({"phase": "text", **cache_details})
            elif config["detect_only"]:
                synchronize()
                start = clock()
                load_profile(engine, store, config["prompts"])
                synchronize()
                cold["encodings"].append({"phase": "text", "profile_load_ms": (clock() - start) * 1000})
        for phase, reference in phases:
            pending = [item for item in dataset["images"] if completed[(phase, int(item["id"]))] is None]
            if not pending:
                continue
            classes = [reference["class_name"]] if reference else config["prompts"]
            category = int(reference["category_id"]) if reference else None
            if reference:
                frame = decoder(local_path(dataset["root"], reference["image_file"]))
                box = reference["bbox_xyxy"]
                if config["reference_variant"] == "loose":
                    box = loose_reference_box(box, frame.shape, config["box_scale"])
                mask = read_reference_mask(reference, masks, frame.shape) \
                    if config["reference_variant"] == "human_mask" else None
                synchronize()
                start = clock()
                if mask is None:
                    engine.set_visual_prompt(frame, box, reference["class_name"])
                else:
                    engine.set_visual_prompt(frame, box, reference["class_name"], reference_mask=mask)
                synchronize()
                cold["encodings"].append({"phase": phase, "encoding_ms": (clock() - start) * 1000,
                                          "original_image_id": reference["original_image_id"],
                                          "original_annotation_id": reference["original_annotation_id"]})
                del frame, mask
            elif not config["prompt_cache"] and not config["detect_only"]:
                cold["encodings"].append(prepare_text_prompt(
                    engine, store, classes, synchronize, clock, teacher=config["teacher"],
                ))
            first = perturb_image(decoder(image_path(dataset["root"], pending[0])), config["perturbation"])
            synchronize()
            start = clock()
            for _ in range(config["warmup"]):
                engine.detect(first, conf=config["conf"], iou=config["iou"], class_names=classes,
                              max_det=config["max_det"])
            synchronize()
            cold["warmup_ms"] += (clock() - start) * 1000
            del first
            for item in pending:
                frame = decoder(image_path(dataset["root"], item))
                if frame.shape[:2] != (int(item["height"]), int(item["width"])):
                    raise ValueError(f"图片尺寸与原始人工标注不一致：{item['file_name']}")
                frame = perturb_image(frame, config["perturbation"])
                synchronize()
                start = clock()
                detections = engine.detect(frame, conf=config["conf"], iou=config["iou"],
                                           class_names=classes, max_det=config["max_det"])
                synchronize()
                latency_ms = (clock() - start) * 1000
                records = mapped_records(detections, item["id"], config["category_map"], category)
                completed[(phase, int(item["id"]))] = store.save_unit(config, phase, item["id"], records, latency_ms)
                cold["units_executed"] += 1
                del frame, detections
                print(f"{config['name']} | {phase} | {cold['units_executed']} 新增 / {cold['units_reused']} 恢复", flush=True)
        if args.device == "CUDA" and needs_engine:
            memory = {"peak_allocated_mb": torch.cuda.max_memory_allocated(0) / 1024 ** 2,
                      "peak_reserved_mb": torch.cuda.max_memory_reserved(0) / 1024 ** 2}
        predictions = [record for unit in completed.values() for record in unit["predictions"]]
        predictions = cap_predictions(predictions, image_ids, config["max_det"])
        timings = [{"image_id": image_id, "latency_ms": sum(completed[(phase, image_id)]["latency_ms"]
                    for phase, _ in phases)} for image_id in image_ids]
        evaluation = coco_metrics(dataset["annotations"], predictions, image_ids)
        folder = store.output / "modes" / config["name"]
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "predictions.json", predictions)
        write_csv(folder / "timings.csv", timings, ["image_id", "latency_ms"])
        from resource_safety import process_max_rss_mib
        result = {"status": "complete", "fingerprint": store.scenario_fingerprint(config), "config": config,
                  **evaluation, "timing_summary": timing_summary(timings, len(phases)), "memory": memory,
                  "process_max_rss_mib": process_max_rss_mib(),
                  "rss_note": "进程累计 RSS 高水位，不是该场景独立峰值", "cold_session": cold,
                  "prediction_count": len(predictions), "prompt_cache": cache_details,
                  "predictions_file": str((folder / "predictions.json").relative_to(store.output)),
                  "predictions_sha256": sha256_file(folder / "predictions.json"),
                  "teacher_candidate_only": config["teacher"]}
        atomic_json(store.summary_path(config["name"]), result)
        return result
    finally:
        if engine is not None:
            engine.unload()


def compare_cached_predictions(store, result):
    baseline = store.output / "modes" / "text_baseline" / "predictions.json"
    cached = store.output / "modes" / "prompt_cached" / "predictions.json"
    if not baseline.is_file() or not cached.is_file():
        result["baseline_prediction_comparison"] = {"status": "unavailable", "reason": "需要相同配置的完整 text_baseline"}
        return
    left = json.loads(baseline.read_text(encoding="utf-8"))
    right = json.loads(cached.read_text(encoding="utf-8"))
    result["baseline_prediction_comparison"] = {
        "status": "complete", "exact_equal": left == right,
        "baseline_count": len(left), "cached_count": len(right),
        "notice": "FP32 同权重下比较原始预测；exact_equal=False 需进一步定位数值差异，不能宣称缓存精度等价",
    }


def save_report(store, report):
    rows = [{"scenario": name, **entry["metrics"], **entry["timing_summary"], **entry["memory"]}
            for name, entry in report["scenarios"].items() if entry.get("status") == "complete"]
    write_csv(store.output / "comparison.csv", rows,
              ["scenario", *METRIC_NAMES, "mean_ms", "median_ms", "p95_ms", "class_passes_per_image",
               "peak_allocated_mb", "peak_reserved_mb"])
    lines = ["# 小目标额外消融", "", f"状态：{report['status']}；设备：{report['device']}；图片：{report['image_count']}。", "",
             "| 场景 | 状态 | AP | APsmall | 平均 ms/图 | 备注 |", "|---|---|---:|---:|---:|---|"]
    for name, entry in report["scenarios"].items():
        metrics, timing = entry.get("metrics", {}), entry.get("timing_summary", {})
        values = ["—" if value is None else f"{value:.3f}" for value in
                  (metrics.get("AP"), metrics.get("APsmall"), timing.get("mean_ms"))]
        note = entry.get("reason", entry.get("error", ""))
        if entry.get("teacher_candidate_only"):
            note = "未经复核的预标注候选；不是 GT"
        lines.append("| " + " | ".join([name, entry["status"], *values, note.replace("|", "/")]) + " |")
    lines.extend(["", NOTICE, "", "每图/每类检查点在 checkpoints/；完整原始预测在 modes/<场景>/predictions.json。", "",
                  "768/960 只改变网络输入尺寸；原图分辨率未提高，不会恢复已经缺失的细节。", "",
                  "loose box 的宽高各为原框的 1.25 倍，中心不变并裁至原图边界。", "",
                  "置信度、IoU、全图最大预测数在所有场景中保持一致；视觉三类合并后统一截断。", "",
                  "同权重、同词表的文本嵌入持久化并复用，首次编码和后续 profile 加载分别记录。", "",
                  "提示缓存、检测架构转换和教师模型都逐个加载并释放；没有同时驻留多个检测 checkpoint。"])
    temporary = store.output / "summary.md.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(store.output / "summary.md")
    atomic_json(store.output / "report.json", report)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_scenarios:
        print("\n".join(SCENARIOS))
        return 0
    if not 1 <= args.max_det <= 1000 or args.threads > 8:
        raise ValueError("安全模式 max-det 1–1000、threads 1–8")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
    dataset = load_dataset(args.dataset, args.limit)
    masks = mask_provenance(args, dataset)
    configs = [scenario_config(name, dataset, args) for name in args.scenarios]
    from resource_safety import preflight_inference
    for config in configs:
        for image in dataset["images"]:
            preflight_inference("cpu", config["imgsz"], (int(image["height"]), int(image["width"]), 3))
    manifest = manifest_for(args, dataset, masks)
    if args.plan:
        print(json.dumps({"manifest": manifest, "scenarios": configs,
                          "skipped": {config["name"]: reason for config in configs
                                      if (reason := skip_reason(config, args, masks))}}, ensure_ascii=False, indent=2))
        return 0
    store = CheckpointStore(args.output.resolve(), manifest, args.resume)
    report = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(), "device": args.device,
              "image_count": len(dataset["images"]), "all_image_count": dataset["all_image_count"],
              "manifest_fingerprint": store.fingerprint, "notice": NOTICE, "scenarios": {}}
    for config in configs:
        reason = skip_reason(config, args, masks)
        report["scenarios"][config["name"]] = {"status": "skipped", "reason": reason, "config": config} \
            if reason else {"status": "pending", "config": config}
    save_report(store, report)
    active = None
    try:
        if all(entry["status"] == "skipped" for entry in report["scenarios"].values()):
            report["status"] = "complete_with_skips"
            save_report(store, report)
            return 0
        from pycocotools.coco import COCO  # noqa: F401
        import cv2
        cv2.setNumThreads(1)
        import torch
        torch.set_num_threads(args.threads)
        for config in configs:
            name = config["name"]
            if report["scenarios"][name]["status"] == "skipped":
                continue
            active = name
            report["scenarios"][name]["status"] = "running"
            save_report(store, report)
            result = run_scenario(config, dataset, args, store, masks)
            if config["prompt_cache"]:
                compare_cached_predictions(store, result)
                atomic_json(store.summary_path(name), result)
            report["scenarios"][name] = result
            save_report(store, report)
        report["status"] = "complete_with_skips" if any(entry["status"] == "skipped"
                                                          for entry in report["scenarios"].values()) else "complete"
        save_report(store, report)
        print(f"完成：{store.output / 'summary.md'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "stopped"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if active is not None:
            report["scenarios"][active].update(status="stopped", error=report["error"])
        save_report(store, report)
        from resource_safety import handle_cuda_oom
        handle_cuda_oom(exc)
        print(f"批次已停止；已完成图片检查点保留，不会切换设备重试：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
