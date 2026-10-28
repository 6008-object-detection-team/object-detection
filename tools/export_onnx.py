"""Export YOLOE with a text vocabulary fixed at export time to ONNX + manifest.

--help and --plan neither import torch nor load a checkpoint. Export always
runs on CPU, serially, after its own RAM/disk check: the 3 GiB inference budget
is not an export memory guarantee. Ultralytics auto-install is disabled, so a
missing onnx/onnxslim/onnxruntime stops the export instead of changing the
environment. Existing files are never overwritten. .onnx files and manifests
are build products and stay outside Git.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from onnx_engine import MANIFEST_SCHEMA, OUTPUT_FORMAT, manifest_path, sha256_file, validate_manifest

# Same names and order as pan_deng/evaluate_detectors.py (240-image COCO comparison).
COCO10 = ["person", "bicycle", "car", "dog", "backpack", "bottle", "cup", "chair", "laptop", "cell phone"]
VARIANTS = ("seg", "det")
# Detection-branch conversion is limited to the matching 26S architecture, as in
# tools/run_additional_experiments.py, so no tensor is transferred into a wrong shape.
DETECTION_SOURCE = "yoloe-26s-seg.pt"
DETECTION_YAML = "yoloe-26s.yaml"
TEXT_ENCODER = "mobileclip2_b.ts"
TEXT_ENCODER_FILES = {"mobileclip2:b": "mobileclip2_b.ts", "mobileclip:blt": "mobileclip_blt.ts"}
OPSET_RANGE = (11, 21)
MIN_EXPORT_RAM_MB = 3072
MIN_EXPORT_DISK_MB = 1024
LIMITATIONS = [
    "Vocabulary is fixed at export: the ONNX graph has no text encoder, so text or visual prompts cannot change classes.",
    "Raw one-to-many output (xywh + class scores [+ 32 mask coefficients]); NMS runs outside the graph.",
    "Static batch 1 and a square imgsz x imgsz input; the PyTorch app path uses a minimal rectangular letterbox instead.",
    "Only CPUExecutionProvider was checked; onnxruntime-gpu/TensorRT are not part of this environment.",
    "Prompt-free (LRPC, 4585-class) checkpoints are not exported by this tool.",
]


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def parse_classes(value):
    names = [part.strip() for part in value.split(",")]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("类别必须是逗号分隔、非空且不重复的英文名称")
    return names


def parse_variants(value):
    names = [part.strip() for part in value.split(",")]
    if not names or len(set(names)) != len(names) or any(name not in VARIANTS for name in names):
        raise argparse.ArgumentTypeError("变体必须是 seg,det 中不重复的逗号分隔列表")
    return names


def build_parser():
    parser = argparse.ArgumentParser(description="把固定文字类别的 YOLOE 导出为 ONNX（CPU、batch=1、静态尺寸），并写入 manifest。")
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--class-set", choices=("mini_eval", "coco10"), default="mini_eval",
                        help="mini_eval：读取数据集 references.json 的类别；coco10：240 图 COCO 对照的 10 类")
    parser.add_argument("--classes", type=parse_classes, help="自定义类别（覆盖 --class-set），例如 \"person,bottle,cup\"")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/mini_eval"))
    parser.add_argument("--variants", type=parse_variants, default=list(VARIANTS),
                        help="seg：完整分割图（含 mask 原型输出）；det：仅检测分支转换")
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--opset", type=positive_integer, default=17)
    parser.add_argument("--threads", type=positive_integer, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("work/onnx_export"))
    parser.add_argument("--plan", action="store_true", help="只验证参数并输出计划，不加载模型")
    return parser


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def resolve_classes(args):
    if args.classes:
        return list(args.classes), {"class_set": "custom"}
    if args.class_set == "coco10":
        return list(COCO10), {"class_set": "coco10", "order": "COCO category order (pan_deng/evaluate_detectors.py)"}
    references = project_path(args.dataset) / "references.json"
    if not references.is_file():
        raise ValueError(f"找不到数据集类别文件：{references}")
    classes = json.loads(references.read_text(encoding="utf-8")).get("classes", [])
    try:
        parse_classes(",".join(classes))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"数据集类别无效：{exc}") from exc
    return list(classes), {"class_set": "mini_eval", "references_json": str(references),
                           "references_sha256": sha256_file(references)}


def output_stem(model, variant, class_set, imgsz):
    stem = Path(model).stem
    if variant == "det":
        stem = stem[:-4] + "-det" if stem.endswith("-seg") else stem + "-det"
    return f"{stem}-{class_set}-{imgsz}"


def plan_export(args):
    """Validate everything that does not need torch; return the export plan."""
    model = project_path(args.model)
    if not model.is_file() or model.suffix != ".pt":
        raise ValueError(f"找不到 .pt 权重：{model}")
    if model.stat().st_size < 1024 * 1024:
        raise ValueError(f"权重文件过小，可能是 Git LFS 指针：{model}")
    if model.name.endswith("-pf.pt"):
        raise ValueError("无提示（LRPC）权重没有可固定的文字类别；本工具只导出文字提示原版权重")
    if "det" in args.variants and model.name != DETECTION_SOURCE:
        raise ValueError(f"仅检测分支转换只支持 {DETECTION_SOURCE}，避免把权重转移到不匹配的结构")
    if not 32 <= args.imgsz <= 960 or args.imgsz % 32:
        raise ValueError("imgsz 必须在 32–960 之间且是 32 的倍数")
    if not OPSET_RANGE[0] <= args.opset <= OPSET_RANGE[1]:
        raise ValueError(f"opset 必须在 {OPSET_RANGE[0]}–{OPSET_RANGE[1]} 之间")
    classes, class_source = resolve_classes(args)
    output = project_path(args.output_dir).resolve()
    targets = {}
    for variant in args.variants:
        stem = output_stem(model, variant, class_source["class_set"], args.imgsz)
        onnx_file = output / f"{stem}.onnx"
        for path in (onnx_file, manifest_path(onnx_file)):
            if path.exists():
                raise ValueError(f"输出已存在，不覆盖旧文件或 manifest：{path}")
        targets[variant] = str(onnx_file)
    return {"model": str(model), "classes": classes, "class_source": class_source, "variants": list(args.variants),
            "imgsz": args.imgsz, "opset": args.opset, "threads": args.threads, "output_dir": str(output),
            "targets": targets}


def check_export_resources(output, min_ram_mb=MIN_EXPORT_RAM_MB, min_disk_mb=MIN_EXPORT_DISK_MB):
    """Independent export check; the inference budget does not cover export."""
    from resource_safety import ResourceLimitError, available_ram_mb
    free_ram = available_ram_mb()
    if free_ram < min_ram_mb:
        raise ResourceLimitError(f"导出停止：系统可用内存 {free_ram} MiB，低于 {min_ram_mb} MiB")
    free_disk = shutil.disk_usage(output).free // 1024 ** 2
    if free_disk < min_disk_mb:
        raise ResourceLimitError(f"导出停止：输出磁盘可用 {free_disk} MiB，低于 {min_disk_mb} MiB")
    return {"available_ram_mib": free_ram, "output_disk_free_mib": free_disk}


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "ultralytics", "onnx", "onnxslim", "onnxruntime", "numpy", "opencv-python"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def source_hashes():
    return {relative: sha256_file(PROJECT / relative) for relative in (
        "tools/export_onnx.py", "onnx_engine.py", "engines.py", "resource_safety.py")}


def host_info():
    cpu = None
    try:
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), None)
    except OSError:
        pass
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {"cpu": cpu or platform.processor(), "logical_cpus": os.cpu_count(), "loadavg": load}


def file_record(path):
    path = Path(path)
    return {"path": str(path), "resolved_path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def verify_transfer(target_state, source_state):
    """Every tensor of the detection graph must be an exact copy from the checkpoint."""
    import torch
    copied, problems = 0, []
    for key, value in target_state.items():
        source = source_state.get(key)
        if source is None or tuple(source.shape) != tuple(value.shape):
            problems.append(key)
        elif not torch.equal(source.to(value.dtype), value):
            problems.append(key)
        else:
            copied += 1
    if problems:
        raise ValueError(f"检测分支有 {len(problems)} 个张量没有从原权重精确复制，例如 {problems[:3]}")
    return {"target_tensors": len(target_state), "copied_exactly": copied,
            "source_only_tensors": len(set(source_state) - set(target_state))}


def build_detection_model(model_path, classes, embeddings, source_state):
    """Matching detection architecture with the seg checkpoint's weights; no mask branch."""
    from ultralytics import YOLOE
    model = YOLOE(DETECTION_YAML).load(str(model_path))
    model.model.ckpt = None
    if model.task != "detect":
        raise ValueError("匹配 YAML 没有构建检测架构，停止转换")
    transfer = verify_transfer({key: value.detach() for key, value in model.model.state_dict().items()}, source_state)
    model.set_classes(classes, embeddings)
    return model, transfer


def onnx_check(path, classes, imgsz, threads):
    """Structural check, then one CPU ORT run on a constant letterbox-gray input."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    opsets = {item.domain or "ai.onnx": item.version for item in graph.opset_import}
    ir_version = graph.ir_version
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = [{"name": item.name, "shape": list(item.shape), "dtype": item.type} for item in session.get_inputs()]
    outputs = [{"name": item.name, "shape": list(item.shape), "dtype": item.type} for item in session.get_outputs()]
    if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, imgsz, imgsz]:
        raise ValueError(f"ONNX 输入不是静态 [1, 3, {imgsz}, {imgsz}]：{inputs}")
    values = session.run(None, {inputs[0]["name"]: np.full((1, 3, imgsz, imgsz), 114 / 255, np.float32)})
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("ONNX Runtime 输出包含非有限值")
    shape = values[0].shape
    extra = shape[1] - 4 - len(classes)
    if len(shape) != 3 or shape[0] != 1 or extra < 0:
        raise ValueError(f"ONNX 主输出形状与固定类别不一致：{shape}")
    return {"opset_import": opsets, "ir_version": ir_version, "inputs": inputs, "outputs": outputs,
            "providers": session.get_providers(), "constant_input_outputs_finite": True,
            "output_layout": {"format": OUTPUT_FORMAT, "box": "xywh in letterboxed input pixels",
                              "num_classes": len(classes), "extra_channels": int(extra), "anchors": int(shape[2])}}


def export_variant(model, variant, plan, staging, prompt, source_record, encoder_record, transfer=None):
    target = Path(plan["targets"][variant])
    staged = staging / target.name
    # Ultralytics writes next to pt_path; point it at the private staging folder.
    model.model.pt_path = str(staged.with_suffix(".pt"))
    start = time.perf_counter()
    exported = Path(model.export(format="onnx", imgsz=plan["imgsz"], opset=plan["opset"], simplify=True,
                                 dynamic=False, batch=1, device="cpu", verbose=False))
    seconds = time.perf_counter() - start
    if exported.resolve() != staged.resolve() or not staged.is_file():
        raise RuntimeError(f"Ultralytics 输出路径异常：{exported}")
    check = onnx_check(staged, plan["classes"], plan["imgsz"], plan["threads"])
    expected_extra = 32 if variant == "seg" else 0
    if check["output_layout"]["extra_channels"] != expected_extra:
        raise ValueError(f"{variant} 输出附加通道应为 {expected_extra}，实际 {check['output_layout']['extra_channels']}")
    from resource_safety import process_max_rss_mib
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "exported",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "file": target.name, "onnx_sha256": sha256_file(staged), "onnx_bytes": staged.stat().st_size,
        "variant": variant,
        "variant_note": ("Full segmentation graph; output1 mask prototypes are computed but the engine ignores them."
                         if variant == "seg" else
                         "Detection-branch-only conversion (yoloe-26s.yaml + exact tensor copy); no mask branch."),
        "source_model": source_record, "text_encoder": encoder_record,
        "classes": list(plan["classes"]), "class_source": plan["class_source"],
        "prompt_embeddings": prompt,
        "export": {"format": "onnx", "imgsz": plan["imgsz"], "opset": plan["opset"], "batch": 1, "dynamic": False,
                   "simplify": "onnxslim", "nms": None, "end2end": False, "head": "one2many", "precision": "fp32",
                   "device": "cpu", "seconds": seconds},
        "io": {"inputs": check["inputs"], "outputs": check["outputs"]},
        "output_layout": check["output_layout"],
        "preprocess": {"letterbox": "square, centered", "padding_value": 114, "interpolation": "cv2.INTER_LINEAR",
                       "color": "BGR->RGB", "scale": "1/255", "layout": "NCHW float32"},
        "postprocess": {"nms": "outside graph, Ultralytics-compatible greedy NMS (onnx_engine.py)"},
        "onnx_check": {key: check[key] for key in ("opset_import", "ir_version", "providers",
                                                    "constant_input_outputs_finite")},
        "detection_transfer": transfer, "versions": versions(), "source_hashes": source_hashes(),
        "host": host_info(), "process_max_rss_mib_cumulative": process_max_rss_mib(),
        "limitations": LIMITATIONS,
    }
    validate_manifest(manifest)
    temporary = staging / (manifest_path(target).name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for path in (target, manifest_path(target)):
        if path.exists():
            raise ValueError(f"输出在导出期间出现，不覆盖：{path}")
    os.replace(staged, target)
    os.replace(temporary, manifest_path(target))
    return manifest


def run_export(plan):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    # Never let Ultralytics pip-install exporter dependencies into the environment.
    os.environ["YOLO_AUTOINSTALL"] = "False"
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    resources = check_export_resources(output)
    import torch
    torch.set_num_threads(plan["threads"])
    from ultralytics import YOLOE
    model_path = Path(plan["model"])
    source_record = file_record(model_path)
    staging = output / f".staging-{os.getpid()}"
    staging.mkdir()
    manifests = []
    previous = Path.cwd()
    try:
        os.chdir(PROJECT)  # The text encoder is resolved relative to the project.
        model = YOLOE(str(model_path))
        if hasattr(model.model.model[-1], "lrpc"):
            raise ValueError("这是无提示 LRPC 权重，不能固定文字类别")
        encoder_name = getattr(model.model, "text_model", None)
        encoder = PROJECT / TEXT_ENCODER_FILES.get(encoder_name, "")
        encoder_record = {"variant": encoder_name, **(file_record(encoder) if encoder.is_file() else {"status": "not found"})}
        start = time.perf_counter()
        embeddings = model.get_text_pe(plan["classes"]).detach().cpu()
        encoding_ms = (time.perf_counter() - start) * 1000
        prompt = {"sha256": tensor_sha256(embeddings), "shape": list(embeddings.shape),
                  "encoding_ms": encoding_ms, "encoded_by": "source checkpoint get_text_pe, before export"}
        source_state = {key: value.detach().float().clone() for key, value in model.model.state_dict().items()} \
            if "det" in plan["variants"] else None
        if "seg" in plan["variants"]:
            model.set_classes(plan["classes"], embeddings)
            manifests.append(export_variant(model, "seg", plan, staging, prompt, source_record, encoder_record))
        del model
        gc.collect()
        if "det" in plan["variants"]:
            check_export_resources(output)
            model, transfer = build_detection_model(model_path, plan["classes"], embeddings, source_state)
            del source_state
            manifests.append(export_variant(model, "det", plan, staging, prompt, source_record,
                                            encoder_record, transfer))
            del model
        return {"status": "complete", "resources_before": resources, "text_encoding_ms": encoding_ms,
                "files": [str(output / item["file"]) for item in manifests]}
    finally:
        os.chdir(previous)
        # Only this run's private staging folder; finished files were moved out.
        shutil.rmtree(staging, ignore_errors=True)
        gc.collect()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = plan_export(args)
    except (ValueError, OSError) as exc:
        print(f"导出前已停止：{exc}", file=sys.stderr)
        return 1
    if args.plan:
        print(json.dumps({"status": "planned_not_executed", **plan}, ensure_ascii=False, indent=2))
        return 0
    try:
        result = run_export(plan)
    except Exception as exc:
        print(f"导出已停止（未重试、未切换设备）：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
