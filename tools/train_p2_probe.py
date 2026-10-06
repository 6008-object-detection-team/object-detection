"""Prepare a CPU-only, short YOLO26n/P2 training functionality comparison.

The default command only validates and prints a plan. Training requires an
explicit --run. Imports, --help and --plan never import torch, download weights
or allocate a model. This detector is distinct from the application's YOLOE.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluate_mini_dataset import atomic_json, sha256_file, versions
from resource_safety import available_ram_mb, process_max_rss_mib
from tools.prepare_training_probe import jpeg_dimensions, yolo_labels

ASSET_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
SOURCE_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
MAX_WEIGHT_BYTES = 16 * 1024 ** 2
MAX_DATASET_BYTES = 40 * 1024 ** 2
CLASSES = ["bottle", "cup", "cell phone"]
VARIANTS = {"standard": "yolo26n.yaml", "p2": "yolo26n-p2.yaml"}
NOTICE = (
    "Only a training functionality probe: 3 epochs and a tiny public development split cannot "
    "establish P2 accuracy gains. P2 has no matching fully pretrained extra head; initialization "
    "and transferred tensor coverage differ. Name/shape matches do not establish semantic "
    "equivalence, and coverage excludes partial class-row remapping. Both models use the same "
    "12 development validation images, not the original 24 diagnostic images or 3 references. "
    "This is closed-set YOLO26, not YOLOE; its weights cannot directly replace the open-vocabulary "
    "UI. Ultralytics mAP and its evaluation defaults must not be equated with earlier COCO AP. "
    "CPU time depends on other host jobs. Training keeps framework EMA copies, but comparison "
    "architectures are loaded and released sequentially. The report status and per-variant "
    "epoch records distinguish planning, interrupted runs, and completed functionality probes."
)


def positive_integer(value):
    value = int(value)
    if not 1 <= value <= 3:
        raise argparse.ArgumentTypeError("流程探针只允许 1 到 3 个 epoch")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="CPU P2 短训练准备；默认只计划，显式 --run 才训练。")
    parser.add_argument("--data", type=Path, default=Path("datasets/mini_train_probe/data.yaml"))
    parser.add_argument("--source-annotations", type=Path, default=Path("work/dataset_cache/instances_val2017.json"))
    parser.add_argument("--exclude-dataset", type=Path, default=Path("datasets/mini_eval"))
    parser.add_argument("--weights", type=Path, default=Path("models/training/yolo26n.pt"))
    parser.add_argument("--output", type=Path, default=Path("work/additional_experiments/p2_training_probe"))
    parser.add_argument("--epochs", type=positive_integer, default=3)
    parser.add_argument("--device", choices=("CPU",), default="CPU")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--run", action="store_true", help="未来手动确认资源后启动真实 CPU 短训练")
    return parser


def resource_check(output, minimum_ram_mb=3072, ram_reader=available_ram_mb, disk_reader=shutil.disk_usage):
    free_ram = ram_reader()
    if free_ram < minimum_ram_mb:
        raise RuntimeError(f"CPU 训练已停止：可用 RAM {free_ram} MiB，低于 {minimum_ram_mb} MiB")
    existing = Path(output).resolve()
    while not existing.exists():
        existing = existing.parent
    free_disk = disk_reader(existing).free // 1024 ** 2
    if free_disk < 1024:
        raise RuntimeError("CPU 训练已停止：输出磁盘可用空间不足 1 GiB")
    return {"available_ram_mib": free_ram, "free_disk_mib": free_disk}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"数据文件不存在或越界：{relative}")
    return path


def validate_dataset(args, expected_source_sha=SOURCE_SHA256):
    """Verify every label against the authentic original COCO annotations."""
    import yaml  # no torch; installed Ultralytics dependency

    root = args.data.resolve().parent
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = config.get("names")
    names = [names[index] for index in range(3)] if isinstance(names, dict) and set(names) == {0, 1, 2} else names
    if names != CLASSES or config.get("train") != "images/train" or config.get("val") != "images/val" \
            or "download" in config or config.get("path", str(root)) != str(root):
        raise ValueError("只允许本地 3 类、固定 images/train 与 images/val 的训练数据")
    manifest = json.loads(inside(root, "manifest.json").read_text(encoding="utf-8"))
    source_hash = sha256_file(args.source_annotations)
    if source_hash != expected_source_sha or manifest.get("source_sha256") != source_hash:
        raise ValueError("原始人工标注来源 SHA256 不匹配，停止训练")
    source = json.loads(args.source_annotations.read_text(encoding="utf-8"))
    source_images = {int(image["id"]): image for image in source["images"]}
    source_annotations = {int(ann["id"]): ann for ann in source["annotations"]}
    recorded_images = {record["image_file"].replace("\\", "/"): record["sha256"]
                       for record in manifest.get("images", [])}
    original_targets = {}
    for ann in source["annotations"]:
        if ann["category_id"] in (44, 47, 77):
            original_targets.setdefault(int(ann["image_id"]), []).append(ann)
    evaluation = json.loads((args.exclude_dataset / "annotations.json").read_text(encoding="utf-8"))
    references = json.loads((args.exclude_dataset / "references.json").read_text(encoding="utf-8"))
    forbidden = {int(image["id"]) for image in evaluation["images"]} \
        | {int(ref["original_image_id"]) for ref in references["references"]}
    hashes, split_ids, summaries = {}, {}, {}
    all_image_hashes = set()
    for split, required_count in (("train", 48), ("val", 12)):
        annotation_path = inside(root, f"annotations/{split}.json")
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if [(cat["id"], cat["name"]) for cat in annotations["categories"]] != list(zip((44, 47, 77), CLASSES)):
            raise ValueError("COCO 类别映射不一致")
        ids = [int(image["id"]) for image in annotations["images"]]
        if len(ids) != required_count or len(set(ids)) != len(ids) or forbidden.intersection(ids):
            raise ValueError("训练/验证数量或 ID 隔离检查失败；不能使用原测试或视觉参考图")
        split_ids[split] = ids
        expected_annotations = {ann["id"]: ann for image_id in ids for ann in original_targets.get(image_id, [])}
        actual_annotations = {ann["id"]: ann for ann in annotations["annotations"]}
        if len(actual_annotations) != len(annotations["annotations"]) or actual_annotations != expected_annotations \
                or any(source_annotations.get(key) != ann for key, ann in actual_annotations.items()):
            raise ValueError("训练标注没有完整保留原 COCO 人工真值；禁止模型伪标签充当 GT")
        labels, skipped, clipped = yolo_labels(annotations)
        for image in annotations["images"]:
            if source_images.get(int(image["id"])) != image:
                raise ValueError("原始图片 ID/元数据被修改")
            path = inside(root, f"images/{split}/{image['file_name']}")
            if jpeg_dimensions(path.read_bytes()) != (int(image["width"]), int(image["height"])):
                raise ValueError("图片实际尺寸与原人工标注不一致")
            image_hash = sha256_file(path)
            if recorded_images.get(path.relative_to(root).as_posix()) != image_hash:
                raise ValueError("图片与原下载 manifest 的 SHA256 不一致")
            if image_hash in all_image_hashes:
                raise ValueError("训练/验证存在重复图片内容")
            all_image_hashes.add(image_hash)
            label = inside(root, f"labels/{split}/{Path(image['file_name']).stem}.txt")
            if label.read_bytes() != labels[int(image["id"])]:
                raise ValueError("YOLO 标签与原始人工 bbox 转换结果不一致")
            hashes[path.relative_to(root).as_posix()] = image_hash
            hashes[label.relative_to(root).as_posix()] = sha256_file(label)
        actual_image_names = {path.name for path in (root / "images" / split).iterdir() if path.is_file()}
        if actual_image_names != {image["file_name"] for image in annotations["images"]}:
            raise ValueError("images 目录混入未审计图片")
        counts = Counter(ann["category_id"] for ann in annotations["annotations"] if not ann.get("iscrowd", 0))
        summaries[split] = {"images": len(ids), "human_annotations": len(actual_annotations),
                            "yolo_labels": sum(payload.count(b"\n") for payload in labels.values()),
                            "per_class_noncrowd": {name: counts[category] for name, category in zip(CLASSES, (44, 47, 77))},
                            "skipped": skipped, "clipped": clipped}
        hashes[f"annotations/{split}.json"] = sha256_file(annotation_path)
    if set(split_ids["train"]) & set(split_ids["val"]):
        raise ValueError("训练与验证 image ID 重叠")
    for relative, record in manifest.get("files", {}).items():
        if sha256_file(inside(root, relative)) != record["sha256"]:
            raise ValueError(f"数据 manifest 文件哈希不一致：{relative}")
    dataset_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if dataset_bytes > MAX_DATASET_BYTES:
        raise ValueError("训练数据超过 40 MiB 限额")
    hashes["manifest.json"] = sha256_file(root / "manifest.json")
    hashes["data.yaml"] = sha256_file(args.data)
    return {"root": str(root), "source_sha256": source_hash, "files": hashes,
            "dataset_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "split_ids": split_ids, "splits": summaries, "excluded_ids": sorted(forbidden), "bytes": dataset_bytes}


def training_options(args, variant, weights):
    return {"model": VARIANTS[variant], "pretrained": str(Path(weights).resolve()),
            "data": str(args.data.resolve()), "device": "cpu", "epochs": args.epochs,
            "imgsz": 416, "batch": 1, "workers": 0, "cache": False, "amp": False,
            "plots": False, "save": True, "save_period": -1, "val": True,
            "project": str(args.output.resolve() / "models"), "name": variant, "exist_ok": False,
            "optimizer": "AdamW", "lr0": 0.001, "nbs": 1, "warmup_epochs": 0.0,
            "patience": 0, "seed": 6008, "deterministic": True, "mosaic": 0.0,
            "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "close_mosaic": 0,
            "resume": False, "compile": False, "verbose": False}


def ensure_weights(path, opener=urllib.request.urlopen):
    path = Path(path).resolve()
    if path.name != "yolo26n.pt":
        raise ValueError("本工具只允许轻量 yolo26n.pt；不会自动下载其他模型")
    downloaded = not path.is_file()
    if downloaded:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.part")
        try:
            request = urllib.request.Request(ASSET_URL, headers={"User-Agent": "EE6008-CPU-training-probe"})
            with opener(request, timeout=30) as response, temporary.open("wb") as stream:
                declared = int(response.headers.get("Content-Length", "0"))
                if declared > MAX_WEIGHT_BYTES:
                    raise ValueError("官方轻量权重超过下载限额")
                size = 0
                while chunk := response.read(64 * 1024):
                    size += len(chunk)
                    if size > MAX_WEIGHT_BYTES:
                        raise ValueError("权重下载超过 16 MiB 限额")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if size < 1024:
                raise ValueError("权重下载结果过小")
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    if not 1024 <= path.stat().st_size <= MAX_WEIGHT_BYTES:
        raise ValueError("本地权重大小不符合轻量模型限额")
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
              "url": ASSET_URL if downloaded else None,
              "provenance": "downloaded_official_asset" if downloaded else "preexisting_local_file"}
    metadata = path.with_suffix(".pt.source.json")
    if metadata.is_file() and not downloaded:
        saved = json.loads(metadata.read_text())
        if saved["sha256"] != record["sha256"]:
            raise ValueError("本地模型与原下载来源 SHA256 不一致")
        record = saved
    else:
        atomic_json(metadata, record)
    return record


def tensor_coverage(source_shapes, target_shapes):
    matched = {key for key, shape in target_shapes.items() if source_shapes.get(key) == shape}
    total_values = sum(math.prod(shape) for shape in target_shapes.values())
    matched_values = sum(math.prod(target_shapes[key]) for key in matched)
    return {"matched_tensors": len(matched), "target_tensors": len(target_shapes),
            "tensor_fraction": len(matched) / len(target_shapes) if target_shapes else None,
            "matched_scalar_values": matched_values, "target_scalar_values": total_values,
            "scalar_fraction": matched_values / total_values if total_values else None,
            "unmatched_keys": sorted(set(target_shapes) - matched),
            "notice": "精确名称/shape 的保守覆盖；未计入按类别名称局部重映射，不代表语义等价"}


def apply_thread_limit(torch_module, cv2_module, torch_utils_module=None):
    """Prevent select_device(cpu) from resetting Torch to its larger default."""
    if torch_utils_module is not None:
        torch_utils_module.NUM_THREADS = 2
    torch_module.set_num_threads(2)
    cv2_module.setNumThreads(1)


def before_probe_batch(trainer):
    """The installed native catch raises immediately when this counter is 3.

    Arm only while a batch is running. Leaving the counter armed after success
    could affect the native epoch-restart condition; the end callback clears it.
    """
    trainer._oom_retries = 3


def after_probe_batch(trainer):
    """A successful batch must not request a native epoch restart."""
    trainer._oom_retries = 0


def epoch_progress(current, started):
    record = {"elapsed_seconds": time.perf_counter() - started,
              "process_max_rss_mib": process_max_rss_mib()}
    if int(current.epoch) >= int(current.epochs):
        record.update(phase="best_checkpoint_validation", after_training_epochs=int(current.epochs))
    else:
        record.update(phase="training_epoch", epoch=int(current.epoch) + 1)
    return record


def source_hashes():
    result = {relative: sha256_file(PROJECT / relative) for relative in (
        "tools/train_p2_probe.py", "resource_safety.py", "evaluate_mini_dataset.py",
        "tools/prepare_training_probe.py",
    )}
    installed = importlib.metadata.distribution("ultralytics")
    for relative in (
        "ultralytics/engine/trainer.py", "ultralytics/engine/validator.py",
        "ultralytics/models/yolo/detect/train.py", "ultralytics/utils/torch_utils.py",
        "ultralytics/nn/tasks.py", "ultralytics/cfg/default.yaml",
        "ultralytics/cfg/models/26/yolo26.yaml", "ultralytics/cfg/models/26/yolo26-p2.yaml",
    ):
        result["installed/" + relative] = sha256_file(installed.locate_file(relative))
    return result


def trainer_factory(options, progress):
    import torch
    import cv2
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import torch_utils
    def limit_threads():
        # Installed 8.4.172 select_device resets Torch threads to this module
        # constant. Change only this separate probe process, not user settings.
        apply_thread_limit(torch, cv2, torch_utils)
    limit_threads()

    class ProbeTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            source_shapes = {key: tuple(value.shape) for key, value in weights.state_dict().items()} if weights else {}
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            limit_threads()
            self.probe_transfer = tensor_coverage(source_shapes, {key: tuple(value.shape) for key, value in model.state_dict().items()})
            return model

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            # The stock trainer doubles the validation batch. This probe keeps
            # both loaders at one, even when the framework requests two.
            loader = super().get_dataloader(dataset_path, batch_size=1, rank=rank, mode=mode)
            limit_threads()
            return loader

    trainer = ProbeTrainer(overrides=options)
    limit_threads()
    def guard(_trainer):
        limit_threads()
        resource_check(options["project"], minimum_ram_mb=1536)
    def before_batch(current):
        before_probe_batch(current)
        guard(current)
    trainer.add_callback("on_train_batch_start", before_batch)
    trainer.add_callback("on_train_batch_end", after_probe_batch)
    trainer.add_callback("on_train_epoch_start", guard)
    def epoch_end(current):
        progress.append(epoch_progress(current, started))
    started = time.perf_counter()
    trainer.add_callback("on_fit_epoch_end", epoch_end)
    return trainer


def plain_json(value):
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if hasattr(value, "item"):
        return plain_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def release_trainer(trainer):
    if trainer is not None:
        for name in ("model", "ema", "optimizer", "validator", "train_loader", "test_loader"):
            if hasattr(trainer, name):
                setattr(trainer, name, None)
    gc.collect()


def run_variants(args, dataset, weight_record, factory=trainer_factory, checker=resource_check):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("训练输出目录非空；使用新的 --output，避免覆盖实验")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(), "notice": NOTICE,
              "dataset": dataset, "weights": weight_record, "versions": versions(),
              "source_hashes": source_hashes(), "native_oom_retries": 0, "variants": {}}
    atomic_json(output / "report.json", report)
    for variant in VARIANTS:
        trainer = None
        epochs = []
        options = training_options(args, variant, weight_record["path"])
        report["variants"][variant] = {"status": "running", "options": options}
        atomic_json(output / "report.json", report)
        started = time.perf_counter()
        try:
            checker(output, minimum_ram_mb=3072)
            trainer = factory(options, epochs)
            trainer.train()
            metrics = trainer.validator.metrics
            checkpoints = {name: {"path": str(path), "sha256": sha256_file(path)}
                           for name in ("best", "last") if (path := Path(getattr(trainer, name))).is_file()}
            report["variants"][variant] = {"status": "complete", "options": options,
                "wall_seconds": time.perf_counter() - started,
                "process_max_rss_mib": process_max_rss_mib(),
                "rss_note": "进程累计 RSS 高水位，不是每个模型独立峰值", "epochs": epochs,
                "transfer": trainer.probe_transfer, "metrics": plain_json(metrics.results_dict),
                "per_class": plain_json(metrics.summary()), "checkpoints": checkpoints}
            atomic_json(output / "report.json", report)
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "stopped"
            report["variants"][variant].update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            atomic_json(output / "report.json", report)
            raise
        finally:
            release_trainer(trainer)
            trainer = None
    report["status"] = "complete_functionality_probe"
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    # A separate invocation hides all GPUs before any optional library import.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
    plan = {"status": "prepared_not_executed", "device": "CPU", "cuda_hidden": True,
            "threads": {"torch": 2, "opencv": 1}, "notice": NOTICE,
            "source_hashes": source_hashes(), "native_oom_retries": 0,
            "weights": {"path": str(args.weights.resolve()), "download_url_if_missing": ASSET_URL,
                        "download_limit_bytes": MAX_WEIGHT_BYTES},
            "variants": {name: training_options(args, name, args.weights) for name in VARIANTS}}
    if args.data.is_file():
        plan["resources"] = resource_check(args.output)
        plan["dataset"] = validate_dataset(args)
    else:
        plan["dataset"] = {"status": "awaiting_dataset", "data": str(args.data.resolve())}
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if not args.data.is_file():
        raise ValueError("训练数据尚未完成；保持准备状态")
    weight_record = ensure_weights(args.weights)
    # Only the one explicit small asset is downloaded above. The trainer then
    # stays offline and uses that local file for both architectures.
    os.environ["YOLO_OFFLINE"] = "True"
    import cv2
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(2)
    report = run_variants(args, plan["dataset"], weight_record)
    print(f"完成流程验证：{args.output.resolve() / 'report.json'}；{report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
