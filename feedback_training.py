"""Explicit, isolated head-only YOLOE correction training, without prompt fusion.

Retains the original model class, SAVPE, RepRTA and mask branches. Does not use
the closed-set PE trainers. No torch import, allocation or download at import.
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from feedback import PROJECT, atomic_json, safe_path, sha256_file
from i18n import message, exception_message
from model_families import MODEL_FAMILIES, record_family

TRAIN_CHILDREN = ("cv2", "cv3", "cv4", "one2one_cv2", "one2one_cv3", "one2one_cv4")


class TrainingCancelled(RuntimeError):
    pass


def validate_dataset(path):
    path = Path(path).absolute()
    if sha256_file(path/"manifest.json") != (path/"manifest.sha256").read_text(encoding="ascii").strip():
        raise ValueError("数据清单被修改；不要覆盖旧指纹，请重新导出新版本")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mode = manifest.get("training_mode", "strict")
    if mode not in ("strict", "correction_demo"):
        raise ValueError("训练数据模式损坏，请重新导出")
    demo = mode == "correction_demo"
    files = manifest["files"]
    for relative, expected in files.items():
        if sha256_file(safe_path(path, relative)) != expected:
            raise ValueError("导出后数据被修改；请生成新的数据版本")
    config = json.loads((path / "data.yaml").read_text(encoding="utf-8"))
    if config["names"] != manifest["classes"] or "download" in config:
        raise ValueError("类别清单损坏")
    if demo and (manifest.get("independent_validation") is not False or "val" in config or "test" in config):
        raise ValueError("同图演示不能伪装为独立验证或测试")
    groups, images = {}, set()
    for row in manifest["samples"]:
        if row["status"] not in (("reviewed", "demo_accepted") if demo else ("reviewed",)) or row["image_id"] in images:
            raise ValueError("训练包包含未审核或重复样本")
        if demo and row["split"] != "train":
            raise ValueError("同图演示包只能包含明确采用的训练图")
        images.add(row["image_id"])
        scoped = manifest.get("per_sample_review_scope", False)
        if (not scoped and not set(manifest["classes"]).issubset(row["classes"])) or \
                (scoped and (not row["classes"] or not set(row["classes"]).issubset(manifest["classes"]))):
            raise ValueError("样本未审核全部类别")
        if scoped and record_family(row) != manifest.get("model_family"):
            raise ValueError("训练包混入其它基础模型的纠错，拒绝更新")
        for key in ("scene:" + row["scene"], "origin:" + row["provenance"]["origin_group"]):
            if groups.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("场景/来源跨训练、验证或测试")
        relative = f"images/{row['split']}/{row['image_id']}.png"
        label = f"labels/{row['split']}/{row['image_id']}.txt"
        if relative not in files or label not in files or files[relative] != row["image_sha256"]:
            raise ValueError("样本没有完整的图像/标签指纹")
    if demo and not manifest["samples"]:
        raise ValueError("请先打开图片并修改框，再点击一键更新。")
    if not demo and not all(any(row["split"] == split for row in manifest["samples"]) for split in ("train", "val")):
        raise ValueError("需要独立的训练与验证场景")
    return manifest


def resource_check(output, device, torch_module=None, initial=False):
    from resource_safety import available_ram_mb
    required = 4096 if initial else 1536
    if available_ram_mb() < required:
        raise RuntimeError(message("训练停止：系统可用内存低于 {required} MiB", required=required))
    if shutil.disk_usage(output).free < (3 if initial else 1) * 1024**3:
        raise RuntimeError("训练停止：输出磁盘可用空间不足")
    if device == "CUDA":
        if torch_module is None or not torch_module.cuda.is_available():
            raise RuntimeError("CUDA 不可用；不会自动切换 CPU")
        free, total = torch_module.cuda.mem_get_info(0)
        if free < (8192 if initial else 1024) * 1024**2:
            raise RuntimeError("训练停止：开始时需 8 GiB 空闲显存，运行时保留至少 1 GiB")
        if initial:
            torch_module.cuda.set_per_process_memory_fraction(min(1., 6144 * 1024**2 / total), 0)


def prompt_modules_present(model):
    head = model.model[-1]
    if hasattr(head, "lrpc") or getattr(head, "is_fused", False) or not hasattr(head, "savpe") or not hasattr(head, "reprta") \
            or not callable(getattr(model, "get_text_pe", None)) or not callable(getattr(model, "get_visual_pe", None)):
        raise ValueError("必须使用保留文字和视觉提示的未重参数化 YOLOE；不支持 DINO 或固定类别模型")


def configure_trainable(model):
    prompt_modules_present(model)
    model.requires_grad_(False)
    head = model.model[-1]
    selected = []
    for name in TRAIN_CHILDREN:
        child = getattr(head, name, None)
        if child is not None:
            child.requires_grad_(True)
            selected.append(name)
    if not selected:
        raise ValueError("模型没有可训练的检测分支")
    return selected


def set_training_mode(model):
    # With batch=1 / one photo, drifting running statistics can flood inference
    # with false positives even while training-mode loss decreases. Preserve
    # pretrained BN statistics in trainable heads too (affine may still learn).
    import torch
    model.eval()
    head = model.model[-1]
    head.training = True
    for name in TRAIN_CHILDREN:
        child = getattr(head, name, None)
        if child is not None:
            child.train()
            for module in child.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()


def batch_norm_statistics_fingerprint(model):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, tensor in module.named_buffers(recurse=False):
                digest.update((name + "." + key).encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    prefix = "model." + str(len(model.model)-1) + "."
    for name, tensor in model.state_dict().items():
        if any(name.startswith(prefix + child + ".") for child in TRAIN_CHILDREN):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def changed_head_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    prefix = "model." + str(len(model.model)-1) + "."
    for name, tensor in model.state_dict().items():
        if any(name.startswith(prefix + child + ".") for child in TRAIN_CHILDREN):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_batch(root, row, classes, imgsz, torch_module, device):
    from visual_engine import prepare_reference
    raw = cv2.imdecode(np.fromfile(str(root / f"images/{row['split']}/{row['image_id']}.png"), np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("训练图像无法读取")
    # Same letterbox as visual inference. Only actual human bounding boxes are used.
    image, _ = prepare_reference(raw, (0, 0, raw.shape[1], raw.shape[0]), imgsz)
    gain = min(imgsz/raw.shape[0], imgsz/raw.shape[1])
    left = round((imgsz-round(raw.shape[1]*gain))/2 - .1)
    top = round((imgsz-round(raw.shape[0]*gain))/2 - .1)
    boxes, ids = [], []
    for box in row["boxes"]:
        if box["label"] not in classes:
            continue
        x1, y1, x2, y2 = box["xyxy"]
        boxes.append([((x1+x2)*gain/2+left)/imgsz, ((y1+y2)*gain/2+top)/imgsz,
                      (x2-x1)*gain/imgsz, (y2-y1)*gain/imgsz])
        ids.append(classes.index(box["label"]))
    tensor = torch_module.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(device).float()/255.
    return {"img": tensor,
            "batch_idx": torch_module.zeros(len(ids), device=device),
            "cls": torch_module.tensor(ids, device=device, dtype=torch_module.float32).reshape(-1, 1),
            "bboxes": torch_module.tensor(boxes, device=device, dtype=torch_module.float32).reshape(-1, 4)}


def verify_prompt_capabilities(wrapper, imgsz=128):
    """Functional gates, not accuracy or retention-of-quality claims."""
    import torch
    from visual_engine import prepare_reference
    from ultralytics.data.augment import LoadVisualPrompt
    model = wrapper.model
    prompt_modules_present(model)
    parameter = next(model.parameters())
    raw = np.full((imgsz, imgsz, 3), 96, np.uint8)
    image, mapped = prepare_reference(raw, (imgsz/4, imgsz/4, 3*imgsz/4, 3*imgsz/4), imgsz)
    tensor = torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(parameter.device).float()/255.
    with torch.inference_mode():
        for names in (["person", "bicycle", "chair"], ["umbrella", "backpack"]):
            wrapper.set_classes(names)
            model.eval()
            model.predict(tensor)
            if model.model[-1].nc != len(names) or not torch.isfinite(model.pe).all():
                raise RuntimeError("更换任意文字提示的功能检查失败")
        visuals = LoadVisualPrompt().get_visuals(np.array([0], np.int32), image.shape[:2], bboxes=mapped)
        visuals = visuals.unsqueeze(0).to(parameter.device)
        embedding = wrapper.get_visual_pe(tensor, visuals)
        if tuple(embedding.shape[:2]) != (1, 1) or not torch.isfinite(embedding).all():
            raise RuntimeError("参考图片框选功能检查失败")
        wrapper.set_classes(["reference_object"], embedding)
        model.predict(tensor)
    return {"text_prompt": True, "visual_prompt": True, "accuracy_verified": False,
            "notice": "更换文字类别与参考图编码/前向测试通过；不是任意未知类别精度不下降的证明。"}


def train(dataset, parent, output, device="CPU", epochs=3, imgsz=640, display_name=None):
    dataset, parent, output = Path(dataset).absolute(), Path(parent).absolute(), Path(output).absolute()
    manifest = validate_dataset(dataset)
    demo = manifest.get("training_mode") == "correction_demo"
    learning_rate = .0001
    if device not in ("CPU", "CUDA") or not 1 <= epochs <= 100 or imgsz not in (128, 320, 640):
        raise ValueError("训练配置超出安全范围")
    if not parent.is_file() or parent.stat().st_size < 1024**2 or parent.suffix != ".pt":
        raise ValueError("请选择本地实际 YOLOE 权重，不接受 LFS 指针")
    if output.exists():
        raise ValueError("输出版本已存在；不覆盖旧版本")
    output.mkdir(parents=True, exist_ok=False)
    original_sha = sha256_file(parent)
    status = {"status": "training", "parent": str(parent), "parent_sha256": original_sha,
              "dataset": str(dataset), "dataset_sha256": sha256_file(dataset/"manifest.json"),
              "code_sha256": sha256_file(Path(__file__)), "epochs_requested": epochs,
              "device": device, "imgsz": imgsz, "batch": 1, "classes": manifest["classes"],
              "display_name": display_name or parent.stem + "（纠错型·" + output.name + "）",
              "text_prompt": False, "visual_prompt": False, "accuracy_verified": False,
              "optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": .001, "seed": 6008,
              "training_mode": "correction_demo" if demo else "strict",
              "independent_validation": not demo, "demo_exposure": demo,
              "selection_policy": "same_image_fit_loss" if demo else "validation_loss",
              "source_hashes": {name: sha256_file(PROJECT/name) for name in
                                ("feedback_training.py", "feedback.py", "visual_engine.py", "resource_safety.py")}}
    status["source_hashes"].update({name: sha256_file(PROJECT/name) for name in
                                  ("modified_models.py", "model_families.py", "prompt_modes.py")})
    atomic_json(output/"version.json", status)
    lease = False
    try:
        os.environ["YOLO_AUTOINSTALL"] = "False"
        os.environ["YOLO_OFFLINE"] = "True"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT/".ultralytics"))
        import torch
        from ultralytics import YOLOE
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
        from resource_safety import _acquire_gpu_lease, release_gpu_lease
        torch.set_num_threads(2)
        random.seed(6008)
        torch.manual_seed(6008)
        cv2.setNumThreads(1)
        import importlib.metadata
        status["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "ultralytics", "numpy", "PySide6")}
        status["hardware"] = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "CUDA" and torch.cuda.is_available() else None,
                              "cuda_runtime": torch.version.cuda, "training_gpu_budget_mib": 6144 if device == "CUDA" else None}
        if device == "CUDA":
            _acquire_gpu_lease()
            lease = True
        resource_check(output, device, torch, initial=True)
        shutil.copyfile(parent, output/"parent-copy.pt")
        if sha256_file(output/"parent-copy.pt") != original_sha:
            raise RuntimeError("父模型副本校验失败")
        wrapper = YOLOE(str(output/"parent-copy.pt"))
        model = wrapper.model.float()
        encoder_name = {"mobileclip2:b": "mobileclip2_b.ts", "mobileclip:blt": "mobileclip_blt.ts"}.get(model.text_model)
        if encoder_name is None or not (PROJECT/encoder_name).is_file() or (PROJECT/encoder_name).stat().st_size < 1024**2:
            raise ValueError("文字编码器未在项目中准备好；先手动补齐，不在训练中自动下载")
        configure_trainable(model)
        frozen_before = frozen_fingerprint(model)
        bn_before = batch_norm_statistics_fingerprint(model)
        head_before = changed_head_fingerprint(model)
        target = "cuda:0" if device == "CUDA" else "cpu"
        # Obtain raw CLIP embeddings once; keep RepRTA and SAVPE, never fuse classes.
        raw_text = model.get_text_pe(manifest["classes"], without_reprta=True).detach().clone().to(target)
        model.to(target)
        wrapper.set_classes(manifest["classes"])
        model.args = get_cfg(overrides={"epochs": epochs, "imgsz": imgsz, "batch": 1, "device": target})
        criterion = E2ELoss(model, v8DetectionLoss) if getattr(model.model[-1], "one2one_cv2", None) is not None else v8DetectionLoss(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=.001)
        train_rows = [row for row in manifest["samples"] if row["split"] == "train"]
        val_rows = [row for row in manifest["samples"] if row["split"] == "val"]
        best, history = math.inf, []
        for epoch in range(epochs):
            random.shuffle(train_rows)
            losses = {"train": [], "val": [], "fit": []}
            passes = (("train", train_rows), ("fit", train_rows)) if demo else (("train", train_rows), ("val", val_rows))
            for split, rows in passes:
                for row in rows:
                    if (output/"cancel.request").exists():
                        raise TrainingCancelled("用户要求停止；旧模型未改动，未完成版本不注册")
                    resource_check(output, device, torch)
                    set_training_mode(model) if split == "train" else model.eval()
                    names = row["classes"] if manifest.get("per_sample_review_scope") else manifest["classes"]
                    batch = make_batch(dataset, row, names, imgsz, torch, target)
                    indices = [manifest["classes"].index(name) for name in names]
                    row_text = raw_text[:, indices] if manifest.get("per_sample_review_scope") else raw_text
                    with torch.set_grad_enabled(split == "train"):
                        predictions = model.predict(batch["img"], tpe=row_text)
                        if manifest.get("per_sample_review_scope"):
                            # The head's nc follows this row's embeddings. Build
                            # the matching loss after forward; unreviewed classes
                            # must not silently become negative/background labels.
                            criterion = E2ELoss(model, v8DetectionLoss) if getattr(model.model[-1], "one2one_cv2", None) is not None else v8DetectionLoss(model)
                        loss, _ = criterion(predictions, batch)
                        loss = loss.sum()
                        if not torch.isfinite(loss):
                            raise RuntimeError("训练损失非有限数，停止")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                    losses[split].append(float(loss.detach().cpu()))
                    del batch, predictions, loss, row_text
            summary = {"epoch": epoch+1, "train_loss": sum(losses["train"])/len(losses["train"]),
                       "val_loss": sum(losses["val"])/len(losses["val"]) if losses["val"] else None}
            if demo:
                summary["training_mode"] = "correction_demo"
                summary["fit_loss"] = sum(losses["fit"])/len(losses["fit"])
            history.append(summary)
            print(json.dumps(summary), flush=True)
            selection_loss = summary["fit_loss"] if demo else summary["val_loss"]
            if selection_loss < best:
                best = selection_loss
                model.eval()
                torch.save({"model": model.cpu(), "train_args": {"task": "segment", "imgsz": imgsz},
                            "epoch": epoch, "version": __import__("ultralytics").__version__}, output/"corrected.pt")
                model.to(target)
            status["history"] = history
            atomic_json(output/"version.json", status)
        if (output/"cancel.request").exists():
            raise TrainingCancelled("用户要求停止")
        if frozen_fingerprint(model) != frozen_before:
            raise RuntimeError("冻结的骨干/文字/视觉/分割模块发生变化，拒绝注册")
        if batch_norm_statistics_fingerprint(model) != bn_before:
            raise RuntimeError("归一化统计量发生漂移，拒绝替换修改版")
        # Release the training model before loading the saved candidate for functional gates.
        del optimizer, criterion, raw_text, model
        wrapper.model = None
        wrapper.ckpt = None
        gc.collect()
        if device == "CUDA":
            torch.cuda.empty_cache()
        candidate = YOLOE(str(output/"corrected.pt"))
        if frozen_fingerprint(candidate.model) != frozen_before or changed_head_fingerprint(candidate.model) == head_before \
                or batch_norm_statistics_fingerprint(candidate.model) != bn_before:
            raise RuntimeError("已保存版本没有有效检测头更新或改变了冻结模块，拒绝注册")
        gates = verify_prompt_capabilities(candidate)
        candidate.model = None
        candidate.ckpt = None
        status.update(gates)
        status.update(status="ready", checkpoint_sha256=sha256_file(output/"corrected.pt"),
                      frozen_modules_unchanged=True, parent_unchanged=sha256_file(parent)==original_sha,
                      detection_heads_changed=True,
                      batch_norm_statistics_unchanged=True,
                      notice="同图拟合演示；保留文字与参考图，按更新后的同图拟合损失选候选，没有独立验证，不能据此证明泛化。" if demo else
                             "保留开放文字/视觉接口的检测头纠错版；验证损失选版本，不代表实际检测精度提升；测试组未使用。")
        if not status["parent_unchanged"]:
            raise RuntimeError("父模型被外部修改；拒绝注册")
        atomic_json(output/"version.json", status)
        return status
    except BaseException as exc:
        status.update(status="cancelled" if isinstance(exc, TrainingCancelled) else "failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output/"version.json", status)
        raise
    finally:
        if lease:
            release_gpu_lease()


def train_modified(dataset, family, store_root, output, device="CPU", epochs=3, imgsz=640):
    """Production path: one stable modified slot; snapshots are only temporary."""
    from modified_models import ModifiedModels
    registry = ModifiedModels(store_root)
    manifest = validate_dataset(dataset)
    if manifest.get("model_family") != family or family not in MODEL_FAMILIES:
        raise ValueError("请为当前基础模型重新导出累计纠错数据，不修改旧数据清单。")
    original = PROJECT/(family + "-seg.pt")
    original_sha = sha256_file(original)
    job = safe_path(registry.root, Path(output).absolute().relative_to(registry.root))
    if not job.is_relative_to(registry.root/".staging"):
        raise ValueError("修改训练必须使用模型目录内的临时工作路径")
    with registry.updating(family):
        parent, previous = registry.parent_for(family)
        try:
            status = train(dataset, parent, job, device, epochs, imgsz,
                           display_name=MODEL_FAMILIES[family] + "修改版")
            if sha256_file(original) != original_sha:
                raise RuntimeError("基础原版权重发生变化，不提交修改版")
            status.update(model_family=family, base_original_sha256=original_sha,
                          base_original_unchanged=True, cumulative_samples=len(manifest["samples"]),
                          trained_revisions=[row["revision"] for row in manifest["samples"] if row["split"] == "train"])
            result = registry.publish(family, job, status, previous)
            atomic_json(job/"version.json", dict(result, status="committed", notice="固定修改版已更新；这里只保留日志，不保留历史模型权重。"))
            return result
        finally:
            registry.cleanup_job_weights(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--family", choices=tuple(MODEL_FAMILIES))
    parser.add_argument("--store", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--ui-events", action="store_true", help="Emit language-neutral failure messages for the desktop UI")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or args.imgsz not in (128, 320, 640):
        parser.error("epoch 必须为 1–100；尺寸为 128（流程检查）/320/640")
    validate_dataset(args.dataset)
    if not args.run:
        print("数据校验通过；未训练。显式 --run 才生成独立纠错型权重。")
        return
    os.chdir(PROJECT)
    if args.family:
        if args.store is None:
            parser.error("--family requires --store")
        train_modified(args.dataset, args.family, args.store, args.output, args.device, args.epochs, args.imgsz)
    else:
        # Legacy diagnostics retain their explicit isolated output API; the UI
        # always uses the fixed-slot path, never this version-producing route.
        train(args.dataset, args.parent, args.output, args.device, args.epochs, args.imgsz)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--ui-events" not in sys.argv:
            raise
        event = message("训练失败：{kind}: {error}", kind=type(exc).__name__, error=exception_message(exc))
        print(json.dumps({"ui_message": event.to_payload()}, ensure_ascii=False), flush=True)
        sys.exit(1)
