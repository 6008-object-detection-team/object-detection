"""Detection-branch-only YOLOE graphs for an app that draws boxes, never masks.

A YOLOE ``-seg`` checkpoint holds a detection model plus a mask branch (mask
coefficients and prototypes). The app only keeps boxes, so the mask branch is
computed and then discarded on every frame. This module builds the matching
detection architecture from Ultralytics' own YAML (``yoloe-26s-seg.yaml`` ->
``yoloe-26s.yaml``) and copies the already-loaded checkpoint into it.

The conversion is accepted only when every tensor of the detection graph is an
exact copy of the checkpoint tensor with the same name, the only tensors left
behind belong to the last layer's mask branch (``cv5``, ``one2one_cv5``,
``proto``), and the text encoder, strides and head settings match. Anything
else raises ``ValueError``; the caller then keeps the full segmentation model
and says so. Nothing is downloaded, cached or written to disk.
"""
from contextlib import contextmanager
from pathlib import Path

from i18n import message

MASK_BRANCH_MODULES = frozenset({"cv5", "one2one_cv5", "proto"})
DEFAULT_TEXT_MODEL = "mobileclip:blt"  # Ultralytics' fallback when a checkpoint lacks the attribute
HEAD_ATTRIBUTES = ("nc", "nl", "reg_max", "end2end", "max_det")


def detection_yaml(seg_model):
    """The detection YAML that matches a loaded ``-seg`` checkpoint."""
    yaml = getattr(seg_model, "yaml", None) or {}
    name = Path(str(yaml.get("yaml_file", ""))).name
    if not name.endswith(".yaml") or "-seg" not in name or name.endswith("-pf.yaml"):
        raise ValueError(message("权重没有可识别的分割结构配置（{name}）", name=name or "?"))
    return name.replace("-seg", "", 1)


def _equal(left, right):
    import torch
    return tuple(left.shape) == tuple(right.shape) and torch.equal(left.to(right.dtype), right)


def verify_detection_branch(seg_model, det_model):
    """Exact-copy and mask-branch-only checks; returns a small manifest record."""
    if getattr(seg_model.model[-1], "lrpc", None) is not None:
        raise ValueError(message("无提示 LRPC 检测头不做仅检测分支转换"))
    source, target = seg_model.state_dict(), det_model.state_dict()
    different = [key for key, value in target.items() if key not in source or not _equal(source[key], value)]
    if different:
        raise ValueError(message("检测分支有 {count} 个张量与原权重不一致，例如 {key}",
                                 count=len(different), key=different[0]))
    head = f"model.{len(seg_model.model) - 1}."
    removed = [key for key in source if key not in target]
    unexpected = [key for key in removed
                  if not key.startswith(head) or key[len(head):].split(".")[0] not in MASK_BRANCH_MODULES]
    if unexpected or not removed:
        raise ValueError(message("去掉的张量不只是掩膜分支，例如 {key}",
                                 key=unexpected[0] if unexpected else message("（没有掩膜分支）")))
    if getattr(seg_model, "text_model", DEFAULT_TEXT_MODEL) != getattr(det_model, "text_model", DEFAULT_TEXT_MODEL):
        raise ValueError(message("检测结构的文字编码器与原权重不同"))
    if not _equal(seg_model.stride.float(), det_model.stride.float()):
        raise ValueError(message("检测结构的步长与原权重不同"))
    seg_head, det_head = seg_model.model[-1], det_model.model[-1]
    for name in HEAD_ATTRIBUTES:
        if getattr(seg_head, name, None) != getattr(det_head, name, None):
            raise ValueError(message("检测头设置 {name} 与原权重不同", name=name))
    return {"tensors_copied_exactly": len(target), "mask_branch_tensors_removed": len(removed),
            "source_tensors": len(source)}


@contextmanager
def one_torch_thread():
    """Run the conversion on one torch thread, then restore the count.

    The conversion only builds a model (a tiny 256x256 stride probe plus a
    weight initialisation that the copy overwrites), copies tensors and
    compares them. On the busy test laptop the YAML build took 3.5-3.8 s with
    PyTorch's default thread count and 0.16-0.17 s on one thread (5
    alternating fresh processes each, one lock session); copying and
    comparing on one thread take tens of milliseconds. Inference threads are
    untouched because the count is restored, also on errors.
    """
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def convert_to_detection_branch(seg_wrapper):
    """Return ``(detection YOLOE wrapper, record)``; raise ValueError if not exact.

    ``seg_wrapper`` is a loaded ``ultralytics.YOLOE`` segmentation model on the
    CPU; it is left usable so the caller can fall back to it.
    """
    from ultralytics import YOLOE

    if seg_wrapper.task != "segment":
        raise ValueError(message("仅检测分支转换只适用于分割权重（当前任务：{task}）", task=str(seg_wrapper.task)))
    with one_torch_thread():
        seg_model = seg_wrapper.model
        yaml_name = detection_yaml(seg_model)
        det_wrapper = YOLOE(yaml_name, verbose=False)
        if det_wrapper.task != "detect":
            raise ValueError(message("匹配 YAML 没有构建检测架构（{name}）", name=yaml_name))
        det_model = det_wrapper.model
        # Prompt state outside the state_dict. Equal names before the copy also keep
        # Ultralytics' class-name row remapping a no-op.
        det_model.names = dict(seg_model.names) if isinstance(seg_model.names, dict) else list(seg_model.names)
        if hasattr(seg_model, "pe"):
            det_model.pe = seg_model.pe.detach().clone()
        det_model.model[-1].nc = seg_model.model[-1].nc
        # Copy the loaded tensors; reading the checkpoint file a second time is not needed.
        det_model.load(seg_model, verbose=False)
        det_model.eval()
        det_model.requires_grad_(any(parameter.requires_grad for parameter in seg_model.parameters()))
        if getattr(seg_model, "pt_path", None):
            det_model.pt_path = seg_model.pt_path
        for key in ("imgsz", "data", "single_cls"):
            if key in seg_wrapper.overrides:
                det_wrapper.overrides[key] = seg_wrapper.overrides[key]
        record = verify_detection_branch(seg_model, det_model)
        record.update(yaml=yaml_name, source_task=seg_wrapper.task)
        return det_wrapper, record
