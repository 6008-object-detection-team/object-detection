"""YOLOE SAVPE visual prompts with a single owned checkpoint.

The reference is the complete BGR image plus an object box. Its visual embedding
is computed once, installed in the existing YOLOE model, and reused for video
frames and sequential slices. No colour filtering or text encoding is involved.
"""
import hashlib

import cv2
import numpy as np

from engines import OpenVocabEngine


def prepare_reference(reference_bgr, box_xyxy, imgsz):
    """Letterbox the reference and map its box using Ultralytics' VP geometry.

    Kept separate from the GPU encoder so geometry and input validation can be
    tested without loading a checkpoint. The padding formula is the one used by
    LetterBox and YOLOEVPDetectPredictor._process_single_image.
    """
    if not isinstance(reference_bgr, np.ndarray) or reference_bgr.ndim != 3 \
            or reference_bgr.shape[2] != 3 or reference_bgr.dtype != np.uint8:
        raise ValueError("视觉参考必须是 uint8 BGR 彩色图片")
    height, width = reference_bgr.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("视觉参考图片太小")
    box = np.asarray(box_xyxy, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("视觉提示框必须是四个有限坐标：x1, y1, x2, y2")
    if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
        raise ValueError("视觉提示框必须完整位于参考图内，并且宽高大于零")
    if isinstance(imgsz, bool) or int(imgsz) != imgsz or not 32 <= imgsz <= 960:
        raise ValueError("视觉提示输入尺寸必须是 32 到 960 之间的整数")
    size = int(np.ceil(int(imgsz) / 32) * 32)
    gain = min(size / height, size / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    left = round((size - resized_width) / 2 - 0.1)
    top = round((size - resized_height) / 2 - 0.1)
    right = size - resized_width - left
    bottom = size - resized_height - top
    image = cv2.resize(reference_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    mapped = box.copy() * gain
    mapped[0::2] += left
    mapped[1::2] += top
    return image, mapped[np.newaxis, :]


def prepare_reference_mask(mask, reference_shape, imgsz):
    """Map a human reference mask to the same letterbox, with zero padding."""
    if isinstance(imgsz, bool) or not isinstance(imgsz, (int, np.integer)) or not 32 <= imgsz <= 960:
        raise ValueError("视觉提示输入尺寸必须是 32 到 960 之间的整数")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 \
            or tuple(mask.shape) != tuple(reference_shape[:2]):
        raise ValueError("参考 mask 必须与原始参考图片高宽一致")
    if mask.dtype not in (np.dtype('bool'), np.dtype('uint8')) \
            or not np.isin(mask, (0, 1, 255)).all() or not mask.any():
        raise ValueError("参考 mask 必须为非空二值 bool/uint8 图片")
    height, width = mask.shape
    size = int(np.ceil(int(imgsz) / 32) * 32)
    gain = min(size / height, size / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    left = round((size - resized_width) / 2 - 0.1)
    top = round((size - resized_height) / 2 - 0.1)
    binary = cv2.resize((mask > 0).astype(np.uint8), (resized_width, resized_height),
                        interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(binary, top, size - resized_height - top,
                              left, size - resized_width - left, cv2.BORDER_CONSTANT, value=0)


class VisualPromptEngine(OpenVocabEngine):
    """One reference object encoded by YOLOE's learned visual prompt encoder.

    Call set_visual_prompt in the inference worker after loading/transferring the
    model, never in the GUI thread. A mode switch may transfer the one model
    reference to another engine, setting the previous engine.model to None. Call
    release_visual_prompt before transferring back to text mode; do not unload
    the previous engine during such a transfer, as that releases the GPU lease.
    """

    def __init__(self):
        super().__init__()
        self.visual_label = None
        self.visual_embedding = None  # small CPU copy, not another model
        self.reference_box = None
        self.reference_shape = None
        self._visual_signature = None

    def load_model(self, model_path, device="CPU", graph="seg"):
        """Same graph choice as OpenVocabEngine (``"det"`` = detection branch only), so tools that
        build a visual engine directly can opt in too; the app hands over from a text engine."""
        self.release_visual_prompt()
        return super().load_model(model_path, device, graph=graph)

    def release_visual_prompt(self):
        """Remove visual state before handing the checkpoint back to text mode."""
        was_visual = self.visual_embedding is not None
        self.visual_label = self.visual_embedding = None
        self.reference_box = self.reference_shape = self._visual_signature = None
        self.current_classes = []
        if self.model is not None and was_visual:
            had_predictor = self.model.predictor is not None
            self.model.predictor = None
            underlying = self.model.model
            if hasattr(underlying, "pe"):
                del underlying.pe
            # YOLOE.set_classes skips text encoding when names are unchanged.
            # Clearing names ensures visual "bottle" -> text "bottle" re-encodes.
            underlying.names = {}
            if had_predictor:
                self._release_dropped_predictor()

    def unload(self):
        self.release_visual_prompt()
        return super().unload()

    def _encode_visual_prompt(self, reference_bgr, box_xyxy, reference_mask=None):
        """Encode on the owned model; no Predictor.setup_model/deepcopy."""
        import torch
        from ultralytics.data.augment import LoadVisualPrompt

        image, mapped_box = prepare_reference(reference_bgr, box_xyxy, self.imgsz)
        parameter = next(self.model.model.parameters())
        # Work with the model's actual dtype/device, including FP16 checkpoints.
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(parameter.device)
        image_tensor = image_tensor.permute(0, 3, 1, 2).flip(1).contiguous()
        image_tensor = image_tensor.to(dtype=parameter.dtype).div_(255)
        region = ({"bboxes": mapped_box.copy()} if reference_mask is None else
                  {"masks": prepare_reference_mask(reference_mask, reference_bgr.shape, self.imgsz)[None]})
        visual_tensor = LoadVisualPrompt().get_visuals(
            np.array([0], dtype=np.int32), image.shape[:2], **region,
        ).unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
        self.model.model.eval()
        with torch.inference_mode():
            embedding = self.model.get_visual_pe(image_tensor, visual_tensor).detach()
        # Normal tensors are safe for the predictor's later mutable backend.
        with torch.inference_mode(False):
            return embedding.clone()

    def set_visual_prompt(self, reference_bgr, box_xyxy, label="object", reference_mask=None):
        """Cache/install a real SAVPE embedding; return SUCCESS or raise."""
        from resource_safety import ResourceLimitError, handle_cuda_oom, preflight_inference

        if self.model is None:
            raise RuntimeError("YOLOE 模型尚未加载")
        label = str(label).strip()
        if not label:
            raise ValueError("视觉参考名称不能为空")
        # Validate before hashing or touching CUDA, including complete image/box.
        prepare_reference(reference_bgr, box_xyxy, self.imgsz)
        if reference_mask is not None:
            prepare_reference_mask(reference_mask, reference_bgr.shape, self.imgsz)
        preflight_inference(self.device, self.imgsz, reference_bgr.shape)
        contiguous = np.ascontiguousarray(reference_bgr)
        signature = (
            id(self.model), int(self.imgsz), contiguous.shape,
            tuple(float(value) for value in box_xyxy),
            hashlib.blake2b(memoryview(contiguous), digest_size=16).digest(),
            None if reference_mask is None else hashlib.blake2b(
                memoryview(np.ascontiguousarray(reference_mask > 0)), digest_size=16).digest(),
        )
        if signature == self._visual_signature and label == self.visual_label:
            return "SUCCESS"
        embedding = None
        try:
            if signature == self._visual_signature and self.visual_embedding is not None:
                parameter = next(self.model.model.parameters())
                embedding = self.visual_embedding.to(device=parameter.device, dtype=parameter.dtype)
            else:
                if reference_mask is None:
                    embedding = self._encode_visual_prompt(contiguous, box_xyxy)
                else:
                    embedding = self._encode_visual_prompt(contiguous, box_xyxy, reference_mask)
            # Explicit embeddings bypass get_text_pe even when labels coincide.
            cached_embedding = embedding.detach().cpu().clone()
            had_predictor = self.model.predictor is not None
            self.model.set_classes([label], embedding)
            if had_predictor and self.model.predictor is None:
                # The previous text or visual predictor is now unreferenced.
                self._release_dropped_predictor()
            self.visual_embedding = cached_embedding
            self.visual_label = label
            self.current_classes = [label]
            self.reference_box = tuple(float(value) for value in box_xyxy)
            self.reference_shape = tuple(reference_bgr.shape)
            self._visual_signature = signature
            return "SUCCESS"
        except Exception as exc:
            embedding = None
            if handle_cuda_oom(exc):
                # Stop this job; never retry the same workload on GPU or CPU.
                self.release_visual_prompt()
                raise ResourceLimitError("视觉提示编码已因显存不足停止；请降低输入尺寸或切换 CPU 后重新启动。") from exc
            raise

    def detect(self, frame, conf=0.4, iou=0.45, class_names=None, source="original", max_det=300):
        if self.visual_embedding is None or self.visual_label is None:
            raise RuntimeError("请先框选视觉参考并在工作线程中编码")
        # Keep the installed visual embedding; arbitrary GUI text must never
        # overwrite it. Parent detection retains clipping, caps and GPU guards.
        return super().detect(frame, conf, iou, [self.visual_label], source, max_det)

    def draw_detections(self, frame, detections, visual_template=None, reference_frame=None):
        # A legacy GUI template must not silently filter true SAVPE detections.
        return super().draw_detections(frame, detections, None, reference_frame)
