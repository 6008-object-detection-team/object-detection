"""Open-vocabulary detection, small-object crops and low-light supplementation."""
import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from detection_utils import Detection, box_iou, merge_supplementary, tile_regions
from i18n import message, exception_message
from overlay_labels import draw_label
from resource_safety import (prepare_device, preflight_inference, guarded_inference,
                             release_gpu_lease, handle_cuda_oom)


def validate_detection_options(conf, iou, max_det):
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("置信度和 IoU 必须在 0 到 1 之间")
    if isinstance(max_det, bool) or int(max_det) != max_det or not 1 <= max_det <= 1000:
        raise ValueError("安全模式最大预测框数必须是 1 到 1000 的整数")
    return int(max_det)


class OpenVocabEngine:
    """YOLOE detector driven by text prompts, with unmodified model scores."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.current_classes = []
        self.imgsz = 640
        self._prediction_head_options = {"end2end": False}
        self._precision_uses_quantize = True
        self._inference_options = {
            "agnostic_nms": True, "head_mode": "one2many", "quantize": 32,
        }
        # Set by load_model: which graph serves the checkpoint ("seg" or "det").
        self.model_graph = None

    @property
    def inference_options(self):
        """A copy of the active options, suitable for an experiment manifest."""
        options = dict(self._inference_options)
        if self.model_graph:
            options["graph"] = self.model_graph["graph"]
        return options

    def set_inference_options(self, *, agnostic_nms=None, head_mode=None, quantize=None):
        """Configure a sequential experiment before inference starts.

        Defaults retain YOLOE's cross-class NMS, one-to-many head and FP32.
        ``one2one`` selects the native NMS-free branch, not an extra NMS pass.
        ``quantize=16`` selects native FP16, not integer quantization. Head and
        precision are fixed when the backend is created, so changing either
        releases the old predictor before the next inference builds another.
        Runtime ``agnostic_nms`` changes reuse the backend: they control NMS
        for one2many, or per-anchor class selection for one2one. Apply options
        in the inference worker while it is idle, never concurrently
        with detect(). Existing text/visual prompt embeddings are preserved.
        """
        candidate = dict(self._inference_options)
        if agnostic_nms is not None:
            if not isinstance(agnostic_nms, bool):
                raise ValueError("agnostic_nms 必须是 True 或 False")
            candidate["agnostic_nms"] = agnostic_nms
        if head_mode is not None:
            if head_mode not in ("one2many", "one2one"):
                raise ValueError("检测头必须是 one2many 或 one2one")
            candidate["head_mode"] = head_mode
        if quantize is not None:
            if isinstance(quantize, bool) or not isinstance(quantize, int) or quantize not in (16, 32):
                raise ValueError("原生推理精度必须是整数 16（FP16）或 32（FP32）")
            candidate["quantize"] = quantize
        self._validate_head_available(candidate["head_mode"])
        rebuild = any(candidate[key] != self._inference_options[key]
                      for key in ("head_mode", "quantize"))
        if rebuild:
            self._discard_predictor()
        self._inference_options = candidate
        self._select_head_options()
        return self.inference_options

    def _validate_head_available(self, head_mode=None):
        if self.model is None or (head_mode or self._inference_options["head_mode"]) != "one2one":
            return
        head = self.model.model.model[-1]
        if getattr(head, "one2one_cv2", None) is None:
            # Ultralytics otherwise warns and silently falls back to one2many,
            # which would make a purported one2one comparison misleading.
            raise ValueError("当前模型没有 one2one 检测头；请使用支持该分支的 YOLOE26 权重")

    def _select_head_options(self):
        end2end = self._inference_options["head_mode"] == "one2one"
        if "nms" in self._prediction_head_options:
            self._prediction_head_options = {"nms": False if end2end else None}
        else:
            self._prediction_head_options = {"end2end": end2end}

    def _runtime_prediction_options(self):
        options = {**self._prediction_head_options,
                   "agnostic_nms": self._inference_options["agnostic_nms"]}
        if self._precision_uses_quantize:
            options["quantize"] = self._inference_options["quantize"]
        else:
            options["half"] = self._inference_options["quantize"] == 16
        return options

    def _discard_predictor(self):
        if self.model is None or self.model.predictor is None:
            return
        # Keep the original, unfused model and prompts; backend fusion removes
        # an unused detection branch only from its own deepcopy of that model.
        self.model.predictor = None
        self._release_dropped_predictor()

    def _release_dropped_predictor(self):
        """Free a predictor that was just dropped, before the next one is built.

        Its fused deepcopy of the model holds reference cycles (fused layers
        keep a bound ``forward``), so each prompt or mode change left one more
        copy in VRAM/RAM until Python's cyclic GC happened to run. A full
        collection takes tens of ms with torch loaded; call this only after a
        predictor was actually dropped.
        """
        import gc
        gc.collect()
        if str(self.device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_path, device="CPU", graph="seg"):
        """Load one checkpoint; ``graph="det"`` serves a ``-seg`` checkpoint
        through its detection branch only (exact tensor copy, no mask branch;
        see detection_branch.py). If that conversion is not exact, the full
        segmentation model stays loaded and ``model_graph["fallback"]`` says
        why; nothing is retried. The conversion runs on the CPU before the
        model moves to ``device``.
        """
        try:
            if graph not in ("seg", "det"):
                raise ValueError("计算图必须是 seg（完整分割模型）或 det（仅检测分支）")
            if self.model is not None:
                self.unload()
            target_device = prepare_device(device)
            config_dir = Path(__file__).resolve().parent / ".ultralytics"
            config_dir.mkdir(exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            import time
            import torch
            from ultralytics import YOLOE
            from ultralytics.cfg import DEFAULT_CFG_DICT

            path = Path(model_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            new_model = YOLOE(str(path))
            record = {"requested": graph, "graph": "seg"}
            if graph == "det":
                from detection_branch import convert_to_detection_branch
                started = time.perf_counter()
                try:
                    new_model, record["transfer"] = convert_to_detection_branch(new_model)
                    record["graph"] = "det"
                except MemoryError:
                    raise
                except Exception as exc:
                    # Visible fallback: keep the verified full model and report why.
                    record["fallback"] = exception_message(exc)
                record["conversion_s"] = round(time.perf_counter() - started, 3)
                import gc
                gc.collect()
            new_model.to(target_device)
            self.model, self.device, self.model_graph = new_model, target_device, record
            self._prediction_head_options = (
                {"nms": None} if DEFAULT_CFG_DICT.get("nms", False) is None else {"end2end": False}
            )
            self._precision_uses_quantize = "quantize" in DEFAULT_CFG_DICT
            self._select_head_options()
            self._validate_head_available()
            # A new model must receive prompts even if their text is unchanged.
            self.current_classes = []
            return "SUCCESS"
        except ImportError as exc:
            self.unload()
            return message("缺少模型依赖：{error}。请安装 requirements.txt。", error=exception_message(exc))
        except Exception as exc:
            handle_cuda_oom(exc)
            self.unload()
            return exception_message(exc)

    def unload(self):
        """Release weights before loading another checkpoint, without retrying."""
        import gc
        self.model = None
        self.model_graph = None
        if hasattr(self, "processor"):
            self.processor = None
        self.current_classes = []
        gc.collect()
        # Loading torch only for a previously GPU-owned engine keeps CPU tests light.
        if str(self.device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        release_gpu_lease()

    @staticmethod
    def calculate_color_similarity(image_a, image_b):
        hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        return max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    @guarded_inference
    def detect(self, frame, conf=0.4, iou=0.45, class_names=None, source="original", max_det=300):
        """Return boxes in frame coordinates without drawing on the source image."""
        if self.model is None:
            raise RuntimeError("YOLOE 模型尚未加载")
        max_det = validate_detection_options(conf, iou, max_det)
        preflight_inference(self.device, self.imgsz, frame.shape)
        self._configure_classes(class_names)
        return self._predict_boxes(frame, conf, iou, source, max_det)

    def _configure_classes(self, class_names):
        """Text/visual engines condition this head; prompt-free overrides this."""
        classes = list(class_names or ["person", "vehicle", "animal", "bag", "bottle", "cup", "phone"])
        if classes != self.current_classes:
            had_predictor = getattr(self.model, "predictor", None) is not None
            self.model.set_classes(classes)
            self.current_classes = classes
            if had_predictor and self.model.predictor is None:
                # Free the dropped predictor's model copy before _predict_boxes
                # builds the next one.
                self._release_dropped_predictor()

    def _predict_boxes(self, frame, conf, iou, source, max_det):
        """One shared, guarded caller for box-only predictions in every mode."""
        prediction_options = self._runtime_prediction_options()
        if self.model.task == "segment" and self.model.predictor is None:
            # Keep the checkpoint's segment metadata so NMS does not mistake
            # mask coefficients for classes, but never allocate per-object masks.
            from box_predictor import BoxOnlySegmentationPredictor
            self.model.predictor = BoxOnlySegmentationPredictor(overrides={
                **self.model.overrides, "task": "segment", "mode": "predict",
                "device": self.device, "imgsz": self.imgsz, "batch": 1,
                "verbose": False, **prediction_options,
            }, _callbacks=self.model.callbacks)
            self.model.predictor.setup_model(model=self.model.model, verbose=False)
        result = self.model.predict(
            frame, conf=conf, iou=iou, device=self.device, imgsz=self.imgsz,
            max_det=max_det, verbose=False, **prediction_options,
        )[0]
        height, width = frame.shape[:2]
        detections = []
        # One transfer avoids synchronizing CUDA separately for every box.
        for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().numpy():
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_id)
            detections.append(Detection(
                (float(x1), float(y1), float(x2), float(y2)), float(score), class_id,
                result.names[class_id], source,
            ))
        return detections

    def detect_sliced(self, frame, conf=0.4, iou=0.45, class_names=None, max_det=300,
                      slice_size=512, slice_overlap=0.2, max_slices=9, base_detections=None):
        from sliced_inference import detect_sliced
        return detect_sliced(self, frame, conf, iou, class_names, max_det,
                             slice_size, slice_overlap, max_slices, base_detections)

    def detect_tiles(self, frame, conf, iou, class_names, max_det=300):
        height, width = frame.shape[:2]
        if min(height, width) < 64:
            return []
        detections = []
        for left, top, right, bottom in tile_regions(width, height):
            crop = frame[top:bottom, left:right]
            for item in self.detect(crop, conf, iou, class_names, source="tile", max_det=max_det):
                x1, y1, x2, y2 = item.xyxy
                # Ignore truncation at internal seams, keeping outside edges.
                if ((left > 0 and x1 < 3) or (top > 0 and y1 < 3)
                        or (right < width and x2 > right - left - 3)
                        or (bottom < height and y2 > bottom - top - 3)):
                    continue
                detections.append(replace(item, xyxy=(x1 + left, y1 + top, x2 + left, y2 + top)))
        return detections

    def process_frame(self, frame, conf=0.4, iou=0.45, class_names=None, visual_template=None,
                      enhancer=None, enhancement_strength=0.65, supplement_dark=True, tiled=False,
                      tracker=None, max_det=300, sliced=False, slice_size=512,
                      slice_overlap=0.2, max_slices=9):
        """Always detect original exposure, then append optional evidence."""
        max_det = validate_detection_options(conf, iou, max_det)
        if tiled and sliced:
            raise ValueError("请在四分块与 SAHI 中选择一种补检方式")
        if sliced:
            from sliced_inference import validate_slice_plan
            validate_slice_plan(frame.shape, slice_size, slice_overlap, max_slices)
        detection_floor = tracker.detection_floor(conf) if tracker is not None else conf
        raw_detections = self.detect(frame, detection_floor, iou, class_names, max_det=max_det)
        detections = [item for item in raw_detections if item.score >= conf]
        low_confidence = [item for item in raw_detections if item.score < conf]
        original_count = len(detections)
        if tiled:
            detections = merge_supplementary(detections, self.detect_tiles(frame, conf, iou, class_names, max_det), iou)
        if sliced:
            detections = self.detect_sliced(frame, conf, iou, class_names, max_det,
                                            slice_size, slice_overlap, max_slices, detections)
        tile_count = max(0, len(detections) - original_count)
        display, enhanced_applied = frame, False
        if enhancer is not None:
            enhanced = enhancer.process(frame, enhancement_strength)
            display, enhanced_applied = enhanced.image, enhanced.applied
            if enhanced.applied and supplement_dark:
                # Added exposure may hallucinate texture; use a conservative floor.
                extra = self.detect(display, max(conf, 0.45), iou, class_names, source="enhanced", max_det=max_det)
                detections = merge_supplementary(detections, extra, iou)
        dark_count = max(0, len(detections) - original_count - tile_count)
        detections = sorted(detections, key=lambda item: item.score, reverse=True)[:max_det]
        if tracker is not None:
            # Low-score base detections cannot start a new track, but ByteTrack can use
            # them to bridge a brief confidence drop for an existing object.
            tracking_candidates = merge_supplementary(detections, low_confidence, iou)
            detections = tracker.update(
                tracking_candidates, frame, confidence_threshold=conf, class_names=class_names,
            )
            detections = sorted(detections, key=lambda item: item.score, reverse=True)[:max_det]
        output = self.draw_detections(display, detections, visual_template, reference_frame=frame)
        mode = message("暗光补检" if enhanced_applied and supplement_dark else "暗光预览" if enhanced_applied else "原图")
        tracking_status = message(" / 跟踪显示 {count}", count=len(detections)) if tracker is not None else ""
        slice_label = "SAHI" if sliced else message("分块")
        status = message("{mode} | 原图 {original} / {slice_label}新增 {tiles} / 暗光新增 {dark}{tracking} | 框数上限 {limit}",
                         mode=mode, original=original_count, slice_label=slice_label, tiles=tile_count,
                         dark=dark_count, tracking=tracking_status, limit=max_det)
        return output, detections, status

    def draw_detections(self, frame, detections, visual_template=None, reference_frame=None):
        output = frame.copy()
        reference_frame = frame if reference_frame is None else reference_frame
        labels = []
        for item in detections:
            x1, y1, x2, y2 = map(int, item.xyxy)
            color = {"original": (255, 140, 0), "tile": (0, 200, 255), "slice": (0, 220, 255), "enhanced": (100, 220, 100)}[item.source]
            suffix = {"original": "", "tile": " +T", "slice": " +S", "enhanced": " +E"}[item.source]
            text = f"{item.display_label}: {item.score:.2f}{suffix}"
            if visual_template is not None:
                # Use raw colors before enhancement and before any overlays.
                roi = reference_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (visual_template.shape[1], visual_template.shape[0]))
                similarity = self.calculate_color_similarity(roi, visual_template)
                if similarity <= 0.30:
                    continue
                text = f"{item.display_label}: {item.score:.2f} match:{similarity:.2f}"
                color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            labels.append((text, x1, y1, color))
        # Draw labels last so later rectangle strokes cannot erase their text.
        for text, x, y, color in labels:
            draw_label(output, text, x, y, color)
        return output

    def infer(self, frame, conf=0.4, iou=0.45, class_names=None, visual_template=None, max_det=300):
        return self.process_frame(frame, conf, iou, class_names, visual_template, max_det=max_det)[0]


class PromptFreeEngine(OpenVocabEngine):
    """Official LRPC checkpoint: its full vocabulary is already in the weights.

    Reuses the serial detection, enhancement, slices, tracking and overlay path.
    Never calls set_classes/get_text_pe or constructs a fixed 80-class prompt.
    """

    @property
    def vocabulary_size(self):
        return len(self.model.names) if self.model is not None else 0

    def load_model(self, model_path, device="CPU"):
        # No automatic Ultralytics download/fallback on the inference thread.
        # A manifest-verified local official file must exist before loading.
        if self.model is not None:
            self.unload()
        import json
        from tools.fetch_models import sha256
        root = Path(__file__).resolve().parent
        path = Path(model_path)
        if not path.is_absolute():
            path = root / path
        records = json.loads((root / "model_sources.json").read_text(encoding="utf-8"))["models"]
        record = next((row for row in records if row["file"] == path.name and path.name.endswith("-seg-pf.pt")), None)
        if record is None or not path.is_file():
            return message("无提示权重未就绪：{path}。请运行 python tools/fetch_models.py --prompt-free。", path=str(path))
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            return message("无提示权重校验失败：{path}。未加载或自动替换，请重新下载并校验。", path=str(path))
        status = super().load_model(str(path), device)
        if status != "SUCCESS":
            return status
        head = self.model.model.model[-1]
        if not hasattr(head, "lrpc") or self.vocabulary_size != 4585:
            self.unload()
            return message("权重没有官方 LRPC 无提示检测头或完整词汇表；不使用固定类别替代。")
        return "SUCCESS"

    def _configure_classes(self, class_names):
        if class_names:
            raise ValueError("无提示引擎不接受文字或视觉提示；请切换到对应的原版模型。")
        # Crucially keep the checkpoint's names/head untouched, including IDs >79.

    def process_frame(self, *args, **kwargs):
        image, detections, status = super().process_frame(*args, **kwargs)
        return image, detections, message("无提示（{count} 类） | {status}",
                                          count=self.vocabulary_size, status=status)


class GroundingDINOEngine(OpenVocabEngine):
    """DINO output cap follows thresholding and class-wise NMS.

    max_det does not change the model's learned object-query count.
    Decoded text phrases are retained as open-vocabulary labels.
    """

    def __init__(self):
        super().__init__()
        self.processor = None
        self.text_threshold = 0.25
        self._label_ids = {}

    def load_model(self, model_path="IDEA-Research/grounding-dino-tiny", device="CPU"):
        try:
            if self.model is not None:
                self.unload()
            target_device = prepare_device(device)
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            cache = Path(__file__).resolve().parent / ".cache" / "huggingface" / "hub"
            cache.mkdir(parents=True, exist_ok=True)
            path = Path(model_path)
            if not path.is_absolute() and (Path(__file__).resolve().parent / path).is_dir():
                model_path = str(Path(__file__).resolve().parent / path)
            processor = AutoProcessor.from_pretrained(model_path, cache_dir=str(cache))
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path, cache_dir=str(cache), use_safetensors=True,
                disable_custom_kernels=True,
            ).to(target_device).eval()
            self.model, self.processor, self.device = model, processor, target_device
            self.current_classes, self._label_ids = [], {}
            return "SUCCESS"
        except ImportError as exc:
            self.unload()
            return message("缺少 Grounding DINO 依赖：{error}。请安装 requirements-dino.txt。", error=exception_message(exc))
        except Exception as exc:
            handle_cuda_oom(exc)
            self.unload()
            return message("Grounding DINO 加载失败：{error}", error=exception_message(exc))

    @guarded_inference
    def detect(self, frame, conf=0.4, iou=0.45, class_names=None, source="original", max_det=300):
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("Grounding DINO 模型尚未加载")
        max_det = validate_detection_options(conf, iou, max_det)
        preflight_inference(self.device, self.imgsz, frame.shape)
        classes = [str(item).strip().lower().rstrip(".") for item in (class_names or ["person"])]
        classes = list(dict.fromkeys(item for item in classes if item))
        if not classes:
            raise ValueError("请输入至少一个英文目标提示词")
        if classes != self.current_classes:
            self.current_classes, self._label_ids = classes, {}
        caption = ". ".join(classes) + "."
        if len(self.processor.tokenizer(caption)["input_ids"]) > self.model.config.max_text_len:
            raise ValueError("提示词超过 Grounding DINO 文本长度限制，请减少目标数量")
        height, width = frame.shape[:2]
        inputs = self.processor(
            images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), text=caption,
            size={"shortest_edge": int(self.imgsz), "longest_edge": round(self.imgsz * 1333 / 800)},
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=conf, text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            label = str(label).strip() or "object"
            if label not in self._label_ids:
                self._label_ids[label] = len(self._label_ids)
            x1, y1, x2, y2 = box
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 > x1 and y2 > y1:
                detections.append(Detection(
                    (float(x1), float(y1), float(x2), float(y2)), float(score),
                    self._label_ids[label], label, source,
                ))
        selected = []
        for item in sorted(detections, key=lambda detection: detection.score, reverse=True):
            if not any(previous.class_id == item.class_id and box_iou(previous.xyxy, item.xyxy) > iou
                       for previous in selected):
                selected.append(item)
                if len(selected) == max_det:
                    break
        return selected


class GroundingDinoEngine(GroundingDINOEngine):
    """Preserve the original per-class token scoring API for pan_deng evaluation.

    The upgraded GUI uses GroundingDINOEngine's phrase-decoding backend. Both
    share loading, device/resource guards, and the single-model allocation cap.
    """

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    @staticmethod
    def _build_prompt(classes):
        """Join prompts as "a. b." and record each class name's character span."""
        spans, cursor = [], 0
        for name in classes:
            spans.append((cursor, cursor + len(name)))
            cursor += len(name) + 2  # the ". " separator
        return ". ".join(classes) + ".", spans

    @staticmethod
    def _class_token_groups(offsets, spans):
        """Token positions inside each class name, from the tokenizer's character offsets."""
        return [
            [index for index, (start, end) in enumerate(offsets)
             if end > start and start >= span_start and end <= span_end]
            for span_start, span_end in spans
        ]

    @guarded_inference
    def detect(self, frame, conf=0.4, iou=0.45, class_names=None, source="original", max_det=300):
        """Detect text-prompted objects and return the common Detection format."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Grounding DINO 模型尚未加载")
        max_det = validate_detection_options(conf, iou, max_det)
        preflight_inference(self.device, self.imgsz, frame.shape)
        default_classes = ["person", "vehicle", "animal", "bag", "bottle", "cup", "phone"]
        classes = [name.strip() for name in (class_names or default_classes) if name.strip()]
        if not classes:
            return []

        import torch
        from torchvision.ops import batched_nms, box_convert

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prompt, spans = self._build_prompt(classes)
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        offsets = self.processor.tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
        token_groups = self._class_token_groups(offsets, spans)
        height, width = frame.shape[:2]
        with torch.inference_mode():
            outputs = self.model(**inputs)
            # Score each query against each class by that class's own prompt tokens.
            # The processor's text_threshold phrase decoding can merge several class
            # names into one label ("person bicycle car") at low thresholds, which
            # then maps to the wrong class; this keeps the label independent of conf.
            token_probs = outputs.logits[0].sigmoid()
            class_scores = torch.zeros((token_probs.shape[0], len(classes)), device=token_probs.device)
            for class_id, tokens in enumerate(token_groups):
                if tokens:
                    class_scores[:, class_id] = token_probs[:, tokens].max(dim=1).values
            scores, class_ids = class_scores.max(dim=1)
            above = scores > conf
            boxes = box_convert(outputs.pred_boxes[0][above], "cxcywh", "xyxy")
            boxes = boxes * torch.tensor([width, height, width, height], device=boxes.device)

        boxes = boxes.float().cpu().numpy()
        scores = scores[above].float().cpu().numpy()
        class_ids = class_ids[above].cpu().numpy()
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id = int(class_id)
            text_label = classes[class_id]
            x1, y1, x2, y2 = box.astype(float)
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(
                (float(x1), float(y1), float(x2), float(y2)), float(score), class_id,
                text_label, source,
            ))
        self.current_classes = classes
        if len(detections) < 2:
            return detections
        # Grounding DINO returns filtered queries but does not perform NMS. Match the
        # application's YOLOE semantics and make the existing IoU control effective.
        nms_boxes = torch.tensor([item.xyxy for item in detections], dtype=torch.float32)
        nms_scores = torch.tensor([item.score for item in detections], dtype=torch.float32)
        nms_classes = torch.tensor([item.class_id for item in detections])
        keep = batched_nms(nms_boxes, nms_scores, nms_classes, iou).tolist()
        return [detections[index] for index in keep[:max_det]]
