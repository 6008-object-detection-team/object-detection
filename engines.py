"""Open-vocabulary detection, small-object crops and low-light supplementation."""
import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from detection_utils import Detection, merge_supplementary, tile_regions


class OpenVocabEngine:
    """YOLOE detector driven by text prompts, with unmodified model scores."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.current_classes = []
        self.imgsz = 960

    def load_model(self, model_path, device="CPU"):
        try:
            config_dir = Path(__file__).resolve().parent / ".ultralytics"
            config_dir.mkdir(exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            import torch
            from ultralytics import YOLOE

            if device == "CUDA" and not torch.cuda.is_available():
                return "CUDA_UNAVAILABLE"
            target_device = "cuda:0" if device == "CUDA" else "cpu"
            path = Path(model_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            new_model = YOLOE(str(path))
            new_model.to(target_device)
            self.model, self.device = new_model, target_device
            # A new model must receive prompts even if their text is unchanged.
            self.current_classes = []
            return "SUCCESS"
        except ImportError as exc:
            return f"缺少模型依赖：{exc}。请安装 requirements.txt。"
        except Exception as exc:
            return str(exc)

    @staticmethod
    def calculate_color_similarity(image_a, image_b):
        hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        return max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    def detect(self, frame, conf=0.4, iou=0.45, class_names=None, source="original"):
        """Return boxes in frame coordinates without drawing on the source image."""
        if self.model is None:
            raise RuntimeError("YOLOE 模型尚未加载")
        classes = list(class_names or ["person", "vehicle", "animal", "bag", "bottle", "cup", "phone"])
        if classes != self.current_classes:
            self.model.set_classes(classes)
            self.current_classes = classes
        result = self.model.predict(
            frame, conf=conf, iou=iou, device=self.device, imgsz=self.imgsz,
            end2end=False, verbose=False,
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

    def detect_tiles(self, frame, conf, iou, class_names):
        height, width = frame.shape[:2]
        if min(height, width) < 64:
            return []
        detections = []
        for left, top, right, bottom in tile_regions(width, height):
            crop = frame[top:bottom, left:right]
            for item in self.detect(crop, conf, iou, class_names, source="tile"):
                x1, y1, x2, y2 = item.xyxy
                # Ignore truncation at internal seams, keeping outside edges.
                if ((left > 0 and x1 < 3) or (top > 0 and y1 < 3)
                        or (right < width and x2 > right - left - 3)
                        or (bottom < height and y2 > bottom - top - 3)):
                    continue
                detections.append(replace(item, xyxy=(x1 + left, y1 + top, x2 + left, y2 + top)))
        return detections

    def process_frame(self, frame, conf=0.4, iou=0.45, class_names=None, visual_template=None,
                      enhancer=None, enhancement_strength=0.65, supplement_dark=True, tiled=False):
        """Always detect original exposure, then append optional evidence."""
        detections = self.detect(frame, conf, iou, class_names)
        original_count = len(detections)
        if tiled:
            detections = merge_supplementary(detections, self.detect_tiles(frame, conf, iou, class_names), iou)
        tile_count = len(detections) - original_count
        display, enhanced_applied = frame, False
        if enhancer is not None:
            enhanced = enhancer.process(frame, enhancement_strength)
            display, enhanced_applied = enhanced.image, enhanced.applied
            if enhanced.applied and supplement_dark:
                # Added exposure may hallucinate texture; use a conservative floor.
                extra = self.detect(display, max(conf, 0.45), iou, class_names, source="enhanced")
                detections = merge_supplementary(detections, extra, iou)
        dark_count = len(detections) - original_count - tile_count
        output = self.draw_detections(display, detections, visual_template, reference_frame=frame)
        mode = "暗光补检" if enhanced_applied and supplement_dark else "暗光预览" if enhanced_applied else "原图"
        status = f"{mode} | 原图 {original_count} / 分块新增 {tile_count} / 暗光新增 {dark_count}"
        return output, detections, status

    def draw_detections(self, frame, detections, visual_template=None, reference_frame=None):
        output = frame.copy()
        reference_frame = frame if reference_frame is None else reference_frame
        for item in detections:
            x1, y1, x2, y2 = map(int, item.xyxy)
            color = {"original": (255, 140, 0), "tile": (0, 200, 255), "enhanced": (100, 220, 100)}[item.source]
            suffix = {"original": "", "tile": " +T", "enhanced": " +E"}[item.source]
            text = f"{item.label}: {item.score:.2f}{suffix}"
            if visual_template is not None:
                # Use raw colors before enhancement and before any overlays.
                roi = reference_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (visual_template.shape[1], visual_template.shape[0]))
                similarity = self.calculate_color_similarity(roi, visual_template)
                if similarity <= 0.30:
                    continue
                text = f"{item.label}: {item.score:.2f} match:{similarity:.2f}"
                color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, max(0, y1 - 20)), (min(output.shape[1] - 1, x1 + width), y1), color, -1)
            cv2.putText(output, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return output

    def infer(self, frame, conf=0.4, iou=0.45, class_names=None, visual_template=None):
        return self.process_frame(frame, conf, iou, class_names, visual_template)[0]
