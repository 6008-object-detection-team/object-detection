"""Model and image-enhancement engines."""
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OpenVocabEngine:
    """YOLOE-11 open-vocabulary detector driven by text prompts."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.current_classes = []

    def load_model(self, model_path, device="CPU"):
        try:
            # Keep Ultralytics settings inside this project. This avoids failures
            # on Windows profiles where AppData/Roaming is not writable.
            config_dir = Path(__file__).resolve().parent / ".ultralytics"
            config_dir.mkdir(exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            import torch
            from ultralytics import YOLOE

            if device == "CUDA" and not torch.cuda.is_available():
                return "CUDA_UNAVAILABLE"
            self.device = "cuda:0" if device == "CUDA" else "cpu"
            self.model = YOLOE(model_path)  # Downloads the official checkpoint once if needed.
            self.model.to(self.device)
            return "SUCCESS"
        except ImportError:
            return "NO_ULTRALYTICS"
        except Exception as exc:
            return str(exc)

    @staticmethod
    def calculate_color_similarity(image_a, image_b):
        try:
            hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
            hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
            hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
            return max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        except Exception:
            return 0.0

    def infer(self, frame, conf=0.4, iou=0.45, class_names=None, visual_template=None):
        if self.model is None:
            raise RuntimeError("YOLOE-11 模型尚未加载")

        classes = class_names or ["person", "vehicle", "animal", "bag", "bottle", "cup", "phone"]
        if classes != self.current_classes:
            self.model.set_classes(classes)
            self.current_classes = classes

        result = self.model.predict(frame, conf=conf, iou=iou, device=self.device, verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id] if class_id < len(classes) else "unknown"
            color = (255, 0, 0)

            if visual_template is not None:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (visual_template.shape[1], visual_template.shape[0]))
                score = self.calculate_color_similarity(roi, visual_template)
                if score <= 0.30:
                    continue
                label, color = "Visual Match", (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {score:.2f}"
            (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + width, y1), color, -1)
            cv2.putText(frame, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame


class ZeroDCEEngine:
    """Optional ONNX low-light enhancer with a CLAHE fallback."""

    def __init__(self):
        self.session = None
        self.input_name = None

    def load_model(self, model_path="zero_dce.onnx", device="CPU"):
        if not os.path.exists(model_path):
            return "NO_MODEL"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "CUDA" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            return "SUCCESS"
        except Exception as exc:
            self.session = None
            return str(exc)

    def enhance(self, image):
        if self.session is None:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        tensor = np.expand_dims(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0)
        curve = self.session.run(None, {self.input_name: tensor})[0]
        for index in range(8):
            coefficient = curve[:, index * 3:(index + 1) * 3]
            tensor = tensor + coefficient * (tensor - tensor ** 2)
        return np.clip(np.transpose(tensor[0], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
