"""YOLOE ONNX graph with a text vocabulary fixed at export, run by ONNX Runtime.

Experimental tool backend: the GUI detection path does not use it. Neither
importing this module nor inference imports torch. Pre/post-processing follows
Ultralytics 8.4.172 for a static square input (letterbox, xywh -> xyxy, best
class, greedy NMS, scale back, clip) in NumPy. Only CPUExecutionProvider is
supported here; another provider is refused instead of silently using CPU.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from detection_utils import Detection
from engines import OpenVocabEngine, validate_detection_options
from i18n import message, exception_message
from resource_safety import prepare_device, preflight_inference, release_gpu_lease

MANIFEST_SCHEMA = 1
MANIFEST_SUFFIX = ".manifest.json"
OUTPUT_FORMAT = "raw_one2many_xywh"
MAX_NMS = 30000  # Ultralytics non_max_suppression defaults.
MAX_WH = 7680
PADDING_VALUE = 114


def manifest_path(onnx_path):
    return Path(onnx_path).with_suffix(MANIFEST_SUFFIX)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    """Reject a manifest that cannot describe a fixed-vocabulary raw output."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("ONNX manifest 版本不受支持")
    classes = manifest.get("classes")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes) \
            or not all(isinstance(name, str) and name.strip() for name in classes):
        raise ValueError("ONNX manifest 的类别必须是非空且不重复的名称列表")
    imgsz = manifest.get("export", {}).get("imgsz")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or not 32 <= imgsz <= 960 or imgsz % 32:
        raise ValueError("ONNX manifest 的输入尺寸必须是 32–960 之间 32 的倍数")
    layout = manifest.get("output_layout", {})
    if layout.get("format") != OUTPUT_FORMAT or layout.get("num_classes") != len(classes) \
            or not isinstance(layout.get("extra_channels"), int) or layout["extra_channels"] < 0:
        raise ValueError("ONNX manifest 的输出布局与固定类别不一致")
    return manifest


def read_manifest(onnx_path, verify_hash=True):
    path = Path(onnx_path)
    record = manifest_path(path)
    if not path.is_file() or not record.is_file():
        raise ValueError(message("缺少 ONNX 文件或同名 manifest：{path}", path=str(path)))
    manifest = validate_manifest(json.loads(record.read_text(encoding="utf-8")))
    if manifest.get("file") != path.name:
        raise ValueError("manifest 记录的文件名与 ONNX 文件不一致")
    if verify_hash and sha256_file(path) != manifest.get("onnx_sha256"):
        raise ValueError("ONNX 文件哈希与 manifest 不一致；拒绝加载")
    return manifest


def letterbox(frame, size, padding_value=PADDING_VALUE):
    """Ultralytics LetterBox(auto=False, center=True, scaleup=True) for a square input."""
    height, width = frame.shape[:2]
    ratio = min(size / height, size / width)
    unpadded = round(width * ratio), round(height * ratio)
    dw, dh = (size - unpadded[0]) / 2, (size - unpadded[1]) / 2
    image = frame if (width, height) == unpadded else cv2.resize(frame, unpadded, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding_value,) * 3)


def to_blob(image):
    """BGR HWC uint8 -> RGB NCHW float32 in 0–1, as the Ultralytics predictor."""
    rgb = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
    return (rgb.astype(np.float32) / np.float32(255))[None]


def xywh2xyxy(boxes):
    result = np.empty_like(boxes)
    half = boxes[:, 2:] / 2
    result[:, :2] = boxes[:, :2] - half
    result[:, 2:] = boxes[:, :2] + half
    return result


def greedy_nms(boxes, scores, iou_threshold):
    """torchvision.ops.nms semantics: drop boxes whose IoU is above the threshold."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        index, rest = order[0], order[1:]
        keep.append(index)
        width = np.clip(np.minimum(boxes[index, 2], boxes[rest, 2]) - np.maximum(boxes[index, 0], boxes[rest, 0]), 0, None)
        height = np.clip(np.minimum(boxes[index, 3], boxes[rest, 3]) - np.maximum(boxes[index, 1], boxes[rest, 1]), 0, None)
        intersection = width * height
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap = intersection / (areas[index] + areas[rest] - intersection)
        # A NaN overlap is not "> threshold" in torchvision either; keep it.
        order = rest[~(overlap > iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def decode_predictions(output, num_classes, conf, iou, max_det, agnostic=True):
    """One image (4 + nc + extra, anchors) -> (n, 6) xyxy/score/class in input pixels."""
    output = np.asarray(output, dtype=np.float32)
    if output.ndim != 2 or output.shape[0] < 4 + num_classes:
        raise ValueError("ONNX 输出形状与固定类别数不一致")
    class_scores = output[4:4 + num_classes]
    candidates = class_scores.max(0) > conf
    if not candidates.any():
        return np.zeros((0, 6), np.float32)
    rows = output[:, candidates].T
    class_ids = rows[:, 4:4 + num_classes].argmax(1)
    scores = rows[np.arange(rows.shape[0]), 4 + class_ids]
    boxes = xywh2xyxy(rows[:, :4])
    if rows.shape[0] > MAX_NMS:
        top = np.argsort(-scores, kind="stable")[:MAX_NMS]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]
    offsets = 0 if agnostic else class_ids[:, None].astype(np.float32) * MAX_WH
    keep = greedy_nms(boxes + offsets, scores, iou)[:max_det]
    return np.concatenate([boxes[keep], scores[keep, None], class_ids[keep, None].astype(np.float32)], 1)


def scale_boxes(boxes, input_size, frame_shape):
    """Ultralytics scale_boxes + clip from the square letterbox to the frame."""
    height, width = frame_shape[:2]
    gain = min(input_size / height, input_size / width)
    pad_x = round((input_size - round(width * gain)) / 2 - 0.1)
    pad_y = round((input_size - round(height * gain)) / 2 - 0.1)
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / gain, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / gain, 0, height)
    return boxes


class OnnxRuntimeEngine(OpenVocabEngine):
    """Fixed-vocabulary YOLOE ONNX detector; reuses the shared frame pipeline.

    Classes, input size and output layout come from the export manifest; the
    file hash is checked before loading. Text or visual prompts cannot change
    the exported vocabulary, so other class lists are rejected.
    """

    def __init__(self):
        super().__init__()
        self.session = None
        self.manifest = None
        self.providers = []
        self.last_timing = {}
        self._input_name = self._output_name = None

    def set_inference_options(self, *, agnostic_nms=None, head_mode=None, quantize=None):
        if head_mode not in (None, "one2many"):
            raise ValueError("ONNX 导出图只包含 one2many 检测头")
        if quantize not in (None, 32) or isinstance(quantize, bool):
            raise ValueError("ONNX 导出图固定为 FP32")
        if agnostic_nms is not None:
            if not isinstance(agnostic_nms, bool):
                raise ValueError("agnostic_nms 必须是 True 或 False")
            self._inference_options["agnostic_nms"] = agnostic_nms
        return self.inference_options

    def load_model(self, model_path, device="CPU", threads=None, verify_hash=True):
        try:
            if self.session is not None:
                self.unload()
            if str(device).upper() != "CPU":
                # onnxruntime-gpu is not part of the verified environment.
                return message("ONNX Runtime 工具后端目前只验证 CPUExecutionProvider；不会自动改用 CPU。")
            self.device = prepare_device("CPU")
            path = Path(model_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            manifest = read_manifest(path, verify_hash)
            import onnxruntime as ort
            options = ort.SessionOptions()
            if threads is not None:
                if isinstance(threads, bool) or int(threads) != threads or threads < 1:
                    raise ValueError("线程数必须是正整数")
                options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.log_severity_level = 3
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            if session.get_providers() != ["CPUExecutionProvider"]:
                raise RuntimeError("ONNX Runtime 没有启用所请求的 CPUExecutionProvider")
            inputs, outputs = session.get_inputs(), session.get_outputs()
            size = manifest["export"]["imgsz"]
            channels = 4 + len(manifest["classes"]) + manifest["output_layout"]["extra_channels"]
            if len(inputs) != 1 or list(inputs[0].shape) != [1, 3, size, size] \
                    or not outputs or len(outputs[0].shape) != 3 or outputs[0].shape[1] != channels:
                raise ValueError("ONNX 输入/输出形状与 manifest 不一致")
            self.session, self.manifest, self.providers = session, manifest, session.get_providers()
            self._input_name, self._output_name = inputs[0].name, outputs[0].name
            self.imgsz = size
            self.current_classes = list(manifest["classes"])
            self.model = path  # Marks the engine loaded for the shared pipeline.
            return "SUCCESS"
        except ImportError as exc:
            self.unload()
            return message("缺少 ONNX Runtime：{error}。请安装 requirements.txt。", error=exception_message(exc))
        except Exception as exc:
            self.unload()
            return exception_message(exc)

    def unload(self):
        self.session = self.manifest = self.model = None
        self.providers, self.current_classes, self.last_timing = [], [], {}
        release_gpu_lease()

    def detect(self, frame, conf=0.4, iou=0.45, class_names=None, source="original", max_det=300):
        if self.session is None:
            raise RuntimeError("ONNX 模型尚未加载")
        max_det = validate_detection_options(conf, iou, max_det)
        preflight_inference(self.device, self.imgsz, frame.shape)
        self._configure_classes(class_names)
        return self._predict_boxes(frame, conf, iou, source, max_det)

    def _configure_classes(self, class_names):
        if class_names is not None and list(class_names) != self.current_classes:
            raise ValueError(message("ONNX 模型的类别在导出时已固定为 {fixed}；运行时不能改为 {requested}，请重新导出。",
                                     fixed=", ".join(self.current_classes), requested=", ".join(map(str, class_names))))

    def _predict_boxes(self, frame, conf, iou, source, max_det):
        clock = time.perf_counter
        start = clock()
        blob = to_blob(letterbox(frame, self.imgsz))
        prepared = clock()
        output = self.session.run([self._output_name], {self._input_name: blob})[0]
        inferred = clock()
        rows = decode_predictions(output[0], len(self.current_classes), conf, iou, max_det,
                                  agnostic=self._inference_options["agnostic_nms"])
        rows[:, :4] = scale_boxes(rows[:, :4], self.imgsz, frame.shape)
        detections = []
        for x1, y1, x2, y2, score, class_id in rows:
            if x2 <= x1 or y2 <= y1 or not math.isfinite(score):
                continue
            class_id = int(class_id)
            detections.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(score),
                                        class_id, self.current_classes[class_id], source))
        finished = clock()
        self.last_timing = {"preprocess_ms": (prepared - start) * 1000, "inference_ms": (inferred - prepared) * 1000,
                            "postprocess_ms": (finished - inferred) * 1000}
        return detections
