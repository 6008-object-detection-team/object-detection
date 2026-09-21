"""Camera, video and still-image acquisition with threaded inference."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from tracking import ByteTrackManager

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def read_image_file(path):
    """Read image paths containing non-ASCII characters reliably on Windows."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, ValueError, cv2.error):
        return None


class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)
    error_signal = Signal(str)
    info_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, engine, enhancer, source=0, source_kind=None):
        super().__init__()
        self.engine, self.enhancer, self.source = engine, enhancer, source
        if source_kind is None:
            source_kind = "camera" if isinstance(source, int) else (
                "image" if Path(str(source)).suffix.lower() in IMAGE_SUFFIXES else "video"
            )
        self.source_kind = source_kind
        self.running = True
        self.conf, self.iou = 0.4, 0.45
        self.class_names, self.visual_template = [], None
        self.use_low_light_enhancement = False
        self.imgsz, self.tiled = 960, False
        self.enhancement_strength, self.supplement_dark = 0.65, True
        self.tracker = ByteTrackManager()

    def _open_capture(self):
        if isinstance(self.source, int) and sys.platform == "win32":
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(self.source)

    def _infer(self, frame):
        self.engine.imgsz = self.imgsz
        return self.engine.process_frame(
            frame, self.conf, self.iou, self.class_names, self.visual_template,
            enhancer=self.enhancer if self.use_low_light_enhancement else None,
            enhancement_strength=self.enhancement_strength,
            supplement_dark=self.supplement_dark, tiled=self.tiled,
            tracker=self.tracker,
        )

    def _emit_inference_error(self, exc):
        self.error_signal.emit(
            f"模型推理失败，已停止检测：\n{type(exc).__name__}: {exc}\n\n"
            "显存不足时可降低推理分辨率或选择较小模型；首次文本检测需要下载文本编码器。"
        )

    def _run_image(self):
        frame = read_image_file(self.source)
        if frame is None:
            self.error_signal.emit(f"无法读取图片：{self.source}\n请检查文件是否损坏或格式是否受支持。")
            return
        try:
            start = time.perf_counter()
            self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.status_signal.emit("图片已读取，正在进行检测…")
            output, _, status = self._infer(frame)
            if self.running:
                elapsed = time.perf_counter() - start
                self.frame_ready.emit(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                self.status_signal.emit(
                    f"{status} | 图片 {Path(self.source).name} | "
                    f"输入 {frame.shape[1]}×{frame.shape[0]} | 用时 {elapsed:.2f} 秒"
                )
        except Exception as exc:
            self._emit_inference_error(exc)

    def run(self):
        if self.source_kind == "image":
            self._run_image()
            return

        capture = self._open_capture()
        if not capture.isOpened():
            self.error_signal.emit(f"无法连接视频源：{self.source}\n请检查摄像头是否被其他程序占用。")
            return
        if isinstance(self.source, int):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # More source pixels help tiny objects; cameras may negotiate a lower mode.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        preview_sent = False
        while self.running:
            ok, frame = capture.read()
            if not ok:
                message = "本地视频已播放完毕。" if isinstance(self.source, str) else "读取摄像头画面失败，视频流已中断。"
                (self.info_signal if isinstance(self.source, str) else self.error_signal).emit(message)
                break
            try:
                start = time.perf_counter()
                if not preview_sent:
                    self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.status_signal.emit("输入源已连接，正在预热模型…")
                    preview_sent = True
                output, _, status = self._infer(frame)
                if self.running:
                    self.frame_ready.emit(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                    elapsed = time.perf_counter() - start
                    self.status_signal.emit(f"{status} | 输入 {frame.shape[1]}×{frame.shape[0]} | 处理 {1 / max(elapsed, 0.001):.1f} FPS")
            except Exception as exc:
                self._emit_inference_error(exc)
                break
            self.msleep(5)
        capture.release()

    def stop(self):
        self.running = False
        return self.wait(3000)
