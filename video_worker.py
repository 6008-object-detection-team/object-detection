"""Camera/video acquisition and inference worker."""
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)
    error_signal = Signal(str)
    info_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, engine, enhancer, source=0):
        super().__init__()
        self.engine, self.enhancer, self.source = engine, enhancer, source
        self.running = True
        self.conf, self.iou = 0.4, 0.45
        self.class_names, self.visual_template = [], None
        self.use_low_light_enhancement = False
        self.imgsz, self.tiled = 960, False
        self.enhancement_strength, self.supplement_dark = 0.65, True

    def _open_capture(self):
        if isinstance(self.source, int) and sys.platform == "win32":
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(self.source)

    def run(self):
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
                self.engine.imgsz = self.imgsz
                if not preview_sent:
                    self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.status_signal.emit("摄像头已连接，正在预热模型…")
                    preview_sent = True
                output, _, status = self.engine.process_frame(
                    frame, self.conf, self.iou, self.class_names, self.visual_template,
                    enhancer=self.enhancer if self.use_low_light_enhancement else None,
                    enhancement_strength=self.enhancement_strength,
                    supplement_dark=self.supplement_dark, tiled=self.tiled,
                )
                if self.running:
                    self.frame_ready.emit(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                    elapsed = time.perf_counter() - start
                    self.status_signal.emit(f"{status} | 输入 {frame.shape[1]}×{frame.shape[0]} | 处理 {1 / max(elapsed, 0.001):.1f} FPS")
            except Exception as exc:
                self.error_signal.emit(
                    f"模型推理失败，已停止检测：\n{type(exc).__name__}: {exc}\n\n"
                    "显存不足时可降低推理分辨率或选择较小模型；首次文本检测需要下载文本编码器。"
                )
                break
            self.msleep(5)
        capture.release()

    def stop(self):
        self.running = False
        return self.wait(3000)
