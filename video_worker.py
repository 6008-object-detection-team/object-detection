"""Camera, video and still-image acquisition with threaded inference."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from tracking import ByteTrackManager
from live_video import (CameraReader, CameraRecordingTimeline, DisplayFrame,
                        LatestFrameMailbox)

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
    live_frame_ready = Signal()
    error_signal = Signal(str)
    info_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, engine, enhancer, source=0, source_kind=None, recording_path=None):
        super().__init__()
        self.engine, self.enhancer, self.source = engine, enhancer, source
        if source_kind is None:
            source_kind = "camera" if isinstance(source, int) else (
                "image" if Path(str(source)).suffix.lower() in IMAGE_SUFFIXES else "video"
            )
        self.source_kind = source_kind
        self.running = True
        self.conf, self.iou, self.max_det = 0.4, 0.45, 300
        self.class_names, self.visual_template = [], None
        self.visual_reference, self.visual_box, self.visual_label = None, None, "object"
        self._visual_prepared = False
        self.use_low_light_enhancement = False
        self.imgsz, self.tiled = 640, False
        self.sliced, self.slice_size, self.slice_overlap, self.max_slices = False, 512, 0.2, 9
        self.enhancement_strength, self.supplement_dark = 0.65, True
        self.tracker = ByteTrackManager()
        self.recording_path = Path(recording_path) if recording_path else None
        self.recorded_frames = 0
        self._camera_reader = None
        self._live_display = LatestFrameMailbox()

    def take_live_frame(self):
        return self._live_display.take()

    def _publish_live_frame(self, packet):
        if self._live_display.put(packet):
            self.live_frame_ready.emit()

    @property
    def display_dropped(self):
        return self._live_display.counters()[1]

    def _open_capture(self):
        if isinstance(self.source, int) and sys.platform == "win32":
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(self.source)

    def _infer(self, frame):
        self.engine.imgsz = self.imgsz
        self._prepare_visual_prompt()
        return self.engine.process_frame(
            frame, self.conf, self.iou, self.class_names, self.visual_template,
            enhancer=self.enhancer if self.use_low_light_enhancement else None,
            enhancement_strength=self.enhancement_strength,
            supplement_dark=self.supplement_dark, tiled=self.tiled,
            sliced=self.sliced, slice_size=self.slice_size,
            slice_overlap=self.slice_overlap, max_slices=self.max_slices,
            tracker=self.tracker, max_det=self.max_det,
        )

    def _prepare_visual_prompt(self):
        """Encode the reference once in this worker, never in the Qt UI thread."""
        if self.visual_reference is None or self._visual_prepared:
            return
        if self.visual_box is None or not hasattr(self.engine, "set_visual_prompt"):
            raise ValueError("视觉提示需要 YOLOE 视觉引擎和有效参考方框。")
        self.status_signal.emit("正在提取参考物体的视觉特征（后台，单模型）…")
        status = self.engine.set_visual_prompt(self.visual_reference, self.visual_box, self.visual_label)
        if status != "SUCCESS":
            raise RuntimeError(f"视觉参考初始化失败：{status}")
        self.visual_template = None
        self.class_names = [self.visual_label]
        self._visual_prepared = True

    def _open_recording(self, output, capture, fps=None):
        path = self.recording_path
        if path.exists():
            raise FileExistsError(f"录像文件已存在，请选择新文件名：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = self._capture_fps(capture) if fps is None else fps
        height, width = output.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise OSError(f"无法创建 MP4 录像：{path}\n请检查目录权限、磁盘空间和 MP4 编码支持。")
        return writer

    @staticmethod
    def _capture_fps(capture):
        fps = capture.get(cv2.CAP_PROP_FPS)
        return float(fps) if np.isfinite(fps) and 0 < fps <= 240 else 20.0

    def _run_camera(self, capture):
        # Capture owns no model. Native read/release always stay on one thread.
        reader = CameraReader(capture)
        self._camera_reader = reader
        writer = timeline = None
        started = False
        try:
            fps = self._capture_fps(capture)
            reader.start()
            started = True
            preview_sent = False
            while self.running:
                packet = reader.mailbox.take(wait=True)
                if packet is None:
                    if self.running and reader.error:
                        self.error_signal.emit(reader.error)
                    break
                if not self.running:
                    break
                frame = packet.image
                if not preview_sent:
                    self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.status_signal.emit("摄像头已连接，正在预热模型（最新帧模式）…")
                    preview_sent = True
                inference_started = time.perf_counter()
                output, _, status = self._infer(frame)
                if not self.running:
                    break
                inference_finished = time.perf_counter()
                if self.recording_path is not None:
                    if writer is None:
                        writer = self._open_recording(output, capture, fps=fps)
                        timeline = CameraRecordingTimeline(writer, fps)
                    timeline.write(output, time.perf_counter())
                    self.recorded_frames = timeline.written
                    status += f" | 录像 {timeline.written} 帧（重复 {timeline.repeated}）"
                captured, dropped = reader.mailbox.counters()
                display = DisplayFrame(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                                       packet.read_at, inference_started, inference_finished,
                                       status, captured, dropped)
                # Emit just one wakeup while a result is pending in the GUI.
                self._publish_live_frame(display)
        finally:
            reader.stop()
            if started:
                # If a driver read is stuck, stop() times out in the UI rather
                # than releasing a live capture or unloading an active worker.
                reader.join()
            else:
                capture.release()
            self._camera_reader = None
            if writer is not None:
                writer.release()
                self.status_signal.emit(f"录像已保存：{self.recording_path}（{self.recorded_frames} 帧，无音轨；重复帧保持时长）")

    def _emit_inference_error(self, exc):
        self.error_signal.emit(
            f"模型推理失败，已停止检测：\n{type(exc).__name__}: {exc}\n\n"
            "建议使用 YOLOE-26S + 640，关闭 SAHI、旧分块和暗光补检后重试。"
            "首次文本检测需要下载文本编码器。"
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
        try:
            self.engine.imgsz = self.imgsz
            self._prepare_visual_prompt()
        except Exception as exc:
            self._emit_inference_error(exc)
            return
        if not self.running:
            return
        if self.source_kind == "image":
            self._run_image()
            return

        capture, writer = None, None
        try:
            capture = self._open_capture()
            if not capture.isOpened():
                self.error_signal.emit(f"无法连接视频源：{self.source}\n请检查摄像头是否被其他程序占用。")
                return
            if self.source_kind == "camera":
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Keep capture buffers modest; SAHI can enlarge local details.
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                camera_capture, capture = capture, None
                self._run_camera(camera_capture)
                return

            preview_sent = False
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    message = "本地视频已播放完毕。" if isinstance(self.source, str) else "读取摄像头画面失败，视频流已中断。"
                    (self.info_signal if isinstance(self.source, str) else self.error_signal).emit(message)
                    break
                start = time.perf_counter()
                if not preview_sent:
                    self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.status_signal.emit("输入源已连接，正在预热模型…")
                    preview_sent = True
                output, _, status = self._infer(frame)
                if self.running:
                    if self.recording_path is not None:
                        if writer is None:
                            writer = self._open_recording(output, capture)
                        writer.write(output)
                        self.recorded_frames += 1
                    self.frame_ready.emit(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                    elapsed = time.perf_counter() - start
                    recording_status = f" | 录像 {self.recorded_frames} 帧" if writer is not None else ""
                    self.status_signal.emit(f"{status} | 输入 {frame.shape[1]}×{frame.shape[0]} | 处理 {1 / max(elapsed, 0.001):.1f} FPS{recording_status}")
        except Exception as exc:
            self.error_signal.emit(f"视频检测/录像失败，已停止：\n{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
                self.status_signal.emit(f"录像已保存：{self.recording_path}（{self.recorded_frames} 帧，无音轨）")

    def stop(self):
        self.running = False
        reader = self._camera_reader
        if reader is not None:
            reader.stop()
        return self.wait(3000)
