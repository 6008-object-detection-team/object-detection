"""Camera, video and still-image acquisition with threaded inference."""
import sys
import time
import threading
import uuid
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from tracking import ByteTrackManager
from i18n import message, exception_message
from live_video import (CameraReader, CameraRecordingTimeline, DisplayFrame,
                        LatestFrameMailbox)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
# "default" keeps the historical requests; "mjpg" is opt-in (see configure_camera).
CAMERA_FORMATS = ("default", "mjpg")
MJPG_DRIVER_BUFFERS = 4


def fourcc_text(value):
    """Four printable characters from a CAP_PROP_FOURCC value, else None."""
    try:
        code = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    text = "".join(chr((code >> (8 * shift)) & 0xFF) for shift in range(4))
    return text if code > 0 and text.isascii() and text.isprintable() else None


def configure_camera(capture, camera_format="default"):
    """Apply the camera requests and read back what the driver negotiated.

    ``default`` sends the app's original requests unchanged: one driver
    buffer, 1280x720 and the driver's own pixel format. ``mjpg`` requests MJPG
    before the frame size (V4L2 and DirectShow negotiate sizes per format) and
    keeps a 4-buffer driver queue: with a single buffer the measured webcam
    delivered only every second MJPG frame. The latest-frame reader already
    discards stale frames, so extra driver buffers do not queue old frames.
    If the driver refuses MJPG, the default requests apply and the result
    says so; nothing is retried or reopened.
    """
    if camera_format not in CAMERA_FORMATS:
        raise ValueError(message("未知摄像头格式：{value}", value=str(camera_format)))
    mjpg = False
    if camera_format == "mjpg":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        mjpg = fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)) == "MJPG"
    capture.set(cv2.CAP_PROP_BUFFERSIZE, MJPG_DRIVER_BUFFERS if mjpg else 1)
    # Keep capture buffers modest; SAHI can enlarge local details.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def number(prop):
        try:
            value = float(capture.get(prop))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    fourcc = fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
    return {"requested": camera_format, "fourcc": fourcc, "width": number(cv2.CAP_PROP_FRAME_WIDTH),
            "height": number(cv2.CAP_PROP_FRAME_HEIGHT), "fps": number(cv2.CAP_PROP_FPS),
            "buffers": number(cv2.CAP_PROP_BUFFERSIZE),
            "mjpg_refused": camera_format == "mjpg" and fourcc != "MJPG"}


def camera_status(info):
    """Status-bar text for the negotiated camera format (values are data)."""
    def shown(value):
        return "?" if value is None else f"{value:.0f}"

    text = message("摄像头 {fourcc} {width}×{height} {fps} FPS，缓冲 {buffers}",
                   fourcc=info.get("fourcc") or "?", width=shown(info.get("width")),
                   height=shown(info.get("height")), fps=shown(info.get("fps")),
                   buffers=shown(info.get("buffers")))
    if info.get("mjpg_refused"):
        return message("{camera}（摄像头未接受 MJPG，已按默认设置运行）", camera=text)
    return text


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
    # Retain message templates across the thread boundary until UI rendering.
    error_signal = Signal(object)
    info_signal = Signal(object)
    status_signal = Signal(object)

    def __init__(self, engine, enhancer, source=0, source_kind=None, recording_path=None,
                 camera_format="default"):
        super().__init__()
        if camera_format not in CAMERA_FORMATS:
            raise ValueError(message("未知摄像头格式：{value}", value=str(camera_format)))
        self.camera_format, self.camera_info = camera_format, None
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
        self._feedback_lock = threading.Lock()
        self._feedback_snapshot = None
        self.feedback_metadata = {}
        self._feedback_session = uuid.uuid4().hex
        self._feedback_frame_index = 0

    def get_feedback_snapshot(self):
        """Copy a single original frame on demand; never enqueue raw video frames."""
        with self._feedback_lock:
            if self._feedback_snapshot is None:
                return None
            frame, boxes, metadata = self._feedback_snapshot
            return frame.copy(), [dict(box, xyxy=list(box["xyxy"])) for box in boxes], deepcopy(metadata)

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
        result = self.engine.process_frame(
            frame, self.conf, self.iou, self.class_names, self.visual_template,
            enhancer=self.enhancer if self.use_low_light_enhancement else None,
            enhancement_strength=self.enhancement_strength,
            supplement_dark=self.supplement_dark, tiled=self.tiled,
            sliced=self.sliced, slice_size=self.slice_size,
            slice_overlap=self.slice_overlap, max_slices=self.max_slices,
            tracker=self.tracker, max_det=self.max_det,
        )
        from feedback import prediction_boxes, source_group
        self._feedback_frame_index += 1
        boxes = prediction_boxes(result[1], frame.shape[1], frame.shape[0])
        metadata = dict(self.feedback_metadata, source=str(self.source), source_kind=self.source_kind,
                        origin_group=source_group(self.source_kind, self.source, self._feedback_session),
                        frame_index=self._feedback_frame_index, captured_at=time.time(),
                        prompts=list(self.class_names), visual_prompt=self.visual_reference is not None,
                        conf=self.conf, iou=self.iou, max_det=self.max_det, imgsz=self.imgsz,
                        sliced=self.sliced, tiled=self.tiled, low_light=self.use_low_light_enhancement)
        # OpenVocabEngine draws on a copy. Keep one owned raw-frame reference;
        # copy only when the human explicitly opens the correction editor.
        with self._feedback_lock:
            self._feedback_snapshot = (frame, boxes, metadata)
        return result

    def _prepare_visual_prompt(self):
        """Encode the reference once in this worker, never in the Qt UI thread."""
        if self.visual_reference is None or self._visual_prepared:
            return
        if self.visual_box is None or not hasattr(self.engine, "set_visual_prompt"):
            raise ValueError("视觉提示需要 YOLOE 视觉引擎和有效参考方框。")
        self.status_signal.emit(message("正在提取参考物体的视觉特征（后台，单模型）…"))
        status = self.engine.set_visual_prompt(self.visual_reference, self.visual_box, self.visual_label)
        if status != "SUCCESS":
            raise RuntimeError(message("视觉参考初始化失败：{status}", status=status))
        self.visual_template = None
        self.class_names = [self.visual_label]
        self._visual_prepared = True

    def _open_recording(self, output, capture, fps=None):
        path = self.recording_path
        if path.exists():
            raise FileExistsError(message("录像文件已存在，请选择新文件名：{path}", path=str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = self._capture_fps(capture) if fps is None else fps
        height, width = output.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise OSError(message("无法创建 MP4 录像：{path}\n请检查目录权限、磁盘空间和 MP4 编码支持。", path=str(path)))
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
        # Read back once before streaming; shown with every result.
        camera_text = camera_status(self.camera_info) if self.camera_info else None
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
                    connected = message("摄像头已连接，正在预热模型（最新帧模式）…")
                    self.status_signal.emit(connected if camera_text is None else
                                            message("{status} | {camera}", status=connected, camera=camera_text))
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
                    status = message("{status} | 录像 {frames} 帧（重复 {repeated}）",
                                     status=status, frames=timeline.written, repeated=timeline.repeated)
                if camera_text is not None:
                    status = message("{status} | {camera}", status=status, camera=camera_text)
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
                self.status_signal.emit(message("录像已保存：{path}（{frames} 帧，无音轨；重复帧保持时长）",
                                                path=str(self.recording_path), frames=self.recorded_frames))

    def _emit_inference_error(self, exc):
        self.error_signal.emit(message(
            "模型推理失败，已停止检测：\n{kind}: {error}\n\n建议使用 YOLOE-26S + 640，关闭 SAHI、旧分块和暗光补检后重试。首次文本检测需要下载文本编码器。",
            kind=type(exc).__name__, error=exception_message(exc)))

    def _run_image(self):
        frame = read_image_file(self.source)
        if frame is None:
            self.error_signal.emit(message("无法读取图片：{path}\n请检查文件是否损坏或格式是否受支持。", path=str(self.source)))
            return
        try:
            start = time.perf_counter()
            self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.status_signal.emit(message("图片已读取，正在进行检测…"))
            output, _, status = self._infer(frame)
            if self.running:
                elapsed = time.perf_counter() - start
                self.frame_ready.emit(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                self.status_signal.emit(message("{status} | 图片 {name} | 输入 {width}×{height} | 用时 {elapsed:.2f} 秒",
                    status=status, name=Path(self.source).name, width=frame.shape[1], height=frame.shape[0], elapsed=elapsed))
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
                self.error_signal.emit(message("无法连接视频源：{source}\n请检查摄像头是否被其他程序占用。", source=self.source))
                return
            if self.source_kind == "camera":
                self.camera_info = configure_camera(capture, self.camera_format)
                camera_capture, capture = capture, None
                self._run_camera(camera_capture)
                return

            preview_sent = False
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    value = message("本地视频已播放完毕。" if isinstance(self.source, str) else "读取摄像头画面失败，视频流已中断。")
                    (self.info_signal if isinstance(self.source, str) else self.error_signal).emit(value)
                    break
                start = time.perf_counter()
                if not preview_sent:
                    self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.status_signal.emit(message("输入源已连接，正在预热模型…"))
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
                    recording_status = message(" | 录像 {frames} 帧", frames=self.recorded_frames) if writer is not None else ""
                    self.status_signal.emit(message("{status} | 输入 {width}×{height} | 处理 {fps:.1f} FPS{recording}",
                        status=status, width=frame.shape[1], height=frame.shape[0], fps=1/max(elapsed, .001), recording=recording_status))
        except Exception as exc:
            self.error_signal.emit(message("视频检测/录像失败，已停止：\n{kind}: {error}", kind=type(exc).__name__, error=exception_message(exc)))
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
                self.status_signal.emit(message("录像已保存：{path}（{frames} 帧，无音轨）", path=str(self.recording_path), frames=self.recorded_frames))

    def stop(self):
        self.running = False
        reader = self._camera_reader
        if reader is not None:
            reader.stop()
        return self.wait(3000)
