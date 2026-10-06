"""Bounded camera capture and software display timing; no model ownership."""
from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np


class LatestFrameMailbox:
    """One replaceable item; producers never wait for slow consumers."""

    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self._closed = False
        self.received = self.dropped = 0

    def put(self, item):
        with self._condition:
            if self._closed:
                return False
            notify = self._item is None
            self.received += 1
            if not notify:
                self.dropped += 1
            self._item = item
            self._condition.notify_all()
            return notify

    def take(self, wait=False):
        with self._condition:
            if wait:
                self._condition.wait_for(lambda: self._item is not None or self._closed)
            item, self._item = self._item, None
            return item

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def counters(self):
        with self._condition:
            return self.received, self.dropped


@dataclass
class CapturedFrame:
    image: np.ndarray
    read_at: float


@dataclass
class DisplayFrame:
    image: np.ndarray
    read_at: float
    inference_started: float
    inference_finished: float
    status: str
    captured: int
    capture_dropped: int


class CameraReader(threading.Thread):
    """Only this thread calls read/release after ownership is transferred."""

    def __init__(self, capture, clock=time.perf_counter):
        super().__init__(name="camera-latest-frame", daemon=True)
        self.capture = capture
        self.clock = clock
        self.mailbox = LatestFrameMailbox()
        self.stopping = threading.Event()
        self.error = None

    def run(self):
        try:
            while not self.stopping.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    if not self.stopping.is_set():
                        self.error = "读取摄像头画面失败，视频流已中断。"
                    break
                read_at = self.clock()
                if not self.stopping.is_set():
                    self.mailbox.put(CapturedFrame(frame, read_at))
        except Exception as exc:
            self.error = f"摄像头采集失败：{type(exc).__name__}: {exc}"
        finally:
            try:
                self.capture.release()
            finally:
                self.mailbox.close()

    def stop(self):
        self.stopping.set()
        self.mailbox.close()


class DisplayMetrics:
    """Rolling read-completion -> Qt pixmap-submission timings, not sensor latency."""

    def __init__(self, window=120):
        self._latencies = deque(maxlen=window)
        self._display_times = deque(maxlen=window)

    def record(self, packet, displayed_at):
        age_ms = max(0.0, displayed_at - packet.read_at) * 1000
        self._latencies.append(age_ms)
        self._display_times.append(displayed_at)
        span = self._display_times[-1] - self._display_times[0]
        return {
            "latency_ms": age_ms,
            "p50_ms": float(np.percentile(self._latencies, 50)),
            "p95_ms": float(np.percentile(self._latencies, 95)),
            "display_fps": (len(self._display_times) - 1) / span if span > 0 else 0.0,
            "queue_ms": max(0.0, packet.inference_started - packet.read_at) * 1000,
            "processing_ms": max(0.0, packet.inference_finished - packet.inference_started) * 1000,
            "samples": len(self._latencies),
        }


class CameraRecordingTimeline:
    """Repeat the last processed frame to preserve constant-FPS wall duration."""

    def __init__(self, writer, fps):
        self.writer, self.fps = writer, fps
        self.started = None
        self.previous = None
        self.written = self.repeated = 0

    def write(self, frame, timestamp):
        if self.started is None:
            self.started = timestamp
        slot = int(max(0.0, timestamp - self.started) * self.fps)
        gap = slot - self.written
        # Bound catch-up work when inference or disk stalls for a long time.
        if gap > self.fps * 10:
            raise RuntimeError("摄像头录像超过 10 秒未更新，已停止，避免大量补写旧帧。")
        while self.written < slot:
            self.writer.write(self.previous)
            self.written += 1
            self.repeated += 1
        if self.written == slot:
            self.writer.write(frame)
            self.written += 1
        self.previous = frame
