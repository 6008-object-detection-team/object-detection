"""Headless realtime benchmark of the camera scheduling pipeline.

The application's own camera path runs without a window: VideoWorker.run()
with its CameraReader, LatestFrameMailbox, engine.process_frame (ByteTrack,
drawing), BGR->RGB conversion, display mailbox and DisplayMetrics. A headless
consumer stands in for the Qt slot. Two schedules exist only in this tool and
never change the application's default:

* latest - the unchanged 2026-10-06 path: one replaceable pending frame.
* fifo   - the same CameraReader feeding a FIFO queue, so captured frames are
           processed in order (optionally bounded; a full queue drops the
           incoming frame, like a driver without a free buffer).
* sync   - a port of the pre-2026-10-06 loop: read, infer and publish on one
           thread, then wait 5 ms; frames wait in the (real or modelled) driver
           queue instead of a mailbox.

Sources are a paced replay of an image directory or video file that behaves
like a camera (frame k is captured at t0 + k/fps whether or not anyone reads
it, and a V4L2-like driver queue keeps the oldest frames and drops the rest),
or a real camera. Camera frames are only timed: they are never written,
displayed or uploaded, and each camera run is capped at 60 seconds.

Importing this module and --help do not import torch, PySide6 or a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmark_detection import class_names, percentile, positive_integer, probability, sha256_file
from live_video import CameraReader, DisplayFrame, DisplayMetrics

MODES = ("latest", "fifo", "sync")
SOURCES = ("video", "camera")
CAMERA_MAX_SECONDS = 60.0
PRE_REWORK_SLEEP_S = 0.005
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CODE_FILES = ("tools/benchmark_realtime.py", "live_video.py", "video_worker.py", "engines.py",
              "box_predictor.py", "tracking.py", "resource_safety.py", "detection_utils.py",
              "overlay_labels.py", "feedback.py")
FRAME_FIELDS = (
    "session_id", "run_id", "source", "device", "mode", "repeat", "execution_index",
    "frame_index", "source_slot", "timestamp_basis",
    "capture_s", "read_s", "inference_start_s", "inference_end_s", "result_ready_s", "displayed_s",
    "capture_to_read_ms", "queue_ms", "inference_ms", "read_to_result_ms", "e2e_ms", "display_ms",
    "backlog_after_result", "frames_read_total", "queue_dropped_total", "source_dropped_total",
    "boxes", "settled",
)
RUN_FIELDS = (
    "session_id", "run_id", "source", "device", "mode", "repeat", "execution_index", "status",
    "duration_s", "frames_offered", "frames_read", "frames_processed", "frames_measured",
    "queue_dropped", "source_dropped", "unprocessed_at_stop", "processed_fraction",
    "output_fps", "source_fps", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "e2e_max_ms",
    "read_to_result_p50_ms", "read_to_result_p95_ms", "inference_p50_ms", "inference_p95_ms",
    "queue_p50_ms", "queue_p95_ms", "backlog_max", "timestamp_basis",
    "load1_before", "load1_after", "process_cpu_s", "torch_threads_during_timing",
)
NOTICE = (
    "Timing benchmark of camera scheduling only; no accuracy is measured. e2e latency is "
    "capture -> result ready in the detection worker (after drawing and BGR->RGB), excluding Qt "
    "rendering and screen refresh. For the paced replay the capture time is the scheduled frame "
    "time; for a real camera it is the V4L2 buffer timestamp when plausible, otherwise the time "
    "read() returned; sensor exposure is never included. fifo and sync are tool-only emulations; "
    "the application's default (latest) is unchanged. Numbers hold for this host, model, "
    "settings and load only."
)


# --------------------------------------------------------------------------- statistics
def describe(values):
    """n/mean/std/min/p50/p95/p99/max of the finite values (linear percentiles)."""
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {key: (0 if key == "n" else None)
                for key in ("n", "mean", "std", "min", "p50", "p95", "p99", "max")}
    return {"n": len(values), "mean": statistics.fmean(values), "std": statistics.pstdev(values),
            "min": min(values), "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99), "max": max(values)}


def spread(values):
    """Across-run spread of one per-run metric: median, mean, sample std and range."""
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


# --------------------------------------------------------------------------- frames and queues
class TaggedFrame(np.ndarray):
    """Zero-copy view that carries the source slot and capture timestamp."""
    slot = None
    capture_ts = None
    shared = False


def tag_frame(image, slot, capture_ts, shared):
    view = image.view(TaggedFrame)
    view.slot, view.capture_ts, view.shared = slot, capture_ts, shared
    return view


def untag(image):
    return image.view(np.ndarray) if isinstance(image, TaggedFrame) else image


class FifoFrameQueue:
    """Tool-only FIFO with the LatestFrameMailbox interface.

    capacity=0 means unbounded apart from the max_bytes memory guard. When full,
    the incoming frame is dropped and older frames are kept, like a V4L2 driver
    without a free buffer. Replayed frames are views of an in-memory clip and
    cost no extra memory, so only owned camera frames count towards max_bytes.
    """

    def __init__(self, capacity=0, max_bytes=0):
        if capacity < 0 or max_bytes < 0:
            raise ValueError("capacity and max_bytes must be non-negative")
        self._condition = threading.Condition()
        self._items = deque()
        self._closed = False
        self.capacity, self.max_bytes = int(capacity), int(max_bytes)
        self.received = self.dropped = self.memory_dropped = 0
        self.bytes_queued = self.max_depth = 0

    @staticmethod
    def _cost(item):
        image = getattr(item, "image", None)
        if image is None or getattr(image, "shared", False):
            return 0
        return int(getattr(image, "nbytes", 0))

    def put(self, item):
        with self._condition:
            if self._closed:
                return False
            self.received += 1
            cost = self._cost(item)
            if self.capacity and len(self._items) >= self.capacity:
                self.dropped += 1
                return False
            if self.max_bytes and self.bytes_queued + cost > self.max_bytes:
                self.dropped += 1
                self.memory_dropped += 1
                return False
            notify = not self._items
            self._items.append((item, cost))
            self.bytes_queued += cost
            self.max_depth = max(self.max_depth, len(self._items))
            self._condition.notify_all()
            return notify

    def take(self, wait=False):
        with self._condition:
            if wait:
                self._condition.wait_for(lambda: bool(self._items) or self._closed)
            if not self._items:
                return None
            item, cost = self._items.popleft()
            self.bytes_queued -= cost
            return item

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def counters(self):
        with self._condition:
            return self.received, self.dropped

    def depth(self):
        with self._condition:
            return len(self._items)


def queue_depth(mailbox):
    """Frames waiting in a FIFO, or 0/1 for the application's single-slot mailbox."""
    if mailbox is None:
        return None
    depth = getattr(mailbox, "depth", None)
    if callable(depth):
        return depth()
    with mailbox._condition:  # read-only peek at LatestFrameMailbox's slot
        return int(mailbox._item is not None)


class FifoCameraReader(CameraReader):
    """The application's CameraReader with a FIFO queue in place of its mailbox."""

    def __init__(self, capture, clock=time.perf_counter, capacity=0, max_bytes=0):
        super().__init__(capture, clock)
        self.mailbox = FifoFrameQueue(capacity, max_bytes)


# --------------------------------------------------------------------------- sources
CAP_PROP_POS_MSEC, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = 0, 3, 4
CAP_PROP_FPS, CAP_PROP_FOURCC, CAP_PROP_BUFFERSIZE = 5, 6, 38


class PacedReplayCapture:
    """A cv2.VideoCapture stand-in that produces frames on a fixed real-time schedule.

    Frame k is captured at t0 + k/fps (t0 = first read) whether or not anyone
    reads it. Like a V4L2 queue of ``buffers`` buffers, frames captured while
    the consumer still holds the previously returned buffer have buffers-1
    places; later frames are dropped and the oldest are kept. read() returns the
    oldest waiting frame or blocks until the next one is captured. With the
    application's CAP_PROP_BUFFERSIZE=1 request honoured, a late read therefore
    waits for a fresh frame. Frames are read-only views; nothing is copied.
    """

    def __init__(self, frames, fps, duration=None, buffers=4, honour_buffersize=True,
                 clock=time.perf_counter, sleep=time.sleep):
        if not frames:
            raise ValueError("replay source has no frames")
        if not fps > 0:
            raise ValueError("fps must be positive")
        self.frames, self.fps, self.duration = frames, float(fps), duration
        self.buffers, self.honour_buffersize = int(buffers), honour_buffersize
        self.clock, self.sleep = clock, sleep
        self.t0 = None
        self._next_slot = 0
        self._waiting = deque()
        self._holding = False
        self.produced = self.dropped = self.delivered = 0
        self.released = False
        self.property_requests = []
        self.height, self.width = frames[0].shape[:2]

    def slot_time(self, slot):
        return self.t0 + slot / self.fps

    def slot_limit(self):
        return math.inf if self.duration is None else math.ceil(self.duration * self.fps)

    def slots_due(self, now):
        """Frames captured up to ``now`` (read or not), for counting offered frames."""
        if self.t0 is None:
            return 0
        return int(min(self.slot_limit(), math.floor((now - self.t0) * self.fps + 1e-9) + 1))

    def _capture_due(self, now, places):
        limit = self.slot_limit()
        while self._next_slot < limit and self.slot_time(self._next_slot) <= now:
            self.produced += 1
            if len(self._waiting) < places:
                self._waiting.append(self._next_slot)
            else:
                self.dropped += 1
            self._next_slot += 1

    def read(self):
        if self.released:
            return False, None
        now = self.clock()
        if self.t0 is None:
            self.t0 = now
        self._capture_due(now, max(0, self.buffers - 1) if self._holding else self.buffers)
        self._holding = False
        while not self._waiting:
            if self._next_slot >= self.slot_limit():
                return False, None
            delay = self.slot_time(self._next_slot) - self.clock()
            if delay > 0:
                self.sleep(delay)
            self._capture_due(self.clock(), self.buffers)
        slot = self._waiting.popleft()
        self._holding = True
        self.delivered += 1
        return True, tag_frame(self.frames[slot % len(self.frames)], slot, self.slot_time(slot), True)

    def isOpened(self):
        return not self.released

    def get(self, prop):
        return {CAP_PROP_FPS: self.fps, CAP_PROP_FRAME_WIDTH: float(self.width),
                CAP_PROP_FRAME_HEIGHT: float(self.height),
                CAP_PROP_BUFFERSIZE: float(self.buffers)}.get(prop, 0.0)

    def set(self, prop, value):
        self.property_requests.append([int(prop), float(value)])
        if prop == CAP_PROP_BUFFERSIZE and self.honour_buffersize and int(value) >= 1:
            self.buffers = int(value)
            return True
        return False

    def release(self):
        self.released = True

    def counters(self, now):
        return {"frames_offered": self.slots_due(now), "source_dropped": self.dropped,
                "frames_delivered": self.delivered, "driver_buffers": self.buffers,
                "property_requests": list(self.property_requests)}


class TimedCameraCapture:
    """Proxy for a real cv2.VideoCapture that tags each frame with its capture time.

    The V4L2 buffer timestamp (CAP_PROP_POS_MSEC, CLOCK_MONOTONIC on Linux) is
    used when it is plausible, otherwise capture_ts is None and the time read()
    returned is used instead. ``buffers_override`` replaces the application's
    CAP_PROP_BUFFERSIZE request for an explicit non-default experiment. Frames
    stay in memory; nothing here writes, encodes or displays an image.
    """
    MAX_DRIVER_AGE_S = 2.0

    def __init__(self, capture, clock=time.perf_counter, monotonic=time.monotonic, buffers_override=None):
        self.capture, self.clock, self.monotonic = capture, clock, monotonic
        self.buffers_override, self.property_requests = buffers_override, []
        self.delivered = self.driver_timestamps = self.fallback_timestamps = 0
        self.driver_ages_ms = deque(maxlen=4096)
        self.first_ts = self.last_ts = None

    def read(self):
        ok, frame = self.capture.read()
        perf_now, mono_now = self.clock(), self.monotonic()
        if not ok or frame is None:
            return False, None
        try:
            stamp_ms = float(self.capture.get(CAP_PROP_POS_MSEC))
        except Exception:
            stamp_ms = 0.0
        age = mono_now - stamp_ms / 1000.0
        capture_ts = None
        if stamp_ms > 0 and math.isfinite(age) and 0.0 <= age <= self.MAX_DRIVER_AGE_S:
            capture_ts = perf_now - age
            self.driver_timestamps += 1
            self.driver_ages_ms.append(age * 1000)
        else:
            self.fallback_timestamps += 1
        slot = self.delivered
        self.delivered += 1
        stamp = perf_now if capture_ts is None else capture_ts
        self.first_ts = stamp if self.first_ts is None else self.first_ts
        self.last_ts = stamp
        return True, tag_frame(frame, slot, capture_ts, False)

    def isOpened(self):
        return bool(self.capture.isOpened())

    def get(self, prop):
        return self.capture.get(prop)

    def set(self, prop, value):
        self.property_requests.append([int(prop), float(value)])
        if prop == CAP_PROP_BUFFERSIZE and self.buffers_override is not None:
            value = self.buffers_override
        return self.capture.set(prop, value)

    def getBackendName(self):
        return self.capture.getBackendName()

    def release(self):
        self.capture.release()

    def counters(self, now):
        span = None if self.first_ts is None else self.last_ts - self.first_ts
        return {"frames_offered": None, "source_dropped": None, "frames_delivered": self.delivered,
                "delivered_fps": (self.delivered - 1) / span if span else None,
                "driver_timestamps": self.driver_timestamps,
                "fallback_timestamps": self.fallback_timestamps,
                "property_requests": list(self.property_requests), "buffers_override": self.buffers_override,
                "read_minus_driver_timestamp_ms": describe(self.driver_ages_ms)}


def decode_image(path):
    import cv2
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"无法解码图片：{path}")
    return image


def letterbox(image, size):
    import cv2
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 114, np.uint8)
    top, left = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def load_clip(path, clip_frames=300, hold=12, frame_size=(640, 480)):
    """Decode once into memory: an image directory (each image held for ``hold``
    frames, cycled) or the first ``clip_frames`` frames of a video file."""
    import cv2
    path = Path(path).expanduser().resolve()
    unique, files = [], []
    if path.is_dir():
        images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"目录中没有图片：{path}")
        for item in images:
            files.append({"path": str(item), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
            unique.append(letterbox(decode_image(item), frame_size))
        order = [(index // hold) % len(unique) for index in range(clip_frames)]
    elif path.is_file():
        files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        capture = cv2.VideoCapture(str(path))
        try:
            while len(unique) < clip_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                unique.append(letterbox(frame, frame_size))
        finally:
            capture.release()
        if not unique:
            raise ValueError(f"无法读取视频：{path}")
        order = [index % len(unique) for index in range(clip_frames)]
    else:
        raise ValueError(f"输入不存在：{path}")
    for image in unique:
        image.flags.writeable = False  # the pipeline must not modify source frames
    digest = hashlib.sha256()
    for index in order:
        digest.update(unique[index].tobytes())
    return [unique[index] for index in order], {
        "input": str(path), "files": files, "clip_frames": clip_frames, "unique_frames": len(unique),
        "hold": hold if path.is_dir() else 1, "frame_size": list(frame_size),
        "decoded_clip_sha256": digest.hexdigest()}


# --------------------------------------------------------------------------- one run
@dataclass
class RunSettings:
    classes: list
    conf: float = 0.40
    iou: float = 0.45
    max_det: int = 300
    imgsz: int = 640
    fifo_depth: int = 0
    fifo_max_bytes: int = 512 * 1024 ** 2
    settle_s: float = 1.0


class RunRecorder:
    """Wrap a VideoWorker's _infer/_publish_live_frame and record one row per result."""

    def __init__(self, worker, capture, clock=time.perf_counter):
        self.worker, self.capture, self.clock = worker, capture, clock
        self.rows, self.reader = [], None
        self._pending = None
        infer, publish = worker._infer, worker._publish_live_frame

        def timed_infer(frame):
            tags = (getattr(frame, "slot", None), getattr(frame, "capture_ts", None))
            result = infer(untag(frame))
            self._pending = (tags, len(result[1]))
            return result

        def recorded_publish(packet):
            ready = self.clock()
            (slot, capture_ts), boxes = self._pending or ((None, None), None)
            self._pending = None
            mailbox = self.reader.mailbox if self.reader is not None else None
            packet.benchmark_index = len(self.rows)
            self.rows.append({
                "source_slot": slot, "capture_ts": capture_ts, "read_at": packet.read_at,
                "inference_started": packet.inference_started,
                "inference_finished": packet.inference_finished, "result_ready": ready,
                "displayed": None, "backlog": queue_depth(mailbox),
                "frames_read_total": packet.captured, "queue_dropped_total": packet.capture_dropped,
                "source_dropped_total": getattr(self.capture, "dropped", None), "boxes": boxes})
            publish(packet)

        worker._infer, worker._publish_live_frame = timed_infer, recorded_publish

    @property
    def in_flight_discarded(self):
        return int(self._pending is not None)


class HeadlessDisplay(threading.Thread):
    """Stand-in for the GUI slot: take the latest result and feed DisplayMetrics."""

    def __init__(self, worker, recorder, clock=time.perf_counter):
        super().__init__(name="benchmark-display", daemon=True)
        from PySide6.QtCore import Qt
        self.worker, self.recorder, self.clock = worker, recorder, clock
        self.wake, self.done = threading.Event(), threading.Event()
        self.metrics, self.last, self.displayed = DisplayMetrics(), None, 0
        worker.live_frame_ready.connect(self.wake.set, Qt.DirectConnection)

    def run(self):
        while True:
            self.wake.wait(0.05)
            self.wake.clear()
            packet = self.worker.take_live_frame()
            if packet is not None:
                now = self.clock()
                self.last = self.metrics.record(packet, now)
                self.displayed += 1
                index = getattr(packet, "benchmark_index", None)
                if index is not None:
                    self.recorder.rows[index]["displayed"] = now
            elif self.done.is_set():
                return


def stop_worker(worker):
    """VideoWorker.stop() without QThread.wait: the worker runs on a Python thread here."""
    worker.running = False
    reader = worker._camera_reader
    if reader is not None:
        reader.stop()


def configure_camera_like_app(capture):
    """The CAP_PROP requests VideoWorker.run() makes for cameras (before and after 10-06)."""
    capture.set(CAP_PROP_BUFFERSIZE, 1)
    capture.set(CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(CAP_PROP_FRAME_HEIGHT, 720)


def run_pre_rework_loop(worker, capture, clock=time.perf_counter, sleep=time.sleep):
    """Port of the pre-2026-10-06 camera loop: read -> infer -> publish -> 5 ms on one thread.

    The old loop emitted every RGB result through a queued Qt signal; results go
    through the current display mailbox here, so all modes are timed identically.
    """
    import cv2
    configure_camera_like_app(capture)
    frames_read = 0
    try:
        while worker.running:
            ok, frame = capture.read()
            if not ok:
                break
            read_at = clock()
            frames_read += 1
            started = clock()
            output, _, status = worker._infer(frame)
            if not worker.running:
                break
            finished = clock()
            worker._publish_live_frame(DisplayFrame(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), read_at,
                                                    started, finished, status, frames_read, 0))
            sleep(PRE_REWORK_SLEEP_S)
    finally:
        capture.release()
    return frames_read


def run_once(engine, settings, mode, capture, duration, clock=time.perf_counter):
    """One timed run through the application's worker; returns raw rows and counters."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    from PySide6.QtCore import Qt
    import video_worker

    worker = video_worker.VideoWorker(engine, None, 0, source_kind="camera")
    worker.class_names = list(settings.classes)
    worker.conf, worker.iou, worker.max_det, worker.imgsz = (
        settings.conf, settings.iou, settings.max_det, settings.imgsz)
    worker.use_low_light_enhancement = worker.sliced = worker.tiled = False
    if worker.recording_path is not None:
        raise RuntimeError("benchmark runs must never record frames")
    recorder = RunRecorder(worker, capture, clock)
    display = HeadlessDisplay(worker, recorder, clock)
    errors = []
    worker.error_signal.connect(lambda value: errors.append(str(value)), Qt.DirectConnection)
    readers = []
    original_reader = video_worker.CameraReader

    def make_reader(camera):
        reader = (FifoCameraReader(camera, capacity=settings.fifo_depth, max_bytes=settings.fifo_max_bytes)
                  if mode == "fifo" else original_reader(camera))
        readers.append(reader)
        recorder.reader = reader
        return reader

    load_before, usage_before = os.getloadavg(), resource.getrusage(resource.RUSAGE_SELF)
    frames_read_sync = None
    display.start()
    timer = threading.Timer(duration, stop_worker, args=(worker,))
    started = clock()
    timer.start()
    try:
        if mode == "sync":
            frames_read_sync = run_pre_rework_loop(worker, capture, clock)
        else:
            worker._open_capture = lambda: capture
            video_worker.CameraReader = make_reader
            try:
                worker.run()
            finally:
                video_worker.CameraReader = original_reader
    finally:
        timer.cancel()
        stopped = clock()
        display.done.set()
        display.join(2)
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    reader = readers[-1] if readers else None
    if reader is not None:
        frames_read, queue_dropped = reader.mailbox.counters()
        unprocessed = queue_depth(reader.mailbox)
    else:
        frames_read, queue_dropped, unprocessed = frames_read_sync or 0, 0, 0
    origin = getattr(capture, "t0", None)
    origin = started if origin is None else origin
    return {
        "rows": recorder.rows, "origin": origin, "started": started, "stopped": stopped,
        "errors": errors, "frames_read": frames_read, "queue_dropped": queue_dropped,
        "unprocessed_at_stop": unprocessed + recorder.in_flight_discarded,
        "fifo_max_depth": getattr(reader.mailbox, "max_depth", None) if reader else None,
        "fifo_memory_dropped": getattr(reader.mailbox, "memory_dropped", None) if reader else None,
        "display_dropped": worker.display_dropped, "displayed": display.displayed,
        "display_metrics": display.last, "source": capture.counters(stopped),
        "load_before": list(load_before), "load_after": list(os.getloadavg()),
        "process_cpu_s": (usage_after.ru_utime + usage_after.ru_stime)
                         - (usage_before.ru_utime + usage_before.ru_stime),
    }


# --------------------------------------------------------------------------- summaries
def frame_records(raw, meta, settle_s):
    """Per-frame CSV rows with times relative to the run origin (first source read)."""
    origin, records = raw["origin"], []

    def rel(value):
        return None if value is None else value - origin

    def ms(later, earlier):
        return None if later is None or earlier is None else (later - earlier) * 1000

    for index, row in enumerate(raw["rows"]):
        basis_ts = row["capture_ts"] if row["capture_ts"] is not None else row["read_at"]
        if row["capture_ts"] is None:
            basis = "read_return"
        else:
            basis = "scheduled_capture" if meta["source"] == "video" else "driver_timestamp"
        records.append({
            **meta, "frame_index": index, "source_slot": row["source_slot"], "timestamp_basis": basis,
            "capture_s": rel(row["capture_ts"]), "read_s": rel(row["read_at"]),
            "inference_start_s": rel(row["inference_started"]),
            "inference_end_s": rel(row["inference_finished"]),
            "result_ready_s": rel(row["result_ready"]), "displayed_s": rel(row["displayed"]),
            "capture_to_read_ms": ms(row["read_at"], row["capture_ts"]),
            "queue_ms": ms(row["inference_started"], row["read_at"]),
            "inference_ms": ms(row["inference_finished"], row["inference_started"]),
            "read_to_result_ms": ms(row["result_ready"], row["read_at"]),
            "e2e_ms": ms(row["result_ready"], basis_ts), "display_ms": ms(row["displayed"], row["result_ready"]),
            "backlog_after_result": row["backlog"], "frames_read_total": row["frames_read_total"],
            "queue_dropped_total": row["queue_dropped_total"],
            "source_dropped_total": row["source_dropped_total"], "boxes": row["boxes"],
            "settled": (basis_ts - origin) >= settle_s,
        })
    return records


def summarize_run(raw, records, meta):
    settled = [row for row in records if row["settled"]]
    duration = raw["stopped"] - raw["origin"]
    ready = [row["result_ready_s"] for row in settled]
    output_fps = (len(ready) - 1) / (ready[-1] - ready[0]) if len(ready) > 1 and ready[-1] > ready[0] else None
    source = raw["source"]
    reads = [row["read_s"] for row in records]
    offered = source.get("frames_offered")
    latency = {name: describe(row[name] for row in settled) for name in (
        "e2e_ms", "read_to_result_ms", "inference_ms", "queue_ms", "capture_to_read_ms", "display_ms")}
    bases = sorted({row["timestamp_basis"] for row in settled})
    status = "error" if raw["errors"] else "complete"
    return {
        **meta, "status": status, "errors": raw["errors"], "duration_s": duration,
        "frames_offered": offered, "frames_read": raw["frames_read"], "frames_processed": len(records),
        "frames_measured": len(settled), "queue_dropped": raw["queue_dropped"],
        "source_dropped": source.get("source_dropped"), "unprocessed_at_stop": raw["unprocessed_at_stop"],
        "processed_fraction": len(records) / offered if offered else None,
        "processed_of_read": len(records) / raw["frames_read"] if raw["frames_read"] else None,
        "output_fps": output_fps,
        "source_fps": source_fps(source, raw, duration),
        "read_fps": (len(reads) - 1) / (reads[-1] - reads[0]) if len(reads) > 1 and reads[-1] > reads[0] else None,
        "latency_ms": latency, "backlog": describe(row["backlog_after_result"] for row in records),
        "timestamp_basis": "+".join(bases), "fifo_max_depth": raw["fifo_max_depth"],
        "fifo_memory_dropped": raw["fifo_memory_dropped"], "display_dropped": raw["display_dropped"],
        "displayed": raw["displayed"], "display_metrics_last": raw["display_metrics"],
        "source_counters": source, "load_before": raw["load_before"], "load_after": raw["load_after"],
        "process_cpu_s": raw["process_cpu_s"],
    }


def source_fps(source, raw, duration):
    """Replay: frames offered per second. Camera: rate at which the driver delivered
    frames to the application (consumer-paced in sync mode, where drops are invisible)."""
    offered = source.get("frames_offered")
    if offered is None:
        return source.get("delivered_fps")
    return offered / duration if offered and duration > 0 else None


def run_row(summary):
    latency = summary["latency_ms"]
    flat = {key: summary.get(key) for key in RUN_FIELDS}
    flat.update({
        "e2e_p50_ms": latency["e2e_ms"]["p50"], "e2e_p95_ms": latency["e2e_ms"]["p95"],
        "e2e_p99_ms": latency["e2e_ms"]["p99"], "e2e_max_ms": latency["e2e_ms"]["max"],
        "read_to_result_p50_ms": latency["read_to_result_ms"]["p50"],
        "read_to_result_p95_ms": latency["read_to_result_ms"]["p95"],
        "inference_p50_ms": latency["inference_ms"]["p50"], "inference_p95_ms": latency["inference_ms"]["p95"],
        "queue_p50_ms": latency["queue_ms"]["p50"], "queue_p95_ms": latency["queue_ms"]["p95"],
        "backlog_max": summary["backlog"]["max"], "load1_before": summary["load_before"][0],
        "load1_after": summary["load_after"][0]})
    return flat


AGGREGATE_METRICS = {
    "output_fps": lambda run: run["output_fps"],
    "e2e_p50_ms": lambda run: run["latency_ms"]["e2e_ms"]["p50"],
    "e2e_p95_ms": lambda run: run["latency_ms"]["e2e_ms"]["p95"],
    "e2e_p99_ms": lambda run: run["latency_ms"]["e2e_ms"]["p99"],
    "e2e_max_ms": lambda run: run["latency_ms"]["e2e_ms"]["max"],
    "read_to_result_p50_ms": lambda run: run["latency_ms"]["read_to_result_ms"]["p50"],
    "read_to_result_p95_ms": lambda run: run["latency_ms"]["read_to_result_ms"]["p95"],
    "inference_p50_ms": lambda run: run["latency_ms"]["inference_ms"]["p50"],
    "inference_p95_ms": lambda run: run["latency_ms"]["inference_ms"]["p95"],
    "queue_p50_ms": lambda run: run["latency_ms"]["queue_ms"]["p50"],
    "processed_fraction": lambda run: run["processed_fraction"],
    "processed_of_read": lambda run: run.get("processed_of_read"),
    "source_fps": lambda run: run["source_fps"],
    "dropped_per_s": lambda run: ((run["queue_dropped"] or 0) + (run["source_dropped"] or 0)) / run["duration_s"]
                                  if run["duration_s"] > 0 else None,
    "backlog_max": lambda run: run["backlog"]["max"],
    "load1_before": lambda run: run["load_before"][0],
}


def aggregate(runs, records_by_run):
    """Per (source, device, mode): spread of per-run metrics plus pooled per-frame latency."""
    groups = {}
    for run in runs:
        groups.setdefault((run["source"], run["device"], run["mode"]), []).append(run)
    result = []
    for (source, device, mode), members in groups.items():
        pooled = [row for run in members for row in records_by_run.get(run["run_id"], []) if row["settled"]]
        result.append({
            "source": source, "device": device, "mode": mode, "runs": len(members),
            "complete_runs": sum(run["status"] == "complete" for run in members),
            "metrics": {name: spread(getter(run) for run in members) for name, getter in AGGREGATE_METRICS.items()},
            "pooled_e2e_ms": describe(row["e2e_ms"] for row in pooled),
            "pooled_inference_ms": describe(row["inference_ms"] for row in pooled),
        })
    return result


# --------------------------------------------------------------------------- output files
def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    return value


def append_csv(path, fields, rows):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows({key: csv_value(row.get(key)) for key in fields} for row in rows)


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


# --------------------------------------------------------------------------- environment
def git_state():
    def git(*arguments):
        try:
            return subprocess.run(["git", *arguments], cwd=PROJECT, capture_output=True, text=True,
                                  timeout=20, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no", "--", *CODE_FILES)}


def code_fingerprint():
    hashes = {name: sha256_file(PROJECT / name) for name in CODE_FILES if (PROJECT / name).is_file()}
    return {"files": hashes, "git": git_state(),
            "combined_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def host_metadata():
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    from resource_safety import available_ram_mb
    clock = time.get_clock_info("perf_counter")
    versions = {}
    for package in ("torch", "ultralytics", "opencv-python", "numpy", "PySide6"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "available_ram_mb": available_ram_mb(),
            "loadavg": list(os.getloadavg()), "perf_counter": clock.implementation,
            "packages": versions}


def device_metadata(engine, requested):
    import torch
    actual = str(engine.device)
    # Ultralytics resets CPU threads to min(8, CPUs - 1) when it builds the predictor on the
    # first inference, after this point; each run records the count used for timing.
    meta = {"requested": requested, "actual": actual, "torch_threads_after_load": torch.get_num_threads(),
            "torch_threads_note": "sampled after loading, before the predictor exists; see "
                                  "runs[].torch_threads_during_timing",
            "cuda_available": bool(torch.cuda.is_available()), "torch_cuda": torch.version.cuda}
    if actual.startswith("cuda"):
        properties = torch.cuda.get_device_properties(0)
        meta.update({"gpu": torch.cuda.get_device_name(0), "gpu_memory_mib": properties.total_memory // 1024 ** 2,
                     "capability": f"{properties.major}.{properties.minor}"})
        try:
            meta["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,power.draw",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            meta["nvidia_smi"] = None
    return meta


def torch_threads():
    import torch
    return torch.get_num_threads()


def camera_metadata(index):
    name_path = Path("/sys/class/video4linux") / f"video{index}" / "name"
    try:
        return {"index": index, "model_name": name_path.read_text(encoding="utf-8").strip()}
    except OSError:
        return {"index": index, "model_name": None}


def describe_capture(capture):
    """Negotiated properties after the app's requests; never reads a frame."""
    fourcc = int(capture.get(CAP_PROP_FOURCC) or 0)
    return {"width": capture.get(CAP_PROP_FRAME_WIDTH), "height": capture.get(CAP_PROP_FRAME_HEIGHT),
            "fps_property": capture.get(CAP_PROP_FPS), "buffersize_property": capture.get(CAP_PROP_BUFFERSIZE),
            "fourcc": "".join(chr((fourcc >> (8 * shift)) & 0xFF) for shift in range(4)) if fourcc else None,
            "backend": capture.getBackendName() if hasattr(capture, "getBackendName") else None}


# --------------------------------------------------------------------------- session
def parse_frame_size(value):
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("请使用 宽x高，例如 640x480") from exc
    if not (32 <= width <= 3840 and 32 <= height <= 2160):
        raise argparse.ArgumentTypeError("帧尺寸需在 32x32 到 3840x2160 之间")
    return width, height


def parse_fourcc(value):
    if len(value) != 4 or not value.isascii() or not value.isprintable():
        raise argparse.ArgumentTypeError("像素格式须为 4 个 ASCII 字符，例如 MJPG")
    return value


def parse_modes(value):
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise argparse.ArgumentTypeError(f"模式须为不重复的 {','.join(MODES)}")
    return modes


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="无界面实时调度基准：latest / fifo / sync，回放或真实摄像头，只计时不保存画面。")
    parser.add_argument("--source", choices=SOURCES, help="video=按固定 FPS 回放的视频/图片序列；camera=真实摄像头（只计时）")
    parser.add_argument("--input", type=Path, default=PROJECT / "datasets" / "mini_eval" / "images",
                        help="回放用视频文件或图片目录")
    parser.add_argument("--camera", type=nonnegative_int, default=0, help="摄像头编号（cv2.VideoCapture）")
    parser.add_argument("--camera-fourcc", type=parse_fourcc,
                        help="非默认压力测试：请求像素格式（如 MJPG）；程序本身不设置，默认不请求")
    parser.add_argument("--camera-buffers", type=positive_integer,
                        help="非默认实验：用此缓冲数替换程序的 CAP_PROP_BUFFERSIZE=1 请求")
    parser.add_argument("--fps", type=positive_float, default=30.0, help="回放源的固定帧率")
    parser.add_argument("--clip-frames", type=positive_integer, default=300)
    parser.add_argument("--hold", type=positive_integer, default=12, help="图片目录中每张图连续出现的帧数")
    parser.add_argument("--frame-size", type=parse_frame_size, default=(640, 480))
    parser.add_argument("--driver-buffers", type=positive_integer,
                        help="回放源的驱动缓冲数；默认按程序 CAP_PROP_BUFFERSIZE=1 请求设置")
    parser.add_argument("--modes", type=parse_modes, default=list(MODES))
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--duration", type=positive_float, default=30.0, help="每次运行秒数；摄像头最多 60 秒")
    parser.add_argument("--settle", type=float, default=1.0, help="开始后不计入延迟统计的秒数（仍计数）")
    parser.add_argument("--warmup", type=nonnegative_int, default=10, help="每次运行前不计时的预热推理次数")
    parser.add_argument("--cooldown", type=float, default=3.0, help="两次运行之间的等待秒数")
    parser.add_argument("--fifo-depth", type=nonnegative_int, default=0, help="fifo 队列长度，0 为不限（仍受内存上限）")
    parser.add_argument("--fifo-max-mib", type=positive_integer, default=512, help="fifo 中摄像头帧的内存上限")
    parser.add_argument("--model", type=Path, default=Path("yoloe-26s-seg.pt"))
    parser.add_argument("--classes", type=class_names, default=["person", "bottle", "cup", "cell phone"])
    parser.add_argument("--conf", type=probability, default=0.40)
    parser.add_argument("--iou", type=probability, default=0.45)
    parser.add_argument("--max-det", type=positive_integer, default=300)
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="必须显式指定；CUDA 不可用时停止，不自动改用 CPU")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "validation" / "realtime_benchmark")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--summarize", nargs="+", type=Path, metavar="DIR",
                        help="只汇总已有结果目录（summary.json），输出 Markdown 表格")
    return parser


def validate_args(parser, args):
    if args.summarize:
        return
    if args.source is None:
        parser.error("需要 --source video 或 --source camera")
    if args.source == "camera" and args.duration > CAMERA_MAX_SECONDS:
        parser.error(f"摄像头每次运行最多 {CAMERA_MAX_SECONDS:.0f} 秒")
    if args.settle < 0 or args.settle >= args.duration:
        parser.error("--settle 必须不小于 0 且小于 --duration")
    if args.driver_buffers is not None and args.source != "video":
        parser.error("--driver-buffers 只用于回放源；真实摄像头使用驱动实际缓冲")
    if (args.camera_fourcc is not None or args.camera_buffers is not None) and args.source != "camera":
        parser.error("--camera-fourcc / --camera-buffers 只用于真实摄像头")
    if args.device is None:
        parser.error("需要 --device cpu 或 --device cuda；不自动选择设备")


def resolve_model(path):
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).is_file() else PROJECT / path
    if not path.is_file():
        raise ValueError(f"权重不存在：{path}")
    return path.resolve()


def load_engine(model, device, imgsz):
    if device not in ("cpu", "cuda"):
        raise ValueError("设备必须显式为 cpu 或 cuda")
    import torch
    from engines import OpenVocabEngine
    use_cuda = device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("已指定 CUDA，但当前 PyTorch 无可用 CUDA；不会自动改用 CPU")
    engine = OpenVocabEngine()
    engine.imgsz = imgsz
    status = engine.load_model(str(model), device="CUDA" if use_cuda else "CPU")
    if status != "SUCCESS":
        raise RuntimeError(f"模型加载失败：{status}")
    return engine


def warm_up(engine, frames, settings, count):
    """Untimed inferences on frames of the run's size; tracker off, prompts encoded."""
    import feedback  # noqa: F401  the worker imports it on its first frame
    for index in range(count):
        engine.process_frame(untag(frames[index % len(frames)]), settings.conf, settings.iou,
                             settings.classes, max_det=settings.max_det)
    if str(engine.device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def schedule(modes, repeats):
    """Repeat-major order with the mode order rotated each repeat (ABC, BCA, CAB)."""
    plan = []
    for repeat in range(repeats):
        shift = repeat % len(modes)
        plan.extend((repeat, mode) for mode in modes[shift:] + modes[:shift])
    return plan


def fourcc_code(text):
    return sum(ord(char) << (8 * shift) for shift, char in enumerate(text))


def open_camera(index, fourcc=None):
    """The app's Linux path (VideoWorker._open_capture). A pixel format is only
    requested for an explicit non-default stress run; the app never sets one."""
    import cv2
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头 {index}")
    if fourcc:
        capture.set(CAP_PROP_FOURCC, fourcc_code(fourcc))
    return capture


def run_session(args):
    started = datetime.now(timezone.utc)
    settings = RunSettings(classes=list(args.classes), conf=args.conf, iou=args.iou, max_det=args.max_det,
                           imgsz=args.imgsz, fifo_depth=args.fifo_depth,
                           fifo_max_bytes=args.fifo_max_mib * 1024 ** 2, settle_s=args.settle)
    model = resolve_model(args.model)
    clip, data = (load_clip(args.input, args.clip_frames, args.hold, args.frame_size)
                  if args.source == "video" else (None, None))
    os.chdir(PROJECT)  # YOLOE resolves its bundled text encoder relative to the project
    engine = load_engine(model, args.device, args.imgsz)
    device = "cuda" if str(engine.device).startswith("cuda") else "cpu"
    session_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{args.source}_{device}"
    output = (args.output_root.expanduser().resolve() / session_id)
    output.mkdir(parents=True, exist_ok=False)
    camera = None
    if args.source == "camera":
        probe = TimedCameraCapture(open_camera(args.camera, args.camera_fourcc), buffers_override=args.camera_buffers)
        try:
            configure_camera_like_app(probe)
            camera = {**camera_metadata(args.camera), **describe_capture(probe)}
        finally:
            probe.release()
        size = (int(camera["height"] or 720), int(camera["width"] or 1280))
        warm_frames = [np.full((*size, 3), 114, np.uint8)]
    else:
        warm_frames = clip
    report = {
        "schema": 1, "tool": "tools/benchmark_realtime.py", "session_id": session_id, "notice": NOTICE,
        "created_utc": started.isoformat(), "status": "running",
        "settings": {**json_safe(asdict(settings)), "source": args.source, "modes": args.modes,
                     "repeats": args.repeats, "duration_s": args.duration, "warmup": args.warmup,
                     "cooldown_s": args.cooldown, "fps": args.fps if args.source == "video" else None,
                     "driver_buffers": args.driver_buffers, "camera_index": args.camera if camera else None,
                     "camera_fourcc_request": args.camera_fourcc, "camera_buffers_override": args.camera_buffers,
                     "e2e_definition": "result ready in worker (after draw + BGR->RGB) minus capture time",
                     "display": "headless consumer; Qt rendering excluded",
                     "tracker": "ByteTrack on (application default)", "low_light": False, "sahi": False},
        "fingerprints": {"code": code_fingerprint(), "model": {"path": str(model), "sha256": sha256_file(model)},
                         "data": data},
        "host": host_metadata(), "device": device_metadata(engine, args.device), "camera": camera,
        "schedule": [{"repeat": repeat + 1, "mode": mode} for repeat, mode in schedule(args.modes, args.repeats)],
        "runs": [], "aggregates": [],
    }
    atomic_json(output / "summary.json", report)
    records_by_run = {}
    try:
        for execution_index, (repeat, mode) in enumerate(schedule(args.modes, args.repeats)):
            if execution_index:
                time.sleep(args.cooldown)
            run_id = f"{session_id}_r{repeat + 1}_{mode}"
            meta = {"session_id": session_id, "run_id": run_id, "source": args.source, "device": device,
                    "mode": mode, "repeat": repeat + 1, "execution_index": execution_index}
            warm_up(engine, warm_frames, settings, args.warmup)
            if args.source == "video":
                capture = PacedReplayCapture(clip, args.fps, duration=args.duration + 2.0,
                                             buffers=args.driver_buffers or 4,
                                             honour_buffersize=args.driver_buffers is None)
            else:
                capture = TimedCameraCapture(open_camera(args.camera, args.camera_fourcc),
                                             buffers_override=args.camera_buffers)
            print(f"[{execution_index + 1}/{len(report['schedule'])}] {run_id} ...", flush=True)
            raw = run_once(engine, settings, mode, capture, args.duration)
            records = frame_records(raw, meta, args.settle)
            summary = summarize_run(raw, records, meta)
            summary["torch_threads_during_timing"] = torch_threads()  # the predictor exists by now
            records_by_run[run_id] = records
            report["runs"].append(summary)
            report["aggregates"] = aggregate(report["runs"], records_by_run)
            append_csv(output / "frames.csv", FRAME_FIELDS, records)
            append_csv(output / "runs.csv", RUN_FIELDS, [run_row(summary)])
            atomic_json(output / "summary.json", report)
            e2e = summary["latency_ms"]["e2e_ms"]
            print(f"    fps={fmt(summary['output_fps'])} e2e p50={fmt(e2e['p50'])} p95={fmt(e2e['p95'])} ms "
                  f"processed={summary['frames_processed']} dropped={summary['queue_dropped']}"
                  f"+{summary['source_dropped'] or 0} load1={summary['load_before'][0]:.2f}", flush=True)
            if summary["status"] != "complete":
                raise RuntimeError(f"运行出错，停止整个批次（不重试）：{summary['errors']}")
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["host_after"] = {"loadavg": list(os.getloadavg())}
        atomic_json(output / "summary.json", report)
        engine.unload()
        if not args.no_plots and report["runs"]:
            report["plots"] = write_plots(output, report, records_by_run)
            atomic_json(output / "summary.json", report)
    print(f"已保存：{output}")
    return output


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- plots and tables
MODE_COLORS = {"latest": "#2a78d6", "fifo": "#eb6834", "sync": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def write_plots(output, report, records_by_run):
    """Latency timeline (first repeat per mode) and per-run p50/p95; skipped without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    except ImportError:
        return {"skipped": "matplotlib unavailable"}

    def plain_log_axis(axis):
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    title = f"{report['settings']['source']} source, {report['device']['actual']}"
    paths = {}
    figure, axis = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    for run in report["runs"]:
        if run["repeat"] != 1:
            continue
        rows = [row for row in records_by_run[run["run_id"]] if row["e2e_ms"] is not None]
        x = [row["capture_s"] if row["capture_s"] is not None else row["read_s"] for row in rows]
        y = [row["e2e_ms"] for row in rows]
        if not rows:
            continue
        axis.plot(x, y, color=MODE_COLORS[run["mode"]], linewidth=1.5, label=run["mode"])
        axis.annotate(run["mode"], (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                      color=INK, va="center", fontsize=8)
    plain_log_axis(axis)
    axis.set_xlabel("capture time since start (s)")
    axis.set_ylabel("capture -> result latency (ms, log)")
    axis.set_title(f"Latency over one run per mode ({title})", loc="left")
    axis.grid(True, which="major", color=GRID, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    paths["timeline"] = str(output / "latency_timeline.png")
    figure.savefig(paths["timeline"])
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.0, 3.4), dpi=150)
    modes = report["settings"]["modes"]
    for position, mode in enumerate(modes):
        runs = [run for run in report["runs"] if run["mode"] == mode]
        for offset, (statistic, marker) in enumerate((("p50", "o"), ("p95", "D"))):
            values = [run["latency_ms"]["e2e_ms"][statistic] for run in runs
                      if run["latency_ms"]["e2e_ms"][statistic] is not None]
            axis.scatter([position + (offset - 0.5) * 0.25] * len(values), values, s=36, marker=marker,
                         color=MODE_COLORS[mode], edgecolors=SURFACE, linewidths=1.5,
                         label=statistic if position == 0 else None)
    axis.set_xticks(range(len(modes)), modes)
    plain_log_axis(axis)
    axis.set_ylabel("e2e latency per run (ms, log)")
    axis.set_title(f"Per-run p50 (circle) and p95 (diamond), {title}", loc="left")
    axis.grid(True, axis="y", color=GRID, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    legend = axis.legend(frameon=False, loc="upper left")
    for handle in getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", []):
        handle.set_color(INK_MUTED)
    figure.tight_layout()
    paths["per_run"] = str(output / "latency_per_run.png")
    figure.savefig(paths["per_run"])
    plt.close(figure)
    return paths


def markdown_table(reports):
    """One row per (source, device, mode): across-run median [min-max]."""
    def cell(metric, digits=1, scale=1.0):
        if metric is None:
            return "n/a"
        return (f"{metric['median'] * scale:.{digits}f} "
                f"[{metric['min'] * scale:.{digits}f}–{metric['max'] * scale:.{digits}f}]")

    lines = ["| 源 | 设备 | 模式 | 运行 | 输出 FPS | e2e p50 ms | e2e p95 ms | e2e p99 ms | 推理 p50 ms | 处理比例 %† | 丢弃/替换 帧/s | load1 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        for group in report["aggregates"]:
            metrics = group["metrics"]
            lines.append("| " + " | ".join([
                group["source"], group["device"], group["mode"], str(group["complete_runs"]),
                cell(metrics["output_fps"]), cell(metrics["e2e_p50_ms"], 0), cell(metrics["e2e_p95_ms"], 0),
                cell(metrics["e2e_p99_ms"], 0), cell(metrics["inference_p50_ms"]),
                cell(metrics["processed_fraction"], 0, 100.0) if metrics["processed_fraction"]
                else cell(metrics.get("processed_of_read"), 0, 100.0) + "*", cell(metrics["dropped_per_s"]),
                cell(metrics["load1_before"], 2)]) + " |")
    lines.append("\n† 回放源：已处理 / 源产生帧；* 摄像头：已处理 / 程序读到的帧（驱动内部丢帧不可见）。"
                 "单元格为各次运行的中位数 [最小–最大]。")
    return "\n".join(lines)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.summarize:
        reports = [json.loads((Path(path) / "summary.json").read_text(encoding="utf-8")) for path in args.summarize]
        print(markdown_table(reports))
        return 0
    original_cwd = Path.cwd()
    try:
        run_session(args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"实验失败：{exc}\n")
    finally:
        os.chdir(original_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
