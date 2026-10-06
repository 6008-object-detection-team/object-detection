"""Deterministic realtime scheduling tests: no physical camera or model runs."""
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from live_video import (CameraReader, CameraRecordingTimeline, DisplayFrame,
                        DisplayMetrics, LatestFrameMailbox)
from video_worker import VideoWorker


class MailboxTests(unittest.TestCase):
    def test_slow_consumer_gets_only_latest_and_replacement_count(self):
        mailbox = LatestFrameMailbox()
        self.assertTrue(mailbox.put(1))
        self.assertFalse(mailbox.put(2))
        self.assertFalse(mailbox.put(3))
        self.assertEqual(mailbox.counters(), (3, 2))
        self.assertEqual(mailbox.take(), 3)
        self.assertIsNone(mailbox.take())
        self.assertTrue(mailbox.put(4))

    def test_close_wakes_waiter_and_rejects_new_frames(self):
        mailbox = LatestFrameMailbox()
        result = []
        waiter = threading.Thread(target=lambda: result.append(mailbox.take(wait=True)))
        waiter.start()
        mailbox.close()
        waiter.join(1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [None])
        self.assertFalse(mailbox.put(1))

    def test_closed_producer_preserves_last_captured_frame(self):
        mailbox = LatestFrameMailbox()
        mailbox.put(1)
        mailbox.close()
        self.assertEqual(mailbox.take(wait=True), 1)
        self.assertIsNone(mailbox.take(wait=True))


class CameraReaderTests(unittest.TestCase):
    def test_read_failure_closes_and_releases_capture(self):
        capture = MagicMock()
        capture.read.side_effect = RuntimeError("camera failure")
        reader = CameraReader(capture)
        reader.start()
        reader.join(1)
        self.assertFalse(reader.is_alive())
        self.assertIn("camera failure", reader.error)
        self.assertIsNone(reader.mailbox.take(wait=True))
        capture.release.assert_called_once()

    def test_stop_does_not_release_capture_while_driver_is_reading(self):
        entered, unblock = threading.Event(), threading.Event()
        capture = MagicMock()
        owners = []

        def read():
            owners.append(threading.get_ident())
            entered.set()
            unblock.wait(2)
            return False, None

        capture.read.side_effect = read
        capture.release.side_effect = lambda: owners.append(threading.get_ident())
        reader = CameraReader(capture)
        reader.start()
        try:
            self.assertTrue(entered.wait(1))
            reader.stop()
            self.assertIsNone(reader.mailbox.take(wait=True))
            capture.release.assert_not_called()
        finally:
            unblock.set()
            reader.join(1)
        self.assertFalse(reader.is_alive())
        capture.release.assert_called_once()
        self.assertEqual(owners, [reader.ident, reader.ident])
        self.assertIsNone(reader.error)


class CameraWorkerTests(unittest.TestCase):
    def worker(self):
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            return VideoWorker(MagicMock(), MagicMock(), 0, source_kind="camera")

    def test_slow_inference_skips_stale_frames_but_runs_models_serially(self):
        worker = self.worker()
        first_infer, burst_done, second_infer = (threading.Event() for _ in range(3))
        capture = MagicMock()
        capture.get.return_value = 25.0
        reads = 0
        seen = []

        def read():
            nonlocal reads
            reads += 1
            if reads == 2 and not first_infer.wait(2):
                return False, None
            if reads > 5:
                burst_done.set()
                second_infer.wait(2)
                return False, None
            return True, np.full((8, 8, 3), reads, np.uint8)

        def infer(frame):
            seen.append(int(frame[0, 0, 0]))
            if len(seen) == 1:
                first_infer.set()
                self.assertTrue(burst_done.wait(2))
            else:
                second_infer.set()
            return frame.copy(), [], "ok"

        capture.read.side_effect = read
        errors = []
        worker.error_signal.connect(errors.append)
        with patch.object(worker, "_open_capture", return_value=capture), \
                patch.object(worker, "_infer", side_effect=infer), \
                patch.object(worker, "msleep") as delay:
            worker.run()
        self.assertEqual(seen, [1, 5])
        packet = worker.take_live_frame()
        self.assertEqual(int(packet.image[0, 0, 0]), 5)
        self.assertEqual(packet.capture_dropped, 3)
        capture.release.assert_called_once()
        delay.assert_not_called()
        self.assertIsNone(worker._camera_reader)
        self.assertIn("读取摄像头", errors[0])

    def test_gui_notifications_are_bounded_and_new_results_replace_old(self):
        worker = self.worker()
        notifications = []
        worker.live_frame_ready.connect(lambda: notifications.append(1))
        for value in range(100):
            worker._publish_live_frame(value)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(worker.display_dropped, 99)
        self.assertEqual(worker.take_live_frame(), 99)
        worker._publish_live_frame(100)
        self.assertEqual(len(notifications), 2)

    def test_camera_start_failure_releases_capture(self):
        worker, capture = self.worker(), MagicMock()
        capture.get.return_value = 25.0
        with patch("video_worker.CameraReader.start", side_effect=RuntimeError("start failed")):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                worker._run_camera(capture)
        capture.release.assert_called_once()
        self.assertIsNone(worker._camera_reader)


class MetricsAndRecordingTests(unittest.TestCase):
    def test_latency_includes_queue_inference_and_gui_delay_and_window_is_bounded(self):
        image = np.zeros((8, 8, 3), np.uint8)
        metrics = DisplayMetrics(window=2)
        packet = DisplayFrame(image, 1.0, 1.01, 1.05, "ok", 1, 0)
        first = metrics.record(packet, 1.1)
        self.assertAlmostEqual(first["latency_ms"], 100)
        self.assertAlmostEqual(first["queue_ms"], 10)
        self.assertAlmostEqual(first["processing_ms"], 40)
        packet = DisplayFrame(image, 1.2, 1.21, 1.25, "ok", 2, 0)
        second = metrics.record(packet, 1.3)
        self.assertAlmostEqual(second["display_fps"], 5)
        self.assertAlmostEqual(second["p95_ms"], 100)
        metrics.record(packet, 1.5)
        self.assertEqual(len(metrics._latencies), 2)

    def test_camera_recording_repeats_frames_to_preserve_elapsed_time(self):
        writer = MagicMock()
        timeline = CameraRecordingTimeline(writer, fps=10)
        first, latest = object(), object()
        timeline.write(first, 0.0)
        timeline.write(latest, 0.5)
        self.assertEqual(timeline.written, 6)
        self.assertEqual(timeline.repeated, 4)
        self.assertEqual([call.args[0] for call in writer.write.call_args_list],
                         [first] * 5 + [latest])

    def test_fast_updates_do_not_increase_recorded_playback_speed(self):
        writer = MagicMock()
        timeline = CameraRecordingTimeline(writer, fps=10)
        timeline.write("first", 0.0)
        timeline.write("second", 0.02)
        timeline.write("latest", 0.2)
        self.assertEqual(timeline.written, 3)
        self.assertEqual([call.args[0] for call in writer.write.call_args_list],
                         ["first", "second", "latest"])

    def test_recording_catch_up_work_is_bounded(self):
        writer = MagicMock()
        timeline = CameraRecordingTimeline(writer, fps=10)
        timeline.write("first", 0.0)
        with self.assertRaisesRegex(RuntimeError, "10 秒"):
            timeline.write("late", 12.0)
        writer.write.assert_called_once_with("first")


class LiveUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_gui_takes_latest_result_and_reports_measured_latency(self):
        from PySide6.QtCore import Qt
        from main import AIApp
        window = AIApp()
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            worker = VideoWorker(MagicMock(), MagicMock(), 0)
        window.worker = worker
        window.current_video_label = window.video_label_tab1
        window._live_metrics = DisplayMetrics()
        worker.live_frame_ready.connect(window.update_live_video_label, Qt.QueuedConnection)
        try:
            for value in range(100):
                worker._publish_live_frame(DisplayFrame(
                    np.full((8, 8, 3), value, np.uint8), 1.0, 1.01, 1.05,
                    f"frame {value}", 100, 50))
            with patch("main.time.perf_counter", return_value=1.1):
                self.app.processEvents()
            message = window.statusBar().currentMessage()
            self.assertIn("frame 99", message)
            self.assertIn("读帧→界面 100 ms", message)
            self.assertIn("显示 99", message)
            self.assertEqual(len(window._live_metrics._latencies), 1)
            self.assertIsNone(worker.take_live_frame())
        finally:
            window.worker = None
            window.close()

    def test_late_notification_from_old_worker_does_not_update_current_view(self):
        from PySide6.QtCore import Qt
        from main import AIApp
        window = AIApp()
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            old = VideoWorker(MagicMock(), MagicMock(), 0)
            current = VideoWorker(MagicMock(), MagicMock(), 0)
        window.worker = current
        window.current_video_label = window.video_label_tab1
        window._live_metrics = DisplayMetrics()
        old.live_frame_ready.connect(window.update_live_video_label, Qt.QueuedConnection)
        try:
            window.statusBar().showMessage("current stream")
            old._publish_live_frame(DisplayFrame(np.zeros((8, 8, 3), np.uint8),
                                                1, 1, 1, "old stream", 1, 0))
            self.app.processEvents()
            self.assertEqual(window.statusBar().currentMessage(), "current stream")
            self.assertEqual(len(window._live_metrics._latencies), 0)
        finally:
            window.worker = None
            window.close()


if __name__ == "__main__":
    unittest.main()
