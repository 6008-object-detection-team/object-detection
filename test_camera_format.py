"""Opt-in MJPG camera format: property order, read-back, fallback and wiring.

Cameras and models are fake; no frames are captured, saved or shown.
"""
import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from i18n import render
from video_worker import (CAMERA_FORMATS, MJPG_DRIVER_BUFFERS, VideoWorker, camera_status,
                          configure_camera, fourcc_text)
from tools import benchmark_realtime as bench

MJPG = cv2.VideoWriter_fourcc(*"MJPG")
YUYV = cv2.VideoWriter_fourcc(*"YUYV")


class FakeCamera:
    """Records requests in order; negotiates like a V4L2 webcam with or without MJPG."""

    def __init__(self, supports_mjpg=True):
        self.supports_mjpg = supports_mjpg
        self.requests = []
        self.values = {cv2.CAP_PROP_FOURCC: float(YUYV), cv2.CAP_PROP_FPS: 10.0,
                       cv2.CAP_PROP_FRAME_WIDTH: 640.0, cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                       cv2.CAP_PROP_BUFFERSIZE: 4.0}
        self.reads = 0
        self.released = False

    def set(self, prop, value):
        self.requests.append((prop, value))
        if prop == cv2.CAP_PROP_FOURCC and not self.supports_mjpg:
            return False
        self.values[prop] = float(value)
        if prop == cv2.CAP_PROP_FOURCC:
            self.values[cv2.CAP_PROP_FPS] = 30.0
        return True

    def get(self, prop):
        return self.values.get(prop, 0.0)

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if self.reads > 2:
            return False, None
        return True, np.full((8, 8, 3), self.reads, np.uint8)

    def release(self):
        self.released = True


class ConfigureCameraTests(unittest.TestCase):
    def test_default_keeps_the_original_requests_in_order(self):
        camera = FakeCamera()
        info = configure_camera(camera, "default")
        self.assertEqual(camera.requests, [(cv2.CAP_PROP_BUFFERSIZE, 1), (cv2.CAP_PROP_FRAME_WIDTH, 1280),
                                           (cv2.CAP_PROP_FRAME_HEIGHT, 720)])
        self.assertEqual(info["fourcc"], "YUYV")
        self.assertEqual((info["width"], info["height"], info["fps"], info["buffers"]), (1280, 720, 10, 1))
        self.assertFalse(info["mjpg_refused"])

    def test_mjpg_is_requested_before_the_size_with_a_multi_buffer_queue(self):
        camera = FakeCamera()
        info = configure_camera(camera, "mjpg")
        self.assertEqual(camera.requests, [(cv2.CAP_PROP_FOURCC, MJPG),
                                           (cv2.CAP_PROP_BUFFERSIZE, MJPG_DRIVER_BUFFERS),
                                           (cv2.CAP_PROP_FRAME_WIDTH, 1280), (cv2.CAP_PROP_FRAME_HEIGHT, 720)])
        self.assertGreater(MJPG_DRIVER_BUFFERS, 1)
        self.assertEqual((info["requested"], info["fourcc"], info["fps"], info["buffers"]), ("mjpg", "MJPG", 30, 4))
        self.assertFalse(info["mjpg_refused"])
        self.assertEqual(render(camera_status(info), "zh"), "摄像头 MJPG 1280×720 30 FPS，缓冲 4")
        self.assertEqual(render(camera_status(info), "en"), "Camera MJPG 1280×720 30 FPS, 4 buffer(s)")

    def test_mjpg_lost_when_the_size_is_set_restores_the_default_buffer(self):
        camera = FakeCamera()
        original = camera.set

        def switching_set(prop, value):
            result = original(prop, value)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:  # no MJPG mode at 1280x720 on this driver
                camera.values[cv2.CAP_PROP_FOURCC] = float(YUYV)
            return result

        camera.set = switching_set
        info = configure_camera(camera, "mjpg")
        self.assertEqual(camera.requests, [(cv2.CAP_PROP_FOURCC, MJPG),
                                           (cv2.CAP_PROP_BUFFERSIZE, MJPG_DRIVER_BUFFERS),
                                           (cv2.CAP_PROP_FRAME_WIDTH, 1280), (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                                           (cv2.CAP_PROP_BUFFERSIZE, 1)])
        self.assertTrue(info["mjpg_refused"])
        self.assertEqual((info["fourcc"], info["buffers"]), ("YUYV", 1))

    def test_refused_mjpg_falls_back_to_the_default_requests_visibly(self):
        camera = FakeCamera(supports_mjpg=False)
        info = configure_camera(camera, "mjpg")
        self.assertEqual(camera.requests[1:], [(cv2.CAP_PROP_BUFFERSIZE, 1), (cv2.CAP_PROP_FRAME_WIDTH, 1280),
                                               (cv2.CAP_PROP_FRAME_HEIGHT, 720)])
        self.assertTrue(info["mjpg_refused"])
        self.assertEqual(info["fourcc"], "YUYV")
        self.assertIn("未接受 MJPG", render(camera_status(info), "zh"))
        self.assertIn("refused MJPG", render(camera_status(info), "en"))

    def test_unknown_format_and_unreadable_properties(self):
        with self.assertRaises(ValueError):
            configure_camera(FakeCamera(), "h264")
        with self.assertRaises(ValueError):
            VideoWorker(MagicMock(), None, 0, source_kind="camera", camera_format="h264")
        self.assertEqual(set(CAMERA_FORMATS), {"default", "mjpg"})
        for value in (0, 0.0, None, float("nan"), 25.0, "x"):
            self.assertIsNone(fourcc_text(value))
        self.assertEqual(fourcc_text(float(MJPG)), "MJPG")
        info = configure_camera(MagicMock(**{"get.return_value": float("nan")}), "default")
        self.assertIsNone(info["fps"])
        self.assertEqual(render(camera_status(info), "zh"), "摄像头 ? ?×? ? FPS，缓冲 ?")


class CameraWorkerFormatTests(unittest.TestCase):
    def run_worker(self, camera_format, camera):
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            worker = VideoWorker(MagicMock(), None, 0, source_kind="camera", camera_format=camera_format)
        statuses = []
        worker.status_signal.connect(statuses.append)
        with patch.object(worker, "_open_capture", return_value=camera), \
                patch.object(worker, "_infer", side_effect=lambda frame: (frame.copy(), [], "ok")):
            worker.run()
        return worker, statuses

    def test_run_applies_the_setting_and_reports_the_negotiated_format(self):
        worker, statuses = self.run_worker("mjpg", FakeCamera())
        self.assertEqual(worker.camera_info["fourcc"], "MJPG")
        self.assertIn("摄像头 MJPG 1280×720 30 FPS", render(statuses[0], "zh"))
        packet = worker.take_live_frame()
        self.assertIn("ok | ", render(packet.status, "zh"))
        self.assertIn("Camera MJPG 1280×720 30 FPS", render(packet.status, "en"))

    def test_default_run_sends_no_pixel_format_request(self):
        camera = FakeCamera()
        worker, _ = self.run_worker("default", camera)
        self.assertNotIn(cv2.CAP_PROP_FOURCC, [prop for prop, _ in camera.requests])
        self.assertEqual(worker.camera_info["buffers"], 1)
        self.assertIn("摄像头 YUYV", render(worker.take_live_frame().status, "zh"))

    def test_files_and_images_ignore_the_camera_setting(self):
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            worker = VideoWorker(MagicMock(), None, "clip.mp4", source_kind="video", camera_format="mjpg")
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch.object(worker, "_open_capture", return_value=capture):
            worker.run()
        capture.set.assert_not_called()
        self.assertIsNone(worker.camera_info)


class CameraFormatSettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_combo_defaults_to_the_original_behaviour_and_is_passed_to_the_worker(self):
        from main import AIApp
        window = AIApp()
        self.addCleanup(window.close)
        self.assertEqual(window.camera_format_combo.currentData(), "default")
        self.assertEqual([window.camera_format_combo.itemData(index) for index in range(2)], ["default", "mjpg"])
        window.camera_format_combo.setCurrentIndex(window.camera_format_combo.findData("mjpg"))
        with patch.object(window, "_ensure_detection_model", return_value=True), \
                patch.object(window, "_switch_prompt_engine"), patch("main.VideoWorker") as worker:
            window._start_stream(window.video_label_tab1, ["cup"])
        self.assertEqual(worker.call_args.kwargs["camera_format"], "mjpg")
        window.worker = None
        window.ui_language.set_language("en")
        self.assertEqual(window.camera_format_combo.itemText(1), "MJPG (4 buffers, camera must support it)")


class BenchmarkCameraFormatTests(unittest.TestCase):
    def test_option_is_camera_only_and_exclusive_with_the_raw_overrides(self):
        parser = bench.build_parser()
        for argv in (["--source", "video", "--device", "cpu", "--camera-format", "mjpg"],
                     ["--source", "camera", "--device", "cpu", "--camera-format", "mjpg", "--camera-fourcc", "MJPG"],
                     ["--source", "camera", "--device", "cpu", "--camera-format", "mjpg", "--camera-buffers", "4"]):
            with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.validate_args(parser, parser.parse_args(argv))
        args = parser.parse_args(["--source", "camera", "--device", "cpu", "--camera-format", "mjpg"])
        bench.validate_args(parser, args)
        self.assertEqual(bench.RunSettings(classes=["cup"]).camera_format, "default")

    def test_run_once_drives_the_worker_setting_and_records_the_fallback(self):
        engine = MagicMock()
        engine.process_frame.side_effect = lambda frame, *args, **kwargs: (frame.copy(), [], "ok")
        clip = [np.full((16, 16, 3), index, np.uint8) for index in range(4)]
        for frame in clip:
            frame.flags.writeable = False
        capture = bench.PacedReplayCapture(clip, 100.0, duration=1.5)
        settings = bench.RunSettings(classes=["cup"], settle_s=0.0, camera_format="mjpg")
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            raw = bench.run_once(engine, settings, "latest", capture, 0.3)
        # The replay source accepts no FOURCC, so the app's default requests apply.
        self.assertTrue(raw["camera_negotiated"]["mjpg_refused"])
        self.assertEqual(capture.property_requests[0], [cv2.CAP_PROP_FOURCC, float(MJPG)])
        self.assertEqual(capture.property_requests[1], [cv2.CAP_PROP_BUFFERSIZE, 1.0])


if __name__ == "__main__":
    unittest.main()
