"""CPU-only UI and recording regressions; requires numpy, OpenCV and PySide6.

Run with QT_QPA_PLATFORM=offscreen python -m unittest test_video_recording.
Models and cameras are mocked; no weights or GPU inference are used.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from video_worker import VideoWorker


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), np.uint8)
        self.output = np.full_like(self.frame, 180)
        self.engine = MagicMock()
        self.engine.process_frame.return_value = (self.output, [], "ok")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "recording.mp4"
        self.capture = MagicMock()
        self.capture.isOpened.return_value = True
        self.capture.get.return_value = 25.0
        self.capture.read.side_effect = [(True, self.frame), (False, None)]

    def worker(self, recording=True):
        # ByteTrack construction would otherwise import Ultralytics and torch.
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            return VideoWorker(
                self.engine, MagicMock(), "input.mp4", source_kind="video",
                recording_path=str(self.path) if recording else None,
            )

    def test_max_det_and_confidence_are_passed_independently(self):
        worker = self.worker(False)
        worker.conf, worker.max_det = 0.62, 17
        worker._infer(self.frame)
        self.assertEqual(self.engine.process_frame.call_args.args[1], 0.62)
        self.assertEqual(self.engine.process_frame.call_args.kwargs["max_det"], 17)

    def test_mp4_contains_processed_frames_and_preserves_source_fps(self):
        worker = self.worker()
        self.capture.read.side_effect = [(True, self.frame), (True, self.frame), (False, None)]
        errors = []
        worker.error_signal.connect(errors.append)
        with patch.object(worker, "_open_capture", return_value=self.capture):
            worker.run()
        self.assertEqual(errors, [])
        self.capture.release.assert_called_once()
        self.assertEqual(worker.recorded_frames, 2)
        recorded = cv2.VideoCapture(str(self.path))
        try:
            self.assertTrue(recorded.isOpened())
            self.assertEqual(int(recorded.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            self.assertAlmostEqual(recorded.get(cv2.CAP_PROP_FPS), 25.0, places=1)
            ok, frame = recorded.read()
            self.assertTrue(ok)
            self.assertGreater(frame.mean(), 170)
        finally:
            recorded.release()

    def test_capture_and_writer_release_when_later_inference_fails(self):
        worker, writer = self.worker(), MagicMock()
        self.capture.read.side_effect = [(True, self.frame), (True, self.frame)]
        self.engine.process_frame.side_effect = [
            (self.output, [], "ok"), RuntimeError("test inference failure"),
        ]
        errors = []
        worker.error_signal.connect(errors.append)
        with patch.object(worker, "_open_capture", return_value=self.capture), patch.object(
            worker, "_open_recording", return_value=writer
        ):
            worker.run()
        self.capture.release.assert_called_once()
        writer.release.assert_called_once()
        writer.write.assert_called_once()
        np.testing.assert_array_equal(writer.write.call_args.args[0], self.output)
        self.assertIn("test inference failure", errors[0])

    def test_stop_during_inference_finalizes_existing_recording(self):
        worker, writer = self.worker(), MagicMock()
        self.capture.read.side_effect = [(True, self.frame), (True, self.frame)]
        calls = 0

        def infer(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                worker.running = False
            return self.output, [], "ok"

        self.engine.process_frame.side_effect = infer
        with patch.object(worker, "_open_capture", return_value=self.capture), patch.object(
            worker, "_open_recording", return_value=writer
        ):
            worker.run()
        self.capture.release.assert_called_once()
        writer.release.assert_called_once()
        writer.write.assert_called_once()

    def test_unavailable_capture_is_released(self):
        worker = self.worker()
        self.capture.isOpened.return_value = False
        with patch.object(worker, "_open_capture", return_value=self.capture):
            worker.run()
        self.capture.release.assert_called_once()
        self.engine.process_frame.assert_not_called()

    def test_invalid_source_fps_falls_back_to_twenty(self):
        worker = self.worker()
        self.capture.get.return_value = float("nan")
        with patch("video_worker.cv2.VideoWriter", return_value=MagicMock()) as factory:
            writer = worker._open_recording(self.output, self.capture)
        self.assertEqual(factory.call_args.args[2], 20.0)
        writer.release()

    def test_existing_recording_is_preserved(self):
        worker = self.worker()
        self.path.write_bytes(b"existing recording")
        with patch("video_worker.cv2.VideoWriter") as factory:
            with self.assertRaises(FileExistsError):
                worker._open_recording(self.output, self.capture)
        factory.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"existing recording")

    def test_failed_writer_is_released(self):
        worker, writer = self.worker(), MagicMock()
        writer.isOpened.return_value = False
        with patch("video_worker.cv2.VideoWriter", return_value=writer):
            with self.assertRaises(OSError):
                worker._open_recording(self.output, self.capture)
        writer.release.assert_called_once()


class UIControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_default_model_size_and_independent_live_controls(self):
        from main import AIApp
        window = AIApp()
        try:
            self.assertEqual(window.MODELS[window.model_combo.currentText()], "yoloe-26s-seg.pt")
            self.assertEqual(window.resolution_combo.currentData(), 640)
            self.assertEqual(window.max_det_spin.value(), 300)
            worker = MagicMock()
            window.worker = worker
            window.max_det_spin.setValue(17)
            self.assertEqual(window.conf_slider.value(), 40)
            window.conf_slider.setValue(62)
            self.assertEqual(window.max_det_spin.value(), 17)
            self.assertEqual(worker.conf, 0.62)
            self.assertEqual(worker.max_det, 17)
        finally:
            window.worker = None
            window.close()

    def test_recording_controls_disabled_for_images_and_active_video(self):
        from main import AIApp
        window = AIApp()
        try:
            window.selected_input_kind = "image_folder"
            window._update_recording_controls()
            self.assertFalse(window.record_checkbox.isEnabled())
            window.selected_input_kind = "video"
            window._update_recording_controls()
            self.assertTrue(window.record_checkbox.isEnabled())
            window.worker = MagicMock()
            window.worker.isRunning.return_value = True
            window._update_recording_controls()
            self.assertFalse(window.record_path_button.isEnabled())
            self.assertFalse(window.record_checkbox.isEnabled())
        finally:
            window.worker = None
            window.close()

    def test_model_switch_releases_previous_model_before_loading(self):
        from engines import OpenVocabEngine
        from main import AIApp
        window = AIApp()
        previous, replacement = window.engine, OpenVocabEngine()
        previous.model, previous.processor = MagicMock(), MagicMock()

        def load(*args):
            self.assertIsNone(previous.model)
            self.assertIsNone(previous.processor)
            self.assertIs(window.engine, replacement)
            return "SUCCESS"

        try:
            with patch("main.OpenVocabEngine", return_value=replacement), patch.object(
                replacement, "load_model", side_effect=load
            ), patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.empty_cache"
            ) as empty_cache, patch("main.QMessageBox.information"):
                window.apply_settings()
            empty_cache.assert_called_once()
            self.assertEqual(replacement.imgsz, 640)
        finally:
            window.close()

    def test_failed_load_leaves_no_partial_model_for_inference(self):
        from engines import OpenVocabEngine
        from main import AIApp
        window = AIApp()
        replacement = OpenVocabEngine()

        def load(*args):
            replacement.model, replacement.processor = MagicMock(), MagicMock()
            return "failed fixture load"

        try:
            with patch("main.OpenVocabEngine", return_value=replacement), patch.object(
                replacement, "load_model", side_effect=load
            ), patch("torch.cuda.is_available", return_value=False), patch(
                "main.QMessageBox.critical"
            ) as critical:
                window.apply_settings()
            critical.assert_called_once()
            self.assertIs(window.engine, replacement)
            self.assertIsNone(window.engine.model)
            self.assertIsNone(window.engine.processor)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
