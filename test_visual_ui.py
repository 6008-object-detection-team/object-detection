"""Prompt/SAHI UI regressions with tiny images, mocked models, and no GPU jobs."""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from main import AIApp
from ui import DrawableLabel
from video_worker import VideoWorker
from visual_engine import VisualPromptEngine


class VisualWorkerTests(unittest.TestCase):
    def worker(self):
        engine = MagicMock()
        engine.set_visual_prompt.return_value = "SUCCESS"
        engine.process_frame.return_value = (np.zeros((8, 8, 3), np.uint8), [], "ok")
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            worker = VideoWorker(engine, MagicMock(), "test.mp4", source_kind="video")
        worker.visual_reference = np.zeros((32, 48, 3), np.uint8)
        worker.visual_box = (10, 4, 30, 28)
        worker.visual_label = "cup"
        return worker, engine

    def test_reference_encoded_once_without_colour_filtering_or_text(self):
        worker, engine = self.worker()
        worker.visual_template = np.full((4, 4, 3), 127, np.uint8)
        worker.class_names = ["unrelated text"]
        frame = np.zeros((24, 36, 3), np.uint8)
        worker._infer(frame)
        worker._infer(frame)
        engine.set_visual_prompt.assert_called_once()
        self.assertIs(engine.set_visual_prompt.call_args.args[0], worker.visual_reference)
        self.assertEqual(engine.set_visual_prompt.call_args.args[1:], ((10, 4, 30, 28), "cup"))
        self.assertEqual(engine.process_frame.call_args.args[3], ["cup"])
        self.assertIsNone(engine.process_frame.call_args.args[4])

    def test_sahi_controls_pass_separate_bounded_options(self):
        worker, engine = self.worker()
        worker.sliced, worker.slice_size = True, 640
        worker.conf, worker.max_det = 0.62, 17
        worker._infer(np.zeros((8, 8, 3), np.uint8))
        options = engine.process_frame.call_args.kwargs
        self.assertTrue(options["sliced"])
        self.assertEqual(options["slice_size"], 640)
        self.assertEqual(options["max_slices"], 9)
        self.assertEqual(options["slice_overlap"], 0.2)
        self.assertEqual(options["max_det"], 17)
        self.assertEqual(engine.process_frame.call_args.args[1], 0.62)

    def test_failed_reference_stops_before_camera_or_recording(self):
        worker, engine = self.worker()
        engine.set_visual_prompt.side_effect = RuntimeError("memory guard")
        errors = []
        worker.error_signal.connect(errors.append)
        with patch.object(worker, "_open_capture") as capture:
            worker.run()
        capture.assert_not_called()
        engine.process_frame.assert_not_called()
        self.assertIn("memory guard", errors[0])


class VisualUIControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def window(self):
        window = AIApp()
        self.addCleanup(window.close)
        return window

    def test_whole_reference_box_uses_original_coordinates_and_clips_letterbox(self):
        canvas = DrawableLabel(MagicMock())
        canvas.resize(450, 320)
        canvas.raw_image = np.zeros((300, 600, 3), np.uint8)
        canvas.original_pixmap = QPixmap(600, 300)
        # Image is rendered at 450x225, with 47 pixels of padding above it.
        canvas.box = QRect(0, 47, 450, 225)
        image, box = canvas.get_visual_prompt()
        self.assertEqual(image.shape, (300, 600, 3))
        self.assertEqual(box, (0, 0, 600, 300))
        self.assertIsNot(image, canvas.raw_image)
        canvas.box = QRect(0, 0, 400, 20)
        self.assertIsNone(canvas.get_visual_prompt())
        canvas.box = QRect(0, 0, 450, 100)
        _, clipped = canvas.get_visual_prompt()
        self.assertEqual(clipped[1], 0)
        self.assertLess(clipped[3], 100)

    def test_default_sahi_off_and_old_tiles_are_mutually_exclusive(self):
        window = self.window()
        self.assertFalse(window.sahi_checkbox.isChecked())
        self.assertEqual(window.slice_size_combo.currentData(), 512)
        self.assertNotEqual(window.slice_size_combo.findData(320), -1)
        self.assertEqual(window.prompt_input.text(), "")
        window.tile_checkbox.setChecked(True)
        window.sahi_checkbox.setChecked(True)
        self.assertFalse(window.tile_checkbox.isChecked())
        self.assertTrue(window.sahi_checkbox.isChecked())
        window.tile_checkbox.setChecked(True)
        self.assertTrue(window.tile_checkbox.isChecked())
        self.assertFalse(window.sahi_checkbox.isChecked())

    def test_text_to_visual_transfers_one_model_without_reload(self):
        window = self.window()
        original = window.engine
        model = original.model = MagicMock()
        original.current_classes = ["cup"]
        original.resource_marker = marker = object()
        with patch.object(VisualPromptEngine, "load_model") as load:
            window._switch_prompt_engine(True)
        load.assert_not_called()
        self.assertIsInstance(window.engine, VisualPromptEngine)
        self.assertIs(window.engine.model, model)
        self.assertIsNone(original.model)
        self.assertIs(window.engine.resource_marker, marker)
        self.assertEqual(window.engine.current_classes, [])

    def test_visual_to_text_clears_prompt_before_reusing_same_model(self):
        window = self.window()
        original = VisualPromptEngine()
        original.model = model = MagicMock()
        window.engine = original
        with patch.object(original, "release_visual_prompt") as release:
            window._switch_prompt_engine(False)
        release.assert_called_once()
        self.assertNotIsInstance(window.engine, VisualPromptEngine)
        self.assertIsNone(original.model)
        self.assertIs(window.engine.model, model)
        self.assertEqual(window.engine.current_classes, [])

    def test_image_navigation_preserves_reference_image_box_and_label(self):
        window = self.window()
        reference = np.zeros((16, 24, 3), np.uint8)
        window.image_paths, window.image_index, window.image_folder = ["one.png", "two.png"], 0, "images"
        window.active_detection_request = {
            "target_label": window.video_label_tab2, "prompts": ["cup"], "template": None,
            "visual_reference": reference, "visual_box": (2, 2, 12, 12), "visual_label": "cup",
        }
        with patch.object(window, "_start_stream") as start:
            window._move_image(1)
        self.assertEqual(window.image_index, 1)
        self.assertIs(start.call_args.kwargs["visual_reference"], reference)
        self.assertEqual(start.call_args.kwargs["visual_box"], (2, 2, 12, 12))
        self.assertEqual(start.call_args.kwargs["visual_label"], "cup")

    def test_text_edits_do_not_overwrite_active_visual_class(self):
        window = self.window()
        window.worker = worker = MagicMock()
        worker.class_names = ["cup"]
        window.current_video_label = window.video_label_tab2
        try:
            window.update_worker_classes("bottle, cell phone")
            self.assertEqual(worker.class_names, ["cup"])
        finally:
            window.worker = None

    def test_dino_visual_button_shows_model_requirement(self):
        from engines import GroundingDINOEngine
        window = self.window()
        window.engine = GroundingDINOEngine()
        with patch("main.QMessageBox.warning") as warning, patch.object(window, "_start_stream") as start:
            window.start_visual_stream()
        start.assert_not_called()
        self.assertIn("YOLOE", warning.call_args.args[2])

    def test_release_waits_for_worker_before_unloading(self):
        window = self.window()
        with patch.object(window, "stop_video_stream", return_value=False), patch.object(window.engine, "unload") as unload:
            self.assertFalse(window.release_model())
        unload.assert_not_called()
        with patch.object(window, "stop_video_stream", return_value=True), patch.object(window.engine, "unload") as unload:
            self.assertTrue(window.release_model())
        unload.assert_called_once()

    def test_exception_during_load_releases_partial_model(self):
        from engines import OpenVocabEngine
        window = self.window()
        replacement = OpenVocabEngine()

        def load(*args):
            replacement.model = MagicMock()
            raise RuntimeError("load failed after model assignment")

        with patch("main.OpenVocabEngine", return_value=replacement), patch.object(
            replacement, "load_model", side_effect=load
        ), patch("torch.cuda.is_available", return_value=False), patch("main.QMessageBox.critical"):
            window.apply_settings()
        self.assertIsNone(window.engine.model)


if __name__ == "__main__":
    unittest.main()
