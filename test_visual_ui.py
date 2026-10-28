"""Prompt/SAHI UI regressions with tiny images, mocked models, and no GPU jobs."""
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from engines import OpenVocabEngine
from main import AIApp
from ui import VISUAL_PROMPT_CONF, DrawableLabel
from video_worker import VideoWorker
from visual_engine import VisualPromptEngine

HAN = re.compile(r"[\u4e00-\u9fff]")


def write_exif_rotated_jpeg(path, width=120, height=60, stripe=20):
    """Store a landscape JPEG whose EXIF Orientation=6 shows it rotated 90 degrees clockwise.

    The stored pixels have a red stripe on the left; a viewer that honours EXIF
    (a phone gallery, OpenCV) shows a portrait image with the stripe on top.
    """
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :stripe] = (0, 0, 255)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    # Minimal big-endian TIFF IFD0 with one SHORT entry: 0x0112 Orientation = 6.
    tiff = (b"MM\x00*\x00\x00\x00\x08" + b"\x00\x01"
            + b"\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00" + b"\x00\x00\x00\x00")
    app1 = b"\xff\xe1" + (len(tiff) + 8).to_bytes(2, "big") + b"Exif\x00\x00" + tiff
    raw = data.tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw[:2] + app1 + raw[2:])


def red_fraction(image):
    return float(((image[..., 2] > 200) & (image[..., 1] < 60) & (image[..., 0] < 60)).mean())


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


class ReferenceCanvasTests(unittest.TestCase):
    """Tab 2 canvas: shown pixels, encoded pixels and box coordinates come from one array."""

    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.canvas = DrawableLabel(MagicMock())
        self.canvas.resize(300, 200)

    def test_exif_rotated_photo_under_non_ascii_path_maps_box_to_seen_pixels(self):
        path = self.root / "参考图片" / "手机照片_旋转.jpg"
        write_exif_rotated_jpeg(path)
        # cv2.imread cannot open non-ASCII paths on Windows; the canvas must not need it.
        with patch("cv2.imread", return_value=None) as imread:
            self.assertTrue(self.canvas.set_image(str(path)))
        imread.assert_not_called()
        image = self.canvas.raw_image
        self.assertEqual(image.shape, (120, 60, 3))  # EXIF orientation applied: portrait
        pixmap = self.canvas.original_pixmap
        self.assertEqual((pixmap.width(), pixmap.height()), (60, 120))
        # The pixmap on screen is the array handed to the encoder, pixel for pixel.
        shown = pixmap.toImage()
        for x, y in ((30, 5), (30, 100), (2, 2), (57, 117)):
            colour = shown.pixelColor(x, y)
            self.assertEqual((colour.blue(), colour.green(), colour.red()), tuple(int(v) for v in image[y, x]))
        # The user boxes the red stripe where it is shown: the top sixth of the photo,
        # drawn at 100x200 with a 100-pixel left offset on the 300x200 canvas.
        self.canvas.box = QRect(100, 0, 100, 33)
        reference, (left, top, right, bottom) = self.canvas.get_visual_prompt()
        np.testing.assert_array_equal(reference, image)
        self.assertEqual((left, top, right), (0, 0, 60))
        self.assertLessEqual(bottom, 20)
        self.assertGreater(red_fraction(reference[top:bottom, left:right]), 0.95)
        # The same pixels go to the template crop used for display.
        self.assertGreater(red_fraction(self.canvas.get_cropped_template()), 0.95)

    def test_unreadable_reference_warns_and_keeps_previous_reference(self):
        good = self.root / "参考.png"
        cv2.imencode(".png", np.full((40, 50, 3), 90, np.uint8))[1].tofile(str(good))
        self.assertTrue(self.canvas.set_image(str(good)))
        self.assertEqual(self.canvas.raw_image.shape, (40, 50, 3))
        self.canvas.box = QRect(10, 10, 50, 50)
        previous = (self.canvas.raw_image, self.canvas.original_pixmap, self.canvas.box)
        broken = self.root / "损坏.jpg"
        broken.write_bytes(b"not an image")
        with patch("ui.QMessageBox.warning") as warning:
            self.assertFalse(self.canvas.set_image(str(broken)))
            self.assertFalse(self.canvas.set_image(str(self.root / "不存在.jpg")))
        self.assertEqual(warning.call_count, 2)
        self.assertEqual(warning.call_args_list[0].args[1], "无法读取参考图片")
        self.assertIn("损坏.jpg", warning.call_args_list[0].args[2])
        self.assertIs(self.canvas.raw_image, previous[0])
        self.assertIs(self.canvas.original_pixmap, previous[1])
        self.assertEqual(self.canvas.box, previous[2])


class VisualThresholdTests(unittest.TestCase):
    """Visual prompts use the Tab 2 threshold; text / prompt-free keep the global slider."""

    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def window(self):
        window = AIApp()

        def close():
            window.worker = None
            window.annotation_panel.worker = None
            window.annotation_panel.session = None  # closing must not ask about unexported results
            window.close()

        self.addCleanup(close)
        window.ref_canvas.raw_image = np.zeros((300, 600, 3), np.uint8)
        window.ref_canvas.original_pixmap = QPixmap(600, 300)
        window.ref_canvas.resize(450, 320)
        window.ref_canvas.box = QRect(0, 47, 450, 225)
        window.ref_class_input.setText("cup")
        return window

    @staticmethod
    def start(window, method):
        workers = []

        def make(*args, **kwargs):
            workers.append(MagicMock())
            return workers[-1]

        with patch.object(window, "_ensure_detection_model", return_value=True), \
                patch("main.VideoWorker", side_effect=make):
            method()
        return workers[-1]

    def test_default_range_and_text_slider_unchanged(self):
        window = self.window()
        spin = window.visual_conf_spin
        self.assertEqual(VISUAL_PROMPT_CONF, 0.03)
        self.assertEqual(spin.value(), 0.03)
        self.assertEqual((spin.minimum(), spin.maximum(), spin.decimals()), (0.001, 1.0, 3))
        self.assertEqual(window.conf_slider.value(), 40)
        spin.setValue(0.0)
        self.assertEqual(spin.value(), 0.001)
        spin.setValue(2.0)
        self.assertEqual(spin.value(), 1.0)

    def test_visual_run_uses_tab2_threshold_and_text_run_keeps_slider(self):
        window = self.window()
        visual = self.start(window, window.start_visual_stream)
        self.assertIsInstance(window.engine, VisualPromptEngine)
        self.assertEqual(visual.conf, 0.03)
        # Moving the global slider during a visual run must not override it ...
        window.conf_slider.setValue(55)
        self.assertEqual(visual.conf, 0.03)
        # ... while the Tab 2 value applies immediately.
        window.visual_conf_spin.setValue(0.12)
        self.assertAlmostEqual(visual.conf, 0.12)
        window.prompt_input.setText("cup")
        text = self.start(window, window.start_text_stream)
        self.assertIs(type(window.engine), OpenVocabEngine)
        self.assertAlmostEqual(text.conf, 0.55)
        window.visual_conf_spin.setValue(0.2)
        self.assertAlmostEqual(text.conf, 0.55)
        window.conf_slider.setValue(40)
        self.assertAlmostEqual(text.conf, 0.40)
        # Starting visual detection again picks up the Tab 2 value.
        again = self.start(window, window.start_visual_stream)
        self.assertAlmostEqual(again.conf, 0.2)

    def test_tab4_visual_mode_records_tab2_threshold(self):
        window = self.window()
        panel = window.annotation_panel
        self.assertAlmostEqual(panel._run_options(["cup"], "visual")["conf"], 0.03)
        self.assertAlmostEqual(panel._run_options(["cup"], "text")["conf"], 0.40)
        self.assertAlmostEqual(panel._run_options([], "prompt_free")["conf"], 0.40)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name) / "待标注"
        folder.mkdir()
        for index in range(2):
            cv2.imencode(".png", np.full((40, 50, 3), index * 60, np.uint8))[1].tofile(str(folder / f"图{index}.png"))
        self.assertTrue(panel.load_folder(folder))
        panel.prompt_mode.setCurrentIndex(panel.prompt_mode.findData("visual"))
        window.visual_conf_spin.setValue(0.05)
        worker = MagicMock()
        with patch.object(window, "_ensure_detection_model", return_value=True), \
                patch("auto_annotation_ui.AutoAnnotationWorker", return_value=worker) as factory:
            panel.start()
        options = factory.call_args.args[3]
        self.assertEqual((options["mode"], options["class_names"]), ("visual", ["cup"]))
        self.assertAlmostEqual(options["conf"], 0.05)
        run = panel.session.runs[0]
        self.assertAlmostEqual(run["conf"], 0.05)
        self.assertIn('"conf": 0.05', run["config"])

    def test_threshold_control_is_translated(self):
        window = self.window()
        window.language_combo.setCurrentIndex(window.language_combo.findData("en"))
        self.qt_app.processEvents()
        tooltip = window.visual_conf_spin.toolTip()
        self.assertTrue(tooltip)
        self.assertFalse(HAN.search(tooltip), tooltip)
        labels = [label.text() for label in window.findChildren(type(window.conf_value))]
        self.assertIn("Visual-prompt threshold:", labels)


if __name__ == "__main__":
    unittest.main()
