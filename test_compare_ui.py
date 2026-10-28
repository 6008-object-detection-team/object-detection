"""C4: pin-and-compare two models on one still image — geometry, dialog and PNG export."""
import gc
import os
import re
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QAbstractButton, QLabel, QLineEdit, QWidget

from compare_ui import (CompareModelsDialog, box_iou, class_counts, compose_comparison, header_strip,
                        match_boxes, open_compare_dialog)
from i18n import EN
from ui_language import UiLanguage

HAN = re.compile(r"[一-鿿]")


def box(label, xyxy, score=0.9):
    return {"label": label, "xyxy": list(xyxy), "score": score, "source": "original"}


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])  # the header is drawn with Qt

    def test_box_iou_matches_known_cases(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        # Two 10x10 boxes overlapping in a 5x10 strip: intersection 50, union 150.
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_class_counts_tally_per_label(self):
        boxes = [box("bottle", [0, 0, 1, 1]), box("bottle", [2, 2, 3, 3]), box("cup", [4, 4, 5, 5])]
        self.assertEqual(class_counts(boxes), {"bottle": 2, "cup": 1})
        self.assertEqual(class_counts([]), {})

    def test_match_boxes_pairs_same_class_above_threshold(self):
        a = [box("bottle", [0, 0, 10, 10])]
        b = [box("bottle", [1, 1, 11, 11])]  # high overlap, same class
        matched, only_a, only_b = match_boxes(a, b)
        self.assertEqual(len(matched), 1)
        self.assertEqual((only_a, only_b), ([], []))

    def test_match_boxes_flags_low_iou_as_a_difference(self):
        a = [box("bottle", [0, 0, 10, 10])]
        b = [box("bottle", [8, 8, 18, 18])]  # same class, low overlap
        matched, only_a, only_b = match_boxes(a, b, iou_threshold=0.5)
        self.assertEqual(matched, [])
        self.assertEqual(only_a, [0])
        self.assertEqual(only_b, [0])

    def test_match_boxes_flags_class_mismatch_even_at_perfect_overlap(self):
        a = [box("bottle", [0, 0, 10, 10])]
        b = [box("cup", [0, 0, 10, 10])]  # identical geometry, different class
        matched, only_a, only_b = match_boxes(a, b)
        self.assertEqual(matched, [])
        self.assertEqual((only_a, only_b), ([0], [0]))

    def test_match_boxes_does_not_reuse_a_box_for_two_matches(self):
        a = [box("bottle", [0, 0, 10, 10]), box("bottle", [0, 0, 10, 10])]
        b = [box("bottle", [0, 0, 10, 10])]
        matched, only_a, only_b = match_boxes(a, b)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(only_a), 1)
        self.assertEqual(only_b, [])

    def test_compose_comparison_returns_a_side_by_side_image_and_summary(self):
        frame = np.full((40, 60, 3), 100, np.uint8)
        result_a = {"title": "A", "boxes": [box("bottle", [0, 0, 10, 10]), box("cup", [20, 20, 30, 30])]}
        result_b = {"title": "B", "boxes": [box("bottle", [1, 1, 11, 11])]}
        composite, summary = compose_comparison(frame, result_a, result_b)
        self.assertEqual(composite.ndim, 3)
        self.assertGreater(composite.shape[1], frame.shape[1] * 2 - 1)  # two panels + a gap, side by side
        self.assertEqual(summary, {"matched": 1, "only_a": 1, "only_b": 0,
                                   "counts_a": {"bottle": 1, "cup": 1}, "counts_b": {"bottle": 1}})
        self.assertEqual(composite.shape[0], frame.shape[0] + 32)  # title strip on top
        wide = np.full((40, 300, 3), 100, np.uint8)  # room for the whole title
        chinese, _ = compose_comparison(wide, result_a, result_b, "zh")
        english, _ = compose_comparison(wide, result_a, result_b, "en")
        self.assertEqual(english.shape, chinese.shape)
        self.assertFalse(np.array_equal(english[:32], chinese[:32]))  # "| 2 boxes" vs "| 2 框"

    def test_header_draws_non_ascii_titles_instead_of_question_marks(self):
        chinese = header_strip(300, "YOLOE-26S（实时方案起点） | 3 框")
        marks = header_strip(300, "YOLOE-26S?????????? | 3 ?")
        self.assertEqual(chinese.shape, (32, 300, 3))
        self.assertEqual(chinese.dtype, np.uint8)
        self.assertGreater(int(chinese.max()), 200)  # white text on the dark strip
        self.assertFalse(np.array_equal(chinese, marks))


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def make_app(self):
        app = MagicMock()
        app.model_combo.count.return_value = 3
        titles = ["YOLOE-26S", "YOLOE-26L", "YOLOE-26X"]
        paths = ["yoloe-26s-seg.pt", "yoloe-26l-seg.pt", "yoloe-26x-seg.pt"]
        app.model_combo.itemText.side_effect = lambda i: titles[i]
        app.model_combo.itemData.side_effect = lambda i: paths[i]
        app.model_combo.currentIndex.return_value = 0
        app.selected_input_kind = "image"
        app.training_active, app.annotation_busy.return_value = False, False
        app.stop_video_stream.return_value = True
        app.conf_slider.value.return_value = 40
        app.max_det_spin.value.return_value = 300
        app.resolution_combo.currentData.return_value = 640
        app.tile_checkbox.isChecked.return_value = False
        app.sahi_checkbox.isChecked.return_value = False
        app.slice_size_combo.currentData.return_value = 512
        app.dce_checkbox.isChecked.return_value = False
        app.dark_mode_combo.currentData.return_value = True
        app.strength_slider.value.return_value = 65
        app.active_model_sha256 = "a" * 64
        return app

    def test_dialog_populates_both_combos_from_the_app_model_list(self):
        dialog = CompareModelsDialog(self.make_app())
        self.assertEqual(dialog.model_a.count(), 3)
        self.assertEqual(dialog.model_b.count(), 3)
        self.assertEqual(dialog.model_a.currentIndex(), 0)
        self.assertEqual(dialog.model_b.currentIndex(), 1)  # defaults to two different models
        dialog.close()

    def test_translation_is_complete_in_english(self):
        dialog = CompareModelsDialog(self.make_app())

        def texts():
            values = [dialog.windowTitle()]
            for widget in dialog.findChildren(QWidget):
                if isinstance(widget, (QLabel, QAbstractButton)):
                    values.append(widget.text())
                if isinstance(widget, QLineEdit):
                    values.append(widget.placeholderText())
                if hasattr(widget, "toolTip"):
                    values.append(widget.toolTip())
            return [value for value in values if value]

        for value in texts():
            if HAN.search(value):
                self.assertIn(value, EN, value)
        dialog.close()

    def test_run_comparison_requires_an_image_source(self):
        app = self.make_app()
        app.selected_input_kind = "camera"
        dialog = CompareModelsDialog(app)
        dialog.run_comparison()
        self.assertIn("单张图片", dialog.status.text())
        app._load_model_path.assert_not_called()
        dialog.close()

    def test_run_comparison_requires_two_different_models(self):
        app = self.make_app()
        dialog = CompareModelsDialog(app)
        dialog.model_b.setCurrentIndex(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))
            app.selected_input_source = str(path)
            dialog.run_comparison()
        self.assertIn("两个不同的模型", dialog.status.text())
        app._load_model_path.assert_not_called()
        dialog.close()

    def test_run_comparison_loads_releases_serially_and_shows_results(self):
        app = self.make_app()
        dialog = CompareModelsDialog(app)
        dialog.prompt_input.setText("bottle, cup")  # keep the two selected checkpoints as is
        load_order, unload_order = [], []

        def fake_load(path):
            load_order.append(path)
            app.engine = MagicMock()  # not a PromptFreeEngine instance: the comparison keeps the typed prompt
            app.engine.process_frame.return_value = (None, [], "ok")
            app.engine.unload.side_effect = lambda: unload_order.append(path)
            app.active_model_path = path
            return True

        app._load_model_path.side_effect = fake_load
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.png"
            image = np.zeros((20, 30, 3), np.uint8)
            image[2:10, 2:10] = 255
            cv2.imwrite(str(path), image)
            app.selected_input_source = str(path)
            with patch("compare_ui.detect_candidates") as detect:
                detect.side_effect = [
                    ([{"label": "bottle", "xyxy": [1., 1., 10., 10.], "score": 0.8, "source": "original"}], "ok"),
                    ([{"label": "bottle", "xyxy": [1., 1., 10., 10.], "score": 0.6, "source": "original"},
                      {"label": "cup", "xyxy": [12., 12., 18., 18.], "score": 0.5, "source": "original"}], "ok"),
                ]
                dialog.run_comparison()
        # Serial: model A is unloaded before model B is loaded.
        self.assertEqual(load_order, ["yoloe-26s-seg.pt", "yoloe-26l-seg.pt"])
        self.assertEqual(unload_order, ["yoloe-26s-seg.pt", "yoloe-26l-seg.pt"])
        self.assertIsNotNone(dialog.results)
        self.assertTrue(dialog.export_button.isEnabled())
        self.assertIn("yoloe-26s-seg.pt", dialog.status.text())  # titled by the checkpoint that ran
        self.assertEqual([result["title"] for result in dialog.results], ["yoloe-26s-seg.pt", "yoloe-26l-seg.pt"])
        self.assertFalse(dialog.label_a.pixmap().isNull())
        self.assertIn("独有", dialog.summary.text())
        dialog.close()

    def image_source(self, app):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "x.png"
        cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))
        app.selected_input_source = str(path)

    def test_entries_resolving_to_the_same_checkpoint_are_refused(self):
        app = self.make_app()
        titles = ["YOLOE-26S", "YOLOE-26S prompt-free"]
        paths = ["yoloe-26s-seg.pt", "yoloe-26s-seg-pf.pt"]
        app.model_combo.count.return_value = 2
        app.model_combo.itemText.side_effect = lambda i: titles[i]
        app.model_combo.itemData.side_effect = lambda i: paths[i]
        self.image_source(app)
        for prompt, path in (("cup", "yoloe-26s-seg.pt"), ("", "yoloe-26s-seg-pf.pt")):
            with self.subTest(prompt=prompt):
                dialog = CompareModelsDialog(app)
                dialog.prompt_input.setText(prompt)
                dialog.run_comparison()
                self.assertIn("同一个权重", dialog.status.text())
                self.assertIn(path, dialog.status.text())
                dialog.close()
        app._load_model_path.assert_not_called()
        app.stop_video_stream.assert_not_called()

    def test_dino_with_an_empty_prompt_is_refused_not_run_with_fixed_classes(self):
        app = self.make_app()
        titles = ["YOLOE-26S", "Grounding DINO Tiny"]
        paths = ["yoloe-26s-seg.pt", "IDEA-Research/grounding-dino-tiny"]
        app.model_combo.count.return_value = 2
        app.model_combo.itemText.side_effect = lambda i: titles[i]
        app.model_combo.itemData.side_effect = lambda i: paths[i]
        self.image_source(app)
        dialog = CompareModelsDialog(app)
        dialog.run_comparison()
        self.assertIn("没有对应的无提示版", dialog.status.text())
        app._load_model_path.assert_not_called()
        app.stop_video_stream.assert_not_called()
        dialog.close()

    def test_run_comparison_reports_a_load_failure_without_crashing(self):
        app = self.make_app()
        app._load_model_path.return_value = False
        dialog = CompareModelsDialog(app)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.png"
            cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))
            app.selected_input_source = str(path)
            dialog.run_comparison()
        self.assertIn("对比未完成", dialog.status.text())
        self.assertFalse(dialog.export_button.isEnabled())
        dialog.close()

    def test_export_png_writes_a_file(self):
        dialog = CompareModelsDialog(self.make_app())
        dialog.results = ({"title": "A"}, {"title": "B"})
        dialog._composite = np.full((20, 40, 3), 128, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "out.png"
            with patch("compare_ui.QFileDialog.getSaveFileName", return_value=(str(target), "")):
                dialog.export_png()
            self.assertTrue(target.is_file())
            loaded = cv2.imread(str(target))
            self.assertEqual(loaded.shape, (20, 40, 3))
        self.assertIn("已导出对比图", dialog.status.text())
        dialog.close()

    def test_export_png_without_a_result_is_a_no_op(self):
        dialog = CompareModelsDialog(self.make_app())
        with patch("compare_ui.QFileDialog.getSaveFileName") as chooser:
            dialog.export_png()
        chooser.assert_not_called()
        dialog.close()


class OpenDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_closed_dialog_is_freed_and_language_switching_still_works(self):
        owner = QWidget()
        self.addCleanup(owner.deleteLater)
        language = UiLanguage(owner)
        app = DialogTests.make_app(self)
        app.ui_language = language
        before = set(language._bindings)

        def run(dialog):
            self.assertTrue(any(key[0] is dialog for key in language._bindings))  # captured while open
            dialog.frame = np.zeros((300, 400, 3), np.uint8)
            dialog._composite = np.zeros((332, 804, 3), np.uint8)
            return 0

        with patch.object(CompareModelsDialog, "exec", run):
            dialog = weakref.ref(open_compare_dialog(app))
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        gc.collect()
        self.assertIsNone(dialog())
        self.assertEqual(set(language._bindings), before)
        language.set_language("en")  # no binding points at a deleted widget
        language.set_language("zh")

    def test_open_compare_dialog_execs_and_returns_the_dialog(self):
        app = MagicMock()
        app.model_combo.count.return_value = 0
        with patch("compare_ui.CompareModelsDialog.exec", return_value=0) as exec_mock:
            dialog = open_compare_dialog(app)
        exec_mock.assert_called_once()
        self.assertIsInstance(dialog, CompareModelsDialog)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
