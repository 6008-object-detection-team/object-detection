"""Genuine PF routing regressions; these unit tests do not run GPU inference."""
import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from engines import OpenVocabEngine, PromptFreeEngine
from main import AIApp
from prompt_modes import PROMPT_FREE_PAIRS, resolve_detection_model
from visual_engine import VisualPromptEngine
from i18n import render

PROJECT = Path(__file__).resolve().parent


def wrapper(*, vocabulary=4585, lrpc=True):
    names = {index: f"vocabulary-{index}" for index in range(vocabulary)}
    names[vocabulary - 1] = "non-COCO target"
    head = types.SimpleNamespace(one2one_cv2=object())
    if lrpc:
        head.lrpc = object()
    data = MagicMock()
    data.cpu.return_value.numpy.return_value = np.array([[1, 2, 20, 24, .8, vocabulary - 1]])
    return types.SimpleNamespace(
        names=names, task="segment", predictor=object(), overrides={}, callbacks={},
        model=types.SimpleNamespace(model=[head]), set_classes=MagicMock(),
        predict=MagicMock(return_value=[types.SimpleNamespace(
            names=names, boxes=types.SimpleNamespace(data=data))]),
    )


class RoutingTests(unittest.TestCase):
    def test_blank_routes_each_base_to_its_own_official_pf(self):
        for base, pf in PROMPT_FREE_PAIRS.items():
            self.assertEqual(resolve_detection_model(base, prompt_free=True), pf)
            self.assertEqual(resolve_detection_model(pf, prompt_free=True), pf)

    def test_text_and_visual_route_pf_back_to_original(self):
        for base, pf in PROMPT_FREE_PAIRS.items():
            self.assertEqual(resolve_detection_model(pf), base)
            self.assertEqual(resolve_detection_model(base), base)

    def test_corrected_model_is_not_silently_substituted(self):
        for name in ("feedback_data/models/demo/corrected.pt", str(PROJECT/"feedback_data/yoloe-26s-seg.pt")):
            with self.assertRaises(ValueError):
                resolve_detection_model(name, prompt_free=True)
            self.assertEqual(resolve_detection_model(name), name)

    def test_dino_is_not_presented_as_prompt_free(self):
        with self.assertRaises(ValueError):
            resolve_detection_model("IDEA-Research/grounding-dino-tiny", prompt_free=True)

    def test_all_official_pf_files_have_integrity_records(self):
        manifest = json.loads((PROJECT/"model_sources.json").read_text(encoding="utf-8"))
        records = {row["file"]: row for row in manifest["models"]}
        for name in PROMPT_FREE_PAIRS.values():
            self.assertTrue(records[name]["url"].startswith("https://github.com/ultralytics/assets/releases/download/v8.4.0/"))
            self.assertEqual(len(records[name]["sha256"]), 64)
            self.assertGreater(records[name]["size"], 10000000)
        self.assertIn("yoloe-26s-seg-pf.pt", manifest["default_models"])


class PromptFreeEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PromptFreeEngine()
        self.engine.model = wrapper()
        self.frame = np.zeros((32, 48, 3), np.uint8)
        self.guard = patch("engines.preflight_inference")
        self.preflight = self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_empty_prompts_keep_all_names_and_ids_beyond_80(self):
        names = self.engine.model.names
        for classes in (None, []):
            boxes = self.engine.detect(self.frame, class_names=classes, max_det=17)
            self.assertEqual(boxes[0].class_id, 4584)
            self.assertEqual(boxes[0].label, "non-COCO target")
            self.assertEqual(boxes[0].xyxy, (1., 2., 20., 24.))
        self.assertIs(self.engine.model.names, names)
        self.assertEqual(self.engine.vocabulary_size, 4585)
        self.engine.model.set_classes.assert_not_called()
        self.assertEqual(self.engine.current_classes, [])

    def test_uses_existing_options_and_resource_guard(self):
        self.engine.detect(self.frame, .63, .51, [], source="slice", max_det=27)
        self.preflight.assert_called_once_with("cpu", 640, self.frame.shape)
        options = self.engine.model.predict.call_args.kwargs
        self.assertEqual((options["conf"], options["iou"], options["max_det"]), (.63, .51, 27))
        self.assertTrue(options["agnostic_nms"])
        self.assertEqual(options["quantize"], 32)

    def test_rejects_text_instead_of_filtering_or_fusing_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "不接受"):
            self.engine.detect(self.frame, class_names=["person"])
        self.engine.model.predict.assert_not_called()
        self.engine.model.set_classes.assert_not_called()

    def test_reuses_box_only_backend(self):
        self.engine.model.predictor = None
        predictor = MagicMock()
        with patch("box_predictor.BoxOnlySegmentationPredictor", return_value=predictor) as create:
            self.engine.detect(self.frame, class_names=[])
        self.assertEqual(create.call_args.kwargs["overrides"]["batch"], 1)
        self.assertEqual(create.call_args.kwargs["overrides"]["task"], "segment")
        predictor.setup_model.assert_called_once()
        self.engine.model.set_classes.assert_not_called()

    def test_tracking_receives_empty_prompts_and_original_vocabulary_labels(self):
        tracker = MagicMock()
        tracker.detection_floor.return_value = .1
        tracker.update.side_effect = lambda boxes, *args, **kwargs: boxes
        _, boxes, status = self.engine.process_frame(self.frame, class_names=[], tracker=tracker)
        self.assertEqual(boxes[0].label, "non-COCO target")
        self.assertEqual(tracker.update.call_args.kwargs["class_names"], [])
        self.assertIn("Prompt-free (4585 classes)", render(status, "en"))

    def test_original_text_engine_still_conditions_requested_classes(self):
        engine = OpenVocabEngine()
        engine.model = wrapper()
        engine.detect(self.frame, class_names=["person", "blue mug"])
        engine.model.set_classes.assert_called_once_with(["person", "blue mug"])

    def test_missing_pf_weights_do_not_load_or_download_another_model(self):
        with patch("pathlib.Path.is_file", return_value=False), patch("ultralytics.YOLOE") as load:
            status = PromptFreeEngine().load_model("yoloe-26s-seg-pf.pt")
        self.assertIn("未就绪", str(status))
        load.assert_not_called()

    def test_wrong_size_is_rejected_before_deserialization(self):
        with patch("pathlib.Path.is_file", return_value=True), \
                patch("pathlib.Path.stat", return_value=types.SimpleNamespace(st_size=123)), \
                patch("ultralytics.YOLOE") as load:
            status = PromptFreeEngine().load_model("yoloe-26s-seg-pf.pt")
        self.assertIn("校验失败", str(status))
        load.assert_not_called()

    def verified_load(self, *, vocabulary=4585, lrpc=True, digest_valid=True):
        row = next(row for row in json.loads((PROJECT/"model_sources.json").read_text(encoding="utf-8"))["models"]
                   if row["file"] == "yoloe-26s-seg-pf.pt")
        engine = PromptFreeEngine()
        def load_model(*args):
            engine.model = wrapper(vocabulary=vocabulary, lrpc=lrpc)
            return "SUCCESS"
        with patch("pathlib.Path.is_file", return_value=True), \
                patch("pathlib.Path.stat", return_value=types.SimpleNamespace(st_size=row["size"])), \
                patch("tools.fetch_models.sha256", return_value=row["sha256"] if digest_valid else "0"*64), \
                patch("engines.OpenVocabEngine.load_model", side_effect=load_model) as base_load:
            status = engine.load_model("yoloe-26s-seg-pf.pt")
            return engine, status, base_load.call_count

    def test_bad_sha256_is_rejected_before_model_load(self):
        _, status, loads = self.verified_load(digest_valid=False)
        self.assertIn("校验失败", str(status))
        self.assertEqual(loads, 0)

    def test_only_complete_lrpc_weights_are_accepted(self):
        engine, status, loads = self.verified_load()
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(loads, 1)
        self.assertEqual(engine.vocabulary_size, 4585)

    def test_80_class_or_non_lrpc_weights_are_not_a_prompt_free_substitute(self):
        for vocabulary, lrpc in ((80, True), (4585, False)):
            engine, status, _ = self.verified_load(vocabulary=vocabulary, lrpc=lrpc)
            self.assertNotEqual(status, "SUCCESS")
            self.assertIsNone(engine.model)


class PromptFreeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = AIApp()
        self.window.device_combo.setCurrentText("CPU")
        self.window.selected_input_kind = "image"
        self.window.selected_input_source = str(PROJECT/"datasets/mini_eval/images/000000066771.jpg")

    def tearDown(self):
        self.window.worker = None
        self.window.engine.model = None
        self.window.feedback_panel.dirty = False
        self.window.close()
        self.window.deleteLater()

    def simulated_load(self, path):
        engine = PromptFreeEngine() if path in PROMPT_FREE_PAIRS.values() else OpenVocabEngine()
        engine.model = wrapper()
        self.window.engine = engine
        self.window.active_model_path = path
        self.window.model_combo.setCurrentIndex(self.window.model_combo.findData(path))
        self.window._update_model_indicator()
        return True

    def test_blank_start_uses_pf_and_no_popup_with_full_snapshot_metadata(self):
        window = self.window
        self.assertEqual(window.prompt_input.text(), "")
        with patch.object(window, "_load_model_path", side_effect=self.simulated_load) as load, \
                patch("main.VideoWorker") as worker, patch("main.QMessageBox.warning") as warning, \
                patch("main.QMessageBox.information") as info, patch("main.QMessageBox.critical") as critical:
            window.start_text_stream()
        load.assert_called_once_with("yoloe-26s-seg-pf.pt")
        worker.return_value.start.assert_called_once()
        self.assertEqual(worker.return_value.class_names, [])
        self.assertEqual(worker.return_value.feedback_metadata["detection_mode"], "prompt_free")
        self.assertEqual(worker.return_value.feedback_metadata["vocabulary_size"], 4585)
        self.assertEqual(window.active_detection_request["prompts"], [])
        for popup in (warning, info, critical):
            popup.assert_not_called()

    def test_typing_text_returns_pf_to_matching_original_model(self):
        self.simulated_load("yoloe-26x-seg-pf.pt")
        self.window.prompt_input.setText("person, red backpack")
        with patch.object(self.window, "_load_model_path", side_effect=self.simulated_load) as load, patch("main.VideoWorker") as worker:
            self.window.start_text_stream()
        load.assert_called_once_with("yoloe-26x-seg.pt")
        self.assertEqual(worker.return_value.class_names, ["person", "red backpack"])
        self.assertEqual(worker.return_value.feedback_metadata["detection_mode"], "text")

    def test_visual_reference_returns_pf_to_original_and_never_transfers_pf_head(self):
        window = self.window
        self.simulated_load("yoloe-26l-seg-pf.pt")
        reference = np.zeros((32, 48, 3), np.uint8)
        with patch.object(window.ref_canvas, "get_visual_prompt", return_value=(reference, (1, 2, 20, 24))), \
                patch.object(window, "_load_model_path", side_effect=self.simulated_load) as load, patch("main.VideoWorker") as worker:
            window.start_visual_stream()
        load.assert_called_once_with("yoloe-26l-seg.pt")
        self.assertIsInstance(window.engine, VisualPromptEngine)
        self.assertIs(worker.return_value.visual_reference, reference)
        self.assertEqual(worker.return_value.feedback_metadata["detection_mode"], "visual")

    def test_direct_pf_to_visual_transfer_is_rejected(self):
        self.simulated_load("yoloe-26s-seg-pf.pt")
        with self.assertRaises(ValueError):
            self.window._switch_prompt_engine(True)
        self.assertIsInstance(self.window.engine, PromptFreeEngine)

    def test_visual_navigation_cannot_load_dino_as_a_visual_model(self):
        window = self.window
        window.model_combo.setCurrentIndex(window.model_combo.findData("IDEA-Research/grounding-dino-tiny"))
        with patch.object(window, "_load_model_path") as load:
            self.assertFalse(window._ensure_detection_model(["cup"], visual=True))
        load.assert_not_called()

    def test_existing_corrected_model_keeps_weights_and_explains_blank_in_page(self):
        window = self.window
        path = "feedback_data/models/demo/corrected.pt"
        window.model_combo.addItem("corrected demo", path)
        window.model_combo.setCurrentIndex(window.model_combo.findData(path))
        window.active_model_path = path
        window.engine.model = model = wrapper()
        with patch.object(window, "_load_model_path") as load, patch("main.QMessageBox.warning") as popup, patch("main.VideoWorker") as worker:
            window.start_text_stream()
        load.assert_not_called()
        worker.assert_not_called()
        popup.assert_not_called()
        self.assertIs(window.engine.model, model)
        self.assertEqual(window.active_model_path, path)
        self.assertIn("修改版", window.prompt_mode_hint.text())

    def test_changing_prompt_mode_stops_before_mutating_worker_classes(self):
        window = self.window
        self.simulated_load("yoloe-26s-seg-pf.pt")
        worker = window.worker = MagicMock()
        worker.class_names = []
        window.current_video_label = window.video_label_tab1
        with patch.object(window, "stop_video_stream", return_value=True) as stop:
            window.update_worker_classes("person")
        stop.assert_called_once()
        self.assertEqual(worker.class_names, [])

    def test_clearing_text_does_not_use_legacy_seven_class_fallback(self):
        window = self.window
        self.simulated_load("yoloe-26s-seg.pt")
        worker = window.worker = MagicMock()
        worker.class_names = ["person"]
        window.current_video_label = window.video_label_tab1
        with patch.object(window, "stop_video_stream", return_value=False) as stop:
            window.update_worker_classes("")
        stop.assert_called_once()
        self.assertEqual(worker.class_names, ["person"])

    def test_model_and_mode_labels_translate_without_a_model_reload(self):
        window = self.window
        self.simulated_load("yoloe-26s-seg-pf.pt")
        model = window.engine.model
        with patch.object(window, "_load_model_path") as load:
            window.ui_language.set_language("en")
        self.assertIn("Prompt-free (4585 classes)", window.detection_mode_label.text())
        self.assertIn("prompt-free", window.model_combo.currentText())
        self.assertEqual(window.model_combo.currentData(), "yoloe-26s-seg-pf.pt")
        self.assertIs(window.engine.model, model)
        load.assert_not_called()

    def test_switch_releases_old_model_before_loading_pf_without_success_popup(self):
        window = self.window
        previous = window.engine
        previous.model = object()

        def load(replacement, path, device):
            self.assertIsNone(previous.model)
            self.assertIs(window.engine, replacement)
            replacement.model = wrapper()
            return "SUCCESS"

        with patch.object(PromptFreeEngine, "load_model", autospec=True, side_effect=load), \
                patch("torch.cuda.is_available", return_value=False), patch("main.QMessageBox.information") as popup:
            self.assertTrue(window._load_model_path("yoloe-26s-seg-pf.pt"))
        self.assertEqual(window.active_model_path, "yoloe-26s-seg-pf.pt")
        popup.assert_not_called()

    def test_stuck_worker_prevents_mode_reload(self):
        with patch.object(self.window, "_stop_input", return_value=False), \
                patch.object(self.window, "_load_model_path") as load:
            self.window.start_text_stream()
        load.assert_not_called()

    def test_folder_navigation_preserves_empty_prompt_and_source_position(self):
        window = self.window
        window.image_paths = ["one.jpg", "two.jpg"]
        window.image_index, window.image_folder = 0, "images"
        window.active_detection_request = {"target_label": window.video_label_tab1, "prompts": [], "template": None}
        with patch.object(window, "_start_stream") as start:
            window._move_image(1)
        self.assertEqual(start.call_args.args[1], [])
        self.assertEqual(window.image_index, 1)

    def test_pf_source_corrections_train_matching_promptable_family(self):
        self.simulated_load("yoloe-26s-seg-pf.pt")
        panel = self.window.feedback_panel
        panel.exported = PROJECT/"datasets/mini_train_probe"
        with patch("feedback_ui.QProcess") as process, patch("feedback_ui.QMessageBox.question", return_value=0) as confirmation:
            panel.start_training()
        process.assert_not_called()
        confirmation.assert_called_once()
        self.assertIn("YOLOE-26S", str(confirmation.call_args.args[2]))


if __name__ == "__main__":
    unittest.main()
