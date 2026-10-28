"""Detection-branch-only loading: exact-copy checks, visible fallback and app wiring.

Tiny torch modules stand in for YOLOE; no checkpoint is read and no GPU is used.
The real-weight checks (byte-identical predictions on mini_eval and the
240-image COCO subset, timing) are run separately and documented in README.
"""
import os
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from torch import nn
from PySide6.QtWidgets import QApplication

import detection_branch
from detection_branch import detection_yaml, verify_detection_branch
from engines import OpenVocabEngine, PromptFreeEngine
from i18n import render


class Head(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.cv2, self.cv3 = nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1)
        self.one2one_cv2 = nn.Conv2d(4, 4, 1)
        self.nc, self.nl, self.reg_max, self.end2end, self.max_det = 80, 3, 1, False, 300
        if seg:
            self.cv5, self.one2one_cv5, self.proto = nn.Conv2d(4, 2, 1), nn.Conv2d(4, 2, 1), nn.Conv2d(4, 8, 1)


class TinyYoloe(nn.Module):
    def __init__(self, seg=True, text_model="mobileclip2:b"):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), Head(seg)])
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.text_model = text_model
        self.yaml = {"yaml_file": "yoloe-26s-seg.yaml" if seg else "yoloe-26s.yaml"}
        self.names = {0: "person"}


def pair():
    seg = TinyYoloe(seg=True)
    det = TinyYoloe(seg=False)
    det.load_state_dict({key: value for key, value in seg.state_dict().items() if key in det.state_dict()})
    return seg, det


class VerifyTests(unittest.TestCase):
    def test_exact_copy_without_mask_branch_is_accepted(self):
        seg, det = pair()
        record = verify_detection_branch(seg, det)
        self.assertEqual(record["tensors_copied_exactly"], len(det.state_dict()))
        self.assertEqual(record["mask_branch_tensors_removed"], 6)

    def test_any_changed_tensor_is_rejected(self):
        seg, det = pair()
        with torch.no_grad():
            det.model[1].running_mean[0] += 1e-6
        with self.assertRaisesRegex(ValueError, "1 个张量"):
            verify_detection_branch(seg, det)

    def test_left_over_tensors_outside_the_mask_branch_are_rejected(self):
        seg, det = pair()
        det.model[2].one2one_cv2 = None  # the detection graph lost a real branch
        with self.assertRaisesRegex(ValueError, "不只是掩膜分支"):
            verify_detection_branch(seg, det)
        seg, det = pair()
        with self.assertRaises(ValueError):
            verify_detection_branch(det, det)  # nothing removed: not a seg checkpoint

    def test_text_encoder_stride_head_and_prompt_free_mismatches(self):
        for change, pattern in [(lambda det: setattr(det, "text_model", "mobileclip:blt"), "文字编码器"),
                                (lambda det: setattr(det, "stride", torch.tensor([8.0, 16.0, 64.0])), "步长"),
                                (lambda det: setattr(det.model[2], "reg_max", 16), "reg_max")]:
            seg, det = pair()
            change(det)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                verify_detection_branch(seg, det)
        seg, det = pair()
        seg.model[2].lrpc = nn.ModuleList()
        with self.assertRaisesRegex(ValueError, "LRPC"):
            verify_detection_branch(seg, det)

    def test_yaml_mapping(self):
        self.assertEqual(detection_yaml(types.SimpleNamespace(yaml={"yaml_file": "yoloe-26l-seg.yaml"})), "yoloe-26l.yaml")
        self.assertEqual(detection_yaml(types.SimpleNamespace(yaml={"yaml_file": "/x/yoloe-11l-seg.yaml"})), "yoloe-11l.yaml")
        for yaml in ({}, {"yaml_file": "yoloe-26s.yaml"}, {"yaml_file": "yoloe-26s-seg-pf.yaml"}):
            with self.subTest(yaml=yaml), self.assertRaises(ValueError):
                detection_yaml(types.SimpleNamespace(yaml=yaml))


class ConversionBuildTests(unittest.TestCase):
    def test_yaml_build_runs_on_one_thread_and_restores_the_count(self):
        seg = types.SimpleNamespace(task="segment", overrides={},
                                    model=types.SimpleNamespace(yaml={"yaml_file": "yoloe-26s-seg.yaml"}))
        before, seen = torch.get_num_threads(), []

        def build(name, verbose=False):
            seen.append((name, torch.get_num_threads()))
            raise RuntimeError("stop after the build")

        with patch("ultralytics.YOLOE", side_effect=build), self.assertRaisesRegex(RuntimeError, "stop"):
            detection_branch.convert_to_detection_branch(seg)
        self.assertEqual(seen, [("yoloe-26s.yaml", 1)])
        self.assertEqual(torch.get_num_threads(), before)

    def test_only_segmentation_checkpoints_are_converted(self):
        with patch("ultralytics.YOLOE") as build, self.assertRaisesRegex(ValueError, "分割权重"):
            detection_branch.convert_to_detection_branch(types.SimpleNamespace(task="detect"))
        build.assert_not_called()


class FakeWrapper:
    """Just enough of ultralytics.YOLOE for engines.load_model."""

    def __init__(self, task):
        self.task, self.predictor, self.overrides = task, None, {}
        self.model = MagicMock()
        self.model.model = [MagicMock(one2one_cv2=object())]
        self.to = MagicMock(return_value=self)


class EngineGraphTests(unittest.TestCase):
    def load(self, graph, convert):
        seg = FakeWrapper("segment")
        engine = OpenVocabEngine()
        with patch("engines.prepare_device", return_value="cuda:0"), \
                patch("ultralytics.YOLOE", return_value=seg), \
                patch("detection_branch.convert_to_detection_branch", side_effect=convert) as conversion:
            status = engine.load_model("unused-test.pt", "CUDA", graph=graph)
        return engine, seg, status, conversion

    def test_det_is_converted_on_the_cpu_before_the_device_move(self):
        det = FakeWrapper("detect")
        record = {"tensors_copied_exactly": 822, "mask_branch_tensors_removed": 136}
        engine, seg, status, conversion = self.load("det", lambda wrapper: (det, dict(record)))
        self.assertEqual(status, "SUCCESS")
        conversion.assert_called_once_with(seg)
        self.assertIs(engine.model, det)
        det.to.assert_called_once_with("cuda:0")
        seg.to.assert_not_called()
        self.assertEqual(engine.model_graph["graph"], "det")
        self.assertEqual(engine.model_graph["transfer"], record)
        self.assertEqual(engine.inference_options["graph"], "det")
        self.assertEqual(engine.current_classes, [])

    def test_failed_check_keeps_the_full_model_and_records_why(self):
        def refuse(wrapper):
            raise ValueError(detection_branch.message("检测结构的步长与原权重不同"))

        engine, seg, status, _ = self.load("det", refuse)
        self.assertEqual(status, "SUCCESS")
        self.assertIs(engine.model, seg)
        seg.to.assert_called_once_with("cuda:0")
        self.assertEqual(engine.model_graph["graph"], "seg")
        self.assertEqual(engine.model_graph["requested"], "det")
        self.assertIn("different strides", render(engine.model_graph["fallback"], "en"))
        self.assertEqual(engine.inference_options["graph"], "seg")

    def test_seg_request_and_tools_default_never_convert(self):
        engine, seg, status, conversion = self.load("seg", None)
        conversion.assert_not_called()
        self.assertIs(engine.model, seg)
        self.assertEqual(engine.model_graph, {"requested": "seg", "graph": "seg"})
        engine = OpenVocabEngine()
        with patch("engines.prepare_device", return_value="cpu"), \
                patch("ultralytics.YOLOE", return_value=FakeWrapper("segment")), \
                patch("detection_branch.convert_to_detection_branch") as conversion:
            self.assertEqual(engine.load_model("unused-test.pt"), "SUCCESS")
        conversion.assert_not_called()

    def test_invalid_graph_loads_nothing_and_unload_clears_the_record(self):
        engine = OpenVocabEngine()
        with patch("ultralytics.YOLOE") as build:
            status = engine.load_model("unused-test.pt", graph="onnx")
        build.assert_not_called()
        self.assertIn("seg", render(status, "en"))
        engine, *_ = self.load("seg", None)
        engine.unload()
        self.assertIsNone(engine.model_graph)
        self.assertNotIn("graph", engine.inference_options)

    def test_option_changes_do_not_freeze_a_stale_graph(self):
        engine, *_ = self.load("seg", None)
        self.assertEqual(engine.set_inference_options(agnostic_nms=False)["graph"], "seg")
        engine.unload()
        self.assertEqual(engine.inference_options, {"agnostic_nms": False, "head_mode": "one2many", "quantize": 32})


class AppGraphSettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def window(self):
        from main import AIApp
        window = AIApp()
        self.addCleanup(window.close)
        return window

    def test_default_loads_the_detection_branch_and_reports_it(self):
        window = self.window()
        self.assertTrue(window.detection_branch_checkbox.isChecked())

        def load(engine, path, device, graph="seg"):
            engine.model = MagicMock()
            engine.model_graph = {"requested": graph, "graph": graph,
                                  "transfer": {"tensors_copied_exactly": 822, "mask_branch_tensors_removed": 136}}
            return "SUCCESS"

        with patch.object(OpenVocabEngine, "load_model", autospec=True, side_effect=load) as loader, \
                patch("torch.cuda.is_available", return_value=False):
            self.assertTrue(window._load_model_path("yoloe-26s-seg.pt"))
        self.assertEqual(loader.call_args.kwargs["graph"], "det")
        status = window.statusBar().currentMessage()
        self.assertIn("仅检测分支", status)
        self.assertIn("822", status)
        window.ui_language.set_language("en")
        self.assertIn("detection branch only", window.statusBar().currentMessage())
        window.ui_language.set_language("zh")
        window.detection_branch_checkbox.setChecked(False)
        with patch.object(OpenVocabEngine, "load_model", autospec=True, side_effect=load) as loader, \
                patch("torch.cuda.is_available", return_value=False):
            self.assertTrue(window._load_model_path("yoloe-26s-seg.pt"))
        self.assertEqual(loader.call_args.kwargs["graph"], "seg")
        self.assertIn("完整分割模型", window.statusBar().currentMessage())

    def test_fallback_is_shown_in_the_load_status(self):
        window = self.window()

        def load(engine, path, device, graph="seg"):
            engine.model = MagicMock()
            engine.model_graph = {"requested": "det", "graph": "seg", "fallback": "核对失败示例"}
            return "SUCCESS"

        with patch.object(OpenVocabEngine, "load_model", autospec=True, side_effect=load), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("main.QMessageBox.critical") as critical:
            self.assertTrue(window._load_model_path("yoloe-26s-seg.pt"))
        critical.assert_not_called()
        self.assertIn("仅检测分支未通过核对：核对失败示例", window.statusBar().currentMessage())

    def test_prompt_free_loads_are_never_asked_for_a_graph(self):
        window = self.window()
        with patch.object(PromptFreeEngine, "load_model", return_value="SUCCESS") as load, \
                patch("torch.cuda.is_available", return_value=False):
            window._load_model_path("yoloe-26s-seg-pf.pt")
        self.assertEqual(load.call_args.kwargs, {})
        self.assertIsNone(window._wanted_graph("yoloe-26s-seg-pf.pt"))
        self.assertIsNone(window._wanted_graph("IDEA-Research/grounding-dino-tiny"))

    def test_changing_the_setting_reloads_once_at_the_next_start(self):
        window = self.window()
        window.device_combo.setCurrentText("CPU")
        window.engine.model = MagicMock()
        window.engine.device = "cpu"
        window.engine.model_graph = {"requested": "det", "graph": "det"}
        window.active_model_path = "yoloe-26s-seg.pt"
        window.model_combo.setCurrentIndex(window.model_combo.findData("yoloe-26s-seg.pt"))
        with patch.object(window, "_load_model_path", return_value=True) as load:
            self.assertTrue(window._ensure_detection_model(["cup"]))
            load.assert_not_called()
            window.detection_branch_checkbox.setChecked(False)
            self.assertTrue(window._ensure_detection_model(["cup"]))
            load.assert_called_once_with("yoloe-26s-seg.pt")
        # A visible fallback (requested det, got seg) is not reloaded again and again.
        window.detection_branch_checkbox.setChecked(True)
        window.engine.model_graph = {"requested": "det", "graph": "seg", "fallback": "x"}
        with patch.object(window, "_load_model_path", return_value=True) as load:
            self.assertTrue(window._ensure_detection_model(["cup"]))
        load.assert_not_called()
        window.engine.model = None


if __name__ == "__main__":
    unittest.main()
