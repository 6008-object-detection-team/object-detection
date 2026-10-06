"""Backend lifecycle and prompt preservation without loading any checkpoint."""
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from engines import OpenVocabEngine


def empty_result():
    data = MagicMock()
    data.cpu.return_value.numpy.return_value = np.empty((0, 6))
    return types.SimpleNamespace(boxes=types.SimpleNamespace(data=data), names={0: "bottle"})


def fake_wrapper(*, predictor=None, one2one=True):
    return types.SimpleNamespace(
        task="segment", predictor=predictor, overrides={}, callbacks={},
        model=types.SimpleNamespace(model=[types.SimpleNamespace(
            one2one_cv2=object() if one2one else None,
        )], pe=object()),
        set_classes=MagicMock(), predict=MagicMock(return_value=[empty_result()]),
        to=MagicMock(),
    )


class InferenceExperimentTests(unittest.TestCase):
    def setUp(self):
        self.engine = OpenVocabEngine()
        # Match the installed 8.4.172 initialization API without a model load.
        self.engine._prediction_head_options = {"nms": None}
        self.engine.model = fake_wrapper(predictor=object())
        self.engine.current_classes = ["bottle"]
        self.frame = np.zeros((64, 64, 3), np.uint8)
        guard = patch("engines.preflight_inference")
        guard.start()
        self.addCleanup(guard.stop)

    def detect(self):
        return self.engine.detect(self.frame, class_names=["bottle"], max_det=17)

    def test_defaults_retain_baseline_head_precision_and_cross_class_nms(self):
        self.detect()
        kwargs = self.engine.model.predict.call_args.kwargs
        self.assertIs(kwargs["nms"], None)
        self.assertTrue(kwargs["agnostic_nms"])
        self.assertEqual(kwargs["quantize"], 32)
        self.assertEqual(kwargs["max_det"], 17)

    def test_class_aware_nms_updates_postprocessing_without_rebuilding_backend(self):
        previous = self.engine.model.predictor
        self.engine.set_inference_options(agnostic_nms=False)
        self.detect()
        self.assertIs(self.engine.model.predictor, previous)
        self.assertFalse(self.engine.model.predict.call_args.kwargs["agnostic_nms"])

    def test_precision_change_rebuilds_backend_once_and_preserves_prompt(self):
        prompt = self.engine.model.model.pe
        predictor = MagicMock()
        with patch("box_predictor.BoxOnlySegmentationPredictor", return_value=predictor) as create:
            self.engine.set_inference_options(quantize=16)
            self.assertIsNone(self.engine.model.predictor)
            self.detect()
            self.engine.set_inference_options(quantize=16)
            self.detect()
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["overrides"]["quantize"], 16)
        predictor.setup_model.assert_called_once_with(model=self.engine.model.model, verbose=False)
        self.assertEqual(self.engine.model.predict.call_args.kwargs["quantize"], 16)
        self.assertIs(self.engine.model.model.pe, prompt)
        self.assertEqual(self.engine.current_classes, ["bottle"])
        self.engine.model.set_classes.assert_not_called()

    def test_switching_both_heads_builds_each_backend_from_original_model(self):
        original = self.engine.model.model
        first, second = MagicMock(), MagicMock()
        with patch("box_predictor.BoxOnlySegmentationPredictor", side_effect=[first, second]) as create:
            self.engine.set_inference_options(head_mode="one2one")
            self.detect()
            self.engine.set_inference_options(head_mode="one2many")
            self.detect()
        self.assertEqual(create.call_count, 2)
        self.assertIs(create.call_args_list[0].kwargs["overrides"]["nms"], False)
        self.assertIs(create.call_args_list[1].kwargs["overrides"]["nms"], None)
        first.setup_model.assert_called_once_with(model=original, verbose=False)
        second.setup_model.assert_called_once_with(model=original, verbose=False)
        self.assertIs(self.engine.model.model, original)

    def test_noop_and_returned_manifest_cannot_reset_or_mutate_live_options(self):
        previous = self.engine.model.predictor
        result = self.engine.set_inference_options(agnostic_nms=True, head_mode="one2many", quantize=32)
        result["quantize"] = 16
        self.assertIs(self.engine.model.predictor, previous)
        self.assertEqual(self.engine.inference_options["quantize"], 32)

    def test_invalid_combination_is_atomic_and_keeps_existing_backend(self):
        previous = self.engine.model.predictor
        for kwargs in [dict(agnostic_nms=1), dict(head_mode="auto"),
                       dict(quantize=True), dict(quantize=32.0), dict(quantize=8),
                       dict(agnostic_nms=False, quantize=8)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.engine.set_inference_options(**kwargs)
            self.assertIs(self.engine.model.predictor, previous)
            self.assertEqual(self.engine.inference_options, {
                "agnostic_nms": True, "head_mode": "one2many", "quantize": 32,
            })

    def test_missing_one2one_head_rejected_instead_of_silent_fallback(self):
        self.engine.model = fake_wrapper(predictor=object(), one2one=False)
        previous = self.engine.model.predictor
        with self.assertRaisesRegex(ValueError, "没有 one2one"):
            self.engine.set_inference_options(head_mode="one2one", quantize=16)
        self.assertIs(self.engine.model.predictor, previous)
        self.assertEqual(self.engine.inference_options["head_mode"], "one2many")
        self.assertEqual(self.engine.inference_options["quantize"], 32)

    def test_configuration_before_load_is_applied_using_installed_cfg(self):
        engine = OpenVocabEngine()
        engine.set_inference_options(agnostic_nms=False, head_mode="one2one", quantize=16)
        wrapper = fake_wrapper(predictor=object())
        with patch("engines.prepare_device", return_value="cpu"), \
                patch("ultralytics.YOLOE", return_value=wrapper):
            self.assertEqual(engine.load_model("unused-test.pt"), "SUCCESS")
        engine.current_classes = ["bottle"]
        engine.detect(self.frame, class_names=["bottle"])
        options = wrapper.predict.call_args.kwargs
        self.assertIs(options["nms"], False)
        self.assertEqual(options["quantize"], 16)
        self.assertFalse(options["agnostic_nms"])

    def test_legacy_backend_keys_keep_same_experiment_semantics(self):
        self.engine._prediction_head_options = {"end2end": False}
        self.engine._precision_uses_quantize = False
        predictor = MagicMock()
        with patch("box_predictor.BoxOnlySegmentationPredictor", return_value=predictor) as create:
            self.engine.set_inference_options(head_mode="one2one", quantize=16)
            self.detect()
        setup = create.call_args.kwargs["overrides"]
        self.assertTrue(setup["end2end"])
        self.assertTrue(setup["half"])
        self.assertNotIn("quantize", setup)
        options = self.engine.model.predict.call_args.kwargs
        self.assertTrue(options["end2end"])
        self.assertTrue(options["half"])
        self.assertNotIn("nms", options)


class ActualPostprocessingTests(unittest.TestCase):
    """Small CPU tensors check the behavior an experiment intends to change."""

    def predictor(self, *, agnostic=True, end2end=False, max_det=17):
        from box_predictor import BoxOnlySegmentationPredictor

        predictor = BoxOnlySegmentationPredictor(overrides={
            "conf": 0.4, "iou": 0.45, "max_det": max_det,
            "agnostic_nms": agnostic, "task": "segment",
        })
        predictor.model = types.SimpleNamespace(names={0: "bottle", 1: "cup", 2: "cell phone"},
                                               end2end=end2end)
        predictor.batch = (["frame.png"], [np.zeros((48, 64, 3), np.uint8)], [""])
        return predictor

    def test_overlapping_different_classes_are_kept_only_by_class_aware_nms(self):
        import torch

        # The two identical boxes disagree about class. Their mask coefficients
        # are deliberately large to ensure they never become category scores.
        raw = torch.zeros((1, 4 + 3 + 32, 2))
        raw[0, :4, :] = torch.tensor([[20., 20.], [20., 20.], [10., 10.], [12., 12.]])
        raw[0, 4, 0] = 0.9
        raw[0, 5, 1] = 0.8
        raw[0, 7:, :] = 10.0
        image = torch.zeros((1, 3, 48, 64))
        original = np.zeros((48, 64, 3), np.uint8)
        for agnostic, expected in [(True, [0.0]), (False, [0.0, 1.0])]:
            with self.subTest(agnostic=agnostic):
                result = self.predictor(agnostic=agnostic).postprocess(
                    (raw.clone(), object()), image, [original],
                )[0]
                self.assertEqual(result.boxes.cls.tolist(), expected)
                self.assertIsNone(result.masks)

    def test_one2one_fp16_outputs_are_filtered_without_iou_suppression(self):
        import torch

        # Native one2one output is already xyxy/score/class, then coefficients.
        # The two overlapping rows stay even with agnostic_nms=True. A low
        # confidence row is rejected and the public box tensor remains FP32.
        rows = torch.tensor([
            [15., 14., 25., 26., 0.9, 0.],
            [15., 14., 25., 26., 0.8, 1.],
            [40., 30., 45., 40., 0.1, 2.],
        ], dtype=torch.float16)
        raw = torch.cat((rows, torch.full((3, 32), 10., dtype=torch.float16)), dim=1)[None]
        image = torch.zeros((1, 3, 48, 64), dtype=torch.float16)
        original = np.zeros((48, 64, 3), np.uint8)
        for max_det, expected in [(17, [0.0, 1.0]), (1, [0.0])]:
            with self.subTest(max_det=max_det):
                result = self.predictor(end2end=True, max_det=max_det).postprocess(
                    (raw, object()), image, [original],
                )[0]
                self.assertEqual(result.boxes.cls.tolist(), expected)
                self.assertEqual(result.boxes.data.dtype, torch.float32)
                self.assertEqual(result.boxes.data.shape[1], 6)
                self.assertIsNone(result.masks)


if __name__ == "__main__":
    unittest.main()
