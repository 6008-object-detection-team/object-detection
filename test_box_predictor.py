"""Tiny CPU tensors validate box-only segmentation without model/GPU loads."""
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from box_predictor import BoxOnlySegmentationPredictor


class BoxOnlyPredictorTests(unittest.TestCase):
    def predictor(self):
        predictor = BoxOnlySegmentationPredictor(overrides={"conf": 0.4, "iou": 0.45, "max_det": 17, "task": "detect"})
        predictor.model = types.SimpleNamespace(names={0: "bottle", 1: "cup", 2: "cell phone"}, end2end=False)
        predictor.batch = (["frame.png"], [np.zeros((48, 64, 3), np.uint8)], [""])
        return predictor

    def test_constructs_real_box_results_without_masks_and_keeps_true_class_count(self):
        predictor = self.predictor()
        image = torch.zeros((1, 3, 48, 64))
        original = np.zeros((48, 64, 3), np.uint8)
        # NMS keeps mask coefficient columns; box Results must retain only six.
        detections = torch.cat((torch.tensor([[5.0, 7.0, 25.0, 35.0, 0.9, 2.0]]), torch.ones((1, 32))), dim=1)
        raw = torch.zeros((1, 4 + 3 + 32, 4))
        proto = object()  # Would fail if mask reconstruction touched it.
        with patch("ultralytics.models.yolo.detect.predict.nms.non_max_suppression", return_value=[detections]) as nms, \
                patch("ultralytics.utils.ops.process_mask", side_effect=AssertionError("mask allocation")) as masks, \
                patch("ultralytics.utils.ops.process_mask_native", side_effect=AssertionError("native mask allocation")) as native:
            result = predictor.postprocess((raw, proto), image, [original])[0]
        self.assertEqual(predictor.args.task, "segment")
        self.assertEqual(nms.call_args.kwargs["nc"], 3)
        self.assertIs(nms.call_args.args[0], raw)
        self.assertIsNone(result.masks)
        self.assertEqual(tuple(result.boxes.data.shape), (1, 6))
        np.testing.assert_allclose(result.boxes.xyxy.numpy(), [[5.0, 7.0, 25.0, 35.0]])
        self.assertEqual(result.boxes.cls.item(), 2)
        self.assertAlmostEqual(result.boxes.conf.item(), 0.9, places=6)
        self.assertEqual(result.names[2], "cell phone")
        masks.assert_not_called()
        native.assert_not_called()

    def test_real_nms_does_not_treat_32_mask_coefficients_as_categories(self):
        predictor = self.predictor()
        image = torch.zeros((1, 3, 48, 64))
        original = np.zeros((48, 64, 3), np.uint8)
        raw = torch.zeros((1, 4 + 3 + 32, 1))
        raw[0, :4, 0] = torch.tensor([20.0, 20.0, 10.0, 12.0])  # cx, cy, w, h
        raw[0, 4:7, 0] = torch.tensor([0.05, 0.92, 0.04])
        raw[0, 7:, 0] = 10.0  # Larger than scores: these must never become labels.
        with patch("ultralytics.utils.ops.process_mask", side_effect=AssertionError("mask allocation")), \
                patch("ultralytics.utils.ops.process_mask_native", side_effect=AssertionError("native mask allocation")):
            result = predictor.postprocess((raw, object()), image, [original])[0]
        self.assertIsNone(result.masks)
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes.cls.item(), 1)
        self.assertAlmostEqual(result.boxes.conf.item(), 0.92, places=6)
        np.testing.assert_allclose(result.boxes.xyxy.numpy(), [[15.0, 14.0, 25.0, 26.0]])

    def test_nested_pytorch_predictions_follow_official_segmentation_selection(self):
        predictor = self.predictor()
        raw = torch.zeros((1, 39, 1))
        nested = (raw, object())
        image = torch.zeros((1, 3, 48, 64))
        original = np.zeros((48, 64, 3), np.uint8)
        with patch("ultralytics.models.yolo.detect.predict.nms.non_max_suppression", return_value=[torch.empty((0, 38))]) as nms:
            result = predictor.postprocess((nested, object()), image, [original])[0]
        self.assertIs(nms.call_args.args[0], nested)
        self.assertEqual(nms.call_args.kwargs["nc"], 3)
        self.assertIsNone(result.masks)
        self.assertEqual(tuple(result.boxes.data.shape), (0, 6))

    def test_box_scaling_remains_correct_for_larger_original(self):
        predictor = self.predictor()
        image = torch.zeros((1, 3, 48, 64))
        original = np.zeros((96, 128, 3), np.uint8)
        detections = torch.tensor([[10.0, 8.0, 30.0, 20.0, 0.8, 0.0]])
        with patch("ultralytics.models.yolo.detect.predict.nms.non_max_suppression", return_value=[detections]):
            result = predictor.postprocess((torch.zeros((1, 39, 1)), object()), image, [original])[0]
        np.testing.assert_allclose(result.boxes.xyxy.numpy(), [[20.0, 16.0, 60.0, 40.0]])
        self.assertIsNone(result.masks)


if __name__ == "__main__":
    unittest.main()
