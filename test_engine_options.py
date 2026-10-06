"""Meaningful regression coverage for independent thresholds and box caps."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from detection_utils import Detection
from engines import OpenVocabEngine, validate_detection_options


class IndependentOptionTests(unittest.TestCase):
    def test_predict_receives_independent_confidence_and_output_limit(self):
        engine = OpenVocabEngine()
        engine.model = MagicMock()
        engine.current_classes = ["person"]
        result = MagicMock()
        result.boxes.data.cpu.return_value.numpy.return_value = np.empty((0, 6))
        engine.model.predict.return_value = [result]
        frame = np.zeros((64, 64, 3), np.uint8)
        for conf, max_det in [(0.2, 300), (0.7, 300), (0.7, 5)]:
            engine.detect(frame, conf=conf, class_names=["person"], max_det=max_det)
            kwargs = engine.model.predict.call_args.kwargs
            self.assertEqual(kwargs["conf"], conf)
            self.assertEqual(kwargs["max_det"], max_det)

    def test_combined_supplementary_detections_obey_frame_limit(self):
        engine = OpenVocabEngine()
        original = Detection((5, 5, 15, 15), 0.8, 0, "person")
        tile = Detection((30, 30, 40, 40), 0.9, 0, "person", "tile")
        frame = np.zeros((64, 64, 3), np.uint8)
        with patch.object(engine, "detect", return_value=[original]) as detect, \
                patch.object(engine, "detect_tiles", return_value=[tile]) as tiles:
            _, detections, _ = engine.process_frame(frame, conf=0.4, tiled=True, max_det=1)
        self.assertEqual(detections, [tile])
        self.assertEqual(detect.call_args.kwargs["max_det"], 1)
        self.assertEqual(tiles.call_args.args[-1], 1)

    def test_invalid_box_limit_rejected(self):
        for limit in [0, -1, 1.5, True, 1001]:
            with self.assertRaises(ValueError):
                validate_detection_options(0.4, 0.45, limit)


if __name__ == "__main__":
    unittest.main()
