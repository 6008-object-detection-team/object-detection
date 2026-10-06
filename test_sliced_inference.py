import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from detection_utils import Detection
from resource_safety import ResourceLimitError
from sliced_inference import detect_sliced, merge_sahi, validate_slice_plan


class SahiTests(unittest.TestCase):
    def engine(self):
        engine = MagicMock()
        engine.device, engine.imgsz = 'cpu', 640
        return engine

    def test_shift_coordinates_and_one_call_per_crop(self):
        engine = self.engine()
        engine.detect.return_value = [Detection((10, 20, 30, 40), .8, 0, 'bottle', 'slice')]
        frame = np.zeros((320, 320, 3), np.uint8)
        with patch('sliced_inference.preflight_inference'):
            detections = detect_sliced(engine, frame, slice_size=192, base_detections=[])
        self.assertEqual(engine.detect.call_count, 4)
        self.assertEqual({item.xyxy for item in detections}, {
            (10, 20, 30, 40), (138, 20, 158, 40), (10, 148, 30, 168), (138, 148, 158, 168)})
        self.assertTrue(all(item.source == 'slice' for item in detections))

    def test_official_sahi_nms_is_class_aware_and_caps_final_output(self):
        detections = [Detection((1, 2, 31, 32), .6, 0, 'bottle'),
                      Detection((1, 2, 31, 32), .9, 0, 'bottle', 'slice'),
                      Detection((1, 2, 31, 32), .8, 1, 'cup')]
        merged = merge_sahi(detections, .5, 100)
        self.assertEqual(merged, [detections[1], detections[2]])
        self.assertEqual(merge_sahi(detections, .5, 1), [detections[1]])

    def test_excess_slices_rejected_before_detector_runs(self):
        engine = self.engine()
        with patch('sliced_inference.preflight_inference'), self.assertRaises(ResourceLimitError):
            detect_sliced(engine, np.zeros((2000, 2000, 3), np.uint8), slice_size=320)
        engine.detect.assert_not_called()

    def test_single_slice_reuses_full_image_result(self):
        engine = self.engine()
        base = [Detection((1, 2, 10, 20), .8, 0, 'bottle')]
        with patch('sliced_inference.preflight_inference'):
            result = detect_sliced(engine, np.zeros((128, 128, 3), np.uint8), base_detections=base)
        self.assertEqual(result, base)
        engine.detect.assert_not_called()

    def test_invalid_plan_stops_before_materializing_crops(self):
        for kwargs in ({'slice_size': 127}, {'slice_overlap': 1}, {'max_slices': 10}, {'slice_size': 320.5}):
            with self.assertRaises(ValueError):
                validate_slice_plan((480, 640, 3), **kwargs)


if __name__ == '__main__':
    unittest.main()
