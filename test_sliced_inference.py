import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from detection_utils import Detection
from resource_safety import ResourceLimitError
from sliced_inference import detect_sliced, merge_sahi, touches_internal_edge, validate_slice_plan


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

    def sliced(self, boxes_by_offset, shape=(480, 640), slice_size=320, base=()):
        """Run detect_sliced; the mock engine returns crop-local boxes chosen by the crop's offset."""
        engine = self.engine()

        def detect(crop, *args, source='original', **kwargs):
            # The frame encodes x // 4 in channel 0 and y // 4 in channel 1.
            offset = (int(crop[0, 0, 0]) * 4, int(crop[0, 0, 1]) * 4)
            return [Detection(xyxy, .9, 0, 'bottle', source) for xyxy in boxes_by_offset.get(offset, [])]

        engine.detect.side_effect = detect
        height, width = shape
        frame = np.zeros((height, width, 3), np.uint8)
        frame[..., 0] = (np.arange(width) // 4)[None, :]
        frame[..., 1] = (np.arange(height) // 4)[:, None]
        with patch('sliced_inference.preflight_inference'):
            result = detect_sliced(engine, frame, conf=.05, slice_size=slice_size, base_detections=list(base))
        return engine, {item.xyxy for item in result}

    def test_box_cut_by_an_internal_slice_edge_is_dropped(self):
        # 640x480 at 320: crops start at x 0/256/320 and y 0/160; x=320 and y=320 are internal edges.
        engine, boxes = self.sliced({(0, 0): [(250, 100, 320, 150)],       # right edge of crop 0
                                     (320, 160): [(30, 0, 80, 40)],        # top edge of a lower crop
                                     (256, 0): [(100, 250, 140, 319)]})    # bottom edge inside the image
        self.assertEqual(engine.detect.call_count, 6)
        self.assertEqual(boxes, set())

    def test_box_on_the_outer_image_border_is_kept_and_shifted(self):
        engine, boxes = self.sliced({(0, 0): [(0, 0, 40, 30)],             # image corner
                                     (320, 160): [(250, 280, 320, 320)],   # bottom-right image corner
                                     (256, 0): [(100, 0, 140, 40)]})       # top image border
        self.assertEqual(boxes, {(0, 0, 40, 30), (570, 440, 640, 480), (356, 0, 396, 40)})

    def test_object_cut_by_one_slice_is_kept_from_the_overlapping_slice(self):
        # Global object x 300-340: cut at x=320 by crop 0, whole inside the crop starting at x=256.
        engine, boxes = self.sliced({(0, 0): [(300, 100, 320, 150)], (256, 0): [(44, 100, 84, 150)]})
        self.assertEqual(boxes, {(300, 100, 340, 150)})

    def test_full_image_boxes_are_never_filtered(self):
        base = [Detection((250, 100, 330, 150), .7, 0, 'bottle')]
        engine, boxes = self.sliced({}, base=base)
        self.assertEqual(boxes, {(250, 100, 330, 150)})

    def test_edge_margin_is_three_pixels_on_internal_edges_only(self):
        inside = dict(left=256, top=160, crop_width=320, crop_height=320, width=640, height=640)
        self.assertTrue(touches_internal_edge((2.9, 50, 60, 90), **inside))
        self.assertFalse(touches_internal_edge((3, 3, 317, 317), **inside))
        self.assertTrue(touches_internal_edge((50, 50, 317.1, 90), **inside))
        self.assertTrue(touches_internal_edge((50, 2, 60, 90), **inside))
        self.assertTrue(touches_internal_edge((50, 50, 60, 318), **inside))
        # A crop that covers the whole image has no internal edge.
        border = dict(left=0, top=0, crop_width=320, crop_height=320, width=320, height=320)
        self.assertFalse(touches_internal_edge((0, 0, 320, 320), **border))

    def test_invalid_plan_stops_before_materializing_crops(self):
        for kwargs in ({'slice_size': 127}, {'slice_overlap': 1}, {'max_slices': 10}, {'slice_size': 320.5}):
            with self.assertRaises(ValueError):
                validate_slice_plan((480, 640, 3), **kwargs)


if __name__ == '__main__':
    unittest.main()
