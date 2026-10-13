"""Readable labels at bright backgrounds and all image borders."""
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from detection_utils import Detection
from engines import OpenVocabEngine, GroundingDINOEngine
from overlay_labels import FONT, FONT_SCALE, THICKNESS, draw_label, fit_text, place_label
from visual_engine import VisualPromptEngine

PROJECT = Path(__file__).resolve().parent
BLUE = (255, 140, 0)


class LayoutTests(unittest.TestCase):
    def test_top_edge_moves_panel_inside(self):
        self.assertEqual(place_label(12, 0, 80, 24, (0, 0, 200, 100)), (12, 0, 80, 24))

    def test_prefers_above_when_space_is_available(self):
        self.assertEqual(place_label(12, 40, 80, 24, (0, 0, 200, 100)), (12, 16, 80, 24))

    def test_right_and_bottom_edges_shift_without_cropping(self):
        self.assertEqual(place_label(199, 99, 80, 24, (0, 0, 200, 100)), (120, 75, 80, 24))

    def test_nonzero_letterbox_bounds_are_respected(self):
        self.assertEqual(place_label(900, 20, 80, 24, (100, 50, 500, 350)), (420, 50, 80, 24))

    def test_panel_clamps_to_tiny_image(self):
        self.assertEqual(place_label(5, 0, 80, 24, (0, 0, 2, 1)), (0, 0, 2, 1))


class DetectionLabelTests(unittest.TestCase):
    def test_white_background_has_solid_panel_and_visible_white_text(self):
        frame = np.full((100, 200, 3), 255, np.uint8)
        left, top, right, bottom = draw_label(frame, "person1: 0.73", 0, 0, (255, 140, 0))
        panel = frame[top:bottom, left:right]
        self.assertGreater(bottom - top, 15)
        self.assertGreater(np.count_nonzero(np.all(panel == BLUE, axis=2)), 100)
        self.assertGreater(np.count_nonzero(np.all(panel[1:-1, 1:-1] > 230, axis=2)), 20)

    def test_panel_works_on_dark_gray_and_bright_backgrounds(self):
        for background in (0, 128, 255):
            with self.subTest(background=background):
                frame = np.full((100, 200, 3), background, np.uint8)
                rect = draw_label(frame, "person2: 0.85", 195, 1, (255, 140, 0))
                left, top, right, bottom = rect
                self.assertEqual(right, 200)
                self.assertGreater(bottom, top)
                self.assertTrue(np.any(np.all(frame[top:bottom, left:right] == BLUE, axis=2)))

    def test_top_person_labels_render_in_all_detection_backends(self):
        frame = np.full((120, 320, 3), 255, np.uint8)
        original = frame.copy()
        detections = [Detection((0, 0, 135, 110), .73, 0, "person", display_id=1),
                      Detection((175, 0, 319, 110), .85, 0, "person", display_id=2)]
        saved = deepcopy(detections)
        for engine in (OpenVocabEngine(), GroundingDINOEngine(), VisualPromptEngine()):
            with self.subTest(engine=type(engine).__name__):
                output = engine.draw_detections(frame, detections)
                for left in (0, 175):
                    region = output[1:24, left+1:left+125]
                    self.assertGreater(np.count_nonzero(np.all(region == BLUE, axis=2)), 200)
                np.testing.assert_array_equal(frame, original)
                self.assertEqual(detections, saved)

    def test_text_shortens_to_fit_without_changing_original(self):
        text = "a_very_long_object_name_" * 20
        fitted = fit_text(text, 90)
        self.assertTrue(fitted.endswith("..."))
        self.assertLessEqual(cv2.getTextSize(fitted, FONT, FONT_SCALE, THICKNESS)[0][0], 90)
        self.assertEqual(text, "a_very_long_object_name_" * 20)

    def test_tiny_frames_do_not_crash_or_expand(self):
        for height, width in ((1, 1), (2, 5), (10, 10), (20, 20)):
            frame = np.full((height, width, 3), 255, np.uint8)
            left, top, right, bottom = draw_label(frame, "person1: 0.73", width-1, 0, (255, 140, 0))
            self.assertTrue(0 <= left < right <= width)
            self.assertTrue(0 <= top < bottom <= height)
            self.assertEqual(frame.shape, (height, width, 3))

    def test_panel_matches_all_box_source_colors(self):
        for color in (BLUE, (0, 200, 255), (0, 220, 255), (100, 220, 100)):
            frame = np.full((100, 200, 3), 255, np.uint8)
            left, top, right, bottom = draw_label(frame, "cup1: 0.85", 0, 0, color)
            panel = frame[top:bottom, left:right]
            self.assertGreater(np.count_nonzero(np.all(panel == color, axis=2)), 100)
            self.assertGreater(np.count_nonzero(np.all(panel > 230, axis=2)), 10)
            self.assertTrue(np.any(np.all(panel == (24, 24, 24), axis=2)))


class CorrectionLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_top_and_right_labels_stay_visible_and_original_boxes_unchanged(self):
        from feedback_ui import CorrectionCanvas
        canvas = CorrectionCanvas()
        canvas.resize(640, 640)
        frame = np.full((480, 640, 3), 255, np.uint8)
        boxes = [{"label": "person", "xyxy": [0., 0., 300., 400.]},
                 {"label": "person", "xyxy": [638., 0., 640., 400.]}]
        saved = deepcopy(boxes)
        canvas.set_sample(frame, boxes)
        canvas.show()
        for selected in (-1, 0):
            canvas.index = selected
            canvas.update()
            self.application.processEvents()
            image = canvas.grab().toImage().convertToFormat(QImage.Format_RGB888)
            pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
            pixels = pixels[:, :image.width()*3].reshape(image.height(), image.width(), 3)
            rect, metrics = canvas.image_rect(), canvas.fontMetrics()
            for index, box in enumerate(boxes):
                text = f"{index+1}. person"
                left, top, width, height = map(int, place_label(
                    rect.left()+box["xyxy"][0], rect.top(), metrics.horizontalAdvance(text)+6,
                    metrics.height()+6, (rect.left(), rect.top(), rect.right(), rect.bottom())))
                region = pixels[top:top+height, left:left+width]
                color = (255, 204, 51) if index == selected else (80, 221, 136)
                self.assertGreater(np.count_nonzero(np.all(region == color, axis=2)), 50)
                self.assertGreater(np.count_nonzero(np.all(region > 230, axis=2)), 20)
        self.assertEqual(canvas.boxes, saved)
        np.testing.assert_array_equal(canvas.image, frame)
        np.testing.assert_array_equal(frame, np.full_like(frame, 255))
        canvas.close()


class RealDatasetRenderingTests(unittest.TestCase):
    def test_existing_real_images_with_human_boxes_render_without_changing_inputs(self):
        root = PROJECT/"datasets/mini_eval"
        if not (root/"annotations.json").is_file():
            self.skipTest("Existing mini_eval data is unavailable")
        coco = json.loads((root/"annotations.json").read_text(encoding="utf-8"))
        names = {item["id"]: item["name"] for item in coco["categories"]}
        engine = OpenVocabEngine()
        rendered = 0
        for item in coco["images"]:
            path = root/"images"/item["file_name"]
            encoded = np.fromfile(str(path), np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            self.assertIsNotNone(frame, str(path))
            original = frame.copy()
            detections = []
            for box in coco["annotations"]:
                if box["image_id"] == item["id"]:
                    x, y, w, h = box["bbox"]
                    detections.append(Detection((x, y, x+w, y+h), 1., box["category_id"], "GT " + names[box["category_id"]]))
            output = engine.draw_detections(frame, detections)
            self.assertEqual(output.shape, frame.shape)
            np.testing.assert_array_equal(frame, original)
            rendered += 1
        self.assertEqual(rendered, len(coco["images"]))


if __name__ == "__main__":
    unittest.main()
