"""Regression tests for exposure handling, supplementary detections and live options."""
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
import cv2
import numpy as np

from detection_utils import Detection, merge_supplementary
from enhancement import AdaptiveLowLightEngine
from engines import OpenVocabEngine
from tracking import ByteTrackManager


class EnhancementTests(unittest.TestCase):
    def test_bright_black_and_disabled_frames_unchanged(self):
        engine = AdaptiveLowLightEngine()
        for value, strength in [(180, 0.65), (0, 0.65), (30, 0.0)]:
            original = np.full((64, 96, 3), value, np.uint8)
            result = engine.process(original, strength)
            self.assertFalse(result.applied)
            np.testing.assert_array_equal(result.image, original)

    def test_dark_color_ratios_highlights_and_input_preserved(self):
        original = np.full((100, 120, 3), [12, 24, 48], np.uint8)
        original[:10, :10] = 250
        saved = original.copy()
        result = AdaptiveLowLightEngine().process(original)
        self.assertTrue(result.applied)
        self.assertGreater(result.image[50, 50, 1], original[50, 50, 1])
        b, g, r = result.image[50, 50].astype(float)
        self.assertAlmostEqual(r / g, 2.0, delta=0.12)
        self.assertAlmostEqual(g / b, 2.0, delta=0.12)
        self.assertGreaterEqual(int(result.image[2, 2, 0]), 250)
        np.testing.assert_array_equal(original, saved)

    def test_shadow_noise_is_not_amplified_as_much_as_plain_gain(self):
        rng = np.random.default_rng(4)
        gray = np.clip(30 + rng.normal(0, 2, (128, 128)), 0, 255).astype(np.uint8)
        original = np.repeat(gray[..., None], 3, axis=2)
        result = AdaptiveLowLightEngine().enhance(original)
        gain = result.mean() / original.mean()
        self.assertLess(result.std(), original.std() * gain)

    def test_invalid_input_and_scene_reset(self):
        engine = AdaptiveLowLightEngine()
        with self.assertRaises(ValueError):
            engine.process(np.empty((0, 0, 3), np.uint8))
        engine.process(np.full((40, 40, 3), 20, np.uint8))
        engine.process(np.full((40, 40, 3), 200, np.uint8))
        self.assertIsNone(engine._gamma)


class FusionTests(unittest.TestCase):
    def test_original_box_and_confidence_survive_all_extra_predictions(self):
        original = Detection((10, 10, 40, 50), 0.51, 0, "bottle")
        higher_score = replace(original, score=0.98, source="enhanced")
        conflicting = replace(higher_score, class_id=1, label="cup")
        additional = Detection((70, 70, 90, 95), 0.61, 0, "bottle", "enhanced")
        merged = merge_supplementary([original], [higher_score, conflicting, additional, additional])
        self.assertEqual(merged, [original, additional])

    def test_preview_only_and_normal_light_skip_second_detection(self):
        engine = OpenVocabEngine()
        for brightness, supplement in [(180, True), (25, False)]:
            with patch.object(engine, "detect", return_value=[]) as detect:
                engine.process_frame(np.full((60, 80, 3), brightness, np.uint8),
                                     enhancer=AdaptiveLowLightEngine(), supplement_dark=supplement)
                self.assertEqual(detect.call_count, 1)

    def test_dark_second_pass_receives_enhanced_image_and_conservative_threshold(self):
        engine = OpenVocabEngine()
        original = np.full((60, 80, 3), 25, np.uint8)
        raw_box = Detection((5, 5, 20, 30), 0.42, 0, "cup")
        with patch.object(engine, "detect", side_effect=[[raw_box], []]) as detect:
            _, detections, _ = engine.process_frame(original, conf=0.4, enhancer=AdaptiveLowLightEngine())
            self.assertEqual(detect.call_count, 2)
            np.testing.assert_array_equal(detect.call_args_list[0].args[0], original)
            self.assertGreater(detect.call_args_list[1].args[0].mean(), original.mean())
            self.assertEqual(detect.call_args_list[1].args[1], 0.45)
            self.assertEqual(detections, [raw_box])

    def test_tiles_translate_coordinates_and_ignore_internal_truncation(self):
        engine = OpenVocabEngine()
        good = Detection((10, 10, 30, 30), 0.8, 0, "cup", "tile")
        seam = Detection((0, 10, 20, 30), 0.8, 0, "cup", "tile")
        with patch.object(engine, "detect", side_effect=[[good], [good, seam], [good], [good]]):
            boxes = engine.detect_tiles(np.zeros((100, 200, 3), np.uint8), 0.4, 0.45, ["cup"])
        self.assertEqual([item.xyxy for item in boxes],
                         [(10, 10, 30, 30), (90, 10, 110, 30), (10, 50, 30, 70), (90, 50, 110, 70)])

    def test_model_reload_reapplies_unchanged_prompts(self):
        from unittest.mock import MagicMock
        engine = OpenVocabEngine()
        engine.current_classes = ["cup"]
        fake_model = MagicMock()
        fake_model.predict.return_value = [MagicMock()]
        fake_model.predict.return_value[0].boxes.data.cpu.return_value.numpy.return_value = np.empty((0, 6))
        with patch("ultralytics.YOLOE", return_value=fake_model):
            self.assertEqual(engine.load_model("test.pt"), "SUCCESS")
            engine.detect(np.zeros((64, 64, 3), np.uint8), class_names=["cup"])
        fake_model.set_classes.assert_called_once_with(["cup"])

    def test_tracking_receives_merged_high_and_low_score_detections(self):
        from unittest.mock import MagicMock
        engine = OpenVocabEngine()
        high = Detection((10, 10, 30, 40), 0.80, 0, "person")
        low = Detection((50, 10, 70, 40), 0.15, 0, "person")
        numbered = replace(high, track_id=4, display_id=1)
        tracker = MagicMock()
        tracker.detection_floor.return_value = 0.10
        tracker.update.return_value = [numbered]
        frame = np.zeros((60, 80, 3), np.uint8)
        with patch.object(engine, "detect", return_value=[high, low]) as detect:
            _, detections, status = engine.process_frame(
                frame, conf=0.4, class_names=["person"], tracker=tracker,
            )
        self.assertEqual(detect.call_args.args[1], 0.10)
        self.assertEqual(tracker.update.call_args.args[0], [high, low])
        self.assertEqual(detections, [numbered])
        self.assertIn("跟踪显示 1", status)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((160, 240, 3), np.uint8)
        self.tracker = ByteTrackManager(track_buffer=30)

    def test_same_object_keeps_person_number_while_moving(self):
        first = Detection((20, 20, 70, 130), 0.90, 0, "person")
        moved = replace(first, xyxy=(24, 20, 74, 130), score=0.82)
        first_result = self.tracker.update([first], self.frame, 0.4, ["person"])
        second_result = self.tracker.update([moved], self.frame, 0.4, ["person"])
        self.assertEqual(first_result[0].display_label, "person1")
        self.assertEqual(second_result[0].display_label, "person1")
        self.assertEqual(first_result[0].track_id, second_result[0].track_id)

    def test_each_class_receives_its_own_compact_numbering(self):
        detections = [
            Detection((10, 10, 50, 120), 0.9, 0, "person"),
            Detection((80, 10, 120, 120), 0.9, 0, "person"),
            Detection((150, 30, 175, 70), 0.9, 1, "bottle"),
            Detection((190, 30, 215, 70), 0.9, 1, "bottle"),
        ]
        result = self.tracker.update(detections, self.frame, 0.4, ["person", "bottle"])
        self.assertEqual(
            {item.display_label for item in result},
            {"person1", "person2", "bottle1", "bottle2"},
        )

    def test_low_confidence_reconnects_but_cannot_start_track(self):
        high = Detection((20, 20, 70, 130), 0.90, 0, "person")
        low = replace(high, xyxy=(22, 20, 72, 130), score=0.20)
        self.tracker.update([high], self.frame, 0.4, ["person"])
        continued = self.tracker.update([low], self.frame, 0.4, ["person"])
        self.assertEqual([item.display_label for item in continued], ["person1"])

        fresh = ByteTrackManager()
        self.assertEqual(fresh.update([low], self.frame, 0.4, ["person"]), [])

    def test_prompt_change_starts_new_numbering_session(self):
        person = Detection((20, 20, 70, 130), 0.90, 0, "person")
        first = self.tracker.update([person], self.frame, 0.4, ["person"])
        restarted = self.tracker.update([person], self.frame, 0.4, ["person", "bag"])
        self.assertEqual(first[0].display_label, "person1")
        self.assertEqual(restarted[0].display_label, "person1")


class WorkerAndUiTests(unittest.TestCase):
    def test_worker_keeps_raw_frame_and_forwards_enhancement_options(self):
        from unittest.mock import MagicMock
        from video_worker import VideoWorker
        frame = np.full((32, 48, 3), 20, np.uint8)
        capture, engine = MagicMock(), MagicMock()
        capture.read.side_effect = [(True, frame), (False, None)]
        engine.process_frame.return_value = (frame.copy(), [], "ok")
        enhancer = AdaptiveLowLightEngine()
        worker = VideoWorker(engine, enhancer, "fixture.avi")
        worker.use_low_light_enhancement, worker.tiled = True, True
        worker.imgsz = 1280
        with patch.object(worker, "_open_capture", return_value=capture):
            worker.run()
        capture.release.assert_called_once()
        self.assertEqual(engine.imgsz, 1280)
        self.assertTrue(engine.process_frame.call_args.kwargs["tiled"])
        self.assertIs(engine.process_frame.call_args.kwargs["enhancer"], enhancer)
        self.assertIs(engine.process_frame.call_args.kwargs["tracker"], worker.tracker)
        np.testing.assert_array_equal(engine.process_frame.call_args.args[0], frame)

    def test_worker_releases_capture_after_inference_error(self):
        from unittest.mock import MagicMock
        from video_worker import VideoWorker
        capture, engine = MagicMock(), MagicMock()
        capture.read.return_value = (True, np.zeros((32, 32, 3), np.uint8))
        engine.process_frame.side_effect = RuntimeError("test inference failure")
        worker = VideoWorker(engine, AdaptiveLowLightEngine(), "fixture.avi")
        with patch.object(worker, "_open_capture", return_value=capture):
            worker.run()
        capture.release.assert_called_once()

    def test_image_worker_reads_one_image_without_opening_video_capture(self):
        from unittest.mock import MagicMock
        from video_worker import VideoWorker
        frame = np.full((40, 60, 3), 120, np.uint8)
        engine = MagicMock()
        engine.process_frame.return_value = (frame.copy(), [], "ok")
        worker = VideoWorker(engine, AdaptiveLowLightEngine(), "测试图片.jpg", source_kind="image")
        with patch("video_worker.read_image_file", return_value=frame), patch.object(
            worker, "_open_capture"
        ) as open_capture:
            worker.run()
        open_capture.assert_not_called()
        engine.process_frame.assert_called_once()
        self.assertIs(engine.process_frame.call_args.kwargs["tracker"], worker.tracker)

    def test_live_controls_and_stop_timeout_keep_worker_alive(self):
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QApplication
        from main import AIApp
        app = QApplication.instance() or QApplication([])
        window = AIApp()
        worker = MagicMock()
        window.worker = worker
        window.conf_slider.setValue(55)
        window.resolution_combo.setCurrentIndex(window.resolution_combo.findData(960))
        window.tile_checkbox.setChecked(True)
        window.dce_checkbox.setChecked(True)
        self.assertEqual(worker.conf, 0.55)
        self.assertEqual(worker.imgsz, 960)
        self.assertTrue(worker.tiled)
        self.assertTrue(worker.use_low_light_enhancement)
        worker.stop.return_value = False
        self.assertFalse(window.stop_video_stream())
        self.assertIs(window.worker, worker)
        worker.stop.return_value = True
        self.assertTrue(window.stop_video_stream())
        window.close()

    def test_ui_renames_input_source_and_offers_image_modes(self):
        from PySide6.QtWidgets import QApplication, QLabel
        from main import AIApp
        app = QApplication.instance() or QApplication([])
        window = AIApp()
        labels = {label.text() for label in window.findChildren(QLabel)}
        self.assertIn("输入源：", labels)
        self.assertNotIn("视频输入源:", labels)
        self.assertEqual(
            [window.source_combo.itemData(i) for i in range(window.source_combo.count())],
            ["camera", "video", "image", "image_folder"],
        )
        window.close()

    def test_image_folder_navigation_wraps_and_updates_selected_path(self):
        from PySide6.QtWidgets import QApplication
        from main import AIApp
        app = QApplication.instance() or QApplication([])
        window = AIApp()
        window.image_paths = ["C:/photos/a.jpg", "C:/photos/b.jpg"]
        window.image_index, window.image_folder = 0, "C:/photos"
        window.selected_input_kind = "image_folder"
        window.next_image()
        self.assertEqual(window.image_index, 1)
        self.assertEqual(window.selected_input_source, "C:/photos/b.jpg")
        self.assertEqual(window.image_position_label.text(), "2 / 2")
        window.next_image()
        self.assertEqual(window.image_index, 0)
        window.previous_image()
        self.assertEqual(window.image_index, 1)
        window.close()


if __name__ == "__main__":
    unittest.main()
