"""Synthetic CPU-only correction/ledger/UI tests; no real training or camera."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from detection_utils import Detection
from feedback import FeedbackStore, image_id, safe_path, sha256_file, source_group
from feedback_training import validate_dataset
from feedback_ui import CorrectionCanvas, FeedbackPanel
from video_worker import VideoWorker


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = FeedbackStore(Path(self.temp.name)/"纠错资料")

    def save(self, number, split="train", reviewed=True, boxes=None, classes=None, scene=None, origin=None):
        image = np.full((40, 80, 3), number, np.uint8)
        return self.store.save(image, [{"label": "cup", "xyxy": [8, 4, 40, 20]}] if boxes is None else boxes,
            classes=classes or ["cup"], scene=scene or "scene"+str(number), split=split, reviewed=reviewed,
            provenance={"source": "synthetic", "origin_group": origin or "origin"+str(number), "synthetic": True},
            candidates=[{"label": "bottle", "xyxy": [8, 4, 40, 20], "score": .7}])

    def test_append_only_revision_and_lossless_original(self):
        one = self.save(17, reviewed=False)
        two = self.save(17, reviewed=True)
        self.assertNotEqual(one["revision"], two["revision"])
        self.assertEqual(len(list((self.store.root/"records").glob("*.json"))), 2)
        self.assertEqual(len(self.store.records()), 1)
        self.assertEqual(self.store.records()[0]["status"], "reviewed")
        np.testing.assert_array_equal(self.store.load_image(two), np.full((40, 80, 3), 17, np.uint8))
        self.assertEqual(two["candidates"][0]["label"], "bottle")
        self.assertEqual(two["boxes"][0]["label"], "cup")

    def test_export_excludes_drafts_and_keeps_negative_and_test(self):
        self.save(11)
        self.save(12, "val", boxes=[])
        self.save(13, "test")
        self.save(14, reviewed=False)
        output = self.store.export(["cup"])
        manifest = validate_dataset(output)
        self.assertEqual(len(manifest["samples"]), 3)
        self.assertEqual(manifest["excluded_drafts"], 1)
        line = next((output/"labels/train").glob("*.txt")).read_text().strip()
        self.assertEqual(line, "0 0.3000000000 0.3000000000 0.4000000000 0.4000000000")
        self.assertEqual(next((output/"labels/val").glob("*.txt")).read_text(), "")
        self.assertIn("test", json.loads((output/"data.yaml").read_text()))

    def test_dataset_versions_never_overwrite(self):
        self.save(1); self.save(2, "val")
        one = self.store.export(["cup"])
        first_sha = sha256_file(one/"manifest.json")
        two = self.store.export(["cup"])
        self.assertNotEqual(one, two)
        self.assertEqual(sha256_file(one/"manifest.json"), first_sha)

    def test_requires_train_and_val_and_whole_class_review(self):
        self.save(1)
        with self.assertRaisesRegex(ValueError, "独立场景"):
            self.store.export(["cup"])
        self.save(2, "val")
        with self.assertRaisesRegex(ValueError, "全部类别"):
            self.store.export(["cup", "bottle"])

    def test_duplicate_frame_scene_and_origin_cannot_cross_splits(self):
        self.save(1)
        for kwargs in ({"number": 1, "split": "val"}, {"number": 2, "split": "val", "scene": "scene1"},
                       {"number": 3, "split": "val", "origin": "origin1"}):
            with self.assertRaisesRegex(ValueError, "不能跨"):
                self.save(**kwargs)

    def test_invalid_box_or_unreviewed_class_is_rejected(self):
        for box in ([0, 0, 100, 20], [10, 10, 1, 1], [float("nan"), 0, 20, 20]):
            with self.assertRaises(ValueError):
                self.save(1, boxes=[{"label": "cup", "xyxy": box}])
        with self.assertRaisesRegex(ValueError, "类别"):
            self.save(1, boxes=[{"label": "mouse", "xyxy": [0, 0, 10, 10]}])

    def test_tampering_stops_loading_and_training(self):
        row = self.save(1); self.save(2, "val")
        output = self.store.export(["cup"])
        next((output/"labels/train").glob("*.txt")).write_text("changed")
        with self.assertRaisesRegex(ValueError, "被修改"):
            validate_dataset(output)
        (self.store.root/"images"/(row["image_id"]+".png")).write_bytes(b"broken")
        with self.assertRaises(ValueError):
            self.store.load_image(row)
        with self.assertRaises(ValueError):
            self.save(1)

    def test_path_traversal_and_diagnostic_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(self.store.root, "../escape")
        from feedback import PROJECT
        with self.assertRaisesRegex(ValueError, "开发诊断"):
            self.store.save(np.zeros((8, 8, 3), np.uint8), [], classes=["cup"], scene="a", split="train", reviewed=True,
                provenance={"source": str(PROJECT/"datasets/mini_eval/images/test.jpg"), "origin_group": "a"})

    def test_video_group_is_stable_and_camera_sessions_differ(self):
        self.assertEqual(source_group("video", "test.mp4", "a"), source_group("video", "test.mp4", "b"))
        self.assertNotEqual(source_group("camera", 0, "a"), source_group("camera", 0, "b"))

    def test_only_ready_prompt_compatible_model_versions_are_listed(self):
        root = self.store.root/"models"/"v1"
        root.mkdir(parents=True)
        (root/"corrected.pt").write_bytes(b"placeholder")
        row = {"status": "failed", "display_name": "A correction", "text_prompt": True, "visual_prompt": True,
               "model_family": "yoloe-26s", "parent_unchanged": True, "frozen_modules_unchanged": True,
               "detection_heads_changed": True, "checkpoint_sha256": sha256_file(root/"corrected.pt")}
        (root/"version.json").write_text(json.dumps(row))
        self.assertEqual(self.store.model_versions(), [])
        row["status"] = "ready"
        row["visual_prompt"] = False
        (root/"version.json").write_text(json.dumps(row))
        self.assertEqual(self.store.model_versions(), [])
        row["visual_prompt"] = True
        (root/"version.json").write_text(json.dumps(row))
        self.assertEqual(self.store.model_versions()[0][0], "YOLOE-26S修改版")


class SnapshotTests(unittest.TestCase):
    def test_raw_snapshot_is_independent_and_does_not_use_numbered_labels(self):
        raw = np.full((20, 30, 3), 18, np.uint8)
        drawn = np.full_like(raw, 222)
        engine = MagicMock()
        detection = Detection((2, 3, 20, 18), .9, 0, "person", display_id=2)
        engine.process_frame.return_value = drawn, [detection], "ok"
        worker = VideoWorker(engine, MagicMock(), "demo.jpg")
        worker.feedback_metadata = {"model": "A.pt", "options": {"precision": 32}}
        self.assertIsNone(worker.get_feedback_snapshot())
        worker._infer(raw)
        snapshot, boxes, metadata = worker.get_feedback_snapshot()
        np.testing.assert_array_equal(snapshot, raw)
        self.assertEqual(boxes[0]["label"], "person")
        self.assertEqual(metadata["model"], "A.pt")
        snapshot[:] = 0
        boxes[0]["xyxy"][0] = 99
        metadata["options"]["precision"] = 16
        again, again_boxes, again_metadata = worker.get_feedback_snapshot()
        np.testing.assert_array_equal(again, raw)
        self.assertEqual(again_boxes[0]["xyxy"][0], 2)
        self.assertEqual(again_metadata["options"]["precision"], 32)


class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_letterbox_draw_select_delete(self):
        canvas = CorrectionCanvas()
        canvas.resize(400, 400)
        canvas.set_sample(np.zeros((100, 200, 3), np.uint8), [])
        self.assertIsNone(canvas.to_image(QPointF(20, 20)))
        canvas.mode, canvas.label = "draw", "cup"
        QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(40, 140))
        QTest.mouseMove(canvas, QPoint(200, 260))
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(200, 260))
        self.assertEqual(canvas.boxes[0]["xyxy"], [20., 20., 100., 80.])
        canvas.mode = "select"
        QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(120, 200))
        self.assertEqual(canvas.index, 0)
        canvas.delete_selected()
        self.assertEqual(canvas.boxes, [])
        canvas.close()

    def test_editing_revokes_review_and_saving_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MagicMock()
            app.active_model_path = "A.pt"
            panel = FeedbackPanel(app, FeedbackStore(Path(directory)/"data"))
            snapshot = (np.zeros((20, 40, 3), np.uint8), [{"label": "cup", "xyxy": [1, 1, 20, 10]}],
                        {"origin_group": "one", "source": "synthetic", "prompts": ["cup"]})
            panel.set_snapshot(snapshot)
            panel.reviewed.setChecked(True)
            panel.save_sample()
            self.assertIsNone(panel.process)
            self.assertEqual(panel.store.records()[0]["status"], "reviewed")
            panel.category.setText("bottle")
            panel.canvas.index = 0
            panel.relabel()
            self.assertFalse(panel.reviewed.isChecked())
            self.assertTrue(panel.dirty)
            panel.close()

    def test_delete_and_d_shortcuts_but_text_fields_keep_their_own_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = FeedbackPanel(MagicMock(), FeedbackStore(Path(directory)/"data"))
            panel.show()
            panel.canvas.set_sample(np.zeros((20, 40, 3), np.uint8), [{"label": "cup", "xyxy": [1, 1, 10, 10]}])
            panel.canvas.index = 0
            panel.canvas.setFocus()
            self.qt_app.processEvents()
            QTest.keyClick(panel.canvas, Qt.Key_Delete)
            self.qt_app.processEvents()
            self.assertEqual(panel.canvas.boxes, [])
            self.assertEqual(panel.mode.currentIndex(), 0)
            QTest.keyClick(panel.canvas, Qt.Key_D)
            self.qt_app.processEvents()
            self.assertEqual(panel.mode.currentIndex(), 1)
            self.assertEqual(panel.canvas.mode, "draw")
            # A text field keeps its own "d" and Delete keys; the shortcuts must not fire.
            panel.category.setText("cat")
            panel.category.setCursorPosition(0)
            panel.category.setFocus()
            self.qt_app.processEvents()
            QTest.keyClick(panel.category, Qt.Key_Delete)
            self.qt_app.processEvents()
            self.assertEqual(panel.category.text(), "at")
            self.assertEqual(panel.mode.currentIndex(), 1)  # unchanged: D did not fire either
            panel.close()

    def test_new_second_box_selection_survives_list_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = FeedbackPanel(MagicMock(), FeedbackStore(directory))
            panel.canvas.set_sample(np.zeros((20, 40, 3), np.uint8), [
                {"label":"bottle", "xyxy":[1,1,10,10]}, {"label":"cup", "xyxy":[20,1,30,10]}])
            panel.canvas.index = 1
            panel.rebuild_boxes()
            self.assertEqual(panel.canvas.index, 1)
            self.assertEqual(panel.box_list.currentRow(), 1)
            self.assertEqual(panel.category.text(), "cup")
            panel.close()

    def test_training_requires_explicit_yes_and_separate_process(self):
        from feedback import PROJECT
        with tempfile.TemporaryDirectory() as directory:
            app = MagicMock()
            app.MODELS = {"A": str(PROJECT/"yoloe-26s-seg.pt")}
            app.model_combo.currentText.return_value = "A"
            app.model_combo.currentData.return_value = str(PROJECT/"yoloe-26s-seg.pt")
            app.device_combo.currentText.return_value = "CPU"
            app.release_model.return_value = True
            panel = FeedbackPanel(app, FeedbackStore(directory))
            panel.exported = Path(directory)
            with patch("feedback_ui.QMessageBox.question", return_value=QMessageBox.No), patch("feedback_ui.QProcess") as factory:
                panel.start_training()
                factory.assert_not_called()
                app.release_model.assert_not_called()
            with patch("feedback_ui.QMessageBox.question", return_value=QMessageBox.Yes), patch("feedback_ui.QProcess") as factory:
                panel.start_training()
                args = factory.return_value.start.call_args.args[1]
                self.assertIn("--run", args)
                self.assertIn(str(PROJECT/"yoloe-26s-seg.pt"), args)
                self.assertTrue(app.training_active)
                self.assertNotEqual(panel.training_output, PROJECT)
                panel.training_finished(1)
                self.assertFalse(app.training_active)
            panel.close()


class TrainingRuleTests(unittest.TestCase):
    def model(self):
        import torch
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.is_fused = False
                self.cv2 = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.BatchNorm1d(2))
                self.cv3 = torch.nn.Linear(2,2)
                self.cv4 = torch.nn.Linear(2,2)
                self.savpe = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.BatchNorm1d(2))
                self.reprta = torch.nn.Linear(2,2)
                self.proto = torch.nn.Linear(2,2)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2)), Head()])
            def get_text_pe(self):
                pass
            def get_visual_pe(self):
                pass
        return Model()

    def test_open_prompt_and_backbone_frozen_but_detection_trainable(self):
        from feedback_training import configure_trainable, set_training_mode, frozen_fingerprint
        model = self.model()
        before = frozen_fingerprint(model)
        configure_trainable(model)
        set_training_mode(model)
        self.assertTrue(all(p.requires_grad for p in model.model[-1].cv3.parameters()))
        for child in (model.model[0], model.model[-1].savpe, model.model[-1].reprta, model.model[-1].proto):
            self.assertFalse(child.training)
            self.assertFalse(any(p.requires_grad for p in child.parameters()))
        self.assertTrue(model.model[-1].cv2.training)
        self.assertFalse(model.model[-1].cv2[1].training)
        self.assertEqual(before, frozen_fingerprint(model))

    def test_batch_one_head_training_preserves_bn_statistics(self):
        import torch
        from feedback_training import configure_trainable, set_training_mode, batch_norm_statistics_fingerprint
        model = self.model()
        configure_trainable(model)
        before = batch_norm_statistics_fingerprint(model)
        set_training_mode(model)
        model.model[-1].cv2(torch.full((4, 2), 100.))
        self.assertEqual(before, batch_norm_statistics_fingerprint(model))

    def test_fused_or_missing_visual_module_is_rejected(self):
        from feedback_training import configure_trainable
        model = self.model()
        model.model[-1].is_fused = True
        with self.assertRaisesRegex(ValueError, "保留文字和视觉"):
            configure_trainable(model)
        model.model[-1].is_fused = False
        del model.model[-1].savpe
        with self.assertRaises(ValueError):
            configure_trainable(model)


if __name__ == "__main__":
    unittest.main()
