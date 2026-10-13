"""One-click edits, explicit demo data, quiet failures and fixed-slot retests."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import QApplication

from feedback import FeedbackStore, PROJECT, atomic_json, sha256_file
from feedback_training import validate_dataset
from modified_models import ModifiedModels
from i18n import render


class DemoDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = FeedbackStore(Path(directory.name)/"feedback")

    def demo(self, number=1, label="cup", family="yoloe-26s", accepted=True):
        return self.store.save_demo(np.full((32, 48, 3), number, np.uint8),
            [{"label": label, "xyxy": [2, 3, 25, 30]}], classes=[label],
            provenance={"source": str(PROJECT/"datasets/mini_eval/images/fixture.jpg"),
                        "origin_group": "same-directory", "model_family": family}, accepted=accepted)

    def strict(self, number, split):
        return self.store.save(np.full((32, 48, 3), number, np.uint8), [], classes=["bottle"],
            scene="strict"+str(number), split=split, reviewed=True,
            provenance={"source": "new-personal-scenes", "origin_group": "origin"+str(number), "model_family": "yoloe-26s"})

    def test_one_image_exports_real_training_data_without_independent_validation(self):
        self.demo()
        dataset = self.store.export_demo("yoloe-26s")
        manifest = validate_dataset(dataset)
        config = json.loads((dataset/"data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["training_mode"], "correction_demo")
        self.assertFalse(manifest["independent_validation"])
        self.assertNotIn("val", config)
        self.assertEqual(len(manifest["samples"]), 1)
        self.assertEqual(manifest["samples"][0]["status"], "demo_accepted")

    def test_demo_does_not_relabel_or_move_strict_test_record(self):
        row = self.strict(1, "test")
        path = Path(self.store.records()[0]["record_path"])
        original_hash = sha256_file(path)
        self.demo(1)
        validate_dataset(self.store.export_demo("yoloe-26s"))
        self.assertEqual(sha256_file(path), original_hash)
        self.assertEqual(self.store.records()[0]["split"], "test")
        self.assertEqual(self.store.records()[0]["revision"], row["revision"])

    def test_replays_previous_demo_edits_and_only_strict_training_rows(self):
        self.strict(3, "train"); self.strict(4, "val"); self.strict(5, "test")
        self.demo(1); self.demo(2, "pot"); self.demo(6, family="yoloe-26x")
        self.demo(7, accepted=False)
        manifest = validate_dataset(self.store.export_demo("yoloe-26s"))
        self.assertEqual(len(manifest["samples"]), 3)
        self.assertEqual(set(manifest["classes"]), {"bottle", "cup", "pot"})
        self.assertEqual({row["split"] for row in manifest["samples"]}, {"train"})

    def test_new_revision_replaces_old_edit_for_same_image_without_duplication(self):
        first = self.demo(1)
        second = self.demo(1, "pot")
        manifest = validate_dataset(self.store.export_demo("yoloe-26s"))
        self.assertEqual(len(manifest["samples"]), 1)
        self.assertEqual(manifest["classes"], ["pot"])
        self.assertNotEqual(first["revision"], second["revision"])

    def test_draft_does_not_implicitly_start_or_join_training(self):
        self.demo(1); self.demo(1, "pot", accepted=False)
        with self.assertRaises(ValueError):
            self.store.export_demo("yoloe-26s")
        self.assertEqual(self.store.records(), [])

    def test_demo_cannot_claim_independent_validation_even_with_self_consistent_hash(self):
        self.demo()
        dataset = self.store.export_demo("yoloe-26s")
        manifest = json.loads((dataset/"manifest.json").read_text(encoding="utf-8"))
        manifest["independent_validation"] = True
        atomic_json(dataset/"manifest.json", manifest)
        (dataset/"manifest.sha256").write_text(sha256_file(dataset/"manifest.json"), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "独立验证"):
            validate_dataset(dataset)

    def test_strict_export_still_requires_independent_validation(self):
        self.strict(1, "train")
        with self.assertRaises(ValueError):
            self.store.export_cumulative("yoloe-26s")

    def test_demo_export_retains_resource_guards(self):
        self.demo()
        with patch("resource_safety.available_ram_mb", return_value=100):
            with self.assertRaisesRegex(RuntimeError, "内存"):
                self.store.export_demo("yoloe-26s")


class OneClickUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        from main import AIApp
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.app = AIApp()
        self.panel = self.app.feedback_panel
        self.panel.store = FeedbackStore(Path(directory.name)/"feedback")
        self.panel.refresh()
        self.app.device_combo.setCurrentText("CPU")
        self.app.show()
        self.app.tabs.setCurrentWidget(self.panel)
        self.qt.processEvents()
        self.image = np.full((32, 48, 3), 125, np.uint8)
        self.panel.set_snapshot((self.image, [{"label": "cup", "xyxy": [2, 3, 25, 30]}],
            {"source": "synthetic", "origin_group": "same-directory", "model_family": "yoloe-26s", "prompts": ["cup"]}))

    def tearDown(self):
        self.app.worker = None
        self.app.engine.model = None
        self.app.training_active = False
        self.app._feedback_retest = None
        self.panel.process = None
        self.panel.dirty = False
        self.app.close()
        self.app.deleteLater()
        self.qt.processEvents()

    def test_default_page_hides_review_split_save_and_export_controls(self):
        self.assertTrue(self.panel.quick_button.isVisible())
        for control in (self.panel.reviewed, self.panel.scene, self.panel.split, self.panel.sample_list, self.panel.train_button):
            self.assertFalse(control.isVisible())
        self.panel.advanced.setChecked(True)
        self.assertTrue(self.panel.reviewed.isVisible())
        self.assertTrue(self.panel.train_button.isVisible())

    def test_one_click_commits_typed_label_auto_adds_class_saves_exports_and_starts(self):
        self.panel.canvas.index = 0
        self.panel.rebuild_boxes()
        self.panel.category.setText("pot")
        # No Relabel, Save, Review or Export click.
        with patch.object(self.panel, "_launch_training") as launch, patch("feedback_ui.QMessageBox") as dialogs:
            self.panel.quick_button.click()
            launch.assert_called_once()
            dialogs.question.assert_not_called(); dialogs.warning.assert_not_called()
        rows = self.panel.store.demo_records("yoloe-26s")
        self.assertEqual(rows[0]["boxes"][0]["label"], "pot")
        self.assertIn("pot", rows[0]["classes"])
        self.assertEqual(self.panel.store.records(), [])
        self.assertFalse(self.panel.dirty)
        self.assertEqual(validate_dataset(self.panel.exported)["training_mode"], "correction_demo")
        self.assertEqual(launch.call_args.args[-1], 20)

    def test_click_starts_one_child_process_and_blocks_duplicate_updates(self):
        with patch("feedback_ui.QProcess") as factory, patch("feedback_ui.QMessageBox") as dialogs:
            self.panel.quick_button.click()
            self.panel.quick_update()
            factory.return_value.start.assert_called_once()
            args = factory.return_value.start.call_args.args[1]
            self.assertIn("--run", args); self.assertIn("--family", args)
            self.assertFalse(self.panel.quick_button.isEnabled())
            self.assertTrue(self.app.training_active)
            dialogs.question.assert_not_called()
            self.panel.training_finished(1)
        self.assertFalse(self.app.training_active)
        self.assertTrue(self.panel.quick_button.isEnabled())

    def test_missing_image_and_unsupported_model_fail_inline_without_confirmation(self):
        self.panel.canvas.image = None
        with patch("feedback_ui.QMessageBox") as dialogs, patch.object(self.panel, "_launch_training") as launch:
            self.panel.quick_update()
            launch.assert_not_called(); dialogs.warning.assert_not_called(); dialogs.question.assert_not_called()
        self.assertIn("请先打开图片", self.panel.summary.text())

    def test_switching_images_stashes_draft_without_training_or_popup(self):
        self.panel.canvas.index = 0; self.panel.rebuild_boxes(); self.panel.category.setText("pot")
        with patch("feedback_ui.QMessageBox") as dialogs:
            self.assertTrue(self.panel.may_replace())
            dialogs.question.assert_not_called()
        self.assertIsNone(self.panel.process)
        row = self.panel.store.demo_records()[0]
        self.assertEqual(row["status"], "demo_draft")
        self.assertEqual(row["boxes"][0]["label"], "pot")

    def test_demo_saved_item_reopens_from_separate_image_ledger(self):
        with patch.object(self.panel, "_launch_training"):
            self.panel.quick_update()
        self.panel.refresh()
        item = self.panel.sample_list.item(0)
        self.panel.load_selected(item)
        np.testing.assert_array_equal(self.panel.canvas.image, self.image)

    def test_english_translates_new_workflow_and_demo_progress(self):
        self.app.language_combo.setCurrentIndex(self.app.language_combo.findData("en"))
        self.assertEqual(self.panel.quick_button.text(), "Update model & retest this image")
        self.assertIn("Advanced samples", self.panel.advanced.text())
        process = self.panel.process = MagicMock()
        process.readAllStandardOutput.return_value = json.dumps({"epoch": 1, "train_loss": .3, "val_loss": None,
            "training_mode": "correction_demo"}).encode()+b"\n"
        self.panel.read_training_output()
        self.assertIn("no independent validation", self.panel.log.toPlainText())

    def test_automatic_retest_selects_modified_checkpoint_and_uses_only_image_and_text(self):
        store = self.panel.store
        row = store.save_demo(self.image, self.panel.canvas.boxes, classes=["cup"], provenance=self.panel.provenance)
        registry = ModifiedModels(store.root)
        job = registry.root/".staging/test"
        job.mkdir(parents=True)
        (job/"corrected.pt").write_bytes(b"unit-test-checkpoint")
        status = {"status": "ready", "text_prompt": True, "visual_prompt": True, "parent_unchanged": True,
            "frozen_modules_unchanged": True, "detection_heads_changed": True, "demo_exposure": True,
            "checkpoint_sha256": sha256_file(job/"corrected.pt")}
        with registry.updating("yoloe-26s"):
            result = registry.publish("yoloe-26s", job, status)
        quick = {"record": row, "family": "yoloe-26s", "classes": ["cup"], "audit_dir": job,
            "image_path": store.root/"demo/images"/(row["image_id"]+".png")}
        def start():
            self.app.worker = MagicMock()
        with patch.object(self.app, "start_text_stream", side_effect=start) as launch:
            self.app.retest_feedback(quick)
            launch.assert_called_once()
        self.assertEqual(self.app.model_combo.currentText(), "YOLOE-26S修改版")
        self.assertEqual(self.app.model_combo.currentData(), str(registry.slot("yoloe-26s")/"modified.pt"))
        self.assertEqual(self.app.prompt_input.text(), "cup")
        self.app._feedback_cached = (self.image, [], {"model_sha256": result["checkpoint_sha256"]})
        self.app.finish_feedback_retest()
        audit = json.loads((job/"retest.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["human_boxes_used_in_inference"])
        self.assertEqual(audit["predictions"], [])  # Not replaced by the human box.
        self.assertIn("本图已自动重测", self.panel.summary.text())


if __name__ == "__main__":
    unittest.main()
