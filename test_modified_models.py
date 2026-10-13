"""Fixed-slot updates, cumulative review scopes and fail-safe publication."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication

from feedback import FeedbackStore, atomic_json, sha256_file
from feedback_training import validate_dataset
from modified_models import ModifiedModels
from model_families import model_family


class FixedSlotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = FeedbackStore(Path(self.directory.name)/"data")
        self.registry = ModifiedModels(self.store.root)

    def candidate(self, index, *, valid=True):
        job = self.registry.root/".staging"/("job"+str(index))
        job.mkdir(parents=True)
        (job/"corrected.pt").write_bytes(b"candidate"+bytes([index]))
        (job/"parent-copy.pt").write_bytes(b"temporary-parent")
        status = {"status": "ready" if valid else "failed", "text_prompt": True, "visual_prompt": True,
            "parent_unchanged": True, "frozen_modules_unchanged": True, "detection_heads_changed": True,
            "checkpoint_sha256": sha256_file(job/"corrected.pt"), "parent": "original", "parent_sha256": "parent"}
        return job, status

    def commit(self, index, family="yoloe-26s"):
        job, status = self.candidate(index)
        previous = self.registry.current(family)
        with self.registry.updating(family):
            result = self.registry.publish(family, job, status, previous)
        self.registry.cleanup_job_weights(job)
        return result

    def test_two_updates_keep_one_name_one_path_one_completed_checkpoint(self):
        first = self.commit(1)
        path = self.registry.listed()[0][1]
        second = self.commit(2)
        listed = self.registry.listed()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0][0], "YOLOE-26S修改版")
        self.assertEqual(listed[0][1], path)
        self.assertEqual(second["update_count"], 2)
        self.assertNotEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
        self.assertEqual(len(list(self.registry.root.rglob("*.pt"))), 1)

    def test_separate_families_have_separate_single_modified_models(self):
        self.commit(1)
        self.commit(2, "yoloe-26l")
        self.assertEqual({row[0] for row in self.registry.listed()}, {"YOLOE-26S修改版", "YOLOE-26L修改版"})

    def test_failed_candidate_does_not_replace_working_weights(self):
        previous = self.commit(1)
        job, status = self.candidate(2, valid=False)
        with self.registry.updating("yoloe-26s"), self.assertRaises(ValueError):
            self.registry.publish("yoloe-26s", job, status, previous)
        self.assertEqual(self.registry.current("yoloe-26s"), previous)

    def test_tampered_candidate_does_not_replace_working_weights(self):
        previous = self.commit(1)
        job, status = self.candidate(2)
        (job/"corrected.pt").write_bytes(b"changed")
        with self.registry.updating("yoloe-26s"), self.assertRaises(ValueError):
            self.registry.publish("yoloe-26s", job, status, previous)
        self.assertEqual(self.registry.current("yoloe-26s"), previous)

    def test_metadata_commit_failure_restores_old_weights_without_new_entry(self):
        previous = self.commit(1)
        job, status = self.candidate(2)
        real = atomic_json
        def fail(path, value):
            if Path(path).name == "current.json":
                raise OSError("simulated metadata failure")
            return real(path, value)
        with self.registry.updating("yoloe-26s"), patch("modified_models.atomic_json", side_effect=fail), self.assertRaises(OSError):
            self.registry.publish("yoloe-26s", job, status, previous)
        self.assertEqual(self.registry.current("yoloe-26s"), previous)

    def test_first_commit_failure_leaves_no_half_registered_model(self):
        job, status = self.candidate(1)
        real = atomic_json
        def fail(path, value):
            if Path(path).name == "current.json":
                raise OSError("simulated metadata failure")
            return real(path, value)
        with self.registry.updating("yoloe-26s"), patch("modified_models.atomic_json", side_effect=fail), self.assertRaises(OSError):
            self.registry.publish("yoloe-26s", job, status)
        self.assertEqual(self.registry.listed(), [])

    def test_uncommitted_journal_is_not_loaded_and_recovers_under_writer_lock(self):
        previous = self.commit(1)
        job, status = self.candidate(2)
        slot = self.registry.slot("yoloe-26s")
        backup = job/"prior.pt"
        backup.write_bytes((slot/"modified.pt").read_bytes())
        new = dict(previous, checkpoint_sha256=status["checkpoint_sha256"], update_count=2)
        (slot/"modified.pt").write_bytes((job/"corrected.pt").read_bytes())
        atomic_json(slot/"publish.pending.json", {"new": new, "previous": previous,
                    "backup": str(backup.relative_to(self.registry.root))})
        with self.assertRaises(ValueError):
            self.registry.current("yoloe-26s")
        with self.registry.updating("yoloe-26s"):
            self.assertEqual(self.registry.current("yoloe-26s")["update_count"], 2)
        self.assertFalse(backup.exists())

    def test_concurrent_update_and_path_escape_are_rejected(self):
        with self.registry.updating("yoloe-26s"):
            with self.assertRaises(RuntimeError):
                with self.registry.updating("yoloe-26s"):
                    pass
        with self.assertRaises(ValueError):
            self.registry.cleanup_job_weights(self.registry.root/"yoloe-26s")
        with self.assertRaises(ValueError):
            self.registry.slot("../escape")

    def test_old_versions_are_deduplicated_read_only(self):
        for index in range(2):
            job, status = self.candidate(index)
            legacy = self.registry.root/("old"+str(index))
            legacy.mkdir()
            (legacy/"corrected.pt").write_bytes((job/"corrected.pt").read_bytes())
            atomic_json(legacy/"version.json", dict(status, model_family="yoloe-26s"))
        before = {str(path): sha256_file(path) for path in self.registry.root.rglob("version.json")}
        self.assertEqual(len(self.registry.listed()), 1)
        self.assertEqual(before, {str(path): sha256_file(path) for path in self.registry.root.rglob("version.json")})


class CumulativeDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = FeedbackStore(self.directory.name)

    def save(self, index, label="cup", family="yoloe-26s", split="train", reviewed=True):
        return self.store.save(np.full((24, 32, 3), index, np.uint8), [{"label": label, "xyxy": [2, 3, 20, 21]}],
            classes=[label], scene="scene"+str(index), split=split, reviewed=reviewed,
            provenance={"source": "synthetic", "origin_group": "origin"+str(index), "model_family": family})

    def test_cumulative_export_keeps_old_and_new_corrections_without_other_families(self):
        self.save(1)
        self.save(2, "bottle")
        self.save(3, split="val")
        self.save(4, family="yoloe-26l")
        self.save(5, reviewed=False)
        manifest = validate_dataset(self.store.export_cumulative("yoloe-26s"))
        self.assertEqual(manifest["classes"], ["cup", "bottle"])
        self.assertEqual(len(manifest["samples"]), 3)
        self.assertTrue(manifest["per_sample_review_scope"])
        self.assertEqual(manifest["excluded_drafts"], 1)
        self.assertEqual({row["model_family"] for row in manifest["samples"]}, {"yoloe-26s"})

    def test_same_image_can_have_independent_corrections_for_two_families(self):
        self.save(1)
        self.save(1, "bottle", "yoloe-26l")
        self.save(1, "cup", "yoloe-26s", reviewed=False)
        self.assertEqual(len(self.store.records()), 2)
        self.assertEqual(self.store.records("yoloe-26l")[0]["boxes"][0]["label"], "bottle")
        self.assertEqual(self.store.records("yoloe-26s")[0]["status"], "draft")

    def test_global_scene_split_safety_is_retained_across_families(self):
        self.save(1)
        with self.assertRaises(ValueError):
            self.save(1, family="yoloe-26l", split="val")

    def test_no_reviewed_corrections_reports_sample_requirement(self):
        self.save(1, reviewed=False)
        with self.assertRaisesRegex(ValueError, "已人工审核的训练图和一组验证图"):
            self.store.export_cumulative("yoloe-26s")

    def test_pf_and_prompted_files_share_family_but_other_shapes_do_not(self):
        self.assertEqual(model_family("yoloe-26s-seg.pt"), model_family("yoloe-26s-seg-pf.pt"))
        self.assertEqual(model_family("models/yoloe-26s/modified.pt"), "yoloe-26s")
        self.assertEqual(model_family("IDEA-Research/grounding-dino-tiny"), "")


class ModifiedUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_updated_bytes_at_same_path_force_model_reload(self):
        from main import AIApp
        from engines import OpenVocabEngine
        app = AIApp()
        path = str(Path(tempfile.gettempdir())/"models/yoloe-26s/modified.pt")
        app.model_combo.addItem("YOLOE-26S修改版", path)
        app._feedback_versions[path] = {"model_family": "yoloe-26s", "checkpoint_sha256": "old"}
        app.model_combo.setCurrentIndex(app.model_combo.findData(path))
        app.active_model_path, app.active_model_sha256 = path, "old"
        app.engine = OpenVocabEngine(); app.engine.model = object()
        app.device_combo.setCurrentText("CPU")
        def refresh():
            app._feedback_versions[path] = {"model_family": "yoloe-26s", "checkpoint_sha256": "new"}
        try:
            with patch.object(app, "refresh_feedback_models", side_effect=refresh), \
                    patch.object(app, "_load_model_path", return_value=True) as load:
                self.assertTrue(app._ensure_detection_model(["cup"]))
            load.assert_called_once_with(path)
        finally:
            app.engine.model = None
            app.close()


if __name__ == "__main__":
    unittest.main()
