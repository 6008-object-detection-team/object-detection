"""Standard-library/mock tests; no network, torch, model loading or training."""
import contextlib
import io
import json
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tools.train_p2_probe import (
    after_probe_batch, apply_thread_limit, before_probe_batch, build_parser,
    ensure_weights, epoch_progress, resource_check, run_variants, sha256_file,
    source_hashes, tensor_coverage, training_options, validate_dataset,
)
from tools.prepare_training_probe import yolo_labels


class ProbeConfigurationTests(unittest.TestCase):
    def test_default_is_plan_and_only_cpu_can_be_selected(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.run)
        self.assertEqual(args.device, "CPU")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--device", "CUDA"])

    def test_probe_epochs_cannot_start_a_long_training_job(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--epochs", "1000"])

    def test_thread_limit_covers_framework_cpu_select_device_reset(self):
        torch_module, cv2_module = MagicMock(), MagicMock()
        torch_utils = types.SimpleNamespace(NUM_THREADS=8)
        apply_thread_limit(torch_module, cv2_module, torch_utils)
        self.assertEqual(torch_utils.NUM_THREADS, 2)
        torch_module.set_num_threads.assert_called_once_with(2)
        cv2_module.setNumThreads.assert_called_once_with(1)

    def test_successful_batches_cannot_trigger_native_epoch_restart(self):
        trainer = types.SimpleNamespace(_oom_retries=0, stop=False)
        for _ in range(3):
            before_probe_batch(trainer)
            self.assertGreaterEqual(trainer._oom_retries, 3)
            after_probe_batch(trainer)
            self.assertFalse(trainer._oom_retries and not trainer.stop)
        self.assertEqual(trainer._oom_retries, 0)

    def test_cpu_oom_raises_on_the_first_attempt_without_end_callback(self):
        trainer = types.SimpleNamespace(_oom_retries=0)
        retries = []
        before_probe_batch(trainer)
        # Match the installed trainer's catch condition, after forward fails.
        with self.assertRaisesRegex(RuntimeError, "CPU out of memory"):
            try:
                raise RuntimeError("CPU out of memory")
            except RuntimeError:
                if trainer._oom_retries >= 3:
                    raise
                retries.append("retry")
        self.assertEqual(retries, [])
        self.assertEqual(trainer._oom_retries, 3)

    def test_final_best_validation_is_not_a_fourth_training_epoch(self):
        trainer = types.SimpleNamespace(epoch=2, epochs=3)
        training = epoch_progress(trainer, 0)
        self.assertEqual((training["phase"], training["epoch"]), ("training_epoch", 3))
        trainer.epoch = 3
        validation = epoch_progress(trainer, 0)
        self.assertEqual(validation["phase"], "best_checkpoint_validation")
        self.assertEqual(validation["after_training_epochs"], 3)
        self.assertNotIn("epoch", validation)

    def test_source_snapshot_includes_native_training_contract_and_model_yaml(self):
        hashes = source_hashes()
        for path in ("tools/train_p2_probe.py", "resource_safety.py", "evaluate_mini_dataset.py",
                     "tools/prepare_training_probe.py", "installed/ultralytics/engine/trainer.py",
                     "installed/ultralytics/cfg/models/26/yolo26-p2.yaml"):
            self.assertEqual(len(hashes[path]), 64)

    def test_both_variants_use_same_small_weights_and_safe_cpu_options(self):
        args = build_parser().parse_args([])
        standard = training_options(args, "standard", args.weights)
        p2 = training_options(args, "p2", args.weights)
        self.assertEqual(standard["pretrained"], p2["pretrained"])
        for options in (standard, p2):
            self.assertEqual((options["device"], options["batch"], options["imgsz"], options["epochs"]), ("cpu", 1, 416, 3))
            self.assertEqual(options["workers"], 0)
            self.assertFalse(options["cache"])
            self.assertFalse(options["amp"])
            self.assertFalse(options["plots"])
            self.assertFalse(options["resume"])

    def test_ram_guard_stops_before_model_factory(self):
        with self.assertRaisesRegex(RuntimeError, "RAM"):
            resource_check(".", ram_reader=lambda: 3000, disk_reader=lambda _: types.SimpleNamespace(free=10 * 1024 ** 3))
        with self.assertRaisesRegex(RuntimeError, "磁盘"):
            resource_check(".", ram_reader=lambda: 4000, disk_reader=lambda _: types.SimpleNamespace(free=100 * 1024 ** 2))

    def test_name_and_shape_coverage_does_not_count_partial_matches(self):
        report = tensor_coverage({"backbone": (4, 4), "head": (80, 4)},
                                 {"backbone": (4, 4), "head": (3, 4), "p2": (8,)})
        self.assertEqual(report["matched_tensors"], 1)
        self.assertEqual(report["matched_scalar_values"], 16)
        self.assertEqual(report["target_scalar_values"], 36)
        self.assertEqual(report["unmatched_keys"], ["head", "p2"])


class FakeResponse(io.BytesIO):
    headers = {}


class SmallAssetTests(unittest.TestCase):
    def test_mock_stream_records_source_and_reuses_verified_local_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "yolo26n.pt"
            opener = MagicMock(return_value=FakeResponse(b"weights" * 1024))
            first = ensure_weights(path, opener=opener)
            self.assertEqual(first["provenance"], "downloaded_official_asset")
            self.assertEqual(first["sha256"], sha256_file(path))
            second = ensure_weights(path, opener=MagicMock(side_effect=AssertionError("network")))
            self.assertEqual(first, second)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ensure_weights(path, opener=MagicMock(side_effect=AssertionError("network")))

    def test_other_model_name_never_reaches_download(self):
        opener = MagicMock()
        with self.assertRaisesRegex(ValueError, "其他模型"):
            ensure_weights("yolo26l.pt", opener=opener)
        opener.assert_not_called()


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "annotations").mkdir()
        categories = [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"}, {"id": 77, "name": "cell phone"}]
        images, annotations = [], []
        for index in range(60):
            image_id = index + 1
            images.append({"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 640, "height": 480})
            annotations.append({"id": image_id, "image_id": image_id, "category_id": categories[index % 3]["id"],
                                "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0})
        source = {"images": images, "annotations": annotations, "categories": categories}
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps(source))
        self.source_sha = sha256_file(self.source)
        for split, selected in (("train", images[:48]), ("val", images[48:])):
            ids = {image["id"] for image in selected}
            subset = {"images": selected, "annotations": [ann for ann in annotations if ann["image_id"] in ids], "categories": categories}
            (self.data / "annotations" / f"{split}.json").write_text(json.dumps(subset))
            (self.data / "images" / split).mkdir(parents=True)
            (self.data / "labels" / split).mkdir(parents=True)
            labels, _, _ = yolo_labels(subset)
            for image in selected:
                # Unique bytes and a sufficient JPEG SOF geometry header.
                payload = b"\xff\xd8\xff\xc0" + struct.pack(">HBHHB", 8, 8, 480, 640, 0) + str(image["id"]).encode() + b"\xff\xd9"
                (self.data / "images" / split / image["file_name"]).write_bytes(payload)
                (self.data / "labels" / split / f"{image['id']:012d}.txt").write_bytes(labels[image["id"]])
        (self.data / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [bottle, cup, cell phone]\n")
        image_records = [{"image_file": str(path.relative_to(self.data)), "sha256": sha256_file(path)}
                         for path in (self.data / "images").rglob("*.jpg")]
        (self.data / "manifest.json").write_text(json.dumps({"source_sha256": self.source_sha,
                                                            "files": {}, "images": image_records}))
        self.excluded = self.root / "excluded"
        self.excluded.mkdir()
        (self.excluded / "annotations.json").write_text(json.dumps({"images": [{"id": 1001}]}))
        (self.excluded / "references.json").write_text(json.dumps({"references": [{"original_image_id": 1002}]}))
        self.args = types.SimpleNamespace(data=self.data / "data.yaml", source_annotations=self.source,
                                          exclude_dataset=self.excluded)

    def test_original_ids_and_human_labels_are_validated_without_models(self):
        report = validate_dataset(self.args, expected_source_sha=self.source_sha)
        self.assertEqual(report["splits"]["train"]["images"], 48)
        self.assertEqual(report["splits"]["val"]["images"], 12)
        self.assertEqual(report["splits"]["train"]["yolo_labels"], 48)

    def test_test_image_leakage_is_rejected(self):
        (self.excluded / "annotations.json").write_text(json.dumps({"images": [{"id": 1}]}))
        with self.assertRaisesRegex(ValueError, "隔离"):
            validate_dataset(self.args, expected_source_sha=self.source_sha)

    def test_pseudo_or_changed_label_is_rejected(self):
        (self.data / "labels" / "train" / "000000000001.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        with self.assertRaisesRegex(ValueError, "人工 bbox"):
            validate_dataset(self.args, expected_source_sha=self.source_sha)


class SequentialLifecycleTests(unittest.TestCase):
    def test_previous_trainer_is_released_before_next_factory_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(["--output", str(Path(temporary) / "results")])
            trainers = []
            def factory(options, progress):
                if trainers:
                    self.assertIsNone(trainers[-1].model)
                    self.assertIsNone(trainers[-1].optimizer)
                trainer = types.SimpleNamespace(model=object(), optimizer=object(), ema=object(),
                    train=MagicMock(), probe_transfer={"matched_tensors": 1},
                    validator=types.SimpleNamespace(metrics=types.SimpleNamespace(
                        results_dict={"metrics/mAP50-95(B)": 0.1}, summary=lambda: [{"Class": "cup", "mAP50-95": 0.1}])),
                    best=Path(temporary) / "missing-best.pt", last=Path(temporary) / "missing-last.pt")
                trainers.append(trainer)
                return trainer
            report = run_variants(args, {"status": "validated"}, {"path": str(args.weights)},
                                  factory=factory, checker=MagicMock())
            self.assertEqual(report["status"], "complete_functionality_probe")
            self.assertEqual(len(trainers), 2)
            self.assertTrue(all(trainer.model is None for trainer in trainers))

    def test_failure_does_not_start_second_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(["--output", str(Path(temporary) / "results")])
            trainer = types.SimpleNamespace(model=object(), train=MagicMock(side_effect=RuntimeError("RAM low")))
            factory = MagicMock(return_value=trainer)
            with self.assertRaisesRegex(RuntimeError, "RAM low"):
                run_variants(args, {}, {"path": str(args.weights)}, factory=factory, checker=MagicMock())
            factory.assert_called_once()
            self.assertIsNone(trainer.model)
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(report["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
