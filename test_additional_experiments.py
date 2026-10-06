"""CPU-only tests for class mapping, perturbations and crash recovery."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from detection_utils import Detection
from tools.run_additional_experiments import (
    CheckpointStore, build_parser, loose_reference_box, mapped_records,
    load_profile, mask_provenance, perturb_image, prepare_text_prompt,
    run_scenario, scenario_config, skip_reason,
)


class MappingAndPerturbationTests(unittest.TestCase):
    def test_synonym_and_distractor_indices_map_to_original_categories(self):
        predictions = [Detection((1, 2, 6, 8), 0.8, 2, "smartphone"),
                       Detection((0, 0, 9, 9), 0.99, 3, "remote control")]
        records = mapped_records(predictions, 123, [44, 47, 77, None, None])
        self.assertEqual(records, [{"image_id": 123, "category_id": 77,
                                    "bbox": [1., 2., 5., 6.], "score": 0.8}])
        with self.assertRaisesRegex(ValueError, "类别 ID"):
            mapped_records([Detection((1, 1, 2, 2), 0.5, 5, "unknown")], 123, [44, 47, 77, None, None])

    def test_visual_class_zero_maps_to_each_reference_category(self):
        result = mapped_records([Detection((1, 1, 5, 7), 0.5, 0, "reference")], 10,
                                [44, 47, 77], visual_category=77)
        self.assertEqual(result[0]["category_id"], 77)

    def test_each_photometric_perturbation_preserves_shape_type_and_source(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(31, 47, 3), dtype=np.uint8)
        original = frame.copy()
        configs = [{"kind": "dark", "factor": 0.6}, {"kind": "motion_blur", "kernel": 9},
                   {"kind": "jpeg", "quality": 35}]
        for config in configs:
            with self.subTest(kind=config["kind"]):
                changed = perturb_image(frame, config)
                self.assertEqual(changed.shape, frame.shape)
                self.assertEqual(changed.dtype, np.uint8)
                self.assertFalse(np.array_equal(changed, frame))
                np.testing.assert_array_equal(frame, original)

    def test_loose_box_expands_around_center_and_clips_to_original_image(self):
        self.assertEqual(loose_reference_box([10, 10, 30, 30], (40, 50, 3)), [7.5, 7.5, 32.5, 32.5])
        self.assertEqual(loose_reference_box([0, 0, 50, 40], (40, 50, 3)), [0., 0., 50., 40.])

    def test_cpu_skips_fp16_without_a_device_fallback(self):
        args = build_parser().parse_args([])
        dataset = {"classes": ["bottle", "cup", "cell phone"], "category_ids": [44, 47, 77]}
        for name in ("text_fp16", "visual_fp16"):
            self.assertIn("等待 CUDA", skip_reason(scenario_config(name, dataset, args), args, {"available": False}))

    def test_human_mask_metadata_must_match_the_independent_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "0.png").write_bytes(b"source-mask")
            metadata = {"references": [{"class_id": 0, "original_image_id": 88,
                                        "original_annotation_id": 901, "file": "0.png"}]}
            (root / "metadata.json").write_text(json.dumps(metadata))
            dataset = {"references": [{"class_id": 0, "original_image_id": 88,
                                       "original_annotation_id": 901}]}
            args = types.SimpleNamespace(reference_masks=root)
            result = mask_provenance(args, dataset)
            self.assertTrue(result["available"])
            self.assertEqual(result["references"]["0"]["file"], "0.png")
            dataset["references"][0]["original_image_id"] = 10
            with self.assertRaisesRegex(ValueError, "独立参考"):
                mask_provenance(args, dataset)


class CheckpointAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "images").mkdir()
        for name in ("one.jpg", "two.jpg"):
            (self.root / "images" / name).write_bytes(b"image-stub")
        self.dataset = {
            "root": self.root, "classes": ["bottle", "cup", "cell phone"],
            "category_ids": [44, 47, 77], "references": [],
            "images": [{"id": index, "file_name": name, "width": 16, "height": 16}
                       for index, name in [(1, "one.jpg"), (2, "two.jpg")]],
            "annotations": {},
        }
        self.args = build_parser().parse_args([])
        self.config = scenario_config("text_baseline", self.dataset, self.args)
        self.manifest = {"dataset_sha256": "dataset-hash", "model_sha256": "weight-hash",
                         "source_hashes": {"runner": "code-hash"}, "device": "CPU"}
        self.store = CheckpointStore(self.root / "result", self.manifest)
        self.engine = types.SimpleNamespace(
            model=types.SimpleNamespace(set_classes=MagicMock()), current_classes=[],
            detect=MagicMock(return_value=[Detection((1, 1, 5, 7), 0.8, 0, "bottle")]),
            unload=MagicMock(),
        )
        self.factory = MagicMock(return_value=self.engine)
        self.decoder = MagicMock(return_value=np.zeros((16, 16, 3), np.uint8))
        prompt = patch("tools.run_additional_experiments.prepare_text_prompt", return_value={
            "phase": "text", "encoding_ms": 1.0, "profile_reused": False,
        })
        prompt.start()
        self.addCleanup(prompt.stop)

    def evaluator_patch(self):
        return patch("tools.run_additional_experiments.coco_metrics", return_value={
            "metrics": {"AP": 0.5, "APsmall": 0.4}, "per_class": {}, "coco_max_dets": [1, 10, 100],
        })

    def run_scenario(self):
        with self.evaluator_patch():
            return run_scenario(self.config, self.dataset, self.args, self.store, {"available": False},
                                engine_factory=self.factory, decoder=self.decoder)

    def test_recovery_rejects_changed_weight_dataset_source_and_device_hashes(self):
        for key in ("model_sha256", "dataset_sha256", "source_hashes", "device"):
            with self.subTest(key=key):
                changed = dict(self.manifest, **{key: "changed"})
                with self.assertRaisesRegex(ValueError, "哈希"):
                    CheckpointStore(self.store.output, changed, resume=True)
        CheckpointStore(self.store.output, self.manifest, resume=True)

    def test_existing_output_requires_explicit_resume(self):
        with self.assertRaisesRegex(ValueError, "--resume"):
            CheckpointStore(self.store.output, self.manifest)

    def test_complete_unit_is_not_used_for_warmup_or_inferred_again(self):
        previous = [{"image_id": 1, "category_id": 44, "bbox": [1., 1., 4., 6.], "score": 0.7}]
        self.store.save_unit(self.config, "text", 1, previous, 12.0)
        result = self.run_scenario()
        # The only new image is decoded for warmup and its timed inference.
        self.assertEqual(self.engine.detect.call_count, 2)
        self.assertTrue(all(call.args[0].name == "two.jpg" for call in self.decoder.call_args_list))
        self.assertEqual(result["cold_session"]["units_reused"], 1)
        self.assertEqual(result["cold_session"]["units_executed"], 1)
        self.engine.unload.assert_called_once()
        predictions = json.loads((self.store.output / result["predictions_file"]).read_text())
        self.assertEqual(predictions[0], previous[0])

    def test_completed_summary_resumes_without_loading_or_decoding(self):
        first = self.run_scenario()
        self.factory.reset_mock()
        self.decoder.reset_mock()
        second = self.run_scenario()
        self.assertEqual(first, second)
        self.factory.assert_not_called()
        self.decoder.assert_not_called()

    def test_all_image_checkpoints_complete_but_missing_summary_needs_no_model(self):
        for item in self.dataset["images"]:
            self.store.save_unit(self.config, "text", item["id"], [], 2.0)
        result = self.run_scenario()
        self.factory.assert_not_called()
        self.decoder.assert_not_called()
        self.assertEqual(result["prediction_count"], 0)
        self.assertEqual(result["cold_session"]["units_reused"], 2)

    def test_checkpoint_rejects_changed_scene_options(self):
        self.store.save_unit(self.config, "text", 1, [], 3.0)
        changed = dict(self.config, imgsz=768)
        with self.assertRaisesRegex(ValueError, "配置"):
            self.store.load_unit(changed, "text", 1)

    def test_checkpoint_rejects_corrupt_prediction_payload(self):
        self.store.save_unit(self.config, "text", 1, [], 3.0)
        path = self.store.unit_path("text_baseline", "text", 1)
        payload = json.loads(path.read_text())
        payload["latency_ms"] = 30.0
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "结果哈希"):
            self.store.load_unit(self.config, "text", 1)

    def test_failure_preserves_completed_image_and_always_unloads_model(self):
        # Warmup, first timed image, then the next timed image fails.
        self.engine.detect.side_effect = [[], [], RuntimeError("resource limit")]
        with self.evaluator_patch(), self.assertRaisesRegex(RuntimeError, "resource limit"):
            run_scenario(self.config, self.dataset, self.args, self.store, {"available": False},
                         engine_factory=self.factory, decoder=self.decoder)
        self.assertIsNotNone(self.store.load_unit(self.config, "text", 1))
        self.assertIsNone(self.store.load_unit(self.config, "text", 2))
        self.engine.unload.assert_called_once()


class ArrayEmbedding:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.values


class PromptProfileTests(unittest.TestCase):
    def test_text_profile_avoids_reencoding_and_rejects_file_tampering(self):
        class PromptModel:
            def __init__(self):
                self.model = types.SimpleNamespace()
                self.encodes = 0
                self.loads = 0

            def set_classes(self, classes):
                self.encodes += 1
                self.classes = list(classes)
                self.model.pe = ArrayEmbedding(np.arange(len(classes) * 4, dtype=np.float32).reshape(1, len(classes), 4))

            def save_prompt_embeddings(self, path):
                np.savez_compressed(path, embeddings=self.model.pe.numpy(), names=np.asarray(self.classes))

            def load_prompt_embeddings(self, path):
                self.loads += 1
                with np.load(path, allow_pickle=False) as payload:
                    self.classes = payload["names"].tolist()
                    self.model.pe = ArrayEmbedding(payload["embeddings"].copy())

        with tempfile.TemporaryDirectory() as temporary:
            classes = ["bottle", "cup", "cell phone"]
            store = CheckpointStore(Path(temporary), {"classes": classes, "model_sha256": "weights",
                                    "dataset_sha256": "data", "source_hashes": {"runner": "code"}})
            first = types.SimpleNamespace(model=PromptModel(), current_classes=[])
            second = types.SimpleNamespace(model=PromptModel(), current_classes=[])
            encoded = prepare_text_prompt(first, store, classes, lambda: None, lambda: 0)
            reused = prepare_text_prompt(second, store, classes, lambda: None, lambda: 0)
            self.assertFalse(encoded["profile_reused"])
            self.assertTrue(reused["profile_reused"])
            self.assertEqual(first.model.encodes, 1)
            self.assertEqual(second.model.encodes, 0)
            self.assertEqual(second.model.loads, 1)
            np.testing.assert_array_equal(first.model.model.pe.numpy(), second.model.model.pe.numpy())
            profile = store.output / "prompt_cache" / "text_prompts.npz"
            with profile.open("ab") as stream:
                stream.write(b"tampering")
            with self.assertRaisesRegex(ValueError, "哈希"):
                load_profile(second, store, classes)
            self.assertEqual(second.model.loads, 1)


if __name__ == "__main__":
    unittest.main()
