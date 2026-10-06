"""CPU COCO evaluator and inference-contract tests; no model or GPU."""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from evaluate_mini_dataset import (
    build_parser, cap_predictions, coco_metrics, infer, load_dataset, main, preflight_dataset, run_mode,
)


def fixture_annotations():
    return {"info": {}, "images": [{"id": 1, "file_name": "one.jpg", "height": 40, "width": 40},
                                    {"id": 2, "file_name": "two.jpg", "height": 40, "width": 40}],
            "categories": [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 44,
                             "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0},
                            {"id": 2, "image_id": 2, "category_id": 47,
                             "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}


def fixture_references():
    return {"classes": ["bottle", "cup"], "references": [
        {"class_id": 0, "category_id": 44, "class_name": "bottle", "original_image_id": 3,
         "image_file": "references/bottle.jpg", "bbox_xyxy": [1, 1, 10, 10]},
        {"class_id": 1, "category_id": 47, "class_name": "cup", "original_image_id": 4,
         "image_file": "references/cup.jpg", "bbox_xyxy": [2, 2, 12, 12]},
    ]}


def create_dataset(root):
    (root / "images").mkdir()
    (root / "references").mkdir()
    for path in ["images/one.jpg", "images/two.jpg", "references/bottle.jpg", "references/cup.jpg"]:
        (root / path).write_bytes(b"test-fixture-no-decoding")
    (root / "annotations.json").write_text(json.dumps(fixture_annotations()))
    (root / "references.json").write_text(json.dumps(fixture_references()))


class COCOMetricTests(unittest.TestCase):
    def test_perfect_human_boxes_yield_one_ap_and_recall_without_mutating_records(self):
        truth = fixture_annotations()
        predictions = [{"image_id": row["image_id"], "category_id": row["category_id"],
                        "bbox": list(row["bbox"]), "score": 0.99} for row in truth["annotations"]]
        before = json.dumps(predictions)
        result = coco_metrics(truth, predictions, [1, 2])
        for name in ["AP", "AP50", "AP75", "APsmall", "AR100"]:
            self.assertAlmostEqual(result["metrics"][name], 1.0)
        self.assertIsNone(result["metrics"]["APmedium"])
        self.assertIsNone(result["metrics"]["APlarge"])
        for category in ["bottle", "cup"]:
            self.assertAlmostEqual(result["per_class"][category]["AP"], 1.0)
        self.assertEqual(json.dumps(predictions), before)

    def test_empty_detections_have_zero_ap_and_recall_instead_of_loadres_crash(self):
        result = coco_metrics(fixture_annotations(), [], [1, 2])
        for name in ["AP", "AP50", "AP75", "APsmall", "AR100"]:
            self.assertEqual(result["metrics"][name], 0)

    def test_limit_evaluates_only_requested_images_and_missing_category_is_not_zero(self):
        truth = fixture_annotations()
        prediction = {"image_id": 1, "category_id": 44, "bbox": [5, 5, 10, 10], "score": 0.99}
        result = coco_metrics(truth, [prediction], [1])
        self.assertAlmostEqual(result["metrics"]["AP"], 1)
        self.assertIsNone(result["per_class"]["cup"]["AP"])


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        create_dataset(self.root)

    def test_coco_basenames_map_to_images_folder_and_hashes_cover_reference_inputs(self):
        result = load_dataset(self.root)
        self.assertEqual(result["category_ids"], [44, 47])
        self.assertEqual(result["classes"], ["bottle", "cup"])
        self.assertEqual(len(result["images"]), 2)
        self.assertIn("images/one.jpg", result["file_hashes"])
        self.assertIn("references/cup.jpg", result["file_hashes"])
        self.assertEqual(result["sha256"], load_dataset(self.root)["sha256"])

    def test_reference_leakage_checked_against_full_dataset_even_with_limit(self):
        references = fixture_references()
        references["references"][0]["original_image_id"] = 2
        (self.root / "references.json").write_text(json.dumps(references))
        with self.assertRaisesRegex(ValueError, "测试真值"):
            load_dataset(self.root, limit=1)

    def test_wrong_category_mapping_and_path_escape_are_rejected(self):
        references = fixture_references()
        references["references"][0]["category_id"] = 47
        (self.root / "references.json").write_text(json.dumps(references))
        with self.assertRaisesRegex(ValueError, "映射"):
            load_dataset(self.root)
        references = fixture_references()
        references["references"][0]["image_file"] = "../escape.jpg"
        (self.root / "references.json").write_text(json.dumps(references))
        with self.assertRaisesRegex(ValueError, "目录内"):
            load_dataset(self.root)


class InferenceContractTests(unittest.TestCase):
    def test_plain_and_sliced_modes_have_identical_thresholds_and_caps(self):
        args = build_parser().parse_args([])
        engine = MagicMock()
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        infer(engine, frame, args, False, ["bottle", "cup"])
        infer(engine, frame, args, True, ["bottle", "cup"])
        engine.detect.assert_called_once_with(frame, source="original", conf=0.05, iou=0.5,
                                               class_names=["bottle", "cup"], max_det=100)
        engine.detect_sliced.assert_called_once_with(frame, conf=0.05, iou=0.5,
                                                      class_names=["bottle", "cup"], max_det=100,
                                                      slice_size=320, slice_overlap=0.2, max_slices=9)

    def test_visual_class_passes_encode_once_map_to_canonical_ids_and_cap_globally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_dataset(root)
            dataset = load_dataset(root)
            args = build_parser().parse_args(["--max-det", "1"])
            engine = MagicMock()
            def detect(frame, **kwargs):
                score = 0.6 if kwargs["class_names"] == ["bottle"] else 0.95
                return [types.SimpleNamespace(xyxy=(1, 1, 12, 12), score=score, class_id=0)]
            engine.detect.side_effect = detect
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            calls, ticks = [], iter(range(100))
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_mode(engine, dataset, args, "visual", lambda: calls.append("sync"),
                                  clock=lambda: next(ticks) / 1000, decoder=lambda path: frame)
            self.assertEqual(engine.set_visual_prompt.call_count, 2)
            self.assertEqual(engine.detect.call_count, 6)  # 1 warmup + 2 images per class
            self.assertEqual(len(result["predictions"]), 2)
            self.assertTrue(all(row["category_id"] == 47 for row in result["predictions"]))
            self.assertEqual(result["timing_summary"]["class_passes_per_image"], 2)
            self.assertAlmostEqual(result["timing_summary"]["mean_ms"], 2)
            self.assertEqual(len(result["reference_encodings"]), 2)
            # Each reference comes before all images of that class, never from GT.
            self.assertEqual([call.args[2] for call in engine.set_visual_prompt.call_args_list], ["bottle", "cup"])

    def test_global_cap_keeps_highest_scores_across_classes_for_each_image(self):
        rows = [{"image_id": 1, "category_id": 44, "score": 0.8},
                {"image_id": 2, "category_id": 44, "score": 0.9},
                {"image_id": 1, "category_id": 47, "score": 0.95}]
        capped = cap_predictions(rows, [1, 2], 1)
        self.assertEqual([(row["image_id"], row["category_id"]) for row in capped], [(1, 47), (2, 44)])


class PreflightTests(unittest.TestCase):
    def test_every_selected_image_is_checked_before_gpu_loading(self):
        args = build_parser().parse_args([])
        dataset = {"images": fixture_annotations()["images"]}
        with patch("resource_safety.preflight_inference") as memory, \
                patch("sliced_inference.validate_slice_plan", return_value=[(0, 0, 40, 40)]) as slices:
            plans = preflight_dataset(args, dataset)
        self.assertEqual(memory.call_count, 2)
        self.assertEqual(slices.call_count, 2)
        memory.assert_called_with("cpu", 640, (40, 40, 3))
        slices.assert_called_with((40, 40, 3), 320, 0.2, 9)
        self.assertEqual(plans, [{"image_id": 1, "slice_count": 1}, {"image_id": 2, "slice_count": 1}])

    def test_invalid_slicing_limits_fail_without_loading_gpu_or_materializing_crops(self):
        for flag, value in [("--slice-size", "64"), ("--slice-overlap", "0.31"), ("--max-slices", "10")]:
            args = build_parser().parse_args([flag, value])
            with self.subTest(flag=flag), patch("resource_safety.preflight_inference") as memory, \
                    self.assertRaises(ValueError):
                preflight_dataset(args, {"images": fixture_annotations()["images"]})
            memory.assert_not_called()

    def test_failed_memory_preflight_stops_cli_before_checkpoint_load(self):
        from resource_safety import ResourceLimitError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_dataset(root)
            with patch("resource_safety.preflight_inference", side_effect=ResourceLimitError("RAM budget")), \
                    patch("engines.OpenVocabEngine.load_model") as load, \
                    contextlib.redirect_stderr(io.StringIO()):
                code = main(["--dataset", str(root), "--device", "CUDA"])
            self.assertEqual(code, 1)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
