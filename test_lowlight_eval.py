"""CPU tests for synthetic darkening, condition wiring, weighted COCO and bootstrap; no model."""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np

from detection_utils import Detection
from engines import OpenVocabEngine
from evaluate_mini_dataset import coco_metrics
from tools.analyze_saved_predictions import resample_images
from resource_safety import MIN_AVAILABLE_RAM_MB, ResourceLimitError
from tools.evaluate_lowlight import (
    DARK_CONDITIONS, SCHEMA, build_parser, check_timing_memory, default_example_image, evaluation_tables,
    git_state, legacy_clahe, main, paired_bootstrap, run_dark_conditions, synthesize_low_light,
    threshold_counts, validate_report, weighted_metrics,
)


def scene(height=96, width=128, seed=3):
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 150, np.uint8)
    image[:] = rng.integers(60, 230, size=(height, width, 3), dtype=np.uint8) // 2 + 70
    return image


def fixture_truth():
    images = [{"id": index, "file_name": f"{index}.png", "height": 100, "width": 100} for index in (1, 2, 3)]
    annotations = [
        {"id": 11, "image_id": 1, "category_id": 44, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 12, "image_id": 1, "category_id": 47, "bbox": [50, 50, 40, 40], "area": 1600, "iscrowd": 0},
        {"id": 13, "image_id": 2, "category_id": 44, "bbox": [5, 5, 12, 12], "area": 144, "iscrowd": 0},
        {"id": 14, "image_id": 2, "category_id": 44, "bbox": [60, 10, 30, 30], "area": 900, "iscrowd": 1},
        {"id": 15, "image_id": 3, "category_id": 47, "bbox": [20, 20, 70, 70], "area": 4900, "iscrowd": 0},
    ]
    return {"info": {}, "images": images, "annotations": annotations,
            "categories": [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"}]}


def fixture_predictions():
    return [
        {"image_id": 1, "category_id": 44, "bbox": [11, 10, 20, 21], "score": 0.91},
        {"image_id": 1, "category_id": 47, "bbox": [52, 48, 40, 40], "score": 0.42},
        {"image_id": 1, "category_id": 44, "bbox": [70, 5, 10, 10], "score": 0.55},
        {"image_id": 2, "category_id": 44, "bbox": [5, 6, 12, 12], "score": 0.33},
        {"image_id": 2, "category_id": 44, "bbox": [61, 11, 29, 29], "score": 0.80},
        {"image_id": 3, "category_id": 47, "bbox": [25, 22, 60, 70], "score": 0.67},
        {"image_id": 3, "category_id": 44, "bbox": [0, 0, 30, 30], "score": 0.12},
    ]


class SynthesisTests(unittest.TestCase):
    def test_darkening_is_monotonic_typed_and_in_range(self):
        image = scene()
        saved = image.copy()
        means = [image.mean()]
        for level, exposure in enumerate((0.25, 0.12, 0.06)):
            dark = synthesize_low_light(image, exposure, [6008, level, 1])
            self.assertEqual((dark.dtype, dark.shape), (np.uint8, image.shape))
            self.assertGreaterEqual(int(dark.min()), 0)
            self.assertLessEqual(int(dark.max()), 255)
            means.append(dark.mean())
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertEqual(len(set(means)), len(means))
        np.testing.assert_array_equal(image, saved)

    def test_noise_free_path_is_exact_gamma_exposure_scaling(self):
        image = scene()
        identity = synthesize_low_light(image, 1.0, 0, full_well=None, read_noise=0)
        np.testing.assert_array_equal(identity, image)
        dark = synthesize_low_light(image, 0.12, 0, full_well=None, read_noise=0)
        expected = np.rint(((image / 255.0) ** 2.2 * 0.12) ** (1 / 2.2) * 255)
        np.testing.assert_array_equal(dark, expected.astype(np.uint8))

    def test_seeded_noise_is_deterministic_and_independent_per_image(self):
        image = scene()
        first = synthesize_low_light(image, 0.06, [6008, 2, 42])
        np.testing.assert_array_equal(first, synthesize_low_light(image, 0.06, [6008, 2, 42]))
        self.assertFalse(np.array_equal(first, synthesize_low_light(image, 0.06, [6008, 2, 43])))
        self.assertFalse(np.array_equal(first, synthesize_low_light(image, 0.06, [6009, 2, 42])))

    def test_relative_shot_noise_grows_as_exposure_falls(self):
        gray = np.full((128, 128, 3), 128, np.uint8)
        bright = synthesize_low_light(gray, 0.25, 1).astype(float)
        dark = synthesize_low_light(gray, 0.06, 1).astype(float)
        self.assertGreater(dark.std() / dark.mean(), bright.std() / bright.mean())

    def test_invalid_inputs_raise(self):
        image = scene()
        for exposure in (0, -0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                synthesize_low_light(image, exposure, 0)
        for bad in (image.astype(np.float32), image[..., 0], np.empty((0, 0, 3), np.uint8)):
            with self.assertRaises(ValueError):
                synthesize_low_light(bad, 0.25, 0)
        with self.assertRaises(ValueError):
            synthesize_low_light(image, 0.25, 0, full_well=0)

    def test_parser_rejects_unordered_exposures(self):
        self.assertEqual(build_parser().parse_args([]).exposures, [0.25, 0.12, 0.06, 0.02])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["--exposures", "0.06,0.25"])


class WeightedCocoTests(unittest.TestCase):
    def test_unit_weights_reproduce_pycocotools_summary(self):
        truth, predictions = fixture_truth(), fixture_predictions()
        reference = coco_metrics(truth, predictions, [1, 2, 3])["metrics"]
        fast = weighted_metrics(evaluation_tables(truth, predictions, [1, 2, 3]))
        for name in ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR100"):
            if reference[name] is None:
                self.assertIsNone(fast[name])
            else:
                self.assertAlmostEqual(fast[name], reference[name], places=12)

    def test_integer_weights_equal_duplicated_images(self):
        truth, predictions = fixture_truth(), fixture_predictions()
        evaluation = evaluation_tables(truth, predictions, [1, 2, 3])
        sampled, detected = resample_images(truth, {"mode": predictions}, [1, 1, 3])
        reference = coco_metrics(sampled, detected["mode"], [item["id"] for item in sampled["images"]])["metrics"]
        fast = weighted_metrics(evaluation, [2, 0, 1])
        for name in ("AP", "AP50", "AP75", "AR100"):
            self.assertAlmostEqual(fast[name], reference[name], places=12)

    def test_threshold_counts_ignore_crowd_and_low_scores(self):
        evaluation = evaluation_tables(fixture_truth(), fixture_predictions(), [1, 2, 3])
        counts = threshold_counts(evaluation, 0.40)
        # Four non-crowd GT; >=0.40: bottle@1 TP, cup@1 TP, bottle@1 FP, crowd match ignored, cup@3 TP.
        self.assertEqual((counts["gt_gui"], counts["tp_gui"], counts["fp_gui"]), (4, 3, 1))
        self.assertAlmostEqual(counts["recall_gui"], 0.75)
        self.assertAlmostEqual(counts["precision_gui"], 0.75)
        empty = threshold_counts(evaluation_tables(fixture_truth(), [], [1, 2, 3]), 0.4)
        self.assertEqual(empty["recall_gui"], 0)
        self.assertIsNone(empty["precision_gui"])


class BootstrapTests(unittest.TestCase):
    def evaluations(self, left, right):
        truth = fixture_truth()
        ids = [1, 2, 3]
        tables = {"a": evaluation_tables(truth, left, ids), "b": evaluation_tables(truth, right, ids)}
        return tables, copy.deepcopy(tables)

    def test_identical_conditions_give_zero_difference_and_reproduce(self):
        tables, gui = self.evaluations(fixture_predictions(), fixture_predictions())
        first = paired_bootstrap(tables, gui, [("a", "b")], 300, 7, 0.4)
        second = paired_bootstrap(tables, gui, [("a", "b")], 300, 7, 0.4)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        difference = first["differences"]["a_minus_b"]["AP"]
        self.assertEqual(difference["point_difference"], 0)
        self.assertEqual(difference["percentile_interval"], [0.0, 0.0])
        self.assertTrue(first["paired_across_all_conditions"])

    def test_perfect_versus_empty_interval_excludes_zero(self):
        truth = fixture_truth()
        perfect = [{"image_id": row["image_id"], "category_id": row["category_id"], "bbox": row["bbox"],
                    "score": 0.9} for row in truth["annotations"] if not row["iscrowd"]]
        tables, gui = self.evaluations(perfect, [])
        result = paired_bootstrap(tables, gui, [("a", "b")], 200, 3, 0.4)["differences"]["a_minus_b"]
        self.assertAlmostEqual(result["AP"]["point_difference"], 1.0)
        self.assertGreater(result["AP"]["percentile_interval"][0], 0)
        self.assertGreater(result["recall_gui"]["percentile_interval"][0], 0)
        # Draws missing one class entirely are not scored as zero.
        self.assertLessEqual(result["AP"]["valid_samples"], 200)


def brightness_detector(truth_by_shape, bright=0.25, dim=0.12):
    """Finds the human boxes only when the frame is bright enough; dim frames give weak scores."""
    def detect(frame, conf=0.4, iou=0.45, class_names=None, source="original", max_det=300):
        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() / 255
        score = 0.9 if luma >= bright else 0.3 if luma >= dim else 0.0
        boxes = [Detection(box, score, class_id, class_names[class_id], source)
                 for box, class_id in truth_by_shape[frame.shape[:2]] if score]
        if dim <= luma < bright:
            boxes.append(Detection((1.0, 1.0, 9.0, 9.0), 0.2, 0, class_names[0], source))
        return [item for item in boxes if item.score >= conf][:max_det]
    return detect


class ConditionWiringTests(unittest.TestCase):
    def test_each_condition_receives_its_frame_and_dual_uses_app_merge(self):
        frame = scene(80, 120)
        dark = synthesize_low_light(frame, 0.02, [1, 2, 3])
        engine = OpenVocabEngine()
        box = ((10.0, 10.0, 40.0, 50.0), 0)
        engine.detect = MagicMock(side_effect=brightness_detector({(80, 120): [box]}, bright=0.15))
        ap, gui, info, images = run_dark_conditions(engine, dark, ["bottle"], 0.05, 0.40, 0.45, 300, 0.65)
        calls = engine.detect.call_args_list
        self.assertEqual(len(calls), 7)
        np.testing.assert_array_equal(calls[0].args[0], dark)
        np.testing.assert_array_equal(calls[1].args[0], legacy_clahe(dark))
        np.testing.assert_array_equal(calls[2].args[0], images["dark_adaptive"])
        self.assertTrue(info["applied"])
        self.assertGreater(info["luma"]["dark_adaptive"], info["luma"]["dark"])
        # process_frame: original exposure first at the requested floor, then the enhanced frame at >=0.45.
        for index, conf in ((3, 0.05), (5, 0.40)):
            np.testing.assert_array_equal(calls[index].args[0], dark)
            self.assertEqual(calls[index].args[1], conf)
            np.testing.assert_array_equal(calls[index + 1].args[0], images["dark_adaptive"])
            self.assertEqual((calls[index + 1].args[1], calls[index + 1].kwargs["source"]), (0.45, "enhanced"))
        self.assertEqual(ap["dark"], [])
        self.assertEqual([item.source for item in ap["dark_dual"]], ["enhanced"])
        self.assertEqual([item.source for item in gui["dark_dual"]], ["enhanced"])
        self.assertTrue(all(item.score >= 0.40 for item in gui["dark_adaptive"]))
        self.assertEqual(set(ap), set(DARK_CONDITIONS))

    def test_example_image_choice_is_fixed_before_scoring(self):
        truth = fixture_truth()
        truth["images"][2]["license"] = 4
        self.assertEqual(default_example_image(truth), 3)
        truth["images"][2]["license"] = 1
        self.assertEqual(default_example_image(truth), 1)


class EndToEndReportTests(unittest.TestCase):
    def test_mock_run_writes_valid_report_summary_plot_and_grid(self):
        shapes = [(96, 128), (80, 120), (90, 140)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            images, annotations, truth_by_shape = [], [], {}
            for index, (height, width) in enumerate(shapes, 1):
                cv2.imwrite(str(root / "images" / f"{index}.png"), scene(height, width, index))
                box = (10.0 + index, 12.0, 50.0 + index, 60.0)
                class_id = index % 2
                truth_by_shape[(height, width)] = [(box, class_id)]
                images.append({"id": index, "file_name": f"{index}.png", "height": height, "width": width,
                               "license": 4})
                annotations.append({"id": 100 + index, "image_id": index, "category_id": (44, 47)[class_id],
                                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                                    "area": (box[2] - box[0]) * (box[3] - box[1]), "iscrowd": 0})
            (root / "annotations.json").write_text(json.dumps({
                "images": images, "annotations": annotations,
                "categories": [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"}]}))
            engine = OpenVocabEngine()
            engine.detect = MagicMock(side_effect=brightness_detector(truth_by_shape))
            output = root / "out"
            code = main(["--source", f"fixture={root / 'annotations.json'},{root / 'images'}",
                         "--output", str(output), "--exposures", "0.25,0.06",
                         "--bootstrap-samples", "60", "--timing-repeats", "3", "--save-images"],
                        engine_factory=lambda args: engine)
            self.assertEqual(code, 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(validate_report(report))
            self.assertEqual((report["schema"], report["status"]), (SCHEMA, "complete"))
            self.assertEqual(report["dataset"]["image_count"], 3)
            self.assertTrue(report["fast_evaluator_matches_pycocotools"])
            self.assertEqual(report["consistency"]["gui_threshold_filter_vs_dual_base_mismatched_frames"], 0)
            results = report["results"]
            self.assertAlmostEqual(results["original"]["metrics"]["AP"], 1.0)
            self.assertEqual(results["L2/dark"]["gui"]["recall_gui"], 0)
            self.assertEqual(results["L2/dark_dual"]["gui"]["recall_gui"], 1)
            self.assertEqual(report["timing"]["repeats"], 3)
            self.assertEqual(len(report["timing"]["runs"]), 3)
            self.assertEqual(len(report["synthesis"]["dark_image_sha256"]), 6)
            self.assertEqual(report["outputs"]["example_image"]["license_id"], 4)
            self.assertIn("tracked_source_changes", report["fingerprints"]["git"])
            for name in ("summary.md", "per_image.csv", report["outputs"]["example_grid"],
                         "predictions/L1_dark_dual.json", "images/L2/000000000001_dark.png"):
                self.assertTrue((output / name).is_file(), name)
            if report["outputs"]["plot_status"]["status"] == "created":
                self.assertTrue((output / "lowlight_ap.png").is_file())
            broken = dict(report)
            broken.pop("bootstrap")
            with self.assertRaises(ValueError):
                validate_report(broken)
            self.assertGreater(report["timing"]["frames_held_mib"], 0)
            # A second run never overwrites the first report or mixes prediction files.
            before = (output / "report.json").read_bytes()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["--source", f"fixture={root / 'annotations.json'},{root / 'images'}",
                             "--output", str(output), "--exposures", "0.25"], engine_factory=lambda args: engine)
            self.assertEqual(code, 1)
            self.assertIn("非空", stderr.getvalue())
            self.assertEqual((output / "report.json").read_bytes(), before)

    def test_timing_frames_are_checked_against_free_ram_before_allocation(self):
        images = [{"height": 480, "width": 640}] * 84
        needed = 84 * 4 * 480 * 640 * 3 / 1024 ** 2
        self.assertAlmostEqual(check_timing_memory(images, 4, available_mb=6000), needed)
        with self.assertRaises(ResourceLimitError):
            check_timing_memory(images, 4, available_mb=int(MIN_AVAILABLE_RAM_MB + needed) - 1)

    def test_git_state_uses_one_schema_with_or_without_git(self):
        self.assertIn("tracked_source_changes", git_state())
        with patch("tools.evaluate_lowlight.subprocess.run", side_effect=OSError("no git")):
            self.assertEqual(git_state(), {"commit": None, "tracked_source_changes": None})


if __name__ == "__main__":
    unittest.main()
