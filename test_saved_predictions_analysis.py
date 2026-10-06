"""Meaningful CPU checks for matching, missing positives and bootstrap IDs."""
import copy
import unittest

from tools.analyze_saved_predictions import (
    box_iou, coco_ap, confidence_reliability, diagnose, interval, paired_bootstrap,
    pixel_audit, resample_images,
)


def fixture():
    return {"info": {}, "images": [{"id": 11, "width": 320, "height": 160, "file_name": "1.jpg"},
                                    {"id": 22, "width": 320, "height": 160, "file_name": "2.jpg"}],
            "categories": [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"}],
            "annotations": [{"id": 101, "image_id": 11, "category_id": 44,
                             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                            {"id": 102, "image_id": 22, "category_id": 47,
                             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}]}


def predictions(truth):
    return [{"image_id": row["image_id"], "category_id": row["category_id"],
             "bbox": list(row["bbox"]), "score": 0.9} for row in truth["annotations"]]


class DiagnosticTests(unittest.TestCase):
    def test_score_order_one_to_one_duplicate_class_and_localization(self):
        truth = fixture()
        rows = [
            {"image_id": 11, "category_id": 44, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"image_id": 11, "category_id": 44, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 22, "category_id": 44, "bbox": [0, 0, 10, 10], "score": 0.7},
            {"image_id": 22, "category_id": 47, "bbox": [5, 0, 10, 10], "score": 0.6},
            {"image_id": 22, "category_id": 44, "bbox": [5, 0, 10, 10], "score": 0.5},
            {"image_id": 22, "category_id": 44, "bbox": [80, 80, 10, 10], "score": 0.4},
        ]
        result = diagnose(truth, rows)
        self.assertEqual([row["kind"] for row in result["predictions"]],
                         ["tp", "duplicate", "class_confusion", "localization",
                          "class_and_localization", "background"])
        self.assertEqual(result["predictions"][0]["prediction_index"], 1)
        self.assertEqual(result["per_class"]["bottle"]["tp"], 1)
        self.assertEqual(result["per_class"]["bottle"]["fp"], 4)
        self.assertEqual(result["per_class"]["cup"]["fn"], 1)

    def test_crowd_excluded_from_false_negatives_and_false_positives(self):
        truth = fixture()
        truth["annotations"][0]["iscrowd"] = 1
        # A small prediction inside crowd is ignored using IoA, not standard IoU.
        row = {"image_id": 11, "category_id": 44, "bbox": [1, 1, 2, 2], "score": 0.9}
        result = diagnose(truth, [row])
        self.assertEqual(result["predictions"][0]["kind"], "ignored")
        self.assertEqual(result["per_class"]["bottle"]["fn"], 0)
        self.assertEqual(result["per_class"]["bottle"]["fp"], 0)

    def test_pixel_audit_uses_letterbox_ratio_and_actual_matching(self):
        truth = fixture()
        rows = predictions(truth)
        diagnostics = {f"text@{threshold:.2f}": diagnose(truth, rows, threshold)
                       for threshold in (0.05, 0.25)}
        pixels = pixel_audit(truth, {"text": rows}, diagnostics)
        self.assertEqual(pixels[0]["letterbox_scale"], 2)
        self.assertEqual(pixels[0]["letterbox_bbox_width"], 20)
        self.assertEqual(pixels[0]["letterbox_min_side_p3_cells"], 2.5)
        self.assertTrue(pixels[0]["text_matched_conf_0.25"])

    def test_iou_handles_zero_union_and_xywh_coordinates(self):
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 10, 10]), 1 / 3)


class BootstrapTests(unittest.TestCase):
    def test_repeated_images_get_distinct_truth_and_prediction_ids(self):
        truth = fixture()
        original = copy.deepcopy(truth)
        original_predictions = predictions(truth)
        sampled, detected = resample_images(truth, {"text": original_predictions}, [11, 11])
        self.assertEqual([row["id"] for row in sampled["images"]], [1, 2])
        self.assertEqual([row["id"] for row in sampled["annotations"]], [1, 2])
        self.assertEqual([row["image_id"] for row in detected["text"]], [1, 2])
        self.assertEqual([row["source_annotation_id"] for row in sampled["annotations"]], [101, 101])
        self.assertEqual(truth, original)
        self.assertEqual(original_predictions, predictions(truth))

    def test_missing_class_is_null_empty_predictions_with_gt_are_zero(self):
        truth, detected = resample_images(fixture(), {"text": []}, [11, 11])
        metrics = coco_ap(truth, detected["text"], {"all": [44, 47], "small": [44, 47]})
        self.assertIsNone(metrics["metrics"]["AP"])
        self.assertIsNone(metrics["per_class"][47]["APsmall"])
        self.assertEqual(metrics["per_class"][44]["AP"], 0)
        complete = coco_ap(fixture(), [])
        self.assertEqual(complete["metrics"]["AP"], 0)

    def test_paired_bootstrap_identical_predictions_have_zero_difference_and_reproduce(self):
        truth = fixture()
        rows = predictions(truth)
        inputs = {"text": rows, "text_sahi": copy.deepcopy(rows)}
        first = paired_bootstrap(truth, inputs, samples=12, seed=7)
        second = paired_bootstrap(truth, inputs, samples=12, seed=7)
        self.assertEqual(first, second)
        for metric in ("AP", "APsmall"):
            entry = first["differences"]["text_sahi_minus_text"][metric]
            self.assertEqual(entry["percentile_interval"], [0, 0])
            self.assertGreater(entry["valid_samples"], 0)
            self.assertLess(entry["valid_samples"], 12)
        self.assertAlmostEqual(first["point"]["text"]["metrics"]["AP"], 1)

    def test_interval_ignores_missing_positives_but_keeps_real_zero(self):
        result = interval([None, 0, 0.2], 3)
        self.assertEqual(result["valid_samples"], 2)
        self.assertAlmostEqual(result["mean"], 0.1)
        self.assertEqual(interval([None], 1)["percentile_interval"], None)


class ReliabilityTests(unittest.TestCase):
    def test_empty_bins_are_null_boundaries_and_one_belong_to_correct_bins(self):
        diagnostic = {"confidence": 0.05, "match_iou": 0.5,
                      "per_class": {"bottle": {"category_id": 44}, "cup": {"category_id": 47}},
                      "predictions": [
                          {"score": 0.05, "kind": "background", "category_id": 44},
                          {"score": 0.1, "kind": "tp", "category_id": 44},
                          {"score": 1.0, "kind": "tp", "category_id": 44},
                          {"score": 0.5, "kind": "ignored", "category_id": 44}]}
        result = confidence_reliability(diagnostic)
        self.assertEqual(result["overall"]["prediction_count"], 3)
        self.assertEqual(result["overall"]["bins"][0]["count"], 1)
        self.assertEqual(result["overall"]["bins"][1]["count"], 1)
        self.assertEqual(result["overall"]["bins"][9]["count"], 1)
        self.assertIsNone(result["overall"]["bins"][2]["mean_score"])
        self.assertIsNone(result["overall"]["bins"][2]["empirical_tp_rate"])
        self.assertAlmostEqual(result["overall"]["ece"], (0.05 + 0.9) / 3)
        self.assertIsNone(result["per_class"]["cup"]["ece"])

    def test_non_finite_score_rejected_instead_of_written_to_json(self):
        diagnostic = {"confidence": 0.05, "match_iou": 0.5, "per_class": {},
                      "predictions": [{"score": float("nan"), "kind": "tp", "category_id": 44}]}
        with self.assertRaises(ValueError):
            confidence_reliability(diagnostic)


if __name__ == "__main__":
    unittest.main()
