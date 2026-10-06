"""Training probe split isolation, original GT preservation and bounded I/O."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tools.prepare_mini_dataset import PROJECT_ROOT, sha256_file
from tools.prepare_training_probe import (
    MAX_TOTAL_BYTES, excluded_image_ids, jpeg_dimensions, prepare_probe, select_splits, yolo_labels,
)


def fixture():
    images, annotations = [], []
    categories = [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"},
                  {"id": 77, "name": "cell phone"}, {"id": 1, "name": "person"}]
    for category in categories[:3]:
        for index in range(30):
            image_id = category["id"] * 100 + index
            side = 20 if index < 15 else 64 if index < 22 else 150
            images.append({"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 640, "height": 480})
            annotations += [
                {"id": image_id, "image_id": image_id, "category_id": category["id"],
                 "bbox": [10, 10, side, side], "area": side ** 2, "iscrowd": 0},
                {"id": image_id + 100000, "image_id": image_id, "category_id": category["id"],
                 "bbox": [30, 30, 10, 10], "area": 100, "iscrowd": 1},
                {"id": image_id + 200000, "image_id": image_id, "category_id": 1,
                 "bbox": [0, 0, 300, 300], "area": 90000, "iscrowd": 0},
            ]
    return {"images": images, "annotations": annotations, "categories": categories, "licenses": []}


def small_jpeg():
    # A SOF header is enough to exercise geometry validation without image libraries.
    return b"\xff\xd8\xff\xc0" + struct.pack(">HBHHB", 8, 8, 480, 640, 0) + b"\xff\xd9"


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_and_all_selected_target_gt_preserved(self):
        source = fixture()
        excluded = {4400, 4700, 7700}
        splits, selection = select_splits(source, excluded, train_count=48, val_count=12)
        self.assertEqual((splits, selection), select_splits(source, excluded, train_count=48, val_count=12))
        train = {image["id"] for image in splits["train"]["images"]}
        val = {image["id"] for image in splits["val"]["images"]}
        self.assertEqual((len(train), len(val)), (48, 12))
        self.assertFalse(train & val or (train | val) & excluded)
        for split, subset in splits.items():
            expected = [ann for image in subset["images"] for ann in source["annotations"]
                        if ann["image_id"] == image["id"] and ann["category_id"] in {44, 47, 77}]
            self.assertEqual(subset["annotations"], expected)
            for category in (44, 47, 77):
                self.assertTrue(any(ann["category_id"] == category and ann["area"] < 1024 and not ann["iscrowd"]
                                    for ann in subset["annotations"]))

    def test_excluded_too_large_and_crowd_only_images_cannot_fill_slots(self):
        source = fixture()
        for image in source["images"]:
            if image["id"] // 100 == 77:
                image["width"] = 1280
        with self.assertRaisesRegex(ValueError, "Insufficient distinct"):
            select_splits(source, set(), train_count=12, val_count=6)

    def test_yolo_mapping_crowd_skip_and_clipping_do_not_modify_original(self):
        source = fixture()
        splits, _ = select_splits(source, set(), train_count=12, val_count=6)
        subset = splits["train"]
        first = subset["annotations"][0]
        first["bbox"] = [-10, 10, 30, 20]
        labels, skipped, clipped = yolo_labels(subset)
        self.assertEqual(first["bbox"], [-10, 10, 30, 20])
        self.assertEqual(len(skipped), len(subset["images"]))
        self.assertEqual(clipped, [{"annotation_id": first["id"], "image_id": first["image_id"]}])
        actual = set()
        for payload in labels.values():
            for line in payload.decode().splitlines():
                class_id, *numbers = line.split()
                actual.add(int(class_id))
                self.assertEqual(len(numbers), 4)
                self.assertTrue(all(0 <= float(number) <= 1 for number in numbers))
        self.assertEqual(actual, {0, 1, 2})


class PreparationTests(unittest.TestCase):
    def test_geometry_parser_rejects_non_jpeg_and_truncated_header(self):
        self.assertEqual(jpeg_dimensions(small_jpeg()), (640, 480))
        for payload in (b"not a jpeg", b"\xff\xd8\xff\xc0\x00\x08"):
            with self.assertRaises(ValueError):
                jpeg_dimensions(payload)

    def test_hash_mismatch_fails_before_network_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(fixture()))
            with patch("tools.prepare_training_probe.fetch") as request:
                with self.assertRaisesRegex(ValueError, "hash-verified"):
                    prepare_probe(root / "output", source, root / "excluded")
                request.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_bounded_download_reuse_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(fixture()))
            excluded = root / "excluded"
            excluded.mkdir()
            (excluded / "annotations.json").write_text(json.dumps({"images": [{"id": 4400}]}))
            (excluded / "references.json").write_text(json.dumps({"references": [{"original_image_id": 4700}]}))
            digest = sha256_file(source)
            with patch("tools.prepare_training_probe.OFFICIAL_VAL_SHA256", digest), \
                 patch("tools.prepare_training_probe.fetch", return_value=(small_jpeg(), {})) as request:
                manifest = prepare_probe(root / "output", source, excluded, train_count=12, val_count=6)
                self.assertEqual(request.call_count, 18)
                self.assertTrue(all(call.kwargs["max_bytes"] <= 4 * 1024 * 1024 for call in request.call_args_list))
                self.assertLessEqual(manifest["total_bytes"], MAX_TOTAL_BYTES)
                request.reset_mock()
                prepare_probe(root / "output", source, excluded, train_count=12, val_count=6)
                request.assert_not_called()
                with self.assertRaisesRegex(ValueError, "another dataset"):
                    prepare_probe(root / "output", source, excluded, train_count=12, val_count=6, seed=6009)


class InstalledProbeTests(unittest.TestCase):
    def test_saved_probe_is_disjoint_hashed_and_portable(self):
        root = PROJECT_ROOT / "datasets/mini_train_probe"
        if not (root / "manifest.json").is_file():
            self.skipTest("Probe has not been prepared")
        manifest = json.loads((root / "manifest.json").read_text())
        evaluation, references = excluded_image_ids(PROJECT_ROOT / "datasets/mini_eval")
        train, val = map(set, (manifest["split_image_ids"]["train"], manifest["split_image_ids"]["val"]))
        self.assertFalse(train & val or (train | val) & (evaluation | references))
        self.assertEqual((len(train), len(val)), (48, 12))
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        self.assertEqual(size, manifest["total_bytes"])
        self.assertLessEqual(size, MAX_TOTAL_BYTES)
        for image in manifest["images"]:
            self.assertEqual(sha256_file(root / image["image_file"]), image["sha256"])
            self.assertLessEqual(max(image["width"], image["height"]), 640)
        for relative, record in manifest["files"].items():
            self.assertEqual(sha256_file(root / relative), record["sha256"])
        self.assertNotIn(str(PROJECT_ROOT), (root / "data.yaml").read_text())
        for split in ("train", "val"):
            subset = json.loads((root / "annotations" / f"{split}.json").read_text())
            self.assertEqual({cat["id"] for cat in subset["categories"]}, {44, 47, 77})
            self.assertTrue(all(item["noncrowd_small_annotations"] > 0 for item in manifest["splits"][split]["classes"]))

    def test_original_annotation_ids_complete_and_labels_match_human_boxes(self):
        root = PROJECT_ROOT / "datasets/mini_train_probe"
        if not (root / "manifest.json").is_file():
            self.skipTest("Probe has not been prepared")
        source_path = PROJECT_ROOT / "work/dataset_cache/instances_val2017.json"
        if not source_path.is_file():
            self.skipTest("Original annotation cache unavailable")
        source = json.loads(source_path.read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(sha256_file(source_path), manifest["ground_truth"]["original_annotation_sha256"])
        for split in ("train", "val"):
            subset = json.loads((root / "annotations" / f"{split}.json").read_text())
            ids = {image["id"] for image in subset["images"]}
            expected = {ann["id"]: ann for ann in source["annotations"]
                        if ann["image_id"] in ids and ann["category_id"] in {44, 47, 77}}
            self.assertEqual({ann["id"]: ann for ann in subset["annotations"]}, expected)
            mapping = {44: 0, 47: 1, 77: 2}
            for image in subset["images"]:
                annotations = [ann for ann in subset["annotations"]
                               if ann["image_id"] == image["id"] and not ann.get("iscrowd", 0)]
                rows = (root / "labels" / split / (Path(image["file_name"]).stem + ".txt")).read_text().splitlines()
                self.assertEqual(len(rows), len(annotations))
                for row, ann in zip(rows, annotations):
                    class_id, xc, yc, width, height = map(float, row.split())
                    self.assertEqual(int(class_id), mapping[ann["category_id"]])
                    decoded = ((xc - width / 2) * image["width"], (yc - height / 2) * image["height"],
                               width * image["width"], height * image["height"])
                    x, y, w, h = ann["bbox"]
                    clipped = (max(0, x), max(0, y), min(image["width"], x + w) - max(0, x),
                               min(image["height"], y + h) - max(0, y))
                    for actual, target in zip(decoded, clipped):
                        self.assertAlmostEqual(actual, target, places=6)


if __name__ == "__main__":
    unittest.main()
