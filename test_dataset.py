"""Dataset provenance, reference separation and bounded-download checks; no GPU."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from tools.prepare_mini_dataset import (
    ANNOTATION_MEMBER, DEFAULT_CLASSES, OFFICIAL_VAL_SHA256, PROJECT_ROOT, download_val_annotations,
    fetch, prepare_dataset, select_subset, sha256_file, size_group,
)


def source_fixture():
    categories = [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"},
                  {"id": 77, "name": "cell phone"}, {"id": 1, "name": "person"}]
    images, annotations = [], []
    for category in categories[:3]:
        for index in range(15):
            image_id = category["id"] * 100 + index
            side = 20 if index < 6 else 64 if index < 10 else 150
            images.append({"id": image_id, "width": 640, "height": 480,
                           "file_name": f"{image_id:012d}.jpg"})
            annotations.append({"id": image_id, "image_id": image_id, "category_id": category["id"],
                                "bbox": [2, 2, side, side], "area": side ** 2, "iscrowd": 0})
            # Original annotations on a chosen image must not be narrowed to its selection bbox.
            annotations.append({"id": image_id + 100000, "image_id": image_id,
                                "category_id": category["id"], "bbox": [40, 40, 10, 10],
                                "area": 100, "iscrowd": 1})
            annotations.append({"id": image_id + 200000, "image_id": image_id,
                                "category_id": 1, "bbox": [40, 40, 100, 100],
                                "area": 10000, "iscrowd": 0})
    return {"images": images, "annotations": annotations, "categories": categories, "licenses": []}


class SubsetTests(unittest.TestCase):
    def test_size_strata_reproducibility_complete_gt_and_reference_separation(self):
        source = source_fixture()
        subset, references, selection = select_subset(source)
        self.assertEqual(len(subset["images"]), 24)
        self.assertEqual(select_subset(source), (subset, references, selection))
        selected_ids = {image["id"] for image in subset["images"]}
        reference_ids = {item["original_image_id"] for item in references}
        self.assertFalse(selected_ids & reference_ids)
        self.assertEqual(len(reference_ids), 3)
        self.assertEqual([item["class_name"] for item in references], list(DEFAULT_CLASSES))
        self.assertEqual(subset["annotations"], [ann for image in subset["images"]
                         for ann in source["annotations"] if ann["image_id"] == image["id"]
                         and ann["category_id"] in {44, 47, 77}])
        for category in (44, 47, 77):
            self.assertEqual({size_group(ann["area"]) for ann in subset["annotations"]
                              if ann["category_id"] == category and not ann["iscrowd"]},
                             {"small", "medium", "large"})

    def test_insufficient_images_missing_reference_or_unknown_class_fail(self):
        source = source_fixture()
        for kwargs in ({"count": 1}, {"count": 65}, {"classes": ("phone",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_subset(source, **kwargs)
        for ann in source["annotations"]:
            if ann["category_id"] == 77:
                ann["bbox"] = [0, 0, 20, 20]
        with self.assertRaisesRegex(ValueError, "No independent visual reference"):
            select_subset(source)


class DownloadTests(unittest.TestCase):
    def test_modified_local_annotations_cannot_be_claimed_as_official_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "instances_val2017.json"
            source.write_text(json.dumps(source_fixture()))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_dataset(Path(directory) / "output", source)

    def test_extracts_only_validation_member_and_records_source_hash(self):
        archive = io.BytesIO()
        source_bytes = json.dumps(source_fixture()).encode()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as writer:
            writer.writestr("annotations/instances_train2017.json", b"do not download this member")
            writer.writestr(ANNOTATION_MEMBER, source_bytes)
        payload = archive.getvalue()
        requests = []

        def fake_fetch(url, *, byte_range=None, max_bytes=None):
            requests.append(byte_range)
            if byte_range.startswith("-"):
                return payload[-int(byte_range[1:]):], {"ETag": "fixture-etag"}
            start, end = map(int, byte_range.split("-"))
            return payload[start:end + 1], {}

        with tempfile.TemporaryDirectory() as directory, patch("tools.prepare_mini_dataset.fetch", side_effect=fake_fetch):
            cached = download_val_annotations(Path(directory) / "instances_val2017.json")
            self.assertEqual(cached.read_bytes(), source_bytes)
            self.assertEqual(len(requests), 4)
            self.assertEqual(json.loads(cached.with_suffix(".source.json").read_text())["sha256"], sha256_file(cached))
            download_val_annotations(cached)
            self.assertEqual(len(requests), 4, "Verified existing cache must not redownload")

    def test_refuses_server_that_ignores_range_before_reading_full_archive(self):
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        with patch("tools.prepare_mini_dataset.urllib.request.urlopen", return_value=response), \
             patch("tools.prepare_mini_dataset.time.sleep"), self.assertRaisesRegex(RuntimeError, "ignored Range"):
            fetch("https://example.invalid/archive.zip", byte_range="-65536")
        response.read.assert_not_called()


class InstalledDatasetTests(unittest.TestCase):
    def test_real_saved_data_hashes_boxes_class_ids_and_disjoint_references(self):
        root = PROJECT_ROOT / "datasets" / "mini_eval"
        if not (root / "manifest.json").is_file():
            self.skipTest("Mini dataset has not been downloaded")
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["ground_truth"]["original_annotation_sha256"], OFFICIAL_VAL_SHA256)
        coco = json.loads((root / "annotations.json").read_text())
        references = json.loads((root / "references.json").read_text())["references"]
        self.assertEqual({category["id"] for category in coco["categories"]}, {44, 47, 77})
        self.assertEqual(len(coco["images"]), manifest["evaluation_image_count"])
        self.assertFalse({image["id"] for image in coco["images"]} &
                         {item["original_image_id"] for item in references})
        image_map = {image["id"]: image for image in coco["images"]}
        for annotation in coco["annotations"]:
            self.assertIn(annotation["image_id"], image_map)
            x, y, width, height = annotation["bbox"]
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertLessEqual(x + width, image_map[annotation["image_id"]]["width"] + 0.01)
            self.assertLessEqual(y + height, image_map[annotation["image_id"]]["height"] + 0.01)
        for item in manifest["images"]:
            self.assertEqual(sha256_file(root / item["image_file"]), item["sha256"])
        for category in manifest["classes"]:
            self.assertGreater(category["noncrowd_small_annotations"], 0)


if __name__ == "__main__":
    unittest.main()
