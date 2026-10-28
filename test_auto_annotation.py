"""Folder auto-annotation: scanning, review states, YOLO / COCO export, worker and Tab 4."""
import hashlib
import json
import os
import re
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import (QApplication, QAbstractButton, QComboBox, QLabel, QLineEdit,
                               QMessageBox as QtMessageBox, QWidget)

from auto_annotation import (LABEL_PROVENANCE, MANIFEST_NAME, AnnotationSession, box_origin, coco_bbox,
                             detect_candidates, export_coco, export_yolo, read_image_with_sha,
                             scan_folder, yolo_line)
from detection_utils import Detection
from i18n import EN, render
from resource_safety import ResourceLimitError

HAN = re.compile(r"[\u4e00-\u9fff]")
YOLO_LINE = re.compile(r"^\d+( (0|1)\.\d{6}){4}$")


def write_image(path, width=100, height=80, value=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, np.uint8)
    image[5:15, 5:25] = (value + 90) % 256
    ok, data = cv2.imencode(path.suffix, image)
    assert ok
    data.tofile(str(path))
    return image


def candidates():
    return [{"label": "bottle", "xyxy": [10., 20., 30., 60.], "score": 0.9, "source": "original"},
            {"label": "cup", "xyxy": [50., 10., 99., 70.], "score": 0.3, "source": "slice"}]


def detected_session(folder, count=3):
    for index in range(count):
        write_image(Path(folder) / f"img{index}.png", value=index * 40)
    session = AnnotationSession(folder)
    for index, entry in enumerate(session.entries):
        session.set_candidates(index, candidates(), 100, 80, read_image_with_sha(entry.path)[1])
    return session


class ScanAndGeometryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_scan_sorts_supported_images_and_reads_chinese_paths(self):
        folder = self.root / "中文图片"
        for name in ("b.png", "A.JPG", "c.bmp", "d.jpeg", "子目录/f.png", "导出/images/g.png"):
            write_image(folder / name)
        (folder / "notes.txt").write_text("x")
        write_image(folder / "e.webp")
        (folder / "导出" / MANIFEST_NAME).write_text("{}")
        self.assertEqual([path.name for path in scan_folder(folder)], ["A.JPG", "b.png", "c.bmp", "d.jpeg"])
        recursive = [path.relative_to(folder).as_posix() for path in scan_folder(folder, recursive=True)]
        self.assertEqual(recursive, ["A.JPG", "b.png", "c.bmp", "d.jpeg", "子目录/f.png"])
        image, sha256 = read_image_with_sha(folder / "子目录" / "f.png")
        self.assertEqual(image.shape, (80, 100, 3))
        self.assertEqual(sha256, hashlib.sha256((folder / "子目录" / "f.png").read_bytes()).hexdigest())
        (folder / "bad.png").write_bytes(b"not an image")
        self.assertIsNone(read_image_with_sha(folder / "bad.png")[0])
        with self.assertRaises(ValueError):
            scan_folder(self.root / "missing")

    def test_yolo_lines_are_normalised_clipped_and_six_decimal(self):
        self.assertEqual(yolo_line(2, [10, 20, 30, 60], 100, 80), "2 0.200000 0.500000 0.200000 0.500000")
        self.assertEqual(yolo_line(0, [-10, -5, 50, 40], 100, 80), "0 0.250000 0.250000 0.500000 0.500000")
        self.assertEqual(yolo_line(1, [90, 70, 130, 100], 100, 80), "1 0.950000 0.937500 0.100000 0.125000")
        self.assertEqual(yolo_line(0, [0, 0, 100, 80], 100, 80), "0 0.500000 0.500000 1.000000 1.000000")
        self.assertIsNone(yolo_line(0, [120, 10, 150, 20], 100, 80))
        self.assertIsNone(yolo_line(0, [10, 10, 10, 20], 100, 80))
        self.assertRegex(yolo_line(3, [1 / 3, 2 / 3, 50.123456789, 70.987654321], 100, 80), YOLO_LINE)
        self.assertEqual(coco_bbox([-2, 3, 10, 90], 20, 50), [0., 3., 10., 47.])
        self.assertIsNone(coco_bbox([25, 3, 30, 9], 20, 50))


class SessionStateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name) / "源图片"

    def test_review_transitions_never_let_detection_overwrite_human_work(self):
        for index in range(3):
            write_image(self.folder / f"img{index}.png")
        session = AnnotationSession(self.folder)
        entry = session.entries[0]
        self.assertEqual((entry.state, entry.detected, entry.modified), ("pending", False, False))
        with self.assertRaises(ValueError):
            session.accept(0)
        sha = read_image_with_sha(entry.path)[1]
        for index in range(3):
            self.assertTrue(session.set_candidates(index, candidates(), 100, 80, sha))
        self.assertEqual(entry.boxes, entry.candidates)
        self.assertEqual(session.accept(0), "accepted")
        self.assertEqual(entry.review_method, "per_image")
        session.update_boxes(1, session.entries[1].boxes[:1])
        self.assertEqual(session.entries[1].state, "pending")
        self.assertTrue(session.entries[1].modified)
        self.assertEqual(session.accept(1), "edited")
        session.reject(2)
        self.assertEqual(session.entries[2].state, "rejected")
        session.update_boxes(2, session.entries[2].boxes)
        self.assertEqual(session.entries[2].state, "pending")
        # Edits after acceptance stay exported but are recorded as edited.
        session.update_boxes(0, [dict(candidates()[0], xyxy=[11., 20., 30., 60.])])
        self.assertEqual(entry.state, "edited")
        session.update_boxes(0, candidates())
        self.assertEqual(entry.state, "accepted")
        # A late result never touches a reviewed image: boxes and the candidates the person saw stay.
        new = [{"label": "cell phone", "xyxy": [1, 1, 9, 9], "score": 0.8}]
        self.assertFalse(session.set_candidates(0, new, 100, 80, sha))
        self.assertEqual([box["label"] for box in entry.boxes], ["bottle", "cup"])
        self.assertEqual([box["label"] for box in entry.candidates], ["bottle", "cup"])
        session.reset(1)
        self.assertEqual((session.entries[1].state, session.entries[1].boxes), ("pending", session.entries[1].candidates))
        with self.assertRaises(ValueError):
            session.set_image_info(1, 100, 80, "0" * 64)
        counts = session.counts()
        self.assertEqual((counts["accepted"], counts["pending"], counts["exportable"], counts["detected"]), (1, 2, 1, 3))

    def test_result_for_an_image_reviewed_while_queued_is_discarded(self):
        for index in range(3):
            write_image(self.folder / f"img{index}.png", value=index * 30)
        session = AnnotationSession(self.folder)
        shas = [read_image_with_sha(entry.path)[1] for entry in session.entries]
        first = session.begin_run({"config": "A"})
        session.set_candidates(2, [candidates()[0]], 100, 80, shas[2], first)
        run = session.begin_run({"config": "B"})
        # While run B is queued: image 0 is opened and accepted as empty, image 2 is edited.
        session.set_image_info(0, 100, 80, shas[0])
        session.update_boxes(0, [])
        self.assertEqual(session.accept(0), "accepted")
        session.update_boxes(2, [{"label": "cup", "xyxy": [50., 10., 99., 70.]}])
        for index in range(3):
            session.set_candidates(index, candidates(), 100, 80, shas[index], run)
        accepted, untouched, edited = session.entries
        self.assertEqual((accepted.state, accepted.boxes, accepted.candidates, accepted.run_index, accepted.detected),
                         ("accepted", [], [], None, False))
        self.assertFalse(accepted.modified)
        self.assertEqual((untouched.run_index, untouched.boxes), (run, untouched.candidates))
        self.assertEqual((edited.run_index, [box["label"] for box in edited.candidates]), (first, ["bottle"]))
        self.assertEqual([box_origin(box, edited.candidates) for box in edited.boxes], ["human"])
        self.assertEqual((session.runs[run]["processed"], session.runs[run]["results_discarded_after_review"]), (3, 2))
        # A result delivered after the run was finalised does not change its processed count.
        session.finish_run(run, processed=3)
        session.reset(2)
        session.set_candidates(2, candidates(), 100, 80, shas[2], run)
        self.assertEqual((session.runs[run]["processed"], edited.run_index), (3, run))

    def test_targets_follow_configuration_and_skip_reviewed_or_changed_images(self):
        session = detected_session(self.folder, 4)
        run = session.begin_run({"config": "A"})
        for index, entry in enumerate(session.entries):
            session.set_candidates(index, candidates(), 100, 80, entry.sha256, run)
        self.assertEqual(session.runs[run]["processed"], 4)
        self.assertEqual(session.targets("A"), [])
        session.accept(0)
        session.update_boxes(1, [])
        session.set_error(3, "decode failed")
        self.assertEqual([index for index, _ in session.targets("A")], [3])
        self.assertEqual([index for index, _ in session.targets("B")], [2, 3])

    def test_bulk_accept_keeps_only_confident_candidates_and_leaves_empty_images_pending(self):
        session = detected_session(self.folder, 3)
        session.set_candidates(2, [dict(candidates()[1])], 100, 80, session.entries[2].sha256)
        session.update_boxes(1, [])
        self.assertEqual(session.bulk_accept(0.5, dry_run=True), 1)
        self.assertEqual(session.entries[0].state, "pending")
        self.assertEqual(session.bulk_accept(0.5), 1)
        self.assertEqual(session.entries[0].state, "accepted")
        self.assertEqual(session.entries[0].review_method, "bulk_min_score")
        self.assertEqual([box["label"] for box in session.entries[0].boxes], ["bottle"])
        self.assertEqual((session.entries[1].state, session.entries[2].state), ("pending", "pending"))
        with self.assertRaises(ValueError):
            session.bulk_accept(1.5)

    def test_class_list_and_rename(self):
        session = detected_session(self.folder, 2)
        session.set_classes(["cup", "bottle", "cup", " "])
        self.assertEqual(session.classes, ["cup", "bottle"])
        session.accept(0)
        self.assertEqual(session.rename_class("bottle", "water bottle"), 2)
        self.assertEqual(session.classes, ["cup", "water bottle"])
        self.assertEqual(session.entries[0].state, "edited")
        self.assertEqual(session.entries[1].state, "pending")
        self.assertEqual(session.entries[0].candidates[0]["label"], "bottle")
        self.assertEqual(session.labels_in_use(), ["water bottle", "cup"])

    def test_rename_keeps_state_of_images_without_that_class_and_of_bulk_acceptances(self):
        session = detected_session(self.folder, 3)
        self.assertEqual(session.bulk_accept(0.5), 3)  # the 0.3 cup is trimmed: boxes != candidates
        session.reset(2)
        self.assertEqual(session.accept(2), "accepted")  # opened and accepted unchanged
        self.assertEqual(session.rename_class("phone", "cell phone"), 0)
        self.assertEqual([(entry.state, entry.review_method) for entry in session.entries],
                         [("accepted", "bulk_min_score")] * 2 + [("accepted", "per_image")])
        self.assertEqual(session.rename_class("bottle", "jar"), 3)
        self.assertEqual([(entry.state, entry.review_method) for entry in session.entries],
                         [("accepted", "bulk_min_score")] * 2 + [("edited", "per_image")])
        self.assertEqual([box["label"] for box in session.entries[0].boxes], ["jar"])


class ExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "源图片"
        self.session = session = detected_session(self.folder, 4)
        run = session.begin_run({"model": "yoloe-26s-seg.pt", "model_sha256": "ab" * 32, "prompts": ["bottle", "cup"],
                                 "conf": 0.25, "iou": 0.45, "max_det": 300, "imgsz": 640, "config": "x"})
        for index, entry in enumerate(session.entries):
            session.set_candidates(index, candidates(), 100, 80, entry.sha256, run)
        session.finish_run(run, processed=4, model_sha256="ab" * 32)
        session.set_classes(["bottle", "cup"])
        session.accept(0)
        # Image 1: delete the bottle, move the low-score cup, draw a phone.
        moved = dict(candidates()[1], xyxy=[45., 10., 95., 70.])
        session.update_boxes(1, [moved, {"label": "cell phone", "xyxy": [60., 50., 80., 79.]}])
        session.accept(1)
        session.reject(2)
        self.before = sorted(path.name for path in self.folder.iterdir())

    def test_yolo_exports_only_accepted_or_edited_images(self):
        target = self.root / "out" / "yolo"
        result = export_yolo(self.session, target, min_score=0.5)
        self.assertEqual((result["images"], result["boxes"], result["skipped"]), (2, 4, 0))
        self.assertEqual(sorted(path.name for path in (target / "images").iterdir()), ["img0.png", "img1.png"])
        self.assertEqual(sorted(path.name for path in (target / "labels").iterdir()), ["img0.txt", "img1.txt"])
        self.assertEqual((target / "classes.txt").read_text(encoding="utf-8").splitlines(), ["bottle", "cup", "cell phone"])
        img0 = (target / "labels" / "img0.txt").read_text().splitlines()
        img1 = (target / "labels" / "img1.txt").read_text().splitlines()
        # A person accepted the 0.3 cup: min_score only filters bulk-accepted images.
        self.assertEqual(img0, ["0 0.200000 0.500000 0.200000 0.500000", "1 0.745000 0.500000 0.490000 0.750000"])
        self.assertEqual(img1, ["1 0.700000 0.500000 0.500000 0.750000", "2 0.700000 0.806250 0.200000 0.362500"])
        for line in img0 + img1:
            self.assertRegex(line, YOLO_LINE)
            self.assertTrue(all(0 <= float(value) <= 1 for value in line.split()[1:]))
        self.assertEqual((target / "images.txt").read_text().splitlines(), ["./images/img0.png", "./images/img1.png"])
        data = (target / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("NOT a validation split", data)
        try:
            import yaml
        except ImportError:
            yaml = None
        if yaml is not None:
            parsed = yaml.safe_load(data)
            self.assertEqual((parsed["train"], parsed["nc"], parsed["names"]), ("images.txt", 3, {0: "bottle", 1: "cup", 2: "cell phone"}))
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()), self.before)
        self.assertFalse(self.session.dirty)
        with self.assertRaises(FileExistsError):
            export_yolo(self.session, target)

    def test_coco_schema_ids_and_pycocotools(self):
        target = self.root / "out" / "coco"
        result = export_coco(self.session, target)
        coco = json.loads((target / "annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(set(coco), {"info", "licenses", "images", "annotations", "categories"})
        self.assertIn(LABEL_PROVENANCE, coco["info"]["description"])
        self.assertTrue(coco["licenses"])
        self.assertEqual([(row["id"], row["file_name"], row["width"], row["height"]) for row in coco["images"]],
                         [(1, "img0.png", 100, 80), (2, "img1.png", 100, 80)])
        self.assertEqual([(row["id"], row["name"]) for row in coco["categories"]], [(1, "bottle"), (2, "cup"), (3, "cell phone")])
        self.assertEqual([row["id"] for row in coco["annotations"]], [1, 2, 3, 4])
        self.assertEqual(result["boxes"], 4)
        first = coco["annotations"][0]
        self.assertEqual((first["image_id"], first["category_id"], first["bbox"], first["area"], first["iscrowd"]),
                         (1, 1, [10., 20., 20., 40.], 800., 0))
        for row in coco["annotations"]:
            x, y, width, height = row["bbox"]
            self.assertAlmostEqual(row["area"], width * height, places=3)
            self.assertLessEqual(x + width, 100)
            self.assertLessEqual(y + height, 80)
        self.assertEqual([row["attributes"]["origin"] for row in coco["annotations"]],
                         ["model", "model", "model_edited", "human"])
        try:
            from pycocotools.coco import COCO
        except ImportError:
            self.skipTest("pycocotools is not installed")
        loaded = COCO(str(target / "annotations.json"))
        self.assertEqual(len(loaded.getAnnIds()), 4)
        self.assertEqual(loaded.getCatIds(catNms=["cell phone"]), [3])

    def test_manifest_records_review_provenance_model_and_parameters(self):
        target = self.root / "out" / "coco"
        export_coco(self.session, target, min_score=0.2)
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["label_provenance"], LABEL_PROVENANCE)
        self.assertFalse(manifest["human_ground_truth"])
        self.assertIn("not independent human ground truth", manifest["ground_truth_note"])
        datetime.fromisoformat(manifest["created_at"])
        run = manifest["detection_runs"][0]
        self.assertEqual((run["model"], run["model_sha256"], run["prompts"], run["conf"], run["iou"]),
                         ("yoloe-26s-seg.pt", "ab" * 32, ["bottle", "cup"], 0.25, 0.45))
        self.assertEqual(manifest["runs_used_by_exported_images"], [0])
        self.assertEqual(manifest["export_filter"]["min_score"], 0.2)
        self.assertEqual(manifest["review"]["rejected_not_exported"], 1)
        self.assertEqual(manifest["review"]["pending_not_exported"], 1)
        self.assertEqual([(row["file"], row["state"]) for row in manifest["images"]], [("img0.png", "accepted"), ("img1.png", "edited")])
        self.assertEqual(manifest["images"][1]["box_origins"], {"model": 0, "model_edited": 1, "human": 1})
        self.assertEqual(manifest["images"][0]["source_sha256"], hashlib.sha256((self.folder / "img0.png").read_bytes()).hexdigest())
        self.assertIn("auto_annotation.py", manifest["code_sha256"])

    def test_min_score_filters_only_bulk_accepted_images_and_never_empties_them(self):
        self.assertEqual(self.session.bulk_accept(0.2), 1)  # img3: both candidates, nobody looked
        target = self.root / "out" / "yolo"
        result = export_yolo(self.session, target, min_score=0.5)
        self.assertEqual((result["images"], result["boxes"], result["skipped"]), (3, 5, 0))
        self.assertEqual(len((target / "labels" / "img0.txt").read_text().splitlines()), 2)
        self.assertEqual((target / "labels" / "img3.txt").read_text().splitlines(), ["0 0.200000 0.500000 0.200000 0.500000"])
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        rows = {row["file"]: row for row in manifest["images"]}
        self.assertEqual({name: row["filtered_by_min_score"] for name, row in rows.items()},
                         {"img0.png": 0, "img1.png": 0, "img3.png": 1})
        # At 0.95 no box of img3 is left: it is skipped, not exported with an empty label.
        result = export_coco(self.session, self.root / "out" / "coco", min_score=0.95)
        self.assertEqual((result["images"], result["skipped"]), (2, 1))
        manifest = json.loads((self.root / "out" / "coco" / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["export_filter"]["bulk_images_without_box_at_min_score"], ["img3.png"])
        self.assertEqual(manifest["review"]["bulk_accepted_not_exported_no_box_at_min_score"], 1)
        self.assertEqual([row["file"] for row in manifest["images"]], ["img0.png", "img1.png"])
        bulk_only = detected_session(self.root / "bulk", 1)
        bulk_only.bulk_accept(0.5)
        with self.assertRaisesRegex(ValueError, "批量接受"):
            export_yolo(bulk_only, self.root / "out" / "bulk_only", min_score=0.95)
        self.assertFalse((self.root / "out" / "bulk_only").exists())

    def test_manifest_counts_only_boxes_that_are_written(self):
        # A zero-width box (e.g. a slip of the mouse) cannot be written to YOLO or COCO.
        self.session.update_boxes(0, candidates() + [{"label": "cup", "xyxy": [40., 10., 40., 50.]}])
        self.session.accept(0)
        for fmt, export in (("yolo", export_yolo), ("coco", export_coco)):
            target = self.root / "out" / fmt
            result = export(self.session, target)
            manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
            counted = [row["exported_boxes"] for row in manifest["images"]]
            self.assertEqual(counted, [2, 2])
            self.assertEqual(result["boxes"], sum(counted))
            self.assertEqual([sum(row["box_origins"].values()) for row in manifest["images"]], counted)
        self.assertEqual(len((self.root / "out" / "yolo" / "labels" / "img0.txt").read_text().splitlines()), 2)

    def test_refuses_unsafe_or_empty_exports_and_cleans_staging(self):
        with self.assertRaises(ValueError):
            export_yolo(self.session, self.folder / "labels_here")
        write_image(self.folder / "img0.png", value=200)
        with self.assertRaises(ValueError):
            export_coco(self.session, self.root / "out" / "changed")
        self.assertFalse((self.root / "out").exists() and any((self.root / "out").iterdir()))
        empty = detected_session(self.root / "other", 1)
        with self.assertRaises(ValueError):
            export_yolo(empty, self.root / "out" / "empty")

    def test_yolo_refuses_label_name_clashes_but_coco_keeps_both(self):
        write_image(self.folder / "img0.jpg", value=7)
        session = AnnotationSession(self.folder)
        for index, entry in enumerate(session.entries):
            image, sha = read_image_with_sha(entry.path)
            session.set_candidates(index, candidates(), image.shape[1], image.shape[0], sha)
            session.accept(index)
        with self.assertRaises(ValueError):
            export_yolo(session, self.root / "out" / "yolo")
        self.assertFalse((self.root / "out" / "yolo").exists())
        coco = json.loads((export_coco(session, self.root / "out" / "coco")["path"] / "annotations.json").read_text())
        self.assertIn("img0.jpg", [row["file_name"] for row in coco["images"]])

    def test_recursive_yolo_labels_are_where_ultralytics_reads_them(self):
        source = self.root / "数据集"
        for relative in ("a.png", "子目录/b.png", "子目录/深/c.png"):
            write_image(source / relative)
        session = AnnotationSession(source, recursive=True)
        for index, entry in enumerate(session.entries):
            session.set_candidates(index, candidates(), 100, 80, read_image_with_sha(entry.path)[1])
            session.accept(index)
        target = export_yolo(session, self.root / "out" / "yolo")["path"]
        listed = (target / "images.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(listed, ["./images/a.png", "./images/子目录/b.png", "./images/子目录/深/c.png"])
        try:
            from ultralytics.data.utils import img2label_paths
        except ImportError:
            self.skipTest("ultralytics is not installed")
        images = [str(target / line[2:]) for line in listed]
        for label in img2label_paths(images):
            self.assertTrue(Path(label).is_file(), label)
        # A subfolder named images would make Ultralytics read images/labels/...: refused, no partial output.
        write_image(source / "images" / "子目录" / "d.png")
        session = AnnotationSession(source, recursive=True)
        nested = next(index for index, entry in enumerate(session.entries) if entry.relative.startswith("images/"))
        session.set_candidates(nested, candidates(), 100, 80, read_image_with_sha(session.entries[nested].path)[1])
        session.accept(nested)
        with self.assertRaisesRegex(ValueError, "images"):
            export_yolo(session, self.root / "out" / "yolo_nested")
        self.assertFalse((self.root / "out" / "yolo_nested").exists())
        self.assertEqual(export_coco(session, self.root / "out" / "coco_nested")["images"], 1)

    def test_failed_export_leaves_no_partial_directory(self):
        out = self.root / "out"
        with patch("auto_annotation._copy_image", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                export_yolo(self.session, out / "yolo")
        self.assertEqual(list(out.iterdir()), [])
        self.assertTrue(self.session.dirty)

    def test_box_origin(self):
        model = candidates()
        self.assertEqual(box_origin(model[0], model), "model")
        self.assertEqual(box_origin(dict(model[0], label="cup"), model), "model_edited")
        self.assertEqual(box_origin({"label": "cup", "xyxy": [1, 2, 3, 4]}, model), "human")


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name) / "图片"
        for index in range(3):
            write_image(folder / f"{index}.png")
        (folder / "3_bad.png").write_bytes(b"broken")
        self.jobs = list(enumerate(sorted(folder.iterdir())))
        self.options = {"mode": "text", "class_names": ["bottle"], "conf": 0.35, "iou": 0.45, "max_det": 17,
                        "imgsz": 640, "tiled": False, "sliced": False, "slice_size": 512, "slice_overlap": 0.2,
                        "max_slices": 9, "low_light": False, "supplement_dark": True, "enhancement_strength": 0.65,
                        "model": "yoloe-26s-seg.pt"}
        self.engine = MagicMock()
        self.engine.process_frame.side_effect = lambda frame, *args, **kwargs: (
            frame, [Detection((1., 2., 30., 40.), 0.8, 0, "bottle", "original", track_id=4, display_id=1)], "ok")

    def run_worker(self, **kwargs):
        from auto_annotation_ui import AutoAnnotationWorker
        worker = AutoAnnotationWorker(self.engine, MagicMock(), self.jobs, self.options, model_sha256="f" * 64, **kwargs)
        events = {"done": [], "failed": [], "progress": [], "error": []}
        worker.image_done.connect(lambda index, result: events["done"].append((index, result)))
        worker.image_failed.connect(lambda index, value: events["failed"].append(index))
        worker.progress.connect(lambda done, total: events["progress"].append((done, total)))
        worker.error_signal.connect(events["error"].append)
        return worker, events

    def test_serial_detection_uses_shared_path_without_tracker(self):
        worker, events = self.run_worker()
        worker.run()
        self.assertEqual(self.engine.process_frame.call_count, 3)
        for call in self.engine.process_frame.call_args_list:
            self.assertEqual(call.args[1:5], (0.35, 0.45, ["bottle"], None))
            self.assertIsNone(call.kwargs["tracker"])
            self.assertIsNone(call.kwargs["enhancer"])
            self.assertEqual((call.kwargs["max_det"], call.kwargs["sliced"]), (17, False))
        self.assertEqual([index for index, _ in events["done"]], [0, 1, 2])
        self.assertEqual(events["failed"], [3])
        self.assertEqual(events["progress"][-1], (4, 4))
        box = events["done"][0][1]["boxes"][0]
        self.assertEqual((box["label"], box["xyxy"], box["score"], box["source"]), ("bottle", [1., 2., 30., 40.], 0.8, "original"))
        self.assertEqual(events["done"][0][1]["sha256"], hashlib.sha256(self.jobs[0][1].read_bytes()).hexdigest())
        self.assertEqual((worker.processed, worker.failed), (3, None))

    def test_cancel_stops_before_the_next_image(self):
        worker, events = self.run_worker()
        def once(frame, *args, **kwargs):
            worker.running = False
            return frame, [], "ok"
        self.engine.process_frame.side_effect = once
        worker.run()
        self.assertEqual(self.engine.process_frame.call_count, 1)
        self.assertEqual(len(events["done"]), 1)

    def test_resource_error_stops_without_retry_or_device_switch(self):
        self.engine.process_frame.side_effect = ResourceLimitError("GPU 空闲显存低于 1 GiB，已停止当前推理。")
        worker, events = self.run_worker()
        worker.run()
        self.engine.process_frame.assert_called_once()
        self.engine.load_model.assert_not_called()
        self.assertEqual(events["done"], [])
        self.assertIn("ResourceLimitError", events["error"][0])
        self.assertIn("ResourceLimitError", worker.failed)

    def test_visual_reference_is_encoded_once_in_the_worker(self):
        self.engine.set_visual_prompt.return_value = "SUCCESS"
        self.options["class_names"] = ["cup"]
        reference = np.zeros((20, 20, 3), np.uint8)
        worker, events = self.run_worker(visual=(reference, (2, 2, 10, 10), "cup"))
        worker.run()
        self.engine.set_visual_prompt.assert_called_once_with(reference, (2, 2, 10, 10), "cup")
        self.assertEqual(self.engine.process_frame.call_args.args[3], ["cup"])
        self.engine.set_visual_prompt.return_value = "failed"
        worker, events = self.run_worker(visual=(reference, (2, 2, 10, 10), "cup"))
        calls = self.engine.process_frame.call_count
        worker.run()
        self.assertEqual(self.engine.process_frame.call_count, calls)
        self.assertTrue(events["error"])

    def test_low_light_is_reset_per_independent_image(self):
        self.options["low_light"] = True
        enhancer = MagicMock()
        frame = np.zeros((8, 8, 3), np.uint8)
        detect_candidates(self.engine, frame, self.options, enhancer)
        detect_candidates(self.engine, frame, self.options, enhancer)
        self.assertEqual(enhancer.reset.call_count, 2)
        self.assertIs(self.engine.process_frame.call_args.kwargs["enhancer"], enhancer)

    def test_weights_fingerprint_is_computed_in_the_worker(self):
        from auto_annotation_ui import AutoAnnotationWorker
        worker = AutoAnnotationWorker(self.engine, MagicMock(), [], self.options)
        with patch("auto_annotation_ui.weights_sha256", return_value="c" * 64) as digest:
            worker.run()
        digest.assert_called_once_with("yoloe-26s-seg.pt")
        self.assertEqual(worker.model_sha256, "c" * 64)

    def sized_jobs(self, sizes):
        folder = self.jobs[0][1].parent / "尺寸"
        self.jobs = [(index, folder / f"{index}.png") for index in range(len(sizes))]
        for (_, path), (width, height) in zip(self.jobs, sizes):
            write_image(path, width, height)

    def test_sahi_plan_over_the_slice_cap_skips_only_that_image(self):
        from auto_annotation_ui import SLICE_SKIP_PREFIX
        # 1080p needs 15 slices at 512 px (640 px would fit); a 12 MP photo fits at no offered size.
        self.sized_jobs([(640, 480), (1920, 1080), (640, 480), (4032, 3024), (800, 600)])
        self.options.update(sliced=True, slice_size=512)
        worker, events = self.run_worker()
        messages = {}
        worker.image_failed.connect(lambda index, value: messages.__setitem__(index, value))
        worker.run()
        self.assertEqual([index for index, _ in events["done"]], [0, 2, 4])
        self.assertEqual((events["failed"], worker.skipped), ([1, 3], [1, 3]))
        self.assertEqual((worker.processed, worker.failed, events["error"]), (3, None, []))
        self.assertEqual(self.engine.process_frame.call_count, 3)
        self.assertEqual(events["progress"][-1], (5, 5))
        self.assertTrue(all(value.startswith(SLICE_SKIP_PREFIX) for value in messages.values()))
        self.assertIn("可改用 640 像素切片", messages[1])
        self.assertIn("界面提供的切片尺寸都放不下", messages[3])
        self.assertIn("Use 640 px slices", render(messages[1], "en"))

    def test_real_detection_path_finishes_a_folder_with_one_1080p_image(self):
        # Before, the 1080p image ended the batch after image 0 (scan evidence sahi_batch_abort.txt).
        from engines import OpenVocabEngine
        engine = self.engine = OpenVocabEngine()
        engine.detect = MagicMock(return_value=[])
        self.sized_jobs([(640, 480), (1920, 1080), (640, 480), (800, 600)])
        self.options.update(sliced=True, slice_size=512)
        worker, events = self.run_worker()
        with patch("sliced_inference.preflight_inference"):
            worker.run()
        self.assertEqual([index for index, _ in events["done"]], [0, 2, 3])
        self.assertEqual((events["failed"], events["error"], worker.failed), ([1], [], None))
        # Full image + slices: 2 slices for each 640x480 image, 4 for 800x600 at 512 px.
        self.assertEqual(engine.detect.call_count, (1 + 2) + (1 + 2) + (1 + 4))

    def test_invalid_slice_settings_still_stop_the_batch(self):
        self.options.update(sliced=True, slice_size=100)
        self.engine.process_frame.side_effect = ValueError("切片尺寸必须是 128 到 960 之间的整数")
        worker, events = self.run_worker()
        worker.run()
        self.engine.process_frame.assert_called_once()
        self.assertEqual((worker.skipped, events["done"]), ([], []))
        self.assertIn("ValueError", worker.failed)


class AutoAnnotationTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])
        from main import AIApp
        cls.AIApp = AIApp

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "待标注"
        for index in range(3):
            write_image(self.folder / f"图{index}.png", value=index * 50)
        self.window = self.AIApp()
        self.panel = self.window.annotation_panel
        self.window.show()
        self.qt_app.processEvents()

    def tearDown(self):
        self.window.worker = None
        self.panel.worker = None
        self.panel.session = None  # closing must not ask about unexported results
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def switch(self, language):
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData(language))
        self.qt_app.processEvents()

    def texts(self):
        values = []
        for widget in [self.panel, *self.panel.findChildren(QWidget)]:
            values.append(widget.toolTip())
            if isinstance(widget, (QLabel, QAbstractButton)):
                values.append(widget.text())
            if isinstance(widget, QLineEdit):
                values.append(widget.placeholderText())
            if isinstance(widget, QComboBox) and not widget.isEditable():
                values += [widget.itemText(index) for index in range(widget.count())]
        return [value for value in values if value]

    def simulate_detection(self, index, boxes=None):
        entry = self.panel.session.entries[index]
        self.panel.on_image_done(index, {"boxes": candidates() if boxes is None else boxes, "width": 100, "height": 80,
                                         "sha256": read_image_with_sha(entry.path)[1]})

    def test_tab_replaces_placeholder_and_english_translation_is_complete(self):
        tabs = self.window.tabs
        self.assertIs(tabs.widget(3), self.panel)
        self.assertEqual(tabs.tabText(3), "4. 自动标注")
        self.assertNotIn("开发中", " ".join(self.texts()))
        self.switch("en")
        self.assertEqual(tabs.tabText(3), "4. Auto-labeling")
        untranslated = [value for value in self.texts() if HAN.search(value)]
        self.assertEqual(untranslated, [])
        self.assertIn(self.panel.canvas.empty_text, EN)
        self.assertTrue(self.panel.load_folder(self.folder))
        self.simulate_detection(0)
        item = self.panel.image_list.item(1).text()
        self.assertIn("图1.png", item)
        self.assertIn("Not detected", item)
        self.assertIn("Pending", self.panel.image_list.item(0).text())
        self.assertIn("score 0.90", self.panel.box_list.item(0).text())
        self.assertIn("3 images", self.panel.summary.text())
        self.switch("zh")
        self.assertIn("未检测", self.panel.image_list.item(1).text())
        self.assertIn("分数 0.90", self.panel.box_list.item(0).text())

    def test_buttons_follow_folder_selection_review_and_running_state(self):
        panel = self.panel
        for button in (panel.start_button, panel.stop_button, panel.accept_button, panel.bulk_button,
                       panel.export_yolo_button, panel.export_coco_button):
            self.assertFalse(button.isEnabled())
        self.assertTrue(panel.folder_button.isEnabled())
        panel.load_folder(self.folder)
        self.assertTrue(panel.start_button.isEnabled())
        self.assertFalse(panel.export_yolo_button.isEnabled())
        self.simulate_detection(0)
        self.assertEqual(panel.current, 0)
        self.assertTrue(panel.accept_button.isEnabled())
        self.assertTrue(panel.bulk_button.isEnabled())
        panel.accept_current()
        self.assertEqual(panel.session.entries[0].state, "accepted")
        self.assertTrue(panel.export_yolo_button.isEnabled())
        self.assertTrue(panel.export_coco_button.isEnabled())
        self.assertEqual(panel.current, 1)  # moves on to the next pending image
        panel.worker = MagicMock()
        panel.update_controls()
        self.assertFalse(panel.start_button.isEnabled())
        self.assertTrue(panel.stop_button.isEnabled())
        self.assertFalse(panel.folder_button.isEnabled())
        self.assertFalse(panel.export_yolo_button.isEnabled())
        self.assertFalse(panel.bulk_button.isEnabled())

    def test_start_snapshots_current_settings_and_blocks_parallel_model_use(self):
        window, panel = self.window, self.panel
        panel.load_folder(self.folder)
        panel.prompt_input.setText("bottle, cup")
        window.conf_slider.setValue(33)
        window.max_det_spin.setValue(25)
        window.resolution_combo.setCurrentIndex(1)
        window.active_model_path = "yoloe-26s-seg.pt"
        worker = MagicMock()
        with patch.object(window, "_ensure_detection_model", return_value=True) as ensure, \
                patch("auto_annotation_ui.AutoAnnotationWorker", return_value=worker) as factory:
            panel.start()
        ensure.assert_called_once_with(["bottle", "cup"], visual=False)
        engine, enhancer, jobs, options = factory.call_args.args
        self.assertIs(engine, window.engine)
        self.assertIs(enhancer, window.dce_engine)
        self.assertEqual([index for index, _ in jobs], [0, 1, 2])
        self.assertEqual((options["class_names"], options["conf"], options["max_det"], options["imgsz"], options["mode"]),
                         (["bottle", "cup"], 0.33, 25, 960, "text"))
        run = panel.session.runs[0]
        self.assertEqual((run["model"], run["prompts"], run["engine"], run["tracker"]),
                         ("yoloe-26s-seg.pt", ["bottle", "cup"], "OpenVocabEngine", None))
        self.assertEqual(panel.class_input.text(), "bottle, cup")
        worker.start.assert_called_once()
        self.assertTrue(window.annotation_busy())
        with patch("main.QMessageBox.warning") as warning:
            window.start_text_stream()
            window.apply_settings()
        self.assertEqual(warning.call_count, 2)
        self.assertEqual(warning.call_args.args[1], "自动标注进行中")
        self.assertIsNone(window.worker)
        # The global stop-and-release button also stops auto-annotation first.
        worker.stop.return_value = True
        worker.running, worker.failed, worker.processed, worker.model_sha256 = False, None, 0, "d" * 64
        window.engine = MagicMock()
        self.assertTrue(window.release_model())
        worker.stop.assert_called_once()
        window.engine.unload.assert_called_once()
        self.assertFalse(window.annotation_busy())
        self.assertTrue(run["stopped_by_user"])
        self.assertEqual(run["model_sha256"], "d" * 64)
        worker.stop.return_value = False
        panel.worker = worker
        self.assertFalse(window.release_model())

    def test_start_with_nothing_to_detect_keeps_live_detection_and_model(self):
        window, panel = self.window, self.panel
        panel.load_folder(self.folder)
        panel.prompt_input.setText("bottle")
        window.device_combo.setCurrentText("CPU")
        engine = window.engine = MagicMock()
        engine.device, engine.inference_options = "cpu", {}
        engine.process_frame.side_effect = lambda frame, *args, **kwargs: (
            frame, [Detection((1., 2., 30., 40.), 0.8, 0, "bottle")], "ok")
        window.active_model_path, window.active_model_sha256 = "yoloe-26s-seg.pt", "e" * 64
        with patch.object(window, "_ensure_detection_model", return_value=True), \
                patch("auto_annotation_ui.weights_sha256", return_value="e" * 64):
            panel.start()
            self.assertTrue(panel.worker.wait(10000))
            for _ in range(50):
                self.qt_app.processEvents()
                if panel.worker is None:
                    break
        self.assertEqual(panel.session.counts()["detected"], 3)
        # Same settings and loaded model: nothing to do; live detection and weights are left alone.
        with patch.object(window, "stop_video_stream") as stop, patch.object(window, "_ensure_detection_model") as ensure:
            panel.start()
        stop.assert_not_called()
        ensure.assert_not_called()
        self.assertIn("没有需要检测", panel.status_label.text())
        self.assertEqual(len(panel.session.runs), 1)
        # A changed setting needs a new run, so Start goes on to stop live detection first.
        window.conf_slider.setValue(window.conf_slider.value() + 5)
        with patch.object(window, "stop_video_stream", return_value=False) as stop, \
                patch.object(window, "_ensure_detection_model") as ensure:
            panel.start()
        stop.assert_called_once()
        ensure.assert_not_called()
        # Every image reviewed: nothing to do, even though other weights would have to be loaded.
        for index in range(3):
            panel.session.reject(index)
        window.device_combo.setCurrentText("CUDA")
        with patch.object(window, "stop_video_stream") as stop, patch.object(window, "_ensure_detection_model") as ensure:
            panel.start()
        stop.assert_not_called()
        ensure.assert_not_called()

    def test_empty_prompt_never_falls_back_to_preset_text_classes(self):
        self.panel.load_folder(self.folder)
        with patch.object(self.window, "_ensure_detection_model", return_value=True), \
                patch("auto_annotation_ui.AutoAnnotationWorker") as factory:
            self.panel.start()
        factory.assert_not_called()
        self.assertIn("无提示", self.panel.status_label.text())
        self.window.training_active = True
        with patch.object(self.window, "_ensure_detection_model") as ensure:
            self.panel.start()
        ensure.assert_not_called()
        self.window.training_active = False

    def test_real_thread_delivers_results_and_review_exports(self):
        window, panel = self.window, self.panel
        panel.load_folder(self.folder)
        panel.prompt_input.setText("bottle")
        engine = window.engine = MagicMock()
        engine.device, engine.inference_options = "cpu", {}
        engine.process_frame.side_effect = lambda frame, *args, **kwargs: (
            frame, [Detection((1., 2., 30., 40.), 0.8, 0, "bottle"), Detection((40., 2., 60., 40.), 0.2, 0, "bottle")], "ok")
        window.active_model_path = "yoloe-26s-seg.pt"
        with patch.object(window, "_ensure_detection_model", return_value=True), \
                patch("auto_annotation_ui.weights_sha256", return_value="e" * 64):
            panel.start()
            worker = panel.worker
            self.assertTrue(worker.wait(10000))
            for _ in range(50):
                self.qt_app.processEvents()
                if panel.worker is None:
                    break
        self.assertIsNone(panel.worker)
        self.assertEqual(panel.session.counts()["detected"], 3)
        self.assertEqual(panel.session.runs[0]["model_sha256"], "e" * 64)
        self.assertEqual(panel.current, 0)
        # Delete the low-score box on image 0, accept; reject image 1; bulk-accept the rest.
        panel.canvas.index = 1
        panel.canvas.delete_selected()
        self.assertEqual(panel.session.entries[0].state, "pending")
        panel.accept_current()
        self.assertEqual(panel.session.entries[0].state, "edited")
        panel.reject_current()
        self.assertEqual(panel.session.entries[1].state, "rejected")
        with patch("auto_annotation_ui.QMessageBox.question", return_value=QtMessageBox.Yes) as question:
            panel.bulk_score.setValue(0.5)
            panel.bulk_accept()
        question.assert_called_once()
        self.assertEqual(panel.session.entries[2].review_method, "bulk_min_score")
        self.assertEqual(len(panel.session.entries[2].boxes), 1)
        out = self.root / "导出"
        with patch("auto_annotation_ui.QFileDialog.getExistingDirectory", return_value=str(out)):
            panel.export("yolo")
            panel.export("coco")
        exports = sorted(out.iterdir())
        self.assertEqual([path.name.split("_")[2] for path in exports], ["coco", "yolo"])
        coco = json.loads((exports[0] / "annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(len(coco["images"]), 2)
        self.assertIn("已导出", panel.status_label.text())
        self.assertFalse(panel.session.dirty)

    def test_stop_keeps_the_in_flight_result_in_its_run(self):
        window, panel = self.window, self.panel
        panel.load_folder(self.folder)
        panel.prompt_input.setText("bottle")
        engine = window.engine = MagicMock()
        engine.device, engine.inference_options = "cpu", {}

        def slow(frame, *args, **kwargs):
            time.sleep(0.3)
            return frame, [Detection((1., 2., 30., 40.), 0.8, 0, "bottle")], "ok"
        engine.process_frame.side_effect = slow
        window.active_model_path = "yoloe-26s-seg.pt"
        with patch.object(window, "_ensure_detection_model", return_value=True), \
                patch("auto_annotation_ui.weights_sha256", return_value="e" * 64):
            panel.start()
            run, worker = panel.run_index, panel.worker
            for _ in range(500):
                self.qt_app.processEvents()
                if panel.session.entries[0].detected:
                    break
                time.sleep(0.01)
            time.sleep(0.1)  # the next image is now inside detection
            self.assertTrue(panel.stop_worker())
            entries = panel.session.entries
            detected = [entry for entry in entries if entry.detected]
            # Stored before the run was finalised, not after it with no run.
            self.assertEqual(len(detected), worker.processed)
            self.assertGreaterEqual(len(detected), 2)
            for _ in range(20):
                self.qt_app.processEvents()
        self.assertIsNone(panel.worker)
        self.assertEqual([entry.run_index for entry in entries if entry.detected], [run] * len(detected))
        self.assertEqual(panel.session.runs[run]["processed"], len(detected))
        self.assertTrue(panel.session.runs[run]["stopped_by_user"])
        config = panel.session.runs[run]["config"]
        self.assertEqual([index for index, _ in panel.session.targets(config)],
                         [index for index, entry in enumerate(entries) if not entry.detected])

    def test_unexported_review_asks_before_discarding(self):
        panel = self.panel
        panel.load_folder(self.folder)
        self.simulate_detection(0)
        panel.accept_current()
        with patch("auto_annotation_ui.QMessageBox.question", return_value=QtMessageBox.No) as question:
            self.assertFalse(panel.load_folder(self.folder))
            self.assertFalse(panel.may_discard())
        self.assertEqual(question.call_count, 2)
        self.assertEqual(panel.session.entries[0].state, "accepted")

    def test_subfolder_toggle_asks_before_dropping_unreviewed_detections(self):
        panel = self.panel
        write_image(self.folder / "子目录" / "图9.png")
        panel.load_folder(self.folder)
        self.simulate_detection(0)
        self.simulate_detection(1)
        self.assertFalse(panel.session.dirty)
        session = panel.session
        with patch("auto_annotation_ui.QMessageBox.question", return_value=QtMessageBox.No) as question:
            panel.recursive.setChecked(True)
        question.assert_called_once()
        self.assertIn("2", str(question.call_args.args[2]))
        self.assertFalse(panel.recursive.isChecked())
        self.assertIs(panel.session, session)
        self.assertEqual(panel.session.counts()["detected"], 2)
        with patch("auto_annotation_ui.QMessageBox.question", return_value=QtMessageBox.Yes):
            panel.recursive.setChecked(True)
        self.assertEqual(len(panel.session), 4)
        self.assertEqual(panel.session.counts()["detected"], 0)
        # Nothing detected or reviewed: no question.
        with patch("auto_annotation_ui.QMessageBox.question") as question:
            panel.recursive.setChecked(False)
        question.assert_not_called()
        self.assertEqual(len(panel.session), 3)

    def select_preset(self, key):
        self.panel.preset.setCurrentIndex(self.panel.preset.findData(key))

    def test_accuracy_preset_applies_to_this_run_only_and_is_recorded(self):
        window, panel = self.window, self.panel
        panel.load_folder(self.folder)
        panel.prompt_input.setText("bottle, cup")
        window.conf_slider.setValue(33)
        window.resolution_combo.setCurrentIndex(1)
        window.tile_checkbox.setChecked(True)
        window.dce_checkbox.setChecked(True)
        window.model_combo.setCurrentIndex(window.model_combo.findData("yoloe-26s-seg.pt"))
        self.select_preset("accuracy_first")
        window.active_model_path = "yoloe-26l-seg.pt"  # what the mocked load would leave behind
        worker = MagicMock()
        with patch("auto_annotation_ui.local_weights_present", return_value=True), \
                patch.object(window, "_ensure_detection_model", return_value=True) as ensure, \
                patch("auto_annotation_ui.AutoAnnotationWorker", return_value=worker) as factory:
            panel.start()
        ensure.assert_called_once_with(["bottle", "cup"], visual=False)
        self.assertEqual(window.model_combo.currentData(), "yoloe-26l-seg.pt")
        options = factory.call_args.args[3]
        expected = {"imgsz": 640, "conf": 0.40, "sliced": True, "slice_size": 320, "slice_overlap": 0.2,
                    "max_slices": 9, "tiled": False, "low_light": False, "preset": "accuracy_first"}
        self.assertEqual({key: options[key] for key in expected}, expected)
        self.assertEqual(options["max_det"], window.max_det_spin.value())
        run = panel.session.runs[0]
        self.assertEqual((run["preset"], run["model"], run["slice_size"], run["conf"]),
                         ("accuracy_first", "yoloe-26l-seg.pt", 320, 0.40))
        self.assertIn('"preset": "accuracy_first"', run["config"])
        # The left-side settings are not changed by the preset.
        self.assertEqual((window.conf_slider.value(), window.resolution_combo.currentData()), (33, 960))
        self.assertEqual((window.tile_checkbox.isChecked(), window.sahi_checkbox.isChecked(),
                          window.dce_checkbox.isChecked()), (True, False, True))
        self.assertIn("精度优先预设", panel.status_label.text())
        self.assertFalse(panel.preset.isEnabled())
        # Without a preset the run options are the left-side ones, recorded with preset None.
        self.select_preset("")
        self.assertEqual(panel._preset_options({"conf": 0.33}), {"conf": 0.33, "preset": None})

    def test_accuracy_preset_refuses_non_text_prompts_and_missing_weights(self):
        window, panel = self.window, self.panel
        panel.load_folder(self.folder)
        self.select_preset("accuracy_first")
        before = window.model_combo.currentData()
        cases = (("text", "", True, "只用于文字提示"), ("visual", "bottle", True, "只用于文字提示"),
                 ("text", "bottle", False, "未找到本地 YOLOE-26L 权重"))
        for mode, prompt, present, expected in cases:
            with self.subTest(mode=mode, prompt=prompt, present=present):
                panel.prompt_mode.setCurrentIndex(panel.prompt_mode.findData(mode))
                panel.prompt_input.setText(prompt)
                with patch("auto_annotation_ui.local_weights_present", return_value=present), \
                        patch.object(window, "stop_video_stream") as stop, \
                        patch.object(window, "_ensure_detection_model") as ensure:
                    panel.start()
                stop.assert_not_called()
                ensure.assert_not_called()
                self.assertIn(expected, panel.status_label.text())
        self.assertEqual(window.model_combo.currentData(), before)
        self.assertEqual(panel.session.runs, [])

    def test_image_skipped_for_too_many_slices_is_marked_and_listed_in_the_run(self):
        from auto_annotation_ui import AutoAnnotationWorker
        panel = self.panel
        panel.load_folder(self.folder)
        reason = AutoAnnotationWorker(MagicMock(), MagicMock(), [], {"sliced": True, "slice_size": 512}) \
            ._slice_skip_reason(np.zeros((1080, 1920, 3), np.uint8))
        panel.run_index = panel.session.begin_run({"config": "{}"})
        panel.on_image_failed(1, reason)
        self.assertIn("已跳过（切片过多）", panel.image_list.item(1).text())
        worker = panel.worker = MagicMock(skipped=[1], processed=2, failed=None, running=True, model_sha256="a" * 64)
        panel.on_finished(worker)
        self.assertEqual(panel.session.runs[0]["skipped_too_many_slices"], ["图1.png"])
        self.assertIn("1 张因 SAHI 切片超过上限被跳过", panel.status_label.text())
        self.assertEqual(panel.session.counts()["failed"], 1)
        self.switch("en")
        self.assertIn("Skipped (too many slices)", panel.image_list.item(1).text())
        self.assertIn("1 skipped", panel.status_label.text())
        # Starting again retries the skipped image (e.g. after choosing a larger slice size).
        self.assertIn(1, [index for index, _ in panel.session.targets(None)])


if __name__ == "__main__":
    unittest.main()
