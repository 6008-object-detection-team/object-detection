"""ONNX export/benchmark tooling and the NumPy ONNX engine; no real export or model."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from i18n import EN
from onnx_engine import (OnnxRuntimeEngine, decode_predictions, greedy_nms, letterbox, manifest_path,
                         read_manifest, scale_boxes, to_blob, validate_manifest)
from resource_safety import ResourceLimitError
from tools import benchmark_onnx as bench
from tools import export_onnx as export

PROJECT = Path(__file__).resolve().parent
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None and importlib.util.find_spec("torch") is not None


def valid_manifest(name="m.onnx", classes=("bottle", "cup"), extra=32, digest="0" * 64):
    return {"schema": 1, "file": name, "onnx_sha256": digest, "variant": "seg", "classes": list(classes),
            "export": {"imgsz": 640, "opset": 17}, "source_model": {"sha256": "a" * 64},
            "output_layout": {"format": "raw_one2many_xywh", "num_classes": len(classes), "extra_channels": extra}}


def write_onnx_pair(folder, payload=b"fake-onnx", **overrides):
    path = Path(folder) / "m.onnx"
    path.write_bytes(payload)
    manifest = valid_manifest(digest=hashlib.sha256(payload).hexdigest())
    manifest.update(overrides)
    manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def fake_weight(folder, name="yoloe-26s-seg.pt", size=2 * 1024 * 1024):
    path = Path(folder) / name
    with path.open("wb") as stream:
        stream.truncate(size)
    return path


def export_args(folder, *extra):
    weight = fake_weight(folder)
    return export.build_parser().parse_args(["--model", str(weight), "--output-dir", str(Path(folder) / "out"), *extra])


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_and_each_rejected_field(self):
        self.assertEqual(validate_manifest(valid_manifest())["classes"], ["bottle", "cup"])
        broken = [dict(valid_manifest(), schema=2), dict(valid_manifest(), classes=["cup", "cup"]),
                  dict(valid_manifest(), classes=[]), dict(valid_manifest(), export={"imgsz": 650}),
                  dict(valid_manifest(), export={"imgsz": True}),
                  dict(valid_manifest(), output_layout={"format": "end2end", "num_classes": 2, "extra_channels": 0}),
                  dict(valid_manifest(), output_layout={"format": "raw_one2many_xywh", "num_classes": 3,
                                                        "extra_channels": 0})]
        for manifest in broken:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_read_manifest_checks_file_name_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_onnx_pair(folder)
            self.assertEqual(read_manifest(path)["file"], "m.onnx")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "哈希"):
                read_manifest(path)
            self.assertEqual(read_manifest(path, verify_hash=False)["file"], "m.onnx")
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_onnx_pair(folder, file="other.onnx")
            with self.assertRaisesRegex(ValueError, "文件名"):
                read_manifest(path)
            manifest_path(path).unlink()
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_export_variant_writes_manifest_with_fingerprints_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            target, staging = folder / "out" / "yoloe-26s-seg-mini_eval-640.onnx", folder / "out" / ".staging"
            staging.mkdir(parents=True)
            plan = {"targets": {"seg": str(target)}, "imgsz": 640, "opset": 17, "threads": 2,
                    "classes": ["bottle", "cup", "cell phone"], "class_source": {"class_set": "mini_eval"}}
            model = MagicMock()
            model.model.stride = [8, 16, 32]

            def fake_export(**kwargs):
                staged = Path(model.model.pt_path).with_suffix(".onnx")
                staged.write_bytes(b"graph-bytes")
                self.assertEqual((kwargs["format"], kwargs["opset"], kwargs["dynamic"], kwargs["batch"],
                                  kwargs["simplify"], kwargs["device"]), ("onnx", 17, plan.get("dynamic_hw", False),
                                                                          1, True, "cpu"))
                self.assertNotIn("nms", kwargs)  # Default None: raw one2many output, NMS outside.
                return str(staged)

            model.export.side_effect = fake_export
            check = {"opset_import": {"ai.onnx": 17}, "ir_version": 8, "providers": ["CPUExecutionProvider"],
                     "constant_input_outputs_finite": True,
                     "inputs": [{"name": "images", "shape": [1, 3, 640, 640], "dtype": "tensor(float)"}],
                     "outputs": [{"name": "output0", "shape": [1, 39, 8400], "dtype": "tensor(float)"}],
                     "output_layout": {"format": "raw_one2many_xywh", "box": "xywh", "num_classes": 3,
                                       "extra_channels": 32, "anchors_at_square_input": 8400, "anchors_dynamic": False}}
            source = {"path": "yoloe-26s-seg.pt", "sha256": "b" * 64, "bytes": 1}
            with patch.object(export, "onnx_check", return_value=check):
                manifest = export.export_variant(model, "seg", plan, staging, {"sha256": "c" * 64}, source,
                                                 {"status": "not found"})
            self.assertTrue(target.is_file())
            self.assertFalse(any(staging.iterdir()))
            saved = json.loads(manifest_path(target).read_text(encoding="utf-8"))
            self.assertEqual(saved, json.loads(json.dumps(manifest)))
            self.assertEqual(saved["onnx_sha256"], hashlib.sha256(b"graph-bytes").hexdigest())
            self.assertEqual(saved["classes"], ["bottle", "cup", "cell phone"])
            self.assertEqual(saved["source_model"]["sha256"], "b" * 64)
            self.assertEqual((saved["export"]["opset"], saved["export"]["imgsz"], saved["export"]["batch"],
                              saved["export"]["dynamic"], saved["export"]["nms"], saved["export"]["stride"]),
                             (17, 640, 1, False, None, 32))
            for package in ("torch", "ultralytics", "onnx", "onnxruntime"):
                self.assertIn(package, saved["versions"])
            self.assertIn("tools/export_onnx.py", saved["source_hashes"])
            self.assertEqual(read_manifest(target)["file"], target.name)
            with patch.object(export, "onnx_check", return_value=check), self.assertRaisesRegex(ValueError, "不覆盖"):
                export.export_variant(model, "seg", plan, staging, {}, source, {})
            with patch.object(export, "onnx_check", return_value=dict(check, output_layout=dict(
                    check["output_layout"], extra_channels=0))), self.assertRaisesRegex(ValueError, "附加通道"):
                export.export_variant(model, "seg", dict(plan, targets={"seg": str(folder / "other.onnx")}),
                                      staging, {}, source, {})
            # Dynamic height/width: batch is re-fixed to 1 and the manifest selects the rectangle letterbox.
            plan.update(dynamic_hw=True, targets={"det": str(folder / "out" / "yoloe-26s-det-mini_eval-640-dyn.onnx")})
            dynamic_check = dict(check, output_layout=dict(check["output_layout"], extra_channels=0))
            with patch.object(export, "onnx_check", return_value=dynamic_check) as checked, \
                    patch.object(export, "fix_batch_dimension") as fixed:
                dynamic = export.export_variant(model, "det", plan, staging, {}, source, {})
            fixed.assert_called_once()
            self.assertTrue(checked.call_args.args[-1])
            self.assertEqual((dynamic["export"]["dynamic"], dynamic["export"]["dynamic_hw"]), ("height,width", True))
            self.assertIn("rectangle", dynamic["preprocess"]["letterbox"])


class ExportArgumentTests(unittest.TestCase):
    def test_class_and_variant_parsers(self):
        self.assertEqual(export.parse_classes(" bottle, cell phone "), ["bottle", "cell phone"])
        for value in ["", "cup,,bottle", "cup,cup"]:
            with self.subTest(value=value), self.assertRaises(Exception):
                export.parse_classes(value)
        self.assertEqual(export.parse_variants("det"), ["det"])
        for value in ["seg,seg", "mask", ""]:
            with self.subTest(value=value), self.assertRaises(Exception):
                export.parse_variants(value)

    def test_plan_resolves_class_sets_and_targets_without_torch(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"torch": None, "ultralytics": None}):
            plan = export.plan_export(export_args(folder))
            self.assertEqual(plan["classes"], ["bottle", "cup", "cell phone"])
            self.assertEqual(sorted(Path(path).name for path in plan["targets"].values()),
                             ["yoloe-26s-det-mini_eval-640.onnx", "yoloe-26s-seg-mini_eval-640.onnx"])
            dynamic = export.plan_export(export_args(folder, "--variants", "det", "--dynamic-hw"))
            self.assertEqual(Path(dynamic["targets"]["det"]).name, "yoloe-26s-det-mini_eval-640-dyn.onnx")
            self.assertTrue(dynamic["dynamic_hw"])
            coco = export.plan_export(export_args(folder, "--class-set", "coco10", "--variants", "seg"))
            self.assertEqual(coco["classes"][:3], ["person", "bicycle", "car"])
            self.assertEqual(len(coco["classes"]), 10)
            custom = export.plan_export(export_args(folder, "--classes", "person,cup"))
            self.assertEqual((custom["classes"], custom["class_source"]["class_set"]), (["person", "cup"], "custom"))
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(export.main(["--model", str(Path(folder) / "yoloe-26s-seg.pt"),
                                              "--output-dir", str(Path(folder) / "out"), "--plan"]), 0)
            self.assertIn("planned_not_executed", printed.getvalue())

    def test_invalid_export_requests_stop_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = [["--imgsz", "650"], ["--imgsz", "1024"], ["--opset", "9"], ["--opset", "30"]]
            for extra in cases:
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    export.plan_export(export_args(folder, *extra))
            pointer = fake_weight(folder, "yoloe-26l-seg.pt", 200)
            with self.assertRaisesRegex(ValueError, "LFS"):
                export.plan_export(export.build_parser().parse_args(["--model", str(pointer)]))
            large = fake_weight(folder, "yoloe-26l-seg.pt")
            with self.assertRaisesRegex(ValueError, "仅检测分支"):
                export.plan_export(export.build_parser().parse_args(["--model", str(large)]))
            self.assertEqual(export.plan_export(export.build_parser().parse_args(
                ["--model", str(large), "--variants", "seg", "--output-dir", folder]))["variants"], ["seg"])
            free = fake_weight(folder, "yoloe-26s-seg-pf.pt")
            with self.assertRaisesRegex(ValueError, "无提示"):
                export.plan_export(export.build_parser().parse_args(["--model", str(free), "--variants", "seg"]))
            with self.assertRaises(ValueError):
                export.plan_export(export.build_parser().parse_args(["--model", str(Path(folder) / "missing.pt")]))
            args = export_args(folder)
            target = Path(export.plan_export(args)["targets"]["seg"])
            target.parent.mkdir(parents=True, exist_ok=True)
            manifest_path(target).write_text("{}")
            with self.assertRaisesRegex(ValueError, "不覆盖"):
                export.plan_export(args)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(export.main(["--model", str(Path(folder) / "missing.pt"), "--plan"]), 1)

    def test_independent_export_resource_check(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("resource_safety.available_ram_mb", return_value=1024), \
                    self.assertRaisesRegex(ResourceLimitError, "内存"):
                export.check_export_resources(folder)
            with patch("resource_safety.available_ram_mb", return_value=8192), \
                    patch("shutil.disk_usage", return_value=SimpleNamespace(free=10 * 1024 ** 2)), \
                    self.assertRaisesRegex(ResourceLimitError, "磁盘"):
                export.check_export_resources(folder)
            with patch("resource_safety.available_ram_mb", return_value=8192):
                self.assertEqual(export.check_export_resources(folder)["available_ram_mib"], 8192)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch 未安装")
    def test_detection_transfer_requires_exact_copies(self):
        import torch
        source = {"a": torch.ones(2, 3), "b": torch.zeros(4), "proto": torch.ones(1)}
        self.assertEqual(export.verify_transfer({"a": torch.ones(2, 3), "b": torch.zeros(4)}, source),
                         {"target_tensors": 2, "copied_exactly": 2, "source_only_tensors": 1})
        for target in [{"a": torch.ones(2, 3) * 2}, {"a": torch.ones(3, 2)}, {"new": torch.ones(1)}]:
            with self.subTest(target=list(target)), self.assertRaisesRegex(ValueError, "精确复制"):
                export.verify_transfer(target, source)


class BenchmarkArgumentTests(unittest.TestCase):
    def setUp(self):
        from evaluate_mini_dataset import load_dataset
        self.dataset = load_dataset(PROJECT / "datasets" / "mini_eval", limit=2)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.weight = fake_weight(folder.name)  # Real weights are not needed or loaded.

    def parse(self, *extra):
        return bench.build_parser().parse_args(["--model", str(self.weight), *extra])

    def test_backend_parser_and_rotated_order(self):
        self.assertEqual(bench.parse_backends("pt_cpu_square,ort_cpu_seg"), ["pt_cpu_square", "ort_cpu_seg"])
        for value in ["pt_cpu_square,pt_cpu_square", "trt_gpu", ""]:
            with self.subTest(value=value), self.assertRaises(Exception):
                bench.parse_backends(value)
        names = ["a", "b", "c"]
        orders = [bench.run_order(names, repeat) for repeat in range(3)]
        self.assertEqual(orders, [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]])
        self.assertEqual({order[0] for order in orders}, set(names))

    def test_validation_rejects_unfair_or_mismatched_runs(self):
        with self.assertRaisesRegex(ValueError, "至少重复 3 次"):
            bench.validate_inputs(self.parse("--repeats", "2"), self.dataset)
        with self.assertRaisesRegex(ValueError, "参考后端"):
            bench.validate_inputs(self.parse("--backends", "pt_cpu_rect", "--limit", "1"), self.dataset)
        with self.assertRaisesRegex(ValueError, "--onnx-seg"):
            bench.validate_inputs(self.parse("--backends", "pt_cpu_square,ort_cpu_seg"), self.dataset)
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_onnx_pair(folder, classes=["cup", "bottle", "cell phone"],
                                      output_layout={"format": "raw_one2many_xywh", "num_classes": 3, "extra_channels": 32})
            with self.assertRaisesRegex(ValueError, "顺序"):
                bench.validate_inputs(self.parse("--backends", "pt_cpu_square,ort_cpu_seg", "--onnx-seg", str(path)),
                                      self.dataset)
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_onnx_pair(folder, classes=["bottle", "cup", "cell phone"],
                                      output_layout={"format": "raw_one2many_xywh", "num_classes": 3, "extra_channels": 32})
            with self.assertRaisesRegex(ValueError, "同一份"):
                bench.validate_inputs(self.parse("--backends", "pt_cpu_square,ort_cpu_seg", "--onnx-seg", str(path)),
                                      self.dataset)
            with self.assertRaisesRegex(ValueError, "变体"):
                bench.validate_inputs(self.parse("--backends", "pt_cpu_square,ort_cpu_det", "--onnx-det", str(path)),
                                      self.dataset)
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_onnx_pair(folder, classes=["bottle", "cup", "cell phone"], variant="det",
                                      output_layout={"format": "raw_one2many_xywh", "num_classes": 3, "extra_channels": 0})
            with self.assertRaisesRegex(ValueError, "动态尺寸"):  # A square graph is not the rectangle backend.
                bench.validate_inputs(self.parse("--backends", "pt_cpu_square,ort_cpu_det_rect",
                                                 "--onnx-det-rect", str(path)), self.dataset)

    def test_worker_command_and_environment_pin_threads_and_disable_installs(self):
        args = self.parse("--onnx-seg", "a.onnx", "--onnx-det-rect", "b.onnx", "--threads", "3", "--limit", "2")
        command = bench.worker_command(args, "ort_cpu_seg", Path("out.json"))
        self.assertEqual(command[command.index("--worker") + 1], "ort_cpu_seg")
        self.assertEqual(command[command.index("--threads") + 1], "3")
        self.assertEqual(command[command.index("--limit") + 1], "2")
        self.assertTrue(command[command.index("--onnx-seg") + 1].endswith("a.onnx"))
        self.assertNotIn("--onnx-det", command)
        self.assertTrue(command[command.index("--onnx-det-rect") + 1].endswith("b.onnx"))
        environment = bench.worker_environment(3)
        self.assertEqual([environment[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")], ["3", "3"])
        self.assertEqual(environment["YOLO_AUTOINSTALL"], "False")

    def test_plan_prints_order_without_loading(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed, \
                patch.dict(sys.modules, {"torch": None, "ultralytics": None}):
            self.assertEqual(bench.main(["--model", str(self.weight), "--backends", "pt_cpu_rect,pt_cpu_square",
                                         "--plan"]), 0)
        plan = json.loads(printed.getvalue())
        self.assertEqual(plan["order"][1], ["pt_cpu_square", "pt_cpu_rect"])

    def test_report_records_git_commit_and_tracked_changes(self):
        state = bench.git_state()
        self.assertEqual(set(state), {"commit", "tracked_changes"})
        if state["commit"] is not None:
            self.assertRegex(state["commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(state["tracked_changes"], bool)
        with patch("tools.benchmark_onnx.subprocess.run", side_effect=OSError("no git")):
            self.assertEqual(bench.git_state(), {"commit": None, "tracked_changes": None})


class AgreementTests(unittest.TestCase):
    @staticmethod
    def box(image_id, category, bbox, score):
        return {"image_id": image_id, "category_id": category, "bbox": list(bbox), "score": score}

    def test_iou_values(self):
        self.assertAlmostEqual(bench.xywh_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(bench.xywh_iou([0, 0, 10, 10], [5, 0, 10, 10]), 50 / 150)
        self.assertEqual(bench.xywh_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertEqual(bench.xywh_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_matching_requires_same_class_strict_iou_and_one_to_one(self):
        reference = [self.box(1, 44, [0, 0, 100, 100], 0.9), self.box(1, 44, [1, 0, 100, 100], 0.8)]
        candidate = [self.box(1, 44, [0, 0, 100, 100], 0.85)]
        matches = bench.match_image(reference, candidate, 0.9)
        self.assertEqual(len(matches), 1)  # The single candidate is used once.
        self.assertAlmostEqual(matches[0]["score_delta"], -0.05)
        self.assertEqual(bench.match_image(reference, [self.box(1, 47, [0, 0, 100, 100], 0.9)], 0.9), [])
        # IoU = 0.9 exactly is not "> 0.9".
        exact = [self.box(1, 44, [0, 0, 90, 100], 0.9)]
        self.assertAlmostEqual(bench.xywh_iou(reference[0]["bbox"], exact[0]["bbox"]), 0.9)
        self.assertEqual(bench.match_image(reference[:1], exact, 0.9), [])

    def test_agreement_counts_rates_and_score_threshold(self):
        reference = [self.box(1, 44, [0, 0, 50, 50], 0.9), self.box(1, 47, [60, 60, 20, 20], 0.1),
                     self.box(2, 44, [5, 5, 30, 30], 0.6)]
        candidate = [self.box(1, 44, [0, 0, 50, 50], 0.88), self.box(1, 47, [60, 60, 20, 20], 0.3),
                     self.box(2, 44, [5, 5, 30, 30], 0.6), self.box(2, 47, [0, 0, 5, 5], 0.07)]
        result = bench.agreement(reference, candidate, [1, 2], 0.9, 0.0)
        self.assertEqual((result["reference_count"], result["candidate_count"], result["matched"]), (3, 4, 3))
        self.assertEqual((result["unmatched_reference"], result["unmatched_candidate"]), (0, 1))
        self.assertAlmostEqual(result["reference_match_rate"], 1.0)
        self.assertAlmostEqual(result["candidate_match_rate"], 0.75)
        self.assertAlmostEqual(result["max_abs_score_delta"], 0.2)
        self.assertAlmostEqual(result["mean_abs_score_delta"], (0.02 + 0.2 + 0.0) / 3)
        self.assertAlmostEqual(result["mean_signed_score_delta"], (-0.02 + 0.2) / 3)
        thresholded = bench.agreement(reference, candidate, [1, 2], 0.9, 0.25)
        # A box crossing the threshold in only one backend is an unmatched box.
        self.assertEqual((thresholded["reference_count"], thresholded["candidate_count"], thresholded["matched"]), (2, 3, 2))
        empty = bench.agreement([], [], [1], 0.9, 0.0)
        self.assertIsNone(empty["reference_match_rate"])
        self.assertIsNone(empty["mean_iou"])


class LatencyStatisticTests(unittest.TestCase):
    def test_percentile_matches_existing_linear_interpolation(self):
        from tools.run_additional_experiments import timing_summary
        values = [float(value) for value in range(1, 21)]
        self.assertAlmostEqual(bench.percentile(values, 0.95), 19.05)
        expected = timing_summary([{"latency_ms": value} for value in values], 1)["p95_ms"]
        self.assertAlmostEqual(bench.latency_summary(values)["p95_ms"], expected)
        self.assertEqual(bench.percentile([7.0], 0.95), 7.0)

    def test_summary_pooled_and_repeat_spread(self):
        result = bench.repeat_summary([[10, 12, 14], [11, 13, 15], [30, 30, 30]])
        self.assertEqual(result["pooled"]["n"], 9)
        self.assertAlmostEqual(result["pooled"]["median_ms"], 14)
        self.assertEqual(result["repeat_means_ms"], [12, 13, 30])
        self.assertEqual(result["repeat_mean_range_ms"], [12, 30])
        self.assertAlmostEqual(result["repeat_mean_sd_ms"], np.std([12, 13, 30], ddof=1))
        self.assertEqual(result["per_repeat"][0]["min_ms"], 10)
        for values in [[], [1, -1], [1, float("nan")]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                bench.latency_summary(values)

    def test_summarise_compares_backends_with_ground_truth_and_reference(self):
        from evaluate_mini_dataset import load_dataset
        dataset = load_dataset(PROJECT / "datasets" / "mini_eval", limit=2)
        args = bench.build_parser().parse_args(["--backends", "pt_cpu_square,ort_cpu_seg", "--limit", "2"])
        truth = [{"image_id": row["image_id"], "category_id": row["category_id"], "bbox": row["bbox"], "score": 0.9}
                 for row in dataset["annotations"]["annotations"]
                 if row["image_id"] in {int(item["id"]) for item in dataset["images"]}]
        ids = [int(item["id"]) for item in dataset["images"]]
        runs = []
        for repeat in range(3):
            for name, latency in (("pt_cpu_square", 10.0), ("ort_cpu_seg", 5.0 + repeat)):
                runs.append({"backend": name, "repeat": repeat, "status": "complete", "predictions": truth,
                             "latency_ms": [latency] * len(ids), "image_ids": ids, "stage_ms": [None] * len(ids),
                             "model_load_ms": 1, "prompt_encoding_ms": 0, "warmup_ms": 1, "process_max_rss_mib": 100,
                             "rss_after_load_mib": 90, "torch_imported": name.startswith("pt"),
                             "loadavg_before": [0.5, 0.5, 0.5]})
        report = bench.summarise({"runs": runs}, dataset, args)
        self.assertAlmostEqual(report["backends"]["ort_cpu_seg"]["metrics"]["AP"], 1.0)
        self.assertTrue(report["backends"]["ort_cpu_seg"]["predictions_identical_across_repeats"])
        speed = report["latency_vs_reference"]["ort_cpu_seg"]
        self.assertAlmostEqual(speed["median_ratio"], 0.6)
        self.assertTrue(speed["every_repeat_mean_below_reference_min"])
        first = report["agreement"][0]
        self.assertEqual((first["candidate"], first["reference"]), ("ort_cpu_seg", "pt_cpu_square"))
        self.assertEqual(first["thresholds"][0]["matched"], len(truth))


class OnnxEngineMathTests(unittest.TestCase):
    def test_letterbox_square_padding_and_blob(self):
        frame = np.full((480, 640, 3), 7, np.uint8)
        image = letterbox(frame, 640)
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertTrue((image[:80] == 114).all() and (image[-80:] == 114).all() and (image[80:560] == 7).all())
        tall = letterbox(np.zeros((427, 640, 3), np.uint8), 640)
        self.assertEqual(tall.shape, (640, 640, 3))
        frame[..., 0] = 255  # Blue in BGR becomes the last RGB channel.
        blob = to_blob(letterbox(frame, 640))
        self.assertEqual((blob.shape, blob.dtype), ((1, 3, 640, 640), np.float32))
        self.assertAlmostEqual(float(blob[0, 2, 320, 320]), 1.0)
        self.assertAlmostEqual(float(blob[0, 0, 0, 0]), 114 / 255, places=6)

    def test_greedy_nms_and_class_aware_offsets(self):
        boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], np.float32)
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        self.assertEqual(greedy_nms(boxes, scores, 0.5).tolist(), [0, 2])
        self.assertEqual(greedy_nms(boxes, scores, 0.95).tolist(), [0, 1, 2])
        raw = np.zeros((4 + 2, 3), np.float32)
        raw[:4] = [[5, 6, 55], [5, 5, 55], [10, 10, 10], [10, 10, 10]]
        raw[4, 0], raw[5, 1], raw[4, 2] = 0.9, 0.8, 0.04
        agnostic = decode_predictions(raw, 2, 0.05, 0.5, 100, agnostic=True)
        aware = decode_predictions(raw, 2, 0.05, 0.5, 100, agnostic=False)
        self.assertEqual(len(agnostic), 1)
        self.assertEqual(sorted(aware[:, 5].tolist()), [0.0, 1.0])
        np.testing.assert_allclose(agnostic[0, :4], [0, 0, 10, 10])
        self.assertEqual(len(decode_predictions(raw, 2, 0.95, 0.5, 100)), 0)
        self.assertEqual(len(decode_predictions(raw, 2, 0.05, 0.5, 1, agnostic=False)), 1)
        with self.assertRaises(ValueError):
            decode_predictions(raw[:5], 2, 0.05, 0.5, 100)

    def test_scale_boxes_inverts_letterbox_and_clips(self):
        boxes = np.array([[100, 80 + 100, 200, 80 + 200], [-5, 0, 700, 700]], np.float32)
        scaled = scale_boxes(boxes, (640, 640), (480, 640))
        np.testing.assert_allclose(scaled[0], [100, 100, 200, 200])
        np.testing.assert_allclose(scaled[1], [0, 0, 640, 480])

    @unittest.skipUnless(HAS_ULTRALYTICS, "Ultralytics/torch 未安装")
    def test_numpy_pipeline_matches_ultralytics_reference_functions(self):
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression
        generator = np.random.default_rng(6008)
        for shape in [(480, 640), (640, 427), (332, 500), (366, 640), (1000, 750)]:
            frame = generator.integers(0, 255, (*shape, 3), dtype=np.uint8)
            expected = LetterBox((640, 640), auto=False, stride=32)(image=frame)
            np.testing.assert_array_equal(letterbox(frame, 640), expected)
            rect = LetterBox((640, 640), auto=True, stride=32)(image=frame)
            np.testing.assert_array_equal(letterbox(frame, 640, auto=True, stride=32), rect)
            boxes = np.array([[10, 20, 200, 300], [0, 0, 5, 5]], np.float32)
            reference = ops.scale_boxes(rect.shape[:2], torch.from_numpy(boxes.copy()), shape)
            np.testing.assert_allclose(scale_boxes(boxes, rect.shape, shape), reference.numpy(), rtol=0, atol=1e-3)
        for agnostic in (True, False):
            raw = np.zeros((1, 4 + 3 + 32, 2000), np.float32)
            raw[0, :2] = generator.uniform(0, 640, (2, 2000))
            raw[0, 2:4] = generator.uniform(4, 120, (2, 2000))
            raw[0, 4:7] = generator.uniform(0, 1, (3, 2000)) ** 3
            raw[0, 7:] = generator.normal(size=(32, 2000))
            mine = decode_predictions(raw[0], 3, 0.05, 0.5, 100, agnostic=agnostic)
            theirs = non_max_suppression(torch.from_numpy(raw), 0.05, 0.5, agnostic=agnostic, max_det=100, nc=3)[0]
            np.testing.assert_allclose(mine, theirs[:, :6].numpy(), rtol=0, atol=1e-4)
            scaled = scale_boxes(mine[:, :4], (640, 640), (427, 640))
            reference = ops.scale_boxes((640, 640), torch.from_numpy(mine[:, :4].copy()), (427, 640))
            np.testing.assert_allclose(scaled, reference.numpy(), rtol=0, atol=1e-3)


class OnnxEngineTests(unittest.TestCase):
    def loaded_engine(self, output):
        engine = OnnxRuntimeEngine()
        engine.session = MagicMock()
        engine.session.run.return_value = [output]
        engine.manifest, engine.model, engine.imgsz = valid_manifest(), "m.onnx", 640
        engine.current_classes = ["bottle", "cup"]
        engine._input_name, engine._output_name = "images", "output0"
        return engine

    def test_detect_maps_raw_output_to_frame_detections(self):
        output = np.zeros((1, 4 + 2 + 32, 3), np.float32)
        output[0, :4, 0] = [150, 80 + 150, 100, 100]  # xywh in the 640 letterbox.
        output[0, 5, 0] = 0.8
        output[0, :4, 1] = [400, 400, 50, 50]
        output[0, 4, 1] = 0.3
        engine = self.loaded_engine(output)
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch("onnx_engine.preflight_inference") as preflight:
            detections = engine.detect(frame, conf=0.25, iou=0.5, class_names=["bottle", "cup"], max_det=10)
        preflight.assert_called_once_with("cpu", 640, frame.shape)
        self.assertEqual([(item.label, item.class_id, item.source) for item in detections],
                         [("cup", 1, "original"), ("bottle", 0, "original")])
        np.testing.assert_allclose(detections[0].xyxy, (100, 100, 200, 200))
        feed = engine.session.run.call_args.args
        self.assertEqual(feed[0], ["output0"])
        self.assertEqual(feed[1]["images"].shape, (1, 3, 640, 640))
        self.assertEqual(set(engine.last_timing), {"preprocess_ms", "inference_ms", "postprocess_ms"})
        with patch("onnx_engine.preflight_inference"):
            self.assertEqual(len(engine.detect(frame, conf=0.25, class_names=None, max_det=1)), 1)

    def test_fixed_vocabulary_options_and_unloaded_state(self):
        engine = self.loaded_engine(np.zeros((1, 38, 1), np.float32))
        frame = np.zeros((64, 64, 3), np.uint8)
        with patch("onnx_engine.preflight_inference"), self.assertRaisesRegex(ValueError, "固定"):
            engine.detect(frame, class_names=["person"])
        for options in [{"head_mode": "one2one"}, {"quantize": 16}, {"agnostic_nms": "yes"}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                engine.set_inference_options(**options)
        self.assertFalse(engine.set_inference_options(agnostic_nms=False)["agnostic_nms"])
        engine.unload()
        self.assertIsNone(engine.session)
        with self.assertRaisesRegex(RuntimeError, "尚未加载"):
            engine.detect(frame)

    def test_load_refuses_other_providers_and_bad_files_without_fallback(self):
        engine = OnnxRuntimeEngine()
        with patch("onnx_engine.prepare_device") as prepare:
            status = engine.load_model("m.onnx", "CUDA")
        prepare.assert_not_called()
        self.assertIn("不会自动改用 CPU", status)
        self.assertIsNone(engine.session)
        with tempfile.TemporaryDirectory() as folder, patch("onnx_engine.prepare_device", return_value="cpu"):
            path, _ = write_onnx_pair(folder)
            path.write_bytes(b"tampered")
            self.assertIn("哈希", engine.load_model(path, "CPU"))
            self.assertIsNone(engine.session)

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
                         "onnx/onnxruntime 未安装")
    def test_tiny_real_onnx_graph_through_runtime(self):
        import onnx
        from onnx import TensorProto, helper
        # Toy graph whose output is a constant raw prediction; the input only feeds a zero-weighted mean.
        constant = np.zeros((1, 6, 4), np.float32)
        constant[0, :4, 0] = [32, 32, 16, 16]
        constant[0, 4, 0] = 0.9
        node_a = helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0, axes=[1, 2, 3])
        node_b = helper.make_node("Mul", ["mean", "zero"], ["scaled"])
        node_c = helper.make_node("Add", ["base", "scaled"], ["output0"])
        graph = helper.make_graph(
            [node_a, node_b, node_c], "toy",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 4])],
            [helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
             helper.make_tensor("base", TensorProto.FLOAT, [1, 6, 4], constant.flatten().tolist())])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.onnx"
            onnx.save(model, str(path))
            manifest = valid_manifest(extra=0, digest=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest["export"]["imgsz"] = 64
            manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
            engine = OnnxRuntimeEngine()
            self.assertEqual(engine.load_model(path, "CPU", threads=1), "SUCCESS")
            self.assertEqual(engine.providers, ["CPUExecutionProvider"])
            frame = np.zeros((64, 64, 3), np.uint8)
            detections = engine.detect(frame, conf=0.25, iou=0.5, class_names=["bottle", "cup"], max_det=5)
            self.assertEqual(len(detections), 1)
            self.assertEqual((detections[0].label, detections[0].xyxy), ("bottle", (24.0, 24.0, 40.0, 40.0)))
            engine.unload()


    @unittest.skipUnless(importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
                         "onnx/onnxruntime 未安装")
    def test_dynamic_height_width_graph_receives_the_app_rectangle(self):
        import onnx
        from onnx import TensorProto, helper
        # Output anchors depend on the input width, so a square input would give a different shape.
        nodes = [helper.make_node("ReduceMean", ["images"], ["rows"], keepdims=1, axes=[1, 2]),  # 1x1x1xW
                 helper.make_node("Reshape", ["rows", "shape"], ["row"]),  # 1x1xW
                 helper.make_node("Mul", ["row", "zero"], ["zeros"]),
                 helper.make_node("Add", ["zeros", "base"], ["output0"])]  # broadcast to 1x6xW
        graph = helper.make_graph(
            nodes, "toy_dynamic",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, "height", "width"])],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, "anchors"])],
            [helper.make_tensor("shape", TensorProto.INT64, [3], [1, 1, -1]),
             helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
             helper.make_tensor("base", TensorProto.FLOAT, [1, 6, 1], [32, 32, 16, 16, 0.9, 0.0])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.onnx"
            onnx.save(model, str(path))
            manifest = valid_manifest(extra=0, digest=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest["export"].update(imgsz=128, dynamic_hw=True, stride=32)
            manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
            engine = OnnxRuntimeEngine()
            self.assertEqual(engine.load_model(path, "CPU", threads=1), "SUCCESS")
            run = engine.session.run
            with patch.object(engine.session, "run", side_effect=run) as spy:
                detections = engine.detect(np.zeros((64, 128, 3), np.uint8), conf=0.25, class_names=None)
            self.assertEqual(spy.call_args.args[1]["images"].shape, (1, 3, 64, 128))
            # All 128 anchors carry the same box; NMS keeps one in frame coordinates (no padding here).
            self.assertEqual([(item.label, item.xyxy) for item in detections], [("bottle", (24.0, 24.0, 40.0, 40.0))])
            engine.unload()
            manifest["export"]["dynamic_hw"] = False  # A dynamic graph cannot be described as square.
            manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("形状", engine.load_model(path, "CPU", threads=1))


class OnnxCatalogTests(unittest.TestCase):
    def test_engine_runtime_messages_have_english_entries(self):
        han = re.compile(r"[一-鿿]")
        tree = ast.parse((PROJECT / "onnx_engine.py").read_text(encoding="utf-8"))
        literals = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
                literals.append(node.exc.args[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "message":
                literals.extend(node.args[:1])
        checked = [value.value for value in literals if isinstance(value, ast.Constant)
                   and isinstance(value.value, str) and han.search(value.value)]
        self.assertGreater(len(checked), 10)
        for value in checked:
            with self.subTest(text=value):
                self.assertIn(value, EN)


if __name__ == "__main__":
    unittest.main()
