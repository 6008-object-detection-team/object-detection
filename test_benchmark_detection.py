"""Control-variable, input reuse, timer and report tests; no GPU/model download."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from benchmark_detection import (
    InputFrame, SweepConfig, build_configurations, build_parser,
    load_inputs, measure, sha256_file, summarize, write_report,
)


class DummyFrame:
    shape = (2, 3, 3)
    size = 18

    def __init__(self, value=0):
        self.value = value

    def tobytes(self):
        return bytes([self.value] * self.size)


def arguments(*extra):
    return build_parser().parse_args([
        "--engine", "yoloe", "--model", "fixture.pt", "--source", "fixture.avi",
        "--classes", "person,bottle", *extra,
    ])


class SweepTests(unittest.TestCase):
    def test_both_modes_are_independent_and_not_cartesian(self):
        configs = build_configurations(arguments(
            "--conf-values", "0.1,0.6", "--max-det-values", "10,100",
            "--fixed-conf", "0.35", "--fixed-max-det", "200",
        ))
        self.assertEqual(configs, [
            SweepConfig("confidence", 0.1, 200), SweepConfig("confidence", 0.6, 200),
            SweepConfig("max-det", 0.35, 10), SweepConfig("max-det", 0.35, 100),
        ])

    def test_single_mode_only_has_its_own_configs(self):
        for mode in ("confidence", "max-det"):
            configs = build_configurations(arguments("--mode", mode))
            self.assertTrue(all(item.sweep == mode for item in configs))

    def test_invalid_thresholds_frame_indices_limits_and_nonenglish_names_fail(self):
        cases = [("--fixed-conf", "nan"), ("--iou", "1.01"), ("--max-det-values", "0,10"),
                 ("--frames", "-1"), ("--frames", "0,0"), ("--classes", "person,,bottle"),
                 ("--classes", "人,杯子"), ("--warmup", "0"), ("--repeat", "0")]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(flag, value)


class TimingTests(unittest.TestCase):
    def test_preheated_synchronized_detect_only_and_identical_frame_reused(self):
        args = arguments("--warmup", "2", "--repeat", "2", "--save-detections")
        frame = DummyFrame()
        input_frame = InputFrame("input_0000", frame, {})
        detection = types.SimpleNamespace(xyxy=(0, 0, 2, 2), score=0.7,
                                          class_id=0, label="person", source="original")
        engine = Mock()
        engine.detect.return_value = [detection]
        events = []
        engine.detect.side_effect = lambda *args, **kwargs: (events.append("detect") or [detection])
        clock = Mock(side_effect=[1.0, 1.01, 2.0, 2.02])
        def synchronize():
            events.append("sync")
        with contextlib.redirect_stdout(io.StringIO()):
            results = measure(engine, [input_frame], [SweepConfig("confidence", 0.2, 55)],
                              args, synchronize, clock=clock)
        self.assertEqual(engine.detect.call_count, 4)
        for call in engine.detect.call_args_list:
            self.assertIs(call.args[0], frame)
            self.assertEqual(call.kwargs, {"conf": 0.2, "iou": 0.45, "class_names": ["person", "bottle"],
                                           "source": "original", "max_det": 55})
        self.assertEqual(events, ["detect", "detect", "sync", "sync", "detect", "sync", "sync", "detect", "sync"])
        self.assertEqual(clock.call_count, 4)
        self.assertAlmostEqual(results["samples"][0]["latency_ms"], 10)
        self.assertAlmostEqual(results["samples"][1]["latency_ms"], 20)
        self.assertAlmostEqual(results["summary"][-1]["mean_ms"], 15)
        self.assertEqual(results["samples"][0]["counts_by_class"], {"person": 1})
        self.assertEqual(len(results["first_predictions"]), 1)

    def test_shuffled_order_is_reproducible_and_all_settings_visit_all_frames(self):
        args = arguments("--warmup", "1", "--repeat", "1", "--seed", "17")
        frames = [InputFrame(f"input_{index}", DummyFrame(index), {}) for index in range(2)]
        configs = build_configurations(args)
        engine = Mock()
        engine.detect.return_value = []
        with contextlib.redirect_stdout(io.StringIO()):
            first = measure(engine, frames, configs, args, lambda: None)
            second = measure(engine, frames, configs, args, lambda: None)
        self.assertEqual(first["execution_order"], second["execution_order"])
        self.assertEqual(len(first["samples"]), len(configs) * len(frames))
        pairs = {(row["config_id"], row["input_id"]) for row in first["samples"]}
        self.assertEqual(len(pairs), len(configs) * len(frames))
        self.assertTrue(all(row["box_count"] == 0 and row["mean_confidence"] is None for row in first["samples"]))

    def test_summary_p95_and_count_stats_use_measured_calls(self):
        result = summarize([{"latency_ms": 10, "box_count": 0}, {"latency_ms": 20, "box_count": 4}])
        self.assertEqual(result["median_ms"], 15)
        self.assertAlmostEqual(result["p95_ms"], 19.5)
        self.assertEqual(result["mean_box_count"], 2)
        self.assertEqual(result["std_ms"], 5)
        self.assertAlmostEqual(result["detection_calls_per_second"], 1000 / 15)


class InputTests(unittest.TestCase):
    def test_real_image_unicode_path_and_directory_use_same_pixels(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("cv2/numpy 尚未安装；纯逻辑测试仍可运行")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((4, 6, 3), (20, 40, 80), dtype=np.uint8)
            encoded, content = cv2.imencode(".png", image)
            self.assertTrue(encoded)
            source = root / "场景.png"
            source.write_bytes(content.tobytes())
            single, _ = load_inputs(source, "auto", [0])
            folder, _ = load_inputs(root, "auto", [0])
            np.testing.assert_array_equal(single[0].image, image)
            self.assertEqual(single[0].metadata["width"], 6)
            self.assertEqual(single[0].metadata["height"], 4)
            self.assertEqual(single[0].metadata["frame_index"], None)
            self.assertEqual(single[0].metadata["decoded_bgr_sha256"], folder[0].metadata["decoded_bgr_sha256"])

    def test_sequential_video_selects_exact_frames_and_releases_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.avi"
            source.write_bytes(b"fixture video")
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.side_effect = [(True, DummyFrame(index)) for index in range(4)]
            cv2 = types.SimpleNamespace(VideoCapture=Mock(return_value=capture))
            with patch.dict("sys.modules", {"cv2": cv2, "numpy": types.SimpleNamespace()}):
                frames, metadata = load_inputs(source, "video", [3, 1])
            self.assertEqual([frame.metadata["frame_index"] for frame in frames], [1, 3])
            self.assertEqual([frame.image.value for frame in frames], [1, 3])
            self.assertEqual(capture.read.call_count, 4)
            capture.release.assert_called_once()
            self.assertEqual(metadata[0]["sha256"], hashlib.sha256(b"fixture video").hexdigest())
            self.assertEqual(frames[0].metadata["decoded_bgr_sha256"], hashlib.sha256(bytes([1] * 18)).hexdigest())

    def test_missing_requested_frame_fails_instead_of_changing_inputs_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "short.avi"
            source.write_bytes(b"fixture")
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.side_effect = [(True, DummyFrame()), (False, None)]
            cv2 = types.SimpleNamespace(VideoCapture=Mock(return_value=capture))
            with patch.dict("sys.modules", {"cv2": cv2, "numpy": types.SimpleNamespace()}):
                with self.assertRaisesRegex(ValueError, "无法取得要求的帧"):
                    load_inputs(source, "video", [0, 2])
            capture.release.assert_called_once()


class ReportTests(unittest.TestCase):
    def test_csv_settings_match_json_and_existing_directory_not_overwritten(self):
        args = arguments("--warmup", "1", "--repeat", "1")
        engine = Mock()
        engine.detect.return_value = []
        frame = InputFrame("input_0000", DummyFrame(), {})
        with contextlib.redirect_stdout(io.StringIO()):
            measured = measure(engine, [frame], [SweepConfig("max-det", 0.4, 10)], args,
                               lambda: None, clock=Mock(side_effect=[0, 0.01]))
        settings = vars(args).copy()
        settings.update(source="fixture.avi", output="unused", text_threshold=None)
        report = {"run_id": "fixture", "settings": settings,
                  "device": {"actual": "cpu", "cuda_synchronization": False},
                  "inputs": [{"input_id": "input_0000", "path": "fixture.avi", "kind": "video",
                              "frame_index": 0, "width": 3, "height": 2, "decoded_bgr_sha256": "fixture_sha"}],
                  **measured}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            write_report(output, report)
            loaded = json.loads((output / "results.json").read_text(encoding="utf-8"))
            with (output / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = list(csv.DictReader(stream))[0]
            self.assertEqual(row["model"], loaded["settings"]["model"])
            self.assertEqual(json.loads(row["classes_json"]), loaded["settings"]["classes"])
            self.assertEqual(row["max_det"], "10")
            self.assertEqual(row["confidence"], "0.4")
            self.assertEqual(row["decoded_bgr_sha256"], "fixture_sha")
            saved_hash = sha256_file(output / "results.json")
            with self.assertRaises(FileExistsError):
                write_report(output, report)
            self.assertEqual(saved_hash, sha256_file(output / "results.json"))


if __name__ == "__main__":
    unittest.main()
