"""Realtime benchmark tool tests: fake clocks and a fake detector, no model or camera."""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from live_video import CapturedFrame, DisplayFrame, LatestFrameMailbox
from tools import benchmark_realtime as bench

PROJECT = Path(__file__).resolve().parent


class FakeClock:
    def __init__(self, start=100.0):
        self.now, self.sleeps = start, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds


def clip(count=4, shape=(6, 8, 3)):
    frames = [np.full(shape, index, np.uint8) for index in range(count)]
    for frame in frames:
        frame.flags.writeable = False
    return frames


class StatisticsTests(unittest.TestCase):
    def test_describe_uses_linear_percentiles_and_skips_missing_values(self):
        stats = bench.describe([5, 1, 4, 2, 3, None, float("nan")])
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 3)
        self.assertAlmostEqual(stats["std"], math.sqrt(2))
        self.assertEqual((stats["min"], stats["max"]), (1, 5))
        self.assertAlmostEqual(stats["p50"], 3)
        self.assertAlmostEqual(stats["p95"], 4.8)
        self.assertAlmostEqual(stats["p99"], 4.96)

    def test_describe_without_values_reports_nothing(self):
        self.assertEqual(bench.describe([None]), {"n": 0, "mean": None, "std": None, "min": None,
                                                  "p50": None, "p95": None, "p99": None, "max": None})

    def test_spread_reports_median_sample_std_and_range(self):
        result = bench.spread([60, 10, 20])
        self.assertEqual((result["n"], result["median"], result["min"], result["max"]), (3, 20, 10, 60))
        self.assertAlmostEqual(result["mean"], 30)
        self.assertAlmostEqual(result["std"], math.sqrt(700))
        self.assertEqual(bench.spread([4.0])["std"], 0.0)
        self.assertIsNone(bench.spread([None]))


class QueueSemanticsTests(unittest.TestCase):
    def test_latest_keeps_newest_but_fifo_keeps_every_frame_in_order(self):
        latest, fifo = LatestFrameMailbox(), bench.FifoFrameQueue()
        for value in range(10):
            latest.put(value)
            fifo.put(value)
        self.assertEqual(latest.take(), 9)
        self.assertEqual(latest.counters(), (10, 9))
        self.assertEqual(fifo.max_depth, 10)
        self.assertEqual([fifo.take() for _ in range(10)], list(range(10)))
        self.assertEqual(fifo.counters(), (10, 0))
        self.assertIsNone(fifo.take())

    def test_bounded_fifo_drops_incoming_frames_and_keeps_the_oldest(self):
        fifo = bench.FifoFrameQueue(capacity=3)
        notifications = [fifo.put(value) for value in range(10)]
        self.assertEqual(notifications, [True] + [False] * 9)
        self.assertEqual([fifo.take() for _ in range(3)], [0, 1, 2])
        self.assertEqual(fifo.counters(), (10, 7))
        self.assertTrue(fifo.put(10))

    def test_memory_guard_counts_owned_frames_but_not_replay_views(self):
        owned = CapturedFrame(np.zeros((10, 10, 3), np.uint8), 0.0)
        shared = CapturedFrame(bench.tag_frame(np.zeros((10, 10, 3), np.uint8), 0, 0.0, True), 0.0)
        fifo = bench.FifoFrameQueue(max_bytes=500)
        self.assertTrue(fifo.put(owned))
        self.assertFalse(fifo.put(owned))
        fifo.put(shared)
        self.assertEqual((fifo.depth(), fifo.memory_dropped, fifo.bytes_queued), (2, 1, 300))
        fifo.take()
        self.assertEqual(fifo.bytes_queued, 0)

    def test_close_wakes_waiter_and_still_drains_queued_frames(self):
        fifo = bench.FifoFrameQueue()
        result = []
        waiter = threading.Thread(target=lambda: result.append(fifo.take(wait=True)))
        waiter.start()
        time.sleep(0.02)
        fifo.close()
        waiter.join(1)
        self.assertEqual(result, [None])
        fifo = bench.FifoFrameQueue()
        fifo.put(1)
        fifo.put(2)
        fifo.close()
        self.assertFalse(fifo.put(3))
        self.assertEqual([fifo.take(wait=True) for _ in range(3)], [1, 2, None])

    def test_queue_depth_reads_both_queue_types(self):
        latest, fifo = LatestFrameMailbox(), bench.FifoFrameQueue()
        self.assertEqual((bench.queue_depth(latest), bench.queue_depth(fifo)), (0, 0))
        for value in range(3):
            latest.put(value)
            fifo.put(value)
        self.assertEqual((bench.queue_depth(latest), bench.queue_depth(fifo)), (1, 3))
        self.assertIsNone(bench.queue_depth(None))

    def test_fifo_reader_is_the_application_reader_with_a_fifo_queue(self):
        capture = MagicMock()
        capture.read.side_effect = [(True, "a"), (True, "b"), (False, None)]
        reader = bench.FifoCameraReader(capture)
        reader.start()
        reader.join(1)
        self.assertFalse(reader.is_alive())
        self.assertEqual([reader.mailbox.take().image for _ in range(2)], ["a", "b"])
        self.assertIn("读取摄像头", reader.error)
        capture.release.assert_called_once()


class PacedReplayTests(unittest.TestCase):
    def capture(self, **kwargs):
        clock = FakeClock()
        frames = clip()
        return bench.PacedReplayCapture(frames, 10.0, clock=clock, sleep=clock.sleep, **kwargs), clock, frames

    def test_prompt_reads_receive_every_frame_on_schedule_without_copies(self):
        capture, clock, frames = self.capture()
        ok, first = capture.read()
        self.assertTrue(ok)
        self.assertEqual((first.slot, first.capture_ts, clock.sleeps), (0, 100.0, []))
        ok, second = capture.read()
        self.assertEqual(second.slot, 1)
        self.assertAlmostEqual(second.capture_ts, 100.1)
        self.assertAlmostEqual(sum(clock.sleeps), 0.1)
        self.assertTrue(np.shares_memory(second, frames[1]))
        self.assertFalse(second.flags.writeable)
        self.assertTrue(second.shared)
        self.assertIsInstance(bench.untag(second), np.ndarray)
        self.assertNotIsInstance(bench.untag(second), bench.TaggedFrame)
        self.assertEqual(capture.dropped, 0)

    def test_late_read_with_one_buffer_drops_old_frames_and_waits_for_a_fresh_one(self):
        capture, clock, _ = self.capture()
        self.assertTrue(capture.set(bench.CAP_PROP_BUFFERSIZE, 1))
        self.assertEqual(capture.get(bench.CAP_PROP_BUFFERSIZE), 1)
        capture.read()
        clock.advance(0.35)  # slots 1-3 arrive while the consumer holds slot 0
        ok, frame = capture.read()
        self.assertEqual((frame.slot, capture.dropped), (4, 3))
        self.assertAlmostEqual(clock.now, 100.4)

    def test_late_read_with_four_buffers_returns_the_oldest_and_drops_newer_frames(self):
        capture, clock, _ = self.capture(buffers=4, honour_buffersize=False)
        self.assertFalse(capture.set(bench.CAP_PROP_BUFFERSIZE, 1))
        capture.read()
        clock.advance(0.55)  # slots 1-5 arrive; three free buffers keep 1-3
        slots = [capture.read()[1].slot for _ in range(4)]
        self.assertEqual(slots, [1, 2, 3, 6])
        self.assertEqual(capture.dropped, 2)
        self.assertEqual(capture.counters(clock.now)["frames_offered"], 7)

    def test_duration_ends_the_stream(self):
        capture, clock, _ = self.capture(duration=0.25)
        slots = [capture.read()[1].slot for _ in range(3)]
        self.assertEqual(slots, [0, 1, 2])
        self.assertEqual(capture.read(), (False, None))
        self.assertEqual(capture.counters(clock.now + 5)["frames_offered"], 3)
        capture.release()
        self.assertFalse(capture.isOpened())

    def test_property_constants_match_opencv(self):
        import cv2
        self.assertEqual((bench.CAP_PROP_POS_MSEC, bench.CAP_PROP_FRAME_WIDTH, bench.CAP_PROP_FRAME_HEIGHT,
                          bench.CAP_PROP_FPS, bench.CAP_PROP_FOURCC, bench.CAP_PROP_BUFFERSIZE),
                         (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                          cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_BUFFERSIZE))
        capture, _, _ = self.capture()
        self.assertEqual((capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                          capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), (10.0, 8.0, 6.0))


class CameraTimestampTests(unittest.TestCase):
    def camera(self, stamp_ms, ok=True):
        inner = MagicMock()
        inner.read.return_value = (ok, np.zeros((2, 2, 3), np.uint8) if ok else None)
        inner.get.return_value = stamp_ms
        return bench.TimedCameraCapture(inner, clock=lambda: 20.0, monotonic=lambda: 5.0), inner

    def test_plausible_driver_timestamp_becomes_the_capture_time(self):
        capture, _ = self.camera(4950.0)
        ok, frame = capture.read()
        self.assertTrue(ok)
        self.assertAlmostEqual(frame.capture_ts, 19.95)
        self.assertEqual((frame.slot, frame.shared, capture.driver_timestamps), (0, False, 1))
        counters = capture.counters(20.0)
        self.assertAlmostEqual(counters["read_minus_driver_timestamp_ms"]["p50"], 50.0)
        self.assertIsNone(counters["delivered_fps"])

    def test_delivered_rate_uses_capture_times(self):
        inner, times = MagicMock(), iter([1.0, 1.1, 1.2, 1.3])
        inner.read.return_value = (True, np.zeros((2, 2, 3), np.uint8))
        inner.get.return_value = 0.0
        capture = bench.TimedCameraCapture(inner, clock=lambda: next(times), monotonic=lambda: 5.0)
        for _ in range(4):
            capture.read()
        self.assertAlmostEqual(capture.counters(2.0)["delivered_fps"], 10.0)
        self.assertEqual(bench.fourcc_code("MJPG"), 0x47504A4D)

    def test_missing_or_future_timestamp_falls_back_to_read_return(self):
        for stamp in (0.0, 5100.0, 1000.0):
            capture, _ = self.camera(stamp)
            ok, frame = capture.read()
            self.assertIsNone(frame.capture_ts)
            self.assertEqual(capture.fallback_timestamps, 1)

    def test_buffer_override_replaces_only_the_buffersize_request(self):
        capture, inner = self.camera(4950.0)
        capture.buffers_override = 4
        bench.configure_camera_like_app(capture)
        self.assertEqual([call.args for call in inner.set.call_args_list],
                         [(bench.CAP_PROP_BUFFERSIZE, 4), (bench.CAP_PROP_FRAME_WIDTH, 1280),
                          (bench.CAP_PROP_FRAME_HEIGHT, 720)])
        self.assertEqual(capture.counters(0.0)["property_requests"][0], [bench.CAP_PROP_BUFFERSIZE, 1.0])

    def test_failed_read_and_release_pass_through(self):
        capture, inner = self.camera(4950.0, ok=False)
        self.assertEqual(capture.read(), (False, None))
        capture.release()
        inner.release.assert_called_once()


class PreReworkLoopTests(unittest.TestCase):
    def test_port_reads_infers_publishes_and_waits_like_the_old_loop(self):
        worker = MagicMock()
        worker.running = True
        frames = [np.full((4, 4, 3), value, np.uint8) for value in (1, 2)]
        capture = MagicMock()
        capture.read.side_effect = [(True, frames[0]), (True, frames[1]), (False, None)]
        worker._infer.side_effect = lambda frame: (frame, [], "ok")
        clock, sleeps = FakeClock(), []
        self.assertEqual(bench.run_pre_rework_loop(worker, capture, clock=clock, sleep=sleeps.append), 2)
        packets = [call.args[0] for call in worker._publish_live_frame.call_args_list]
        self.assertEqual([packet.captured for packet in packets], [1, 2])
        self.assertTrue(all(isinstance(packet, DisplayFrame) for packet in packets))
        self.assertEqual(sleeps, [0.005, 0.005])
        self.assertEqual([call.args for call in capture.set.call_args_list],
                         [(bench.CAP_PROP_BUFFERSIZE, 1), (bench.CAP_PROP_FRAME_WIDTH, 1280),
                          (bench.CAP_PROP_FRAME_HEIGHT, 720)])
        capture.release.assert_called_once()


class WorkerScheduleTests(unittest.TestCase):
    """The application's VideoWorker, CameraReader and mailboxes with a fake 30 ms detector."""

    @classmethod
    def setUpClass(cls):
        import feedback  # noqa: F401  keep the worker's first-frame import out of the timings

    def run_mode(self, mode, service=0.03, fps=100.0, duration=0.6, **options):
        threads, active, overlap = set(), [0], []

        def process_frame(frame, *args, **kwargs):
            active[0] += 1
            overlap.append(active[0])
            threads.add(threading.get_ident())
            time.sleep(service)
            active[0] -= 1
            return frame.copy(), [], "ok"

        engine = MagicMock()
        engine.process_frame.side_effect = process_frame
        capture = bench.PacedReplayCapture(clip(8, (16, 16, 3)), fps, duration=duration + 1.0)
        settings = bench.RunSettings(classes=["cup"], settle_s=0.0, **options)
        with patch("video_worker.ByteTrackManager", return_value=MagicMock()):
            raw = bench.run_once(engine, settings, mode, capture, duration)
        meta = {"session_id": "s", "run_id": f"s_{mode}", "source": "video", "device": "cpu",
                "mode": mode, "repeat": 1, "execution_index": 0}
        records = bench.frame_records(raw, meta, 0.0)
        self.assertEqual(max(overlap), 1, "detector calls must stay serial")
        return raw, records, bench.summarize_run(raw, records, meta), threads, capture

    def test_latest_skips_stale_frames_and_keeps_latency_bounded(self):
        raw, records, summary, _, _ = self.run_mode("latest")
        slots = [row["source_slot"] for row in records]
        self.assertEqual(summary["status"], "complete", summary["errors"])
        self.assertGreaterEqual(len(records), 5)
        self.assertTrue(all(later > earlier for earlier, later in zip(slots, slots[1:])))
        self.assertTrue(any(later - earlier > 1 for earlier, later in zip(slots, slots[1:])))
        self.assertGreater(summary["queue_dropped"], 0)
        self.assertLessEqual(summary["backlog"]["max"], 1)
        self.assertLess(summary["latency_ms"]["e2e_ms"]["max"], 250)
        self.assertGreater(summary["displayed"], 0)
        self.assertIsNotNone(summary["display_metrics_last"])
        self.assertTrue(all(row["timestamp_basis"] == "scheduled_capture" for row in records))

    def test_fifo_processes_every_frame_in_order_and_latency_grows(self):
        _, records, summary, _, _ = self.run_mode("fifo")
        slots = [row["source_slot"] for row in records]
        self.assertEqual(summary["status"], "complete", summary["errors"])
        self.assertEqual(slots, list(range(len(slots))))
        self.assertEqual(summary["queue_dropped"], 0)
        self.assertGreater(records[-1]["backlog_after_result"], 3)
        self.assertGreater(records[-1]["e2e_ms"], records[0]["e2e_ms"] + 100)
        self.assertGreater(summary["unprocessed_at_stop"], 3)

    def test_bounded_fifo_drops_frames_when_full(self):
        _, records, summary, _, _ = self.run_mode("fifo", fifo_depth=2)
        self.assertGreater(summary["queue_dropped"], 0)
        self.assertLessEqual(summary["backlog"]["max"], 2)

    def test_sync_runs_on_one_thread_and_honours_the_buffer_request(self):
        _, records, summary, threads, capture = self.run_mode("sync")
        slots = [row["source_slot"] for row in records]
        self.assertEqual(summary["status"], "complete", summary["errors"])
        self.assertEqual(threads, {threading.get_ident()})
        self.assertEqual(capture.buffers, 1)
        self.assertGreater(summary["source_dropped"], 0)
        self.assertTrue(all(later > earlier for earlier, later in zip(slots, slots[1:])))
        self.assertIsNone(summary["backlog"]["max"])
        self.assertLess(summary["latency_ms"]["e2e_ms"]["max"], 250)


class OutputTests(unittest.TestCase):
    def raw(self):
        rows = []
        for index in range(4):
            capture = None if index == 3 else 10.0 + index * 0.1
            read = 10.0 + index * 0.1 + 0.002
            rows.append({"source_slot": index, "capture_ts": capture, "read_at": read,
                         "inference_started": read + 0.001, "inference_finished": read + 0.021,
                         "result_ready": read + 0.025, "displayed": read + 0.026 if index else None,
                         "backlog": 0, "frames_read_total": index + 1, "queue_dropped_total": 0,
                         "source_dropped_total": 0, "boxes": 2})
        return {"rows": rows, "origin": 10.0, "started": 9.99, "stopped": 10.5, "errors": [],
                "frames_read": 4, "queue_dropped": 1, "unprocessed_at_stop": 0, "fifo_max_depth": None,
                "fifo_memory_dropped": None, "display_dropped": 0, "displayed": 3, "display_metrics": None,
                "source": {"frames_offered": 5, "source_dropped": 0, "frames_delivered": 5},
                "load_before": [0.5, 0.4, 0.3], "load_after": [0.6, 0.4, 0.3], "process_cpu_s": 0.2}

    meta = {"session_id": "s", "run_id": "s_r1_latest", "source": "video", "device": "cpu",
            "mode": "latest", "repeat": 1, "execution_index": 0}

    def test_records_summary_and_csv_schema(self):
        raw = self.raw()
        records = bench.frame_records(raw, self.meta, settle_s=0.05)
        self.assertEqual([row["settled"] for row in records], [False, True, True, True])
        self.assertAlmostEqual(records[1]["e2e_ms"], 27.0)
        self.assertAlmostEqual(records[1]["queue_ms"], 1.0)
        self.assertAlmostEqual(records[1]["inference_ms"], 20.0)
        self.assertAlmostEqual(records[1]["capture_to_read_ms"], 2.0)
        self.assertEqual(records[3]["timestamp_basis"], "read_return")
        self.assertAlmostEqual(records[3]["e2e_ms"], 25.0)
        summary = bench.summarize_run(raw, records, self.meta)
        self.assertEqual((summary["frames_processed"], summary["frames_measured"]), (4, 3))
        self.assertAlmostEqual(summary["output_fps"], 10.0)
        self.assertAlmostEqual(summary["processed_fraction"], 0.8)
        self.assertAlmostEqual(summary["source_fps"], 10.0)
        self.assertEqual(summary["timestamp_basis"], "read_return+scheduled_capture")
        with tempfile.TemporaryDirectory() as folder:
            frames_path, runs_path = Path(folder) / "frames.csv", Path(folder) / "runs.csv"
            bench.append_csv(frames_path, bench.FRAME_FIELDS, records[:2])
            bench.append_csv(frames_path, bench.FRAME_FIELDS, records[2:])
            bench.append_csv(runs_path, bench.RUN_FIELDS, [bench.run_row(summary)])
            with frames_path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), list(bench.FRAME_FIELDS))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["displayed_s"], "")
            self.assertEqual(rows[3]["capture_s"], "")
            self.assertAlmostEqual(float(rows[1]["e2e_ms"]), 27.0)
            with runs_path.open(encoding="utf-8") as stream:
                run = next(csv.DictReader(stream))
            self.assertEqual(list(run), list(bench.RUN_FIELDS))
            self.assertAlmostEqual(float(run["e2e_p50_ms"]), 27.0)

    def test_aggregate_json_and_markdown_table(self):
        raw = self.raw()
        runs, records_by_run = [], {}
        for repeat, scale in enumerate((1.0, 2.0, 3.0), start=1):
            meta = dict(self.meta, run_id=f"s_r{repeat}_latest", repeat=repeat)
            for row in raw["rows"]:
                row["result_ready"] = row["read_at"] + 0.025 * scale
            records = bench.frame_records(raw, meta, 0.0)
            records_by_run[meta["run_id"]] = records
            runs.append(bench.summarize_run(raw, records, meta))
        groups = bench.aggregate(runs, records_by_run)
        self.assertEqual(len(groups), 1)
        p50 = groups[0]["metrics"]["e2e_p50_ms"]
        self.assertEqual(p50["n"], 3)
        self.assertAlmostEqual(p50["median"], 52.0)
        self.assertLess(p50["min"], p50["median"])
        self.assertEqual(groups[0]["pooled_e2e_ms"]["n"], 12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            bench.atomic_json(path, {"runs": runs, "aggregates": groups, "bad": float("nan")})
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(data["bad"])
        table = bench.markdown_table([data])
        self.assertIn("| video | cpu | latest | 3 |", table)


class CommandLineTests(unittest.TestCase):
    def parse_error(self, argv):
        with redirect_stderr(StringIO()) as stream, self.assertRaises(SystemExit) as raised:
            bench.main(argv)
        self.assertEqual(raised.exception.code, 2)
        return stream.getvalue()

    def test_camera_runs_are_capped_and_driver_buffers_are_replay_only(self):
        self.assertIn("60", self.parse_error(["--source", "camera", "--duration", "61"]))
        self.parse_error(["--source", "camera", "--driver-buffers", "4"])
        self.parse_error(["--source", "video", "--camera-fourcc", "MJPG"])
        self.parse_error(["--source", "camera", "--camera-fourcc", "MJPEG"])
        self.parse_error(["--source", "video", "--camera-buffers", "4"])
        self.parse_error(["--source", "video", "--duration", "5", "--settle", "5"])
        self.parse_error(["--modes", "latest"])

    def test_device_must_be_explicit(self):
        self.assertIn("--device", self.parse_error(["--source", "video"]))
        self.parse_error(["--source", "video", "--device", "auto"])
        with self.assertRaises(ValueError):
            bench.load_engine(Path("unused.pt"), "auto", 640)

    def test_mode_parsing_and_rotated_schedule(self):
        self.assertEqual(bench.parse_modes("latest,fifo"), ["latest", "fifo"])
        for value in ("latest,latest", "latest,queue", ""):
            with self.assertRaises(Exception):
                bench.parse_modes(value)
        self.assertEqual(bench.schedule(["a", "b", "c"], 3), [
            (0, "a"), (0, "b"), (0, "c"), (1, "b"), (1, "c"), (1, "a"), (2, "c"), (2, "a"), (2, "b")])
        self.assertEqual(bench.parse_frame_size("640x480"), (640, 480))

    def test_import_loads_neither_torch_nor_qt(self):
        code = ("import sys, tools.benchmark_realtime; "
                "print(any(name in sys.modules for name in ('torch', 'PySide6', 'cv2', 'ultralytics')))")
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, capture_output=True, text=True,
                                timeout=60, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_tool_has_no_frame_writing_or_display_calls(self):
        source = Path(bench.__file__).read_text(encoding="utf-8")
        for token in ("imwrite", "VideoWriter", "imshow", "imencode", "recording_path ="):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
