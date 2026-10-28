"""ByteTrackManager keeps only live per-class trackers without changing any ID.

CPU only, synthetic detections, no model. The reference below is the previous
unbounded update (one BYTETracker per class key ever seen, updated on every
frame); the bounded manager must reproduce its boxes and IDs exactly.
"""
import os
import unittest
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
import numpy as np

from detection_utils import Detection
from tracking import ByteTrackManager


class UnboundedReference(ByteTrackManager):
    """The update before trackers were bounded, kept verbatim as an oracle."""

    def update(self, detections, frame, confidence_threshold=0.4, class_names=None):
        signature = tuple(class_names or ())
        if self._class_signature is None:
            self._class_signature = signature
        elif signature != self._class_signature:
            self.reset()
            self._class_signature = signature
        grouped = defaultdict(list)
        for item in detections:
            grouped[(item.class_id, item.label)].append(item)
        tracked = []
        class_keys = sorted(set(self._trackers) | set(grouped), key=lambda item: (item[0], item[1]))
        for class_key in class_keys:
            tracker = self._trackers.get(class_key)
            if tracker is None:
                tracker = self._new_tracker(confidence_threshold)
                self._trackers[class_key] = tracker
            else:
                threshold = float(np.clip(confidence_threshold, 0.01, 1.0))
                tracker.args.track_high_thresh = threshold
                tracker.args.track_low_thresh = self.detection_floor(threshold)
                tracker.args.new_track_thresh = threshold
            items = grouped.get(class_key, [])
            rows = np.asarray(
                [[*item.xyxy, item.score, item.class_id] for item in items], dtype=np.float32
            ).reshape(-1, 6)
            for result in tracker.update(self._boxes_class(rows, frame.shape[:2]), frame):
                source_index = int(result[-1])
                if not 0 <= source_index < len(items):
                    continue
                track_id = int(result[4])
                tracked.append(replace(items[source_index], track_id=track_id,
                                       display_id=self._display_number(class_key, track_id)))
        return tracked


def synthetic_sequence(seed, frames=240, classes=60):
    """Moving objects of many classes that appear in bursts, with gaps longer
    than the track buffer, low-score-only classes and threshold changes."""
    rng = np.random.default_rng(seed)
    objects = []
    for index in range(classes * 2):
        class_id = int(rng.integers(0, classes))
        start = int(rng.integers(0, frames))
        objects.append({
            "class_id": class_id,
            "label": f"class-{class_id}",
            "bursts": [(begin, begin + int(rng.integers(1, 25)))
                       for begin in sorted(int(value) for value in rng.integers(start, frames + 30, 3))],
            "xy": rng.uniform(0, 500, 2),
            "velocity": rng.normal(0, 4, 2),
            "size": rng.uniform(12, 90, 2),
            # Some classes only ever produce candidates below the threshold.
            "score": (0.05, 0.3) if index % 7 == 0 else (0.15, 0.95),
        })
    sequence = []
    for frame_index in range(frames):
        conf = 0.4 if frame_index < 100 else 0.3 if frame_index < 170 else 0.5
        detections = []
        for item in objects:
            if not any(begin <= frame_index < end for begin, end in item["bursts"]):
                continue
            if rng.random() < 0.1:
                continue  # a missed detection inside a burst
            x, y = item["xy"] + item["velocity"] * frame_index
            w, h = item["size"]
            score = float(np.float32(rng.uniform(*item["score"])))
            detections.append(Detection((float(x), float(y), float(x + w), float(y + h)), score,
                                        item["class_id"], item["label"], "original"))
        sequence.append((detections, conf))
    return sequence


def replay(manager, sequence, frame, class_names=()):
    outputs, tracker_counts = [], []
    for detections, conf in sequence:
        result = manager.update(detections, frame, conf, list(class_names))
        outputs.append(sorted((item.class_id, item.label, item.xyxy, item.score,
                               item.track_id, item.display_id) for item in result))
        tracker_counts.append(len(manager._trackers))
    return outputs, tracker_counts


class BoundedTrackerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), np.uint8)

    def test_bounded_trackers_reproduce_unbounded_boxes_and_ids(self):
        for seed, buffer in ((0, 5), (1, 30), (2, 3)):
            with self.subTest(seed=seed, track_buffer=buffer):
                sequence = synthetic_sequence(seed)
                # Sequential runs: both managers reset the global track counter.
                expected, unbounded = replay(UnboundedReference(track_buffer=buffer), sequence, self.frame)
                actual, bounded = replay(ByteTrackManager(track_buffer=buffer), sequence, self.frame)
                self.assertGreater(sum(map(len, expected)), 100)
                self.assertEqual(actual, expected)
                self.assertLess(max(bounded), max(unbounded))
                self.assertLess(bounded[-1], unbounded[-1])

    def test_idle_class_trackers_are_dropped_after_the_track_buffer(self):
        manager = ByteTrackManager(track_buffer=5)
        cups = [Detection((10 + 2 * index, 10, 60 + 2 * index, 60), 0.9, 41, "cup") for index in range(3)]
        for cup in cups:
            manager.update([cup], self.frame, 0.4, [])
        self.assertEqual(list(manager._trackers), [(41, "cup")])
        for _ in range(6):
            manager.update([], self.frame, 0.4, [])
        self.assertEqual(list(manager._trackers), [(41, "cup")])  # still lost, can reconnect
        manager.update([], self.frame, 0.4, [])
        manager.update([], self.frame, 0.4, [])
        self.assertEqual(manager._trackers, {})

    def test_low_score_only_class_creates_no_tracker(self):
        manager = ByteTrackManager()
        low = Detection((10, 10, 60, 60), 0.2, 7, "bag")
        self.assertEqual(manager.update([low], self.frame, 0.4, []), [])
        self.assertEqual(manager._trackers, {})

    def test_returning_class_still_needs_a_second_frame_like_before(self):
        cup = Detection((100, 100, 160, 180), 0.9, 41, "cup")
        script = [[cup]] + [[]] * 12 + [[cup], [cup]]
        results = []
        for manager in (UnboundedReference(track_buffer=5), ByteTrackManager(track_buffer=5)):
            results.append([[item.display_label for item in manager.update(frame_detections, self.frame, 0.4, [])]
                            for frame_detections in script])
        self.assertEqual(results[1], results[0])
        # First appearance is confirmed at once; a returning cup waits one frame.
        self.assertEqual(results[1][0], ["cup1"])
        self.assertEqual(results[1][13], [])
        self.assertEqual(results[1][14], ["cup2"])

    def test_reset_forgets_seen_classes(self):
        cup = Detection((100, 100, 160, 180), 0.9, 41, "cup")
        manager = ByteTrackManager(track_buffer=2)
        manager.update([cup], self.frame, 0.4, ["cup"])
        for _ in range(4):
            manager.update([], self.frame, 0.4, ["cup"])
        self.assertEqual(manager._trackers, {})
        # A new prompt starts a fresh session, where a first sighting is confirmed at once.
        restarted = manager.update([cup], self.frame, 0.4, ["cup", "bottle"])
        self.assertEqual([item.display_label for item in restarted], ["cup1"])
        self.assertEqual(manager._seen_classes, {(41, "cup")})


if __name__ == "__main__":
    unittest.main()
