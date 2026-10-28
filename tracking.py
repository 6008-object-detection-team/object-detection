"""Class-aware ByteTrack integration for merged detector outputs."""
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ByteTrackManager:
    """Keep stable track IDs and expose compact per-class display numbers."""

    def __init__(self, track_buffer=30):
        # Ultralytics otherwise writes its settings into the user's roaming profile.
        config_dir = Path(__file__).resolve().parent / ".ultralytics"
        config_dir.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._boxes_class = Boxes
        self._base_track_class = BaseTrack
        self._tracker_class = BYTETracker
        self.track_buffer = int(track_buffer)
        self._class_signature = None
        self.reset()

    @staticmethod
    def detection_floor(confidence_threshold):
        """Low-score floor used only to reconnect an already established track."""
        threshold = float(np.clip(confidence_threshold, 0.01, 1.0))
        return max(0.01, min(0.10, threshold * 0.5))

    def reset(self):
        """Start a fresh numbering session."""
        self._trackers = {}
        # Class keys that appeared since the reset. A class can return after its
        # empty tracker was dropped; its new tracker must then behave like a
        # long-lived one (see update).
        self._seen_classes = set()
        self._display_ids = {}
        self._next_display_id = defaultdict(int)
        self._class_signature = None
        self._base_track_class.reset_id()

    def _new_tracker(self, confidence_threshold):
        threshold = float(np.clip(confidence_threshold, 0.01, 1.0))
        args = SimpleNamespace(
            track_high_thresh=threshold,
            track_low_thresh=self.detection_floor(threshold),
            new_track_thresh=threshold,
            track_buffer=self.track_buffer,
            match_thresh=0.8,
            fuse_score=True,
        )
        # BYTETracker resets Ultralytics' global counter in its constructor.
        # Preserve it because this manager intentionally uses one tracker per class.
        previous_count = self._base_track_class._count
        tracker = self._tracker_class(args)
        self._base_track_class._count = previous_count
        return tracker

    def _display_number(self, class_key, track_id):
        key = class_key, int(track_id)
        if key not in self._display_ids:
            self._next_display_id[class_key] += 1
            self._display_ids[key] = self._next_display_id[class_key]
        return self._display_ids[key]

    def update(self, detections, frame, confidence_threshold=0.4, class_names=None):
        """Assign stable IDs after original/tile/dark detections have been merged.

        Only classes that hold a track or can start one are updated. Prompt-free
        mode sees thousands of class keys over a moving camera; keeping an idle
        BYTETracker for each made every frame slower. A tracker is dropped once
        it holds no tracked or lost track (at most ``track_buffer`` + 1 frames
        after its class was last tracked), and none is created for a class
        whose detections are all below the threshold, as ByteTrack cannot start
        a track from them. The output IDs and boxes are unchanged.
        """
        signature = tuple(class_names or ())
        if self._class_signature is None:
            self._class_signature = signature
        elif signature != self._class_signature:
            self.reset()
            self._class_signature = signature

        grouped = defaultdict(list)
        for item in detections:
            grouped[(item.class_id, item.label)].append(item)

        threshold = float(np.clip(confidence_threshold, 0.01, 1.0))
        rows_by_class = {
            class_key: np.asarray(
                [[*item.xyxy, item.score, item.class_id] for item in items], dtype=np.float32
            ).reshape(-1, 6)
            for class_key, items in grouped.items()
        }
        # Compare the same float32 scores ByteTrack compares with new_track_thresh.
        startable = {class_key for class_key, rows in rows_by_class.items()
                     if (rows[:, 4] >= threshold).any()}
        tracked = []
        class_keys = sorted(set(self._trackers) | startable, key=lambda item: (item[0], item[1]))
        for class_key in class_keys:
            tracker = self._trackers.get(class_key)
            if tracker is None:
                tracker = self._new_tracker(confidence_threshold)
                if class_key in self._seen_classes:
                    # A tracker kept since the class first appeared would be past
                    # its first frame, where ByteTrack confirms new tracks at
                    # once; later new tracks wait for a second matching frame.
                    tracker.frame_id = 1
                self._trackers[class_key] = tracker
            else:
                tracker.args.track_high_thresh = threshold
                tracker.args.track_low_thresh = self.detection_floor(threshold)
                tracker.args.new_track_thresh = threshold

            items = grouped.get(class_key, [])
            rows = rows_by_class.get(class_key, np.empty((0, 6), dtype=np.float32))
            boxes = self._boxes_class(rows, frame.shape[:2])
            for result in tracker.update(boxes, frame):
                source_index = int(result[-1])
                if not 0 <= source_index < len(items):
                    continue
                item = items[source_index]
                track_id = int(result[4])
                tracked.append(replace(
                    item,
                    track_id=track_id,
                    display_id=self._display_number(class_key, track_id),
                ))
            if not tracker.tracked_stracks and not tracker.lost_stracks:
                # Nothing left to match: equivalent to a fresh tracker apart
                # from its frame count, which the branch above restores.
                del self._trackers[class_key]
        self._seen_classes.update(grouped)
        return tracked
