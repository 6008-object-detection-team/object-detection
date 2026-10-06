"""Read at most two Linux V4L2 camera nodes without acquiring frames.

This never sets capture properties, reads/saves frames, imports torch, or runs
a detector. Property values alone cannot establish hardware control support.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def device_key(path):
    match = re.fullmatch(r"video(\d+)", Path(path).name)
    return int(match.group(1)) if match else 1_000_000


def camera_name(path):
    name_path = Path("/sys/class/video4linux") / Path(path).name / "name"
    try:
        return name_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def atomic_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def probe(max_devices=2):
    import cv2
    import math

    started = time.perf_counter()
    devices = sorted(glob.glob("/dev/video*"), key=device_key)
    devices = [path for path in devices if re.fullmatch(r"video\d+", Path(path).name)]
    records = []
    properties = {"width": cv2.CAP_PROP_FRAME_WIDTH, "height": cv2.CAP_PROP_FRAME_HEIGHT,
                  "fps": cv2.CAP_PROP_FPS, "focus": cv2.CAP_PROP_FOCUS,
                  "autofocus": cv2.CAP_PROP_AUTOFOCUS, "exposure": cv2.CAP_PROP_EXPOSURE,
                  "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE}
    for path in devices[:min(max_devices, 2)]:
        record = {"device": path, "model_name": camera_name(path), "opened": False,
                  "backend": "V4L2", "properties": None, "error": None}
        capture = None
        try:
            capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
            record["opened"] = bool(capture.isOpened())
            if record["opened"]:
                record["backend"] = capture.getBackendName()
                values = {name: float(capture.get(key)) for name, key in properties.items()}
                record["properties"] = {name: value if math.isfinite(value) else None
                                         for name, value in values.items()}
            else:
                record["error"] = "Could not open this video node; it may be a metadata-only node, busy, or inaccessible."
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if capture is not None:
                capture.release()
        records.append(record)
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started, "opencv_version": cv2.__version__,
            "cpu_only": True, "torch_imported": "torch" in sys.modules,
            "frames_acquired": 0, "frames_saved": 0, "properties_set": 0,
            "maximum_nodes_opened": min(max_devices, 2), "enumerated_devices": devices,
            "unprobed_devices": devices[min(max_devices, 2):], "cameras": records,
            "status": "read_only_camera_capability_probe",
            "limitations": [
                "A returned zero is not proof a control is supported; it may mean unsupported, unknown or the current value.",
                "Opening a metadata-only node may fail even when the camera capture node works.",
                "These are values after opening, not a verified physical sensor configuration or measured frame rate.",
                "No exposure/focus/lighting comparison was performed; it requires a controlled scene and user-supplied camera samples.",
                "No frames were acquired, stored or uploaded, and no property setter was called.",
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("work/additional_experiments/camera_probe.json"))
    args = parser.parse_args(argv)
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()),
                      "opened": sum(row["opened"] for row in result["cameras"]),
                      "probed_nodes": len(result["cameras"]), "frames_acquired": 0,
                      "torch_imported": result["torch_imported"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
