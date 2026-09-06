"""Reproducible local smoke comparison; counts/scores are NOT accuracy or mAP.

Uses the installed Ultralytics bus sample and a deterministic synthetic small /
dark version. No webcam, private images or external image upload is needed.
"""
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from enhancement import AdaptiveLowLightEngine
from engines import OpenVocabEngine


def old_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    luma = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(luma)
    return cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2BGR)


def main():
    import torch
    import ultralytics
    root = Path(__file__).resolve().parent
    os.chdir(root)
    output = root / "validation"
    output.mkdir(exist_ok=True)
    bus = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    if bus is None:
        raise RuntimeError("Ultralytics bus.jpg sample is unavailable")
    # Shrink known scene into a 1080p frame to exercise small-target sampling.
    small = np.full((1080, 1920, 3), 130, np.uint8)
    tiny = cv2.resize(bus, (162, 216), interpolation=cv2.INTER_AREA)
    small[180:396, 180:342] = tiny
    smaller = np.full_like(small, 130)
    smaller[180:288, 180:261] = cv2.resize(bus, (81, 108), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(21)
    dark = np.clip(bus.astype(np.float32) * 0.16 + rng.normal(0, 1.5, bus.shape), 0, 255).astype(np.uint8)
    very_dark = np.clip(bus.astype(np.float32) * 0.04 + rng.normal(0, 1.5, bus.shape), 0, 255).astype(np.uint8)
    cases = {"normal": bus, "small_1080p": small, "tiny_1080p": smaller,
             "dark_synthetic": dark, "very_dark_synthetic": very_dark}
    report = {"notice": "Unlabelled smoke comparison: counts and confidence are not accuracy, recall or mAP. Synthetic darkness does not replace a real camera test.",
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
              "ultralytics": ultralytics.__version__, "confidence_threshold": 0.4, "rows": []}
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    for model_path, sizes in [("yoloe-11l-seg.pt", [640]), ("yoloe-26l-seg.pt", [640, 960, 1280]), ("yoloe-26x-seg.pt", [960])]:
        engine = OpenVocabEngine()
        status = engine.load_model(model_path, device)
        if status != "SUCCESS":
            raise RuntimeError(status)
        for size in sizes:
            engine.imgsz = size
            engine.detect(bus, class_names=["person", "bus"])  # exclude model/text warmup
            for case_name, frame in cases.items():
                modes = ["original"]
                if "dark_synthetic" in case_name:
                    modes += ["old_clahe", "adaptive_only", "adaptive_supplement"]
                if case_name in ("small_1080p", "tiny_1080p") and model_path == "yoloe-26l-seg.pt" and size == 960:
                    modes += ["tiles"]
                for mode in modes:
                    input_frame = old_clahe(frame) if mode == "old_clahe" else AdaptiveLowLightEngine().enhance(frame) if mode == "adaptive_only" else frame
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        annotated, detections, status = engine.process_frame(
                            input_frame, class_names=["person", "bus"],
                            enhancer=AdaptiveLowLightEngine() if mode == "adaptive_supplement" else None,
                            tiled=mode == "tiles",
                        )
                        times.append((time.perf_counter() - start) * 1000)
                    row = {"model": model_path, "size": size, "case": case_name, "mode": mode,
                           "counts": dict(Counter(item.label for item in detections)),
                           "detections": [asdict(item) for item in detections],
                           "median_ms": round(float(np.median(times)), 2), "status": status}
                    report["rows"].append(row)
                    name = f"{Path(model_path).stem}_{size}_{case_name}_{mode}.jpg"
                    cv2.imwrite(str(output / name), annotated)
                    print(f"{model_path} {size} {case_name} {mode}: {row['counts']} {row['median_ms']} ms", flush=True)
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(output / "dark_before.jpg"), dark)
    cv2.imwrite(str(output / "dark_adaptive.jpg"), AdaptiveLowLightEngine().enhance(dark))
    print("Saved validation/comparison.json", flush=True)


if __name__ == "__main__":
    main()
