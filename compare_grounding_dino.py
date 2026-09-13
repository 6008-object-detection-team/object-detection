"""Repeatable, unlabelled preview comparison for YOLOE and Grounding DINO Tiny."""
import gc
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from engines import GroundingDinoEngine, OpenVocabEngine


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "validation" / "grounding_dino_preview"
CASES = [
    ("sunflowers", ROOT / "OIF.jpg", ["sunflower", "leaf"]),
    ("portrait", ROOT / "mypicture.jpg", ["person", "face", "black shirt"]),
]
CONFIDENCE, IOU, REPEATS = 0.25, 0.45, 5


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(name, engine, model_path):
    load_started = time.perf_counter()
    status = engine.load_model(model_path, "CUDA" if torch.cuda.is_available() else "CPU")
    load_seconds = time.perf_counter() - load_started
    if status != "SUCCESS":
        raise RuntimeError(f"{name} 加载失败：{status}")

    rows = []
    for case_name, image_path, prompts in CASES:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(image_path)
        engine.detect(frame, CONFIDENCE, IOU, prompts)  # warm-up
        timings, detections = [], []
        for _ in range(REPEATS):
            synchronize()
            started = time.perf_counter()
            detections = engine.detect(frame, CONFIDENCE, IOU, prompts)
            synchronize()
            timings.append((time.perf_counter() - started) * 1000)
        rendered = engine.draw_detections(frame, detections)
        cv2.imwrite(str(OUTPUT_DIR / f"{case_name}_{name}.jpg"), rendered)
        rows.append({
            "case": case_name,
            "image": image_path.name,
            "image_size": [frame.shape[1], frame.shape[0]],
            "prompts": prompts,
            "detections": len(detections),
            "labels": [item.label for item in detections],
            "scores": [round(item.score, 4) for item in detections],
            "latency_ms_median": round(statistics.median(timings), 2),
            "latency_ms_p95": round(float(np.percentile(timings, 95)), 2),
            "measured_runs": REPEATS,
        })
    return {"model": name, "load_seconds": round(load_seconds, 2), "cases": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "notice": "Unlabelled preview only: counts and confidence are not accuracy, recall, or mAP.",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "confidence": CONFIDENCE,
        "iou": IOU,
        "models": [],
    }
    configurations = [
        ("yoloe_26s", OpenVocabEngine, ROOT / "yoloe-26s-seg.pt"),
        ("grounding_dino_tiny", GroundingDinoEngine, GroundingDinoEngine.MODEL_ID),
    ]
    for name, engine_type, model_path in configurations:
        print(f"Running {name}...")
        engine = engine_type()
        report["models"].append(benchmark(name, engine, model_path))
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = OUTPUT_DIR / "comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
