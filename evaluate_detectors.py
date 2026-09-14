"""Quantitative mAP comparison of YOLOE and Grounding DINO Tiny on a labelled COCO subset.

Unlike compare_grounding_dino.py (an unlabelled speed/sanity preview), this script
scores detections against real COCO ground-truth boxes with pycocotools, the same
COCOeval tooling and metric definitions (mAP@0.5, mAP@0.5:0.95, AR) used in standard
object-detection benchmarks.

Requires coco_eval/instances_subset.json and coco_eval/images/, produced by
download_coco_subset.py.

Usage:
    python evaluate_detectors.py
"""
import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from engines import GroundingDinoEngine, OpenVocabEngine

ROOT = Path(__file__).resolve().parent
SUBSET_ANNOTATIONS = ROOT / "coco_eval" / "instances_subset.json"
IMAGES_DIR = ROOT / "coco_eval" / "images"
OUTPUT_DIR = ROOT / "validation" / "coco_eval"

CONFIDENCE = 0.05  # Low on purpose: COCOeval sweeps score thresholds itself; a high
                    # cutoff here would silently cap recall before mAP is computed.
IOU = 0.45

# Each engine is asked for exactly these COCO class names as its text prompt, so
# every predicted box can be mapped back to a COCO category id.
MODELS = [
    ("yoloe_26l", OpenVocabEngine, str(ROOT / "yoloe-26l-seg.pt")),
    ("yoloe_26s", OpenVocabEngine, str(ROOT / "yoloe-26s-seg.pt")),
    ("grounding_dino_tiny", GroundingDinoEngine, GroundingDinoEngine.MODEL_ID),
]


def read_image(path):
    """cv2.imread uses the OS narrow-string API on Windows, which mangles non-ASCII
    paths under a non-UTF-8 codepage (this machine's project folder has a Chinese
    name). Read bytes with pathlib instead and let cv2 decode them in memory."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def run_engine(name, engine_type, model_path, coco, class_names, name_to_cat_id):
    engine = engine_type()
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    status = engine.load_model(model_path, device)
    if status != "SUCCESS":
        raise RuntimeError(f"{name} failed to load: {status}")

    results = []
    img_ids = coco.getImgIds()
    started = time.perf_counter()
    for index, img_id in enumerate(img_ids, 1):
        meta = coco.loadImgs([img_id])[0]
        frame = read_image(IMAGES_DIR / meta["file_name"])
        if frame is None:
            continue
        detections = engine.detect(frame, CONFIDENCE, IOU, class_names)
        for det in detections:
            if det.class_id < 0 or det.class_id >= len(class_names):
                continue  # Grounding DINO occasionally returns an unmatched phrase.
            x1, y1, x2, y2 = det.xyxy
            results.append({
                "image_id": img_id,
                "category_id": name_to_cat_id[class_names[det.class_id]],
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(det.score), 4),
            })
        if index % 40 == 0 or index == len(img_ids):
            print(f"  [{name}] {index}/{len(img_ids)} images")
    elapsed = time.perf_counter() - started

    del engine
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, elapsed


def summarize(coco, results, image_ids):
    if not results:
        return None, {}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco.loadRes(results)
        coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_eval.summarize()
    stats = coco_eval.stats.tolist()
    metric_names = [
        "AP@[.5:.95]", "AP@.5", "AP@.75", "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large",
    ]
    return dict(zip(metric_names, [round(v, 4) for v in stats])), buffer.getvalue()


def main():
    if not SUBSET_ANNOTATIONS.exists():
        raise SystemExit(
            "coco_eval/instances_subset.json not found. Run download_coco_subset.py first."
        )
    coco = COCO(str(SUBSET_ANNOTATIONS))
    categories = coco.loadCats(coco.getCatIds())
    class_names = [c["name"] for c in categories]
    name_to_cat_id = {c["name"]: c["id"] for c in categories}
    image_ids = coco.getImgIds()
    print(f"Evaluating on {len(image_ids)} images, {len(class_names)} classes: {class_names}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "notice": (
            "Each model was queried with the same ten COCO class names as its text "
            "prompt and scored against real COCO val2017 ground-truth boxes with "
            "pycocotools COCOeval (mAP@[.5:.95] is the primary metric). Confidence "
            "threshold was kept low (0.05) so COCOeval's own score sweep is not "
            "truncated. Single run, no multi-seed variance; class list limited to "
            "ten common COCO categories, not the full open-vocabulary range either "
            "model supports."
        ),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "image_count": len(image_ids),
        "classes": class_names,
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU,
        "models": {},
    }

    for name, engine_type, model_path in MODELS:
        print(f"Running {name}...")
        results, elapsed = run_engine(name, engine_type, model_path, coco, class_names, name_to_cat_id)
        metrics, summary_text = summarize(coco, results, image_ids)
        print(summary_text)
        (OUTPUT_DIR / f"{name}_detections.json").write_text(
            json.dumps(results), encoding="utf-8"
        )
        report["models"][name] = {
            "detections": len(results),
            "inference_seconds_total": round(elapsed, 1),
            "ms_per_image": round(elapsed / max(len(image_ids), 1) * 1000, 1),
            "metrics": metrics,
        }

    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {report_path}")
    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    main()
