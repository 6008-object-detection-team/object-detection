"""Separate the confidence threshold from the number of boxes kept per image.

In the first comparison both changed together: a lower threshold also meant many more
boxes. Here each model runs once on the COCO subset with a near-zero threshold, and
every setting is applied offline to those saved detections, so no setting needs a
second inference pass:

- Threshold sweep: vary the confidence threshold, box budget fixed at 100 per image.
- Box-budget sweep: vary the boxes kept per image (highest scores first), threshold
  fixed near zero.

Each setting reports COCO mAP/AR, which rank detections and so barely depend on a
cutoff, plus precision, recall and F1 at IoU 0.5 for the detections that survive it --
the numbers that decide where the app's confidence slider should sit.

If coco_eval_results/legacy_text_threshold/ holds the detections made before the
Grounding DINO label fix, the same-threshold runs are also compared with and without
class labels to show whether the boxes or the labels changed.

Usage:
    python pan_deng/evaluate_detectors.py        # first, for the label-fix check
    python pan_deng/sweep_threshold_topk.py      # inference is cached after the first run
    python pan_deng/sweep_threshold_topk.py --rerun
"""
import argparse
import contextlib
import copy
import io
import json
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluate_detectors import MODELS, OUTPUT_DIR, ROOT, SUBSET_ANNOTATIONS, run_engine

CACHE_DIR = ROOT / "coco_eval_data" / "sweep_cache"  # gitignored with the images
SWEEP_DIR = OUTPUT_DIR / "sweep"
LEGACY_DIR = OUTPUT_DIR / "legacy_text_threshold"

RAW_CONFIDENCE = 0.001
MAX_BOXES = 300  # Ultralytics' default max_det; Grounding DINO is capped the same way.
THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
FIXED_BOX_BUDGET = 100
BOX_BUDGETS = [1, 3, 5, 10, 20, 50, 100, 300]
FIXED_THRESHOLD = RAW_CONFIDENCE

LABELS = {
    "yoloe_26l": "YOLOE-26L",
    "yoloe_26s": "YOLOE-26S",
    "grounding_dino_tiny": "Grounding DINO Tiny",
}


def load_raw_detections(name, engine_type, model_path, coco, class_names, name_to_cat_id, rerun):
    path = CACHE_DIR / f"{name}_raw_detections.json"
    if path.exists() and not rerun:
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"Running {name} once at confidence {RAW_CONFIDENCE}...")
    results, _ = run_engine(name, engine_type, model_path, coco, class_names,
                            name_to_cat_id, RAW_CONFIDENCE)
    results = apply_setting(group_by_image(results), RAW_CONFIDENCE, MAX_BOXES)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results), encoding="utf-8")
    return results


def group_by_image(detections):
    by_image = defaultdict(list)
    for det in detections:
        by_image[det["image_id"]].append(det)
    for dets in by_image.values():
        dets.sort(key=lambda det: det["score"], reverse=True)
    return by_image


def apply_setting(by_image, threshold, box_budget):
    kept = []
    for dets in by_image.values():
        kept.extend([det for det in dets if det["score"] >= threshold][:box_budget])
    return kept


def score(coco, detections, image_ids):
    """COCO mAP/AR plus precision/recall/F1 at IoU 0.5 for exactly these detections."""
    if not detections:
        return {"mAP": 0.0, "AP50": 0.0, "AP_small": 0.0, "AR100": 0.0,
                "precision": 0.0, "recall": 0.0, "F1": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(coco, coco.loadRes(copy.deepcopy(detections)), iouType="bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    true_pos = false_pos = positives = 0
    all_areas = coco_eval.params.areaRng[0]
    for item in coco_eval.evalImgs:
        if item is None or item["aRng"] != all_areas:
            continue
        counted = ~item["dtIgnore"][0].astype(bool)  # row 0 is IoU 0.5
        matched = item["dtMatches"][0] > 0
        true_pos += int((matched & counted).sum())
        false_pos += int((~matched & counted).sum())
        positives += int((~item["gtIgnore"].astype(bool)).sum())
    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    stats = coco_eval.stats
    return {
        "mAP": round(float(stats[0]), 4), "AP50": round(float(stats[1]), 4),
        "AP_small": round(float(stats[3]), 4), "AR100": round(float(stats[8]), 4),
        "precision": round(precision, 4), "recall": round(recall, 4), "F1": round(f1, 4),
    }


def sweep(coco, by_image, image_ids, settings):
    rows = []
    for threshold, box_budget in settings:
        kept = apply_setting(by_image, threshold, box_budget)
        rows.append({
            "confidence": threshold, "max_boxes": box_budget,
            "boxes_per_image": round(len(kept) / len(image_ids), 2),
            **score(coco, kept, image_ids),
        })
    return rows


def class_agnostic(coco, detections):
    """The same boxes scored with every label replaced by one class."""
    dataset = copy.deepcopy(coco.dataset)
    dataset["categories"] = [{"id": 1, "name": "object"}]
    for ann in dataset["annotations"]:
        ann["category_id"] = 1
    with contextlib.redirect_stdout(io.StringIO()):
        agnostic = COCO()
        agnostic.dataset = dataset
        agnostic.createIndex()
    return agnostic, [dict(det, category_id=1) for det in detections]


def label_fix_check(coco, image_ids):
    runs = [
        (0.05, "grounding_dino_tiny_detections.json", "grounding_dino_tiny_detections.json"),
        (0.35, "grounding_dino_tiny_t035_detections.json",
         "grounding_dino_tiny_t035_default_threshold_detections.json"),
    ]
    rows = []
    for threshold, legacy_file, current_file in runs:
        for version, path in [("before_fix", LEGACY_DIR / legacy_file),
                              ("after_fix", OUTPUT_DIR / current_file)]:
            if not path.exists():
                return None
            dets = json.loads(path.read_text(encoding="utf-8"))
            agnostic_coco, agnostic_dets = class_agnostic(coco, dets)
            rows.append({
                "confidence": threshold, "version": version, "detections": len(dets),
                "mAP": score(coco, dets, image_ids)["mAP"],
                "class_agnostic_mAP": score(agnostic_coco, agnostic_dets, image_ids)["mAP"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun", action="store_true", help="ignore cached detections")
    args = parser.parse_args()
    if not SUBSET_ANNOTATIONS.exists():
        raise SystemExit(
            "coco_eval_data/instances_subset.json not found. Run download_coco_subset.py first."
        )
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(SUBSET_ANNOTATIONS))
    categories = coco.loadCats(coco.getCatIds())
    class_names = [c["name"] for c in categories]
    name_to_cat_id = {c["name"]: c["id"] for c in categories}
    image_ids = coco.getImgIds()

    report = {
        "notice": (
            f"Each model ran once at confidence {RAW_CONFIDENCE} (at most {MAX_BOXES} "
            "boxes per image); every setting below was applied offline to those "
            "detections. threshold_sweep fixes the box budget at "
            f"{FIXED_BOX_BUDGET} per image; box_budget_sweep fixes the threshold at "
            f"{FIXED_THRESHOLD}. Precision/recall/F1 are at IoU 0.5 with COCO matching. "
            "COCOeval still keeps at most 100 boxes per image per class."
        ),
        "image_count": len(image_ids),
        "classes": class_names,
        "models": {},
    }
    for name, engine_type, model_path, _ in MODELS:
        if name not in LABELS:
            continue
        raw = load_raw_detections(name, engine_type, model_path, coco, class_names,
                                  name_to_cat_id, args.rerun)
        by_image = group_by_image(raw)
        print(f"Sweeping {name}...")
        threshold_rows = sweep(coco, by_image, image_ids,
                               [(t, FIXED_BOX_BUDGET) for t in THRESHOLDS])
        budget_rows = sweep(coco, by_image, image_ids,
                            [(FIXED_THRESHOLD, k) for k in BOX_BUDGETS])
        report["models"][name] = {
            "label": LABELS[name],
            "threshold_sweep": threshold_rows,
            "box_budget_sweep": budget_rows,
            "best_f1": max(threshold_rows, key=lambda row: row["F1"]),
        }

    check = label_fix_check(coco, image_ids)
    if check:
        report["label_fix_check"] = check

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    path = SWEEP_DIR / "sweep.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    for name, result in report["models"].items():
        best = result["best_f1"]
        print(f"  {name}: best F1 {best['F1']:.3f} at confidence {best['confidence']}")


if __name__ == "__main__":
    main()
