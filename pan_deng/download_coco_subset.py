"""Download a small, labelled COCO val2017 subset for quantitative model comparison.

Only images containing at least one instance of TARGET_CLASSES are kept, capped at
MAX_IMAGES. Images are fetched individually from the official COCO image server, so
this never downloads the full 5,000-image validation split or its ~19 GB of images.

Usage:
    python pan_deng/download_coco_subset.py --annotations path/to/instances_val2017.json
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "coco_eval_data"
IMAGES_DIR = OUT_DIR / "images"
SUBSET_ANNOTATIONS = OUT_DIR / "instances_subset.json"

# Mirrors the ten classes used in the demo prompts and the write-up's comparison table.
TARGET_CLASSES = [
    "person", "backpack", "cell phone", "bottle", "cup",
    "laptop", "chair", "dog", "car", "bicycle",
]
MAX_IMAGES = 240
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Path to instances_val2017.json")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    args = parser.parse_args()

    from pycocotools.coco import COCO

    coco = COCO(args.annotations)
    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    found_names = {c["name"] for c in coco.loadCats(cat_ids)}
    missing = set(TARGET_CLASSES) - found_names
    if missing:
        sys.exit(f"These class names are not in COCO's category list: {sorted(missing)}")

    image_ids = set()
    per_class_ids = {}
    for cat_id, name in zip(cat_ids, TARGET_CLASSES):
        ids = coco.getImgIds(catIds=[cat_id])
        per_class_ids[name] = ids
        image_ids.update(ids)
    print(f"{len(image_ids)} candidate images cover all {len(TARGET_CLASSES)} classes.")

    # Round-robin over classes so the capped subset still covers every class,
    # instead of the natural image-id order (which favours whichever class
    # happens to have the most annotated images in COCO).
    selected, seen = [], set()
    exhausted = False
    while len(selected) < args.max_images and not exhausted:
        exhausted = True
        for name in TARGET_CLASSES:
            for img_id in per_class_ids[name]:
                if img_id not in seen:
                    seen.add(img_id)
                    selected.append(img_id)
                    exhausted = False
                    break
            if len(selected) >= args.max_images:
                break
    print(f"Selected {len(selected)} images (cap {args.max_images}).")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images_meta = coco.loadImgs(selected)
    for index, meta in enumerate(images_meta, 1):
        dest = IMAGES_DIR / meta["file_name"]
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = IMAGE_BASE_URL + meta["file_name"]
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"  failed: {meta['file_name']}: {exc}")
            continue
        if index % 20 == 0 or index == len(images_meta):
            print(f"  downloaded {index}/{len(images_meta)}")

    # Trim the annotation file down to just the selected images/categories so the
    # subset is self-contained and small enough to keep in the repo.
    ann_ids = coco.getAnnIds(imgIds=selected, catIds=cat_ids, iscrowd=None)
    subset = {
        "info": {"description": "COCO val2017 subset for YOLOE / Grounding DINO comparison"},
        "licenses": coco.dataset.get("licenses", []),
        "images": images_meta,
        "annotations": coco.loadAnns(ann_ids),
        "categories": coco.loadCats(cat_ids),
    }
    SUBSET_ANNOTATIONS.write_text(json.dumps(subset), encoding="utf-8")
    print(f"Wrote {SUBSET_ANNOTATIONS} ({len(subset['images'])} images, {len(subset['annotations'])} annotations).")


if __name__ == "__main__":
    main()
