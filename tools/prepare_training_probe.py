#!/usr/bin/env python3
"""Prepare disjoint, bounded COCO train/val probes; standard library, no GPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import struct
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prepare_mini_dataset import (
    ANNOTATION_MEMBER, ANNOTATION_URL, DEFAULT_CLASSES, IMAGE_BASE_URL,
    MAX_ANNOTATION_BYTES, OFFICIAL_VAL_SHA256, PROJECT_ROOT,
    fetch, sha256_file, size_group, write_atomic, write_json,
)

MAX_TOTAL_BYTES = 40 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_SIDE = 640


def excluded_image_ids(evaluation_root: Path):
    """Require both original diagnostic images and references to be excluded."""
    annotations = json.loads((evaluation_root / "annotations.json").read_text())
    references = json.loads((evaluation_root / "references.json").read_text())
    evaluation = {int(image["id"]) for image in annotations["images"]}
    reference = {int(item["original_image_id"]) for item in references["references"]}
    if evaluation & reference:
        raise ValueError("Existing evaluation/reference image IDs overlap")
    return evaluation, reference


def select_splits(coco, excluded, train_count=48, val_count=12, seed=6008):
    """Distinct image slots per category, with half selected for small objects."""
    if not 6 <= train_count <= 96 or not 6 <= val_count <= 24:
        raise ValueError("Bounded probe requires train 6..96 and val 6..24 images")
    categories_by_name = {item["name"]: item for item in coco["categories"]}
    categories = [categories_by_name[name] for name in DEFAULT_CLASSES]
    target_ids = {item["id"] for item in categories}
    images = {item["id"]: item for item in coco["images"]}
    all_annotations = defaultdict(list)
    candidates = defaultdict(set)
    for ann in coco["annotations"]:
        if ann["category_id"] not in target_ids:
            continue
        all_annotations[ann["image_id"]].append(ann)
        image = images[ann["image_id"]]
        if (ann["image_id"] in excluded or ann.get("iscrowd", 0)
                or max(image["width"], image["height"]) > MAX_SIDE
                or min(image["width"], image["height"]) < 1
                or ann.get("area", 0) <= 0 or min(ann["bbox"][2:]) < 2):
            continue
        candidates[ann["category_id"], size_group(ann["area"])].add(ann["image_id"])
    rng = random.Random(seed)
    seen = set(excluded)
    splits, selection = {}, []
    for split, count in (("train", train_count), ("val", val_count)):
        chosen = []
        for index, category in enumerate(categories):
            quota = count // len(categories) + int(index < count % len(categories))
            quotas = {"small": quota // 2, "medium": quota // 4,
                      "large": quota - quota // 2 - quota // 4}
            for group, wanted in quotas.items():
                available = sorted(candidates[category["id"], group] - seen)
                rng.shuffle(available)
                if len(available) < wanted:
                    raise ValueError(f"Insufficient distinct {split} {category['name']} {group} images")
                for image_id in available[:wanted]:
                    chosen.append(image_id)
                    seen.add(image_id)
                    selection.append({"split": split, "image_id": image_id,
                                      "selection_class": category["name"],
                                      "selection_size_group": group})
        splits[split] = {
            "info": {**coco.get("info", {}), "description": f"COCO2017 val {split} functional training probe"},
            "licenses": coco.get("licenses", []), "categories": categories,
            "images": [images[image_id] for image_id in chosen],
            "annotations": [ann for image_id in chosen for ann in all_annotations[image_id]],
        }
    return splits, selection


def yolo_labels(subset):
    """Retain original COCO GT; clip only converted labels and log exceptions."""
    mapping = {category["id"]: index for index, category in enumerate(subset["categories"])}
    images = {image["id"]: image for image in subset["images"]}
    labels = {image_id: [] for image_id in images}
    skipped, clipped = [], []
    for ann in subset["annotations"]:
        if ann.get("iscrowd", 0):
            skipped.append({"annotation_id": ann["id"], "image_id": ann["image_id"], "reason": "crowd"})
            continue
        image = images[ann["image_id"]]
        x, y, width, height = map(float, ann["bbox"])
        if not all(map(math.isfinite, (x, y, width, height))) or min(width, height) <= 0:
            skipped.append({"annotation_id": ann["id"], "image_id": ann["image_id"], "reason": "invalid_bbox"})
            continue
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(image["width"]), x + width), min(float(image["height"]), y + height)
        if x2 <= x1 or y2 <= y1:
            skipped.append({"annotation_id": ann["id"], "image_id": ann["image_id"], "reason": "outside_image"})
            continue
        if (x1, y1, x2, y2) != (x, y, x + width, y + height):
            clipped.append({"annotation_id": ann["id"], "image_id": ann["image_id"]})
        values = ((x1 + x2) / (2 * image["width"]), (y1 + y2) / (2 * image["height"]),
                  (x2 - x1) / image["width"], (y2 - y1) / image["height"])
        labels[ann["image_id"]].append(str(mapping[ann["category_id"]]) + " " + " ".join(f"{value:.10f}" for value in values))
    payloads = {image_id: (("\n".join(lines) + "\n") if lines else "").encode()
                for image_id, lines in labels.items()}
    return payloads, skipped, clipped


def jpeg_dimensions(payload):
    """Read JPEG SOF geometry without loading image libraries or a model."""
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("Expected JPEG bytes")
    offset = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ValueError("Invalid JPEG header")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0xD9, 0xDA) or offset + 2 > len(payload):
            break
        length = struct.unpack(">H", payload[offset:offset + 2])[0]
        if length < 2 or offset + length > len(payload):
            raise ValueError("Truncated JPEG header")
        if marker in sof:
            if length < 8:
                raise ValueError("Invalid JPEG SOF")
            height, width = struct.unpack(">HH", payload[offset + 3:offset + 7])
            return width, height
        offset += length
    raise ValueError("JPEG image geometry not found")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def directory_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def prepare_probe(output, annotation_path, evaluation_root, *, train_count=48, val_count=12, seed=6008,
                  max_total_bytes=MAX_TOTAL_BYTES):
    if not 1024 * 1024 <= max_total_bytes <= MAX_TOTAL_BYTES:
        raise ValueError("Dataset total limit must be 1..40 MiB")
    if annotation_path.stat().st_size > MAX_ANNOTATION_BYTES or sha256_file(annotation_path) != OFFICIAL_VAL_SHA256:
        raise ValueError("Use original, hash-verified COCO2017 validation annotations")
    original = json.loads(annotation_path.read_text())
    evaluation, references = excluded_image_ids(evaluation_root)
    splits, selection = select_splits(original, evaluation | references, train_count, val_count, seed)
    plan = {"seed": seed, "source_sha256": OFFICIAL_VAL_SHA256, "max_total_bytes": max_total_bytes,
            "excluded_evaluation_ids": sorted(evaluation), "excluded_reference_ids": sorted(references),
            "split_image_ids": {split: [image["id"] for image in subset["images"]] for split, subset in splits.items()}}
    plan_path = output / "preparation.json"
    if output.exists() and any(output.iterdir()):
        if not plan_path.is_file() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Output contains another dataset/configuration; choose a new directory")
    metadata = {"preparation.json": json_bytes(plan)}
    summaries = {}
    for split, subset in splits.items():
        labels, skipped, clipped = yolo_labels(subset)
        metadata[f"annotations/{split}.json"] = json_bytes(subset)
        for image in subset["images"]:
            metadata[f"labels/{split}/{Path(image['file_name']).stem}.txt"] = labels[image["id"]]
        counts = Counter(ann["category_id"] for ann in subset["annotations"])
        noncrowd = Counter(ann["category_id"] for ann in subset["annotations"] if not ann.get("iscrowd", 0))
        small = Counter(ann["category_id"] for ann in subset["annotations"] if not ann.get("iscrowd", 0) and size_group(ann["area"]) == "small")
        summaries[split] = {"images": len(subset["images"]), "original_annotation_count": len(subset["annotations"]),
                            "yolo_label_count": sum(payload.count(b"\n") for payload in labels.values()),
                            "skipped_yolo_annotations": skipped, "clipped_yolo_annotations": clipped,
                            "classes": [{"class_id": index, "category_id": cat["id"], "name": cat["name"],
                                         "original_annotations": counts[cat["id"]], "noncrowd_annotations": noncrowd[cat["id"]],
                                         "noncrowd_small_annotations": small[cat["id"]]}
                                        for index, cat in enumerate(subset["categories"])]}
    # Omit path: Ultralytics resolves omitted path from the data.yaml location.
    metadata["data.yaml"] = b"# Relative to this YAML directory; original category IDs mapped to 0/1/2.\ntrain: images/train\nval: images/val\nnames:\n  0: bottle\n  1: cup\n  2: cell phone\n"
    image_paths = {f"images/{split}/{image['file_name']}" for split, subset in splits.items() for image in subset["images"]}
    allowed = set(metadata) | image_paths | {"manifest.json"}
    if output.exists():
        unexpected = [path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file() and path.relative_to(output).as_posix() not in allowed]
        if unexpected:
            raise ValueError("Unexpected files in output: " + ", ".join(unexpected[:3]))
    reserve = sum(map(len, metadata.values())) + 512 * 1024
    if reserve >= max_total_bytes:
        raise ValueError("Metadata exceeds total dataset budget")
    output.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    prior = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").is_file() else {}
    prior_hashes = {item["image_file"]: item["sha256"] for item in prior.get("images", [])}
    licenses = {item["id"]: item for item in original.get("licenses", [])}
    records, used_images = [], 0
    for split, subset in splits.items():
        for image in subset["images"]:
            relative = f"images/{split}/{image['file_name']}"
            path = output / relative
            url = IMAGE_BASE_URL + image["file_name"]
            remaining = max_total_bytes - reserve - used_images
            if remaining <= 0:
                raise ValueError("Dataset would exceed total size budget")
            limit = min(MAX_IMAGE_BYTES, remaining)
            if path.is_file():
                if path.stat().st_size > limit:
                    raise ValueError("Existing JPEG exceeds dataset budget")
                payload = path.read_bytes()
                if relative in prior_hashes and sha256_file(path) != prior_hashes[relative]:
                    raise ValueError("Existing JPEG hash differs from completed manifest")
            else:
                payload, _ = fetch(url, max_bytes=limit)
            if jpeg_dimensions(payload) != (image["width"], image["height"]) or max(jpeg_dimensions(payload)) > MAX_SIDE:
                raise ValueError("JPEG dimensions differ from original COCO metadata or exceed 640")
            if not path.is_file():
                write_atomic(path, payload)
            used_images += len(payload)
            records.append({"split": split, "image_file": relative, "original_image_id": image["id"],
                            "source_url": url, "original_coco_url": image.get("coco_url"),
                            "original_flickr_url": image.get("flickr_url"), "width": image["width"], "height": image["height"],
                            "license": licenses.get(image.get("license")), "bytes": len(payload), "sha256": sha256_file(path)})
            print(f"{split}: {image['file_name']} ({len(payload):,} bytes)", flush=True)
    for relative, payload in metadata.items():
        write_atomic(output / relative, payload)
    manifest = {"status": "complete", "dataset": "COCO2017 val disjoint training/validation functional probe",
                "created_utc": datetime.now(timezone.utc).isoformat(), **plan, "selection": selection,
                "splits": summaries, "images": records, "image_bytes": used_images,
                "ground_truth": {"origin": "Original COCO human annotations; no model pseudo-labels",
                                 "source_url": ANNOTATION_URL, "archive_member": ANNOTATION_MEMBER,
                                 "original_annotation_sha256": OFFICIAL_VAL_SHA256,
                                 "annotation_license": "Creative Commons Attribution 4.0",
                                 "terms_url": "https://cocodataset.org/#termsofuse"},
                "files": {name: {"bytes": len(payload), "sha256": sha256_file(output / name)} for name, payload in metadata.items()},
                "limitations": ["Public, tiny training functionality probe; not camera-domain validation or a leaderboard result.",
                                "Existing mini_eval was used for development; excluded images are not an untouched final test claim.",
                                "Image IDs are disjoint across train, val, original diagnostic images and visual references.",
                                "Public validation photographs may have influenced pretrained models; no unseen-source generalization claim.",
                                "Original target-class COCO annotations retained including crowd; YOLO labels skip crowd and record exceptions.",
                                "YOLO annotation bbox clipping affects converted labels only; original COCO bbox/area remain unchanged.",
                                "Image copyrights follow individual original Flickr licenses; preserve provenance.",
                                "No model training, GPU inference or new model evaluation was performed during preparation."]}
    manifest["total_bytes"] = 0
    for _ in range(3):
        manifest["total_bytes"] = directory_bytes(output) - ((output / "manifest.json").stat().st_size if (output / "manifest.json").exists() else 0) + len(json_bytes(manifest))
    if manifest["total_bytes"] > max_total_bytes:
        raise ValueError("Complete dataset exceeds the total size budget")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"total_bytes": directory_bytes(output), "splits": summaries}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "work/dataset_cache/instances_val2017.json")
    parser.add_argument("--exclude-dataset", type=Path, default=PROJECT_ROOT / "datasets/mini_eval")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets/mini_train_probe")
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=6008)
    args = parser.parse_args()
    prepare_probe(args.output, args.annotations, args.exclude_dataset, train_count=args.train_count,
                  val_count=args.val_count, seed=args.seed)


if __name__ == "__main__":
    main()
