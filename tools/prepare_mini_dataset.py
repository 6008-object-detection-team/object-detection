#!/usr/bin/env python3
"""Prepare a small human-annotated COCO val set without using a model or GPU.

Only the ZIP member containing validation annotations and the selected JPEGs
are downloaded. HTTP ranges avoid fetching the 253 MB official archive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import tempfile
import time
import urllib.request
import zlib

ANNOTATION_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"
IMAGE_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017/"
DEFAULT_CLASSES = ("bottle", "cup", "cell phone")
# Original COCO boxes manually checked for clear, complete reference objects.
# Evaluation image selection happens first; a preferred reference is used only
# when its image is absent from that selected test set.
PREFERRED_REFERENCE_ANNOTATIONS = {"bottle": 371373, "cup": 670367, "cell phone": 325126}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_ANNOTATION_BYTES = 32 * 1024 * 1024
# Digest of the unmodified validation JSON fetched from the official HTTPS
# archive. Local annotation overrides must use this original file, too.
OFFICIAL_VAL_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Replace completed files atomically, never leave a half-written label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def fetch(url: str, *, byte_range: str | None = None, max_bytes: int = 16 * 1024 * 1024):
    headers = {"User-Agent": "EE6008-mini-evaluation/1.0"}
    if byte_range:
        headers["Range"] = "bytes=" + byte_range
    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if byte_range and response.status != 206:
                    raise RuntimeError("Server ignored Range; refusing a full archive download")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise RuntimeError(f"Download exceeds bounded limit: {url}")
                return data, dict(response.headers)
        except (OSError, RuntimeError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}: {error}") from error


def download_val_annotations(cache_file: Path) -> Path:
    if cache_file.is_file():
        if cache_file.stat().st_size > MAX_ANNOTATION_BYTES:
            raise ValueError("Cached annotations exceed the validation JSON size limit")
        return cache_file
    tail, headers = fetch(ANNOTATION_URL, byte_range="-65536", max_bytes=65536)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or len(tail) < end + 22:
        raise ValueError("Missing ZIP central directory")
    fields = struct.unpack("<4s4H2IH", tail[end:end + 22])
    directory_size, directory_offset = fields[5:7]
    central, _ = fetch(ANNOTATION_URL, byte_range=f"{directory_offset}-{directory_offset + directory_size - 1}",
                       max_bytes=1024 * 1024)
    offset = 0
    selected = None
    while offset < len(central):
        item = struct.unpack("<4s6H3I5H2I", central[offset:offset + 46])
        if item[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP member header")
        name_length, extra_length, comment_length = item[10:13]
        name = central[offset + 46:offset + 46 + name_length].decode("utf-8")
        if name == ANNOTATION_MEMBER:
            selected = item
        offset += 46 + name_length + extra_length + comment_length
    if selected is None:
        raise ValueError(f"Missing {ANNOTATION_MEMBER}")
    compression, expected_crc, compressed_bytes, expanded_bytes, member_offset = (
        selected[4], selected[7], selected[8], selected[9], selected[16])
    if compression != 8 or expanded_bytes > MAX_ANNOTATION_BYTES:
        raise ValueError("Unexpected annotation ZIP member format or size")
    local, _ = fetch(ANNOTATION_URL, byte_range=f"{member_offset}-{member_offset + 29}", max_bytes=30)
    local_fields = struct.unpack("<4s5H3I2H", local)
    if local_fields[0] != b"PK\x03\x04":
        raise ValueError("Invalid local ZIP header")
    payload_offset = member_offset + 30 + local_fields[9] + local_fields[10]
    compressed, _ = fetch(ANNOTATION_URL, byte_range=f"{payload_offset}-{payload_offset + compressed_bytes - 1}")
    # Reject unexpectedly large expansion even if ZIP metadata was misleading.
    decoder = zlib.decompressobj(-15)
    decoded = decoder.decompress(compressed, MAX_ANNOTATION_BYTES + 1)
    if len(decoded) != expanded_bytes or not decoder.eof or zlib.crc32(decoded) != expected_crc:
        raise ValueError("Annotation length or CRC verification failed")
    write_atomic(cache_file, decoded)
    write_json(cache_file.with_suffix(".source.json"), {
        "source_url": ANNOTATION_URL, "archive_member": ANNOTATION_MEMBER,
        "download_method": "HTTP Range; only validation member",
        "compressed_bytes": compressed_bytes, "annotation_bytes": len(decoded),
        "sha256": hashlib.sha256(decoded).hexdigest(), "archive_etag": headers.get("ETag"),
    })
    print(f"Downloaded official val annotations: {compressed_bytes:,} compressed bytes", flush=True)
    return cache_file


def size_group(area: float) -> str:
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def select_subset(coco: dict, classes=DEFAULT_CLASSES, count=24, seed=6008):
    if count < len(classes) * 3 or count > 64:
        raise ValueError("Use 3..64 images, and at least three images per class")
    categories_by_name = {item["name"]: item for item in coco["categories"]}
    if any(name not in categories_by_name for name in classes):
        raise ValueError("Class names must exactly match COCO category names")
    categories = [categories_by_name[name] for name in classes]
    ids = {item["id"] for item in categories}
    images_by_id = {item["id"]: item for item in coco["images"]}
    by_category = defaultdict(list)
    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in ids:
            annotations_by_image[ann["image_id"]].append(ann)
            if not ann.get("iscrowd", 0) and ann["area"] > 0 and min(ann["bbox"][2:]) >= 2:
                by_category[ann["category_id"]].append(ann)
    rng = random.Random(seed)
    selected = []
    seen = set()
    selection_log = []
    # Each class receives four small, two medium, and two large image slots
    # at the default count, before a deterministic fill if categories overlap.
    for category_index, category in enumerate(categories):
        quota = count // len(categories) + int(category_index < count % len(categories))
        quotas = {"small": quota // 2, "medium": quota // 4,
                  "large": quota - quota // 2 - quota // 4}
        for group, wanted in quotas.items():
            candidates = sorted({ann["image_id"] for ann in by_category[category["id"]]
                                 if size_group(ann["area"]) == group})
            rng.shuffle(candidates)
            chosen = [image_id for image_id in candidates if image_id not in seen][:wanted]
            for image_id in chosen:
                selected.append(image_id)
                seen.add(image_id)
                selection_log.append({"image_id": image_id, "selection_class": category["name"],
                                      "selection_size_group": group})
    remaining = sorted(set(annotations_by_image) - seen)
    rng.shuffle(remaining)
    for image_id in remaining[:count - len(selected)]:
        selected.append(image_id)
        seen.add(image_id)
        selection_log.append({"image_id": image_id, "selection_class": "fill", "selection_size_group": None})
    if len(selected) != count:
        raise ValueError("Source does not contain enough distinct images for the requested subset")
    references = []
    for class_id, category in enumerate(categories):
        candidates = [ann for ann in by_category[category["id"]] if ann["image_id"] not in seen
                      and min(ann["bbox"][2:]) >= 64]
        candidates.sort(key=lambda ann: (-(ann["bbox"][2] * ann["bbox"][3] /
                                          (images_by_id[ann["image_id"]]["width"] *
                                           images_by_id[ann["image_id"]]["height"])), ann["id"]))
        if not candidates:
            raise ValueError(f"No independent visual reference for {category['name']}")
        preferred = PREFERRED_REFERENCE_ANNOTATIONS.get(category["name"])
        reviewed = next((ann for ann in candidates if ann["id"] == preferred), None)
        ann = reviewed or candidates[0]
        seen.add(ann["image_id"])
        x, y, width, height = ann["bbox"]
        references.append({"class_id": class_id, "class_name": category["name"],
                           "category_id": category["id"], "original_image_id": ann["image_id"],
                           "original_annotation_id": ann["id"], "bbox_xyxy": [x, y, x + width, y + height],
                           "bbox_xywh": ann["bbox"],
                           "reference_selection": "manually reviewed original COCO box" if reviewed else "largest remaining box",
                           "image_file": "references/" + images_by_id[ann["image_id"]]["file_name"]})
    subset = {"info": {**coco.get("info", {}), "description": f"Curated {count}-image COCO2017 val diagnostic subset"},
              "licenses": coco.get("licenses", []),
              "images": [images_by_id[image_id] for image_id in selected],
              "annotations": [ann for image_id in selected for ann in annotations_by_image[image_id]],
              "categories": categories}
    return subset, references, selection_log


def prepare_dataset(output: Path, annotations_path: Path, *, count=24, classes=DEFAULT_CLASSES, seed=6008):
    if annotations_path.stat().st_size > MAX_ANNOTATION_BYTES:
        raise ValueError("Expected validation annotations below 32 MiB; do not pass training JSON")
    annotation_sha256 = sha256_file(annotations_path)
    if annotation_sha256 != OFFICIAL_VAL_SHA256:
        raise ValueError("Validation annotation SHA-256 mismatch: use unmodified official instances_val2017.json")
    with annotations_path.open(encoding="utf-8") as stream:
        coco = json.load(stream)
    subset, references, selection_log = select_subset(coco, classes, count, seed)
    image_map = {item["id"]: item for item in coco["images"]}
    licenses = {item["id"]: item for item in coco.get("licenses", [])}
    records = []
    requests = [(image, "evaluation", "images/" + image["file_name"]) for image in subset["images"]]
    requests += [(image_map[ref["original_image_id"]], "visual_reference", ref["image_file"]) for ref in references]
    for image, role, relative_path in requests:
        url = IMAGE_BASE_URL + image["file_name"]
        path = output / relative_path
        if not path.is_file():
            payload, _ = fetch(url, max_bytes=4 * 1024 * 1024)
            if not payload.startswith(b"\xff\xd8\xff"):
                raise ValueError(f"Expected JPEG bytes: {url}")
            write_atomic(path, payload)
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError(f"Cached image exceeds the bounded size limit: {path}")
        with path.open("rb") as stream:
            if not stream.read(3).startswith(b"\xff\xd8\xff"):
                raise ValueError(f"Cached image is not a JPEG: {path}")
        records.append({"image_file": relative_path, "role": role, "original_image_id": image["id"],
                        "source_url": url, "original_coco_url": image.get("coco_url"),
                        "original_flickr_url": image.get("flickr_url"),
                        "width": image["width"], "height": image["height"],
                        "license": licenses.get(image.get("license")),
                        "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        print(f"{role}: {image['file_name']} ({path.stat().st_size:,} bytes)", flush=True)
    counts = Counter(ann["category_id"] for ann in subset["annotations"])
    small_counts = Counter(ann["category_id"] for ann in subset["annotations"]
                           if not ann.get("iscrowd", 0) and size_group(ann["area"]) == "small")
    manifest = {
        "dataset": "COCO2017 val curated small-object diagnostic subset", "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_image_count": len(subset["images"]),
        "visual_reference_image_count": len(references), "annotation_count": len(subset["annotations"]),
        "classes": [{"class_id": index, "category_id": cat["id"], "name": cat["name"],
                     "annotations": counts[cat["id"]], "noncrowd_small_annotations": small_counts[cat["id"]]}
                    for index, cat in enumerate(subset["categories"])],
        "ground_truth": {"origin": "Original human annotations distributed by COCO, without model pseudo-labels",
                         "source_url": ANNOTATION_URL, "archive_member": ANNOTATION_MEMBER,
                         "original_annotation_sha256": annotation_sha256,
                         "annotation_license": "Creative Commons Attribution 4.0",
                         "terms_url": "https://cocodataset.org/#termsofuse"},
        "selection": selection_log, "images": records,
        "limitations": ["Curated small sample; not a representative COCO leaderboard benchmark.",
                        "Public validation data may have influenced pretrained model selection; this does not establish unseen-data generalization.",
                        "These public photos do not establish accuracy on the user's own camera domain.",
                        "Original annotations contain no explicit occlusion severity labels.",
                        "Visual references use entirely different image IDs from evaluation images.",
                        "All target-class annotations in each selected image are retained, including crowd flags and original segmentation areas."],
    }
    write_json(output / "annotations.json", subset)
    write_json(output / "references.json", {"classes": list(classes), "references": references,
                                          "note": "Different COCO images from the evaluation set; original human boxes."})
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"images": len(subset["images"]), "references": len(references),
                      "annotations": len(subset["annotations"]), "classes": manifest["classes"]}, ensure_ascii=False), flush=True)
    return manifest


def extract_video_frames(source: Path, output: Path, *, interval_seconds=5.0, count=24, max_side=1280):
    """Bounded sparse extraction; resulting frames are explicitly unannotated."""
    if interval_seconds <= 0 or not 1 <= count <= 200 or not 64 <= max_side <= 1600:
        raise ValueError("Invalid video extraction limits")
    if (output / "annotations.json").exists():
        raise ValueError("Output already contains annotations; use a new unannotated video output directory")
    import cv2  # Optional dependency, imported only for the video operation.
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    items = []
    try:
        for index in range(count):
            timestamp_ms = index * interval_seconds * 1000
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            filename = f"frame_{index:04d}.jpg"
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            write_atomic(output / "images" / filename, encoded.tobytes())
            items.append({"id": index + 1, "file_name": filename, "height": frame.shape[0],
                          "width": frame.shape[1], "requested_timestamp_ms": timestamp_ms,
                          "actual_timestamp_ms": capture.get(cv2.CAP_PROP_POS_MSEC)})
    finally:
        capture.release()
    if not items:
        raise ValueError("Video yielded no frames")
    write_json(output / "unannotated_manifest.json", {
        "source_video": str(source.resolve()), "source_sha256": sha256_file(source),
        "annotation_status": "UNANNOTATED: no labels; exclude from evaluation until human annotation",
        "images": items, "interval_seconds": interval_seconds,
        "classes": list(DEFAULT_CLASSES), "max_side": max_side})
    print(f"Extracted {len(items)} UNANNOTATED frames to {output}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "mini_eval")
    parser.add_argument("--annotation-json", type=Path, help="Local official instances_val2017.json, skips archive download")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--seed", type=int, default=6008)
    parser.add_argument("--video", type=Path, help="Sparse extraction for later human annotation; never invent GT")
    parser.add_argument("--video-output", type=Path, default=PROJECT_ROOT / "datasets" / "local_to_annotate")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-side", type=int, default=1280)
    return parser


def main():
    args = build_parser().parse_args()
    if args.video:
        extract_video_frames(args.video, args.video_output, interval_seconds=args.interval_seconds,
                             count=args.count, max_side=args.max_side)
    else:
        classes = tuple(name.strip() for name in args.classes.split(",") if name.strip())
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("Classes must be nonempty and distinct")
        annotations = args.annotation_json or download_val_annotations(PROJECT_ROOT / "work" / "dataset_cache" / "instances_val2017.json")
        prepare_dataset(args.output, annotations, count=args.count, classes=classes, seed=args.seed)


if __name__ == "__main__":
    main()
