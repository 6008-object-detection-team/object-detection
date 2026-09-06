"""Detection records and conservative merging in original-image coordinates."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    xyxy: tuple
    score: float
    class_id: int
    label: str
    source: str = "original"


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def merge_supplementary(original, candidates, iou=0.45):
    """Retain original boxes/scores; supplementary boxes can still be false positives."""
    merged = list(original)
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for existing in merged:
            overlap = box_iou(candidate.xyxy, existing.xyxy)
            if (candidate.class_id == existing.class_id and overlap >= iou) or overlap >= 0.8:
                duplicate = True
                break
        if not duplicate:
            merged.append(candidate)
    return merged


def tile_regions(width, height):
    """Four overlapping crops cover objects at crop seams in their neighbors."""
    tile_w, tile_h = min(width, round(width * 0.6)), min(height, round(height * 0.6))
    for top in sorted({0, height - tile_h}):
        for left in sorted({0, width - tile_w}):
            yield left, top, left + tile_w, top + tile_h
