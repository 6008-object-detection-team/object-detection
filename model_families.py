"""Stable logical model identities, separate from their inference profiles."""
from pathlib import Path

MODEL_FAMILIES = {
    "yoloe-26s": "YOLOE-26S",
    "yoloe-26l": "YOLOE-26L",
    "yoloe-26x": "YOLOE-26X",
    "yoloe-11l": "YOLOE-11L",
}


def model_family(path):
    if not isinstance(path, (str, Path)) or not str(path):
        return ""
    path = Path(path)
    for family in MODEL_FAMILIES:
        if path.name in (family + "-seg.pt", family + "-seg-pf.pt"):
            return family
        if path.name in ("modified.pt", "modified-pf.pt") and path.parent.name == family:
            return family
    return ""


def record_family(row):
    explicit = row.get("model_family") or row.get("provenance", {}).get("model_family")
    return explicit if explicit in MODEL_FAMILIES else model_family(row.get("provenance", {}).get("model", ""))
