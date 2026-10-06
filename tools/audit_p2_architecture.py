"""Audit installed YOLO26/P2 YAML without importing Ultralytics or torch.

Stride propagation is deliberately restricted to the known modules in these
two configurations. Unknown or inconsistent modules fail instead of guessing.
No model is constructed, no checkpoint is deserialized, and no training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


def infer_head_outputs(config):
    layers = list(config["backbone"]) + list(config["head"])
    strides = []
    outputs = []

    def source_stride(reference, layer_index):
        index = layer_index + reference if reference < 0 else reference
        if index == -1 and layer_index == 0:
            return Fraction(1)
        if not 0 <= index < layer_index:
            raise ValueError(f"Invalid source layer {reference} at layer {layer_index}")
        value = strides[index]
        if not isinstance(value, Fraction):
            raise ValueError("A detector output cannot feed this spatial-stride parser")
        return value

    for index, layer in enumerate(layers):
        if len(layer) != 4:
            raise ValueError(f"Expected [from,repeats,module,args] at layer {index}")
        source, repeats, module, arguments = layer
        sources = source if isinstance(source, list) else [source]
        source_values = [source_stride(reference, index) for reference in sources]
        if module == "Detect":
            entry = {"layer_index": index, "module": module, "input_layer_indices": sources,
                     "output_strides": [float(value) for value in source_values]}
            outputs.append(entry)
            strides.append(source_values)
            continue
        if module == "Concat":
            if len(set(source_values)) != 1:
                raise ValueError(f"Concat uses inconsistent spatial strides at layer {index}")
            stride = source_values[0]
        elif len(source_values) != 1:
            raise ValueError(f"Unsupported multi-source module {module}")
        elif module == "Conv":
            if repeats != 1 or len(arguments) < 3:
                raise ValueError("Only explicit single Conv strides are supported")
            stride = source_values[0] * Fraction(arguments[2])
        elif module == "nn.Upsample":
            # Ultralytics YAML uses the literal string "None", resolved later
            # by its model parser; PyYAML leaves that particular token as str.
            if repeats != 1 or len(arguments) < 2 or arguments[0] not in (None, "None"):
                raise ValueError("Only explicit scale-factor upsampling is supported")
            stride = source_values[0] / Fraction(arguments[1])
        elif module in ("C3k2", "SPPF", "C2PSA"):
            stride = source_values[0]
        else:
            raise ValueError(f"Unsupported module {module}; stride will not be guessed")
        strides.append(stride)
    if len(outputs) != 1:
        raise ValueError("Expected exactly one Detect output layer")
    return outputs[0]


def installed_config(name):
    distribution = importlib.metadata.distribution("ultralytics")
    path = Path(distribution.locate_file(f"ultralytics/cfg/models/26/{name}"))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def project_weight_inventory(project):
    """Only inventory this project; do not follow symlinks or scan caches/envs."""
    files = []
    excluded = {".git", ".venv", ".cache", "node_modules", "__pycache__"}
    for directory, child_dirs, names in os.walk(project, followlinks=False):
        child_dirs[:] = [name for name in child_dirs if name not in excluded]
        for name in names:
            path = Path(directory) / name
            if path.suffix.lower() == ".pt":
                files.append(str(path.relative_to(project)))
    return {"searched_project": str(project), "excluded_directories": sorted(excluded),
            "local_checkpoint_files": sorted(files),
            "p2_named_checkpoint_files": sorted(path for path in files if "p2" in Path(path).stem.lower()),
            "verification_scope": "Project filename inventory only; no checkpoint contents deserialized or external directories searched."}


def audit(project):
    import yaml
    started = time.perf_counter()
    paths = {"standard": installed_config("yolo26.yaml"), "p2": installed_config("yolo26-p2.yaml")}
    models = {}
    for name, path in paths.items():
        content = path.read_bytes()
        config = yaml.safe_load(content)
        models[name] = {"config_file": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                        "architecture": "YOLO26 detection", "nc": config["nc"],
                        "end2end": config.get("end2end"),
                        "head": infer_head_outputs(config)}
    inventory = project_weight_inventory(project)
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "torch_imported": "torch" in sys.modules, "ultralytics_imported": "ultralytics" in sys.modules,
            "status": "architecture_verified_only_no_training_or_accuracy_experiment",
            "models": models, "local_weights": inventory,
            "p2_stride_4_verified": 4.0 in models["p2"]["head"]["output_strides"],
            "standard_smallest_output_stride": min(models["standard"]["head"]["output_strides"]),
            "p2_named_local_weights_found": bool(inventory["p2_named_checkpoint_files"]),
            "limitations": [
                "Only YAML graph geometry was checked; no model forward pass, precision or memory measurement was performed.",
                "Installed P2 YAML provides architecture, not an evaluated pretrained P2 checkpoint.",
                "Absence of p2-named files does not inspect or establish the architecture of custom-named checkpoint contents.",
                "Current YOLOE-26S segmentation weights cannot simply be relabeled as YOLO26 P2 trained weights.",
                "An accuracy comparison requires suitable trained weights or controlled training and a separate validation set.",
                "P2 emits an additional stride-4 spatial head; this audit does not establish its benefit or acceptable GPU memory on this laptop.",
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("work/additional_experiments/p2_audit.json"))
    args = parser.parse_args(argv)
    result = audit(args.project.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "standard": result["models"]["standard"]["head"],
                      "p2": result["models"]["p2"]["head"],
                      "p2_named_local_weights_found": result["p2_named_local_weights_found"],
                      "torch_imported": result["torch_imported"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
