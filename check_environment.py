"""Check installed dependencies and the local weights without opening the UI.

Required items (ERROR, exit code 1) are what the default app needs: the
packages in requirements.txt plus a PyTorch backend, YOLOE-26S, its official
prompt-free sibling and the MobileCLIP2 text encoder used by YOLOE-26. The
other models and optional packages only print WARN lines.
"""
import argparse
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from prompt_modes import PROMPT_FREE_PAIRS
from tools.fetch_models import weight_state

# import name -> distribution name used for the printed version
REQUIRED_PACKAGES = {"numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python", "onnxruntime": "onnxruntime",
                     "PySide6": "PySide6", "torch": "torch", "torchvision": "torchvision",
                     "ultralytics": "ultralytics", "lap": "lap", "psutil": "psutil", "clip": "clip"}
OPTIONAL_PACKAGES = {
    "sahi": ("sahi", "SAHI slicing and mini-dataset evaluation (requirements-evaluation.txt)"),
    "pycocotools": ("pycocotools", "COCO AP evaluation (requirements-evaluation.txt)"),
    "onnx": ("onnx", "ONNX export (requirements-onnx.txt)"),
    "onnxslim": ("onnxslim", "ONNX export (requirements-onnx.txt)"),
    "transformers": ("transformers", "Grounding DINO comparison (requirements-dino.txt)"),
}
REQUIRED_MODELS = ("yoloe-26s-seg.pt", "yoloe-26s-seg-pf.pt", "mobileclip2_b.ts")
OPTIONAL_MODELS = tuple(name for name in (*PROMPT_FREE_PAIRS, *PROMPT_FREE_PAIRS.values(), "mobileclip_blt.ts")
                        if name not in REQUIRED_MODELS)
STATE_TEXT = {"missing": "missing", "lfs_pointer": "Git LFS pointer, not downloaded",
              "incomplete": "under 1 MiB, incomplete download"}


def package_version(name, distribution):
    """Import the module (a broken install fails here) and report its version."""
    module = importlib.import_module(name)
    if name == "clip" and not callable(getattr(module, "tokenize", None)):
        raise RuntimeError("Install the ultralytics/CLIP dependency from requirements.txt")
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
    return str(version)


def check_packages():
    """(lines, errors, warnings) for required and optional packages."""
    lines, errors, warnings = [], [], []
    for name, distribution in REQUIRED_PACKAGES.items():
        try:
            lines.append(f"OK {name}: {package_version(name, distribution)}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for name, (distribution, purpose) in OPTIONAL_PACKAGES.items():
        try:
            lines.append(f"OK {name} (optional): {package_version(name, distribution)}")
        except Exception as exc:
            warnings.append(f"{name} (optional, {purpose}): {type(exc).__name__}: {exc}")
    return lines, errors, warnings


def check_models(root=ROOT):
    """(lines, errors, warnings): required weights are errors, the others warnings."""
    lines, errors, warnings = [], [], []
    for names, problems, kind in ((REQUIRED_MODELS, errors, "required"), (OPTIONAL_MODELS, warnings, "optional")):
        for name in names:
            path = Path(root) / name
            state = weight_state(path)
            if state == "ok":
                lines.append(f"OK {name} ({kind}): {path.stat().st_size:,} bytes")
            else:
                problems.append(f"{name} ({kind}): {STATE_TEXT[state]}; run python tools/fetch_models.py --models {name}")
    return lines, errors, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dino", action="store_true", help="also check DINO dependencies and cached weights")
    args = parser.parse_args(argv)
    if sys.platform == "win32":
        dll_dir = Path(sys.prefix) / "Library" / "bin"
        if dll_dir.is_dir():
            dll_handle = os.add_dll_directory(str(dll_dir))
    (ROOT / ".ultralytics").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    package_lines, errors, warnings = check_packages()
    model_lines, model_errors, model_warnings = check_models()
    errors += model_errors
    warnings += model_warnings
    for line in package_lines + model_lines:
        print(line)
    if "torch" in sys.modules:
        print(f"CUDA available: {sys.modules['torch'].cuda.is_available()} (CPU also supported)")
        if sys.modules['torch'].cuda.is_available():
            print(f"GPU: {sys.modules['torch'].cuda.get_device_name(0)}")
    if args.dino:
        try:
            from transformers import AutoProcessor, AutoConfig
            cache = ROOT / ".cache" / "huggingface" / "hub"
            model_id = "IDEA-Research/grounding-dino-tiny"
            AutoProcessor.from_pretrained(model_id, cache_dir=str(cache), local_files_only=True)
            AutoConfig.from_pretrained(model_id, cache_dir=str(cache), local_files_only=True)
            from huggingface_hub import try_to_load_from_cache
            weights = try_to_load_from_cache(model_id, "model.safetensors", cache_dir=str(cache))
            if not isinstance(weights, str) or Path(weights).stat().st_size < 1024 * 1024:
                raise RuntimeError("DINO model.safetensors not cached")
            print(f"OK Grounding DINO: {weights}")
        except Exception as exc:
            errors.append(f"Grounding DINO: {exc}")
    for warning in warnings:
        print(f"WARN {warning}")
    for error in errors:
        print(f"ERROR {error}")
    if errors:
        print("Fix the errors above before starting.")
    elif warnings:
        print(f"Required packages and models are ready; {len(warnings)} optional item(s) are missing (WARN lines). "
              "The app marks undownloaded models as '(not downloaded)'.")
    else:
        print("Environment and model files are ready.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
