"""Check installed dependencies and the local weights without opening the UI."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if sys.platform == "win32":
        dll_dir = Path(sys.prefix) / "Library" / "bin"
        if dll_dir.is_dir():
            dll_handle = os.add_dll_directory(str(dll_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    errors = []
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    for name in ("numpy", "cv2", "onnxruntime", "PySide6", "torch", "torchvision", "ultralytics", "clip"):
        try:
            module = importlib.import_module(name)
            if name == "clip" and not callable(getattr(module, "tokenize", None)):
                raise RuntimeError("Install the ultralytics/CLIP dependency from requirements.txt")
            print(f"OK {name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for name in ("yoloe-11l-seg.pt", "yoloe-26s-seg.pt", "yoloe-26l-seg.pt",
                 "yoloe-26x-seg.pt", "mobileclip_blt.ts", "mobileclip2_b.ts"):
        path = ROOT / name
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            errors.append(f"{name}: missing or a Git LFS pointer; run git lfs pull")
        else:
            print(f"OK {name}: {path.stat().st_size:,} bytes")
    if "torch" in sys.modules:
        print(f"CUDA available: {sys.modules['torch'].cuda.is_available()} (CPU also supported)")
    for error in errors:
        print(f"ERROR {error}")
    print("Environment and model files are ready." if not errors else "Fix the errors above before starting.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
