"""Sequential CUDA smoke for text, SAHI, SAVPE and both DINO APIs; no training."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local-deployment" / "windows-upgrade-20261006")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.chdir(ROOT)
    import cv2
    import psutil
    import torch
    from engines import GroundingDINOEngine, GroundingDinoEngine, OpenVocabEngine
    from visual_engine import VisualPromptEngine
    from tracking import ByteTrackManager
    from video_worker import read_image_file

    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = read_image_file(ROOT / "datasets" / "mini_eval" / "images" / "000000090108.jpg")
    if frame is None:
        raise RuntimeError("Diagnostic image missing")
    references = json.loads((ROOT / "datasets" / "mini_eval" / "references.json").read_text(encoding="utf-8"))
    ref = references["references"][0]
    reference = read_image_file(ROOT / "datasets" / "mini_eval" / ref["image_file"])
    classes = ["bottle", "cup", "cell phone"]
    rows = []
    for mode, engine_type, model in [
        ("text", OpenVocabEngine, "yoloe-26s-seg.pt"),
        ("sahi", OpenVocabEngine, "yoloe-26s-seg.pt"),
        ("visual_savpe", VisualPromptEngine, "yoloe-26s-seg.pt"),
        ("dino_gui", GroundingDINOEngine, "IDEA-Research/grounding-dino-tiny"),
        ("dino_legacy_evaluation", GroundingDinoEngine, "IDEA-Research/grounding-dino-tiny"),
    ]:
        print(f"Starting {mode}", flush=True)
        engine = engine_type()
        try:
            status = engine.load_model(model, "CUDA")
            if status != "SUCCESS":
                raise RuntimeError(f"{mode}: {status}")
            engine.imgsz = 640
            if mode == "visual_savpe":
                status = engine.set_visual_prompt(reference, ref["bbox_xyxy"], ref["class_name"])
                if status != "SUCCESS":
                    raise RuntimeError(status)
            torch.cuda.synchronize()
            start = time.perf_counter()
            rendered, detections, detail = engine.process_frame(
                frame, conf=0.25, iou=0.5, class_names=classes,
                tracker=ByteTrackManager(), max_det=100,
                sliced=mode == "sahi", slice_size=320, max_slices=9,
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            ok, encoded = cv2.imencode(".jpg", rendered)
            if not ok:
                raise RuntimeError("Could not save preview")
            encoded.tofile(str(output / f"{mode}.jpg"))
            rows.append({"mode": mode, "device": engine.device,
                         "detections": len(detections),
                         "labels": [d.display_label for d in detections],
                         "status": detail, "cold_process_ms": round(elapsed, 2),
                         "allocated_mib": round(torch.cuda.memory_allocated() / 1024 ** 2, 2)})
            print(f"PASS {mode}: {len(detections)} boxes on {engine.device}", flush=True)
        finally:
            engine.unload()
        # The next engine can only be allocated after the old one has unloaded.
        if engine.model is not None:
            raise RuntimeError("Model was not released")
    (output / "cuda_smoke_report.json").write_text(json.dumps({
        "notice": "Functional smoke only; cold timing, no new AP or stable speed measurement",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "available_ram_mib": psutil.virtual_memory().available // 1024 ** 2,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
