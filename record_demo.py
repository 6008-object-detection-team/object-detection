"""Record prompted detection on a local video without opening the desktop UI."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="local input video")
    parser.add_argument("--output", required=True, help="new MP4 output")
    parser.add_argument("--engine", choices=["yoloe", "dino"], default="dino")
    parser.add_argument("--model", default=None)
    parser.add_argument("--classes", default="person,bus")
    parser.add_argument("--device", choices=["CPU", "CUDA"], default="CUDA")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()
    import cv2
    from app_utils import parse_prompts
    from engines import GroundingDINOEngine, OpenVocabEngine, validate_detection_options

    validate_detection_options(args.conf, 0.45, args.max_det)
    if args.imgsz < 32 or args.max_frames < 1:
        parser.error("imgsz must be >=32 and max-frames must be positive")
    classes = parse_prompts(args.classes)
    if not classes:
        parser.error("at least one class is required")
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        parser.error(f"output or JSON report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = GroundingDINOEngine() if args.engine == "dino" else OpenVocabEngine()
    model = args.model or ("IDEA-Research/grounding-dino-tiny" if args.engine == "dino" else "yoloe-26s-seg.pt")
    status = engine.load_model(model, args.device)
    if status != "SUCCESS":
        raise RuntimeError(status)
    engine.imgsz = args.imgsz
    capture = cv2.VideoCapture(str(Path(args.source).resolve()))
    writer, rows = None, []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if 0 < fps <= 240 else 20.0
    try:
        if not capture.isOpened():
            raise RuntimeError("cannot open input video")
        for index in range(args.max_frames):
            available, frame = capture.read()
            if not available:
                break
            detections = engine.detect(frame, args.conf, 0.45, classes, max_det=args.max_det)
            annotated = engine.draw_detections(frame, detections)
            text = f"{args.engine.upper()} | prompt: {', '.join(classes)} | conf={args.conf:.2f} | cap={args.max_det}"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), (30, 30, 30), -1)
            cv2.putText(annotated, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("MP4 encoder unavailable")
            writer.write(annotated)
            rows.append({"frame_index": index, "detections": [asdict(item) for item in detections]})
            print(f"frame {index + 1}: {len(detections)} boxes", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not rows:
        raise RuntimeError("input contains no decodable frames")
    report = {"settings": vars(args), "model": model, "output_fps": fps,
              "notice": "Processed frames only, no audio. Playback fps is source fps, not inference throughput.",
              "rows": rows}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
