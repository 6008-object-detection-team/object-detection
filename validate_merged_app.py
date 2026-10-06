"""Exercise real CUDA models through the Qt image-folder and video workflows."""
import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation" / "local_integration_20261006")
    parser.add_argument("--show", action="store_true", help="Show the Qt window during the test")
    args = parser.parse_args()
    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    # This run verifies that the downloaded DINO weights work without network access.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.chdir(ROOT)

    import cv2
    import numpy as np
    import torch
    from PySide6.QtWidgets import QApplication
    from main import AIApp
    from video_worker import read_image_file

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    portrait = read_image_file(ROOT / "mypicture.jpg")
    if portrait is None:
        raise RuntimeError("Test image is missing")
    height = 640
    portrait = cv2.resize(portrait, (round(portrait.shape[1] * height / portrait.shape[0]), height))
    pair = np.concatenate([portrait, portrait], axis=1)
    paths = [out / "多人测试1.jpg", out / "多人测试2.jpg"]
    for index, path in enumerate(paths):
        ok, encoded = cv2.imencode(".jpg", np.roll(pair, index * 2, axis=1))
        if not ok:
            raise RuntimeError("Could not encode the image fixture")
        encoded.tofile(str(path))
    # Use an ASCII relative name for OpenCV's Windows video path API.
    video_name = "worker_smoke.avi"
    previous_cwd = Path.cwd()
    os.chdir(out)
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 10, (pair.shape[1], pair.shape[0]))
    if not writer.isOpened():
        raise RuntimeError("Could not open the video fixture writer")
    try:
        for index in range(4):
            writer.write(np.roll(pair, index * 2, axis=1))
    finally:
        writer.release()
        os.chdir(previous_cwd)

    app = QApplication.instance() or QApplication([])
    window = AIApp()
    window.show()
    messages, runs = [], []

    def dialog(level):
        def record(_parent, title, message, *args, **kwargs):
            messages.append({"level": level, "title": title, "message": message})
            if level in {"error", "warning"}:
                print(f"{level}: {title}: {message}", flush=True)
            return 0
        return record

    def wait_for_worker():
        deadline = time.monotonic() + 120
        while window.worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        if window.worker is not None:
            raise RuntimeError("Worker did not finish within 120 seconds")
        if any(message["level"] == "error" for message in messages):
            raise RuntimeError("A model or input error was reported by the UI")

    with patch("main.QMessageBox.information", side_effect=dialog("info")), patch(
        "main.QMessageBox.critical", side_effect=dialog("error")
    ), patch("main.QMessageBox.warning", side_effect=dialog("warning")):
        for backend_index, tag in [(0, "yoloe_26s"), (4, "grounding_dino_tiny"), (0, "yoloe_return")]:
            window.model_combo.setCurrentIndex(backend_index)
            window.device_combo.setCurrentText("CUDA")
            window.resolution_combo.setCurrentIndex(0)
            window.conf_slider.setValue(40)
            window.apply_settings()
            if window.engine.model is None or window.engine.device != "cuda:0":
                raise RuntimeError(f"{tag} did not load on CUDA")
            if any(message["level"] == "error" for message in messages):
                raise RuntimeError(f"{tag} failed to load")
            records = []
            original_process = window.engine.process_frame

            def recorded_process(*call_args, **call_kwargs):
                started = time.perf_counter()
                result = original_process(*call_args, **call_kwargs)
                records.append({
                    "labels": [item.display_label for item in result[1]],
                    "status": result[2],
                    "process_ms": round((time.perf_counter() - started) * 1000, 2),
                })
                return result

            window.engine.process_frame = recorded_process
            window.source_combo.blockSignals(True)
            window.source_combo.setCurrentIndex(3)
            window.source_combo.blockSignals(False)
            window.selected_input_kind = "image_folder"
            window.image_paths = [str(path) for path in paths]
            window.image_folder, window.image_index = str(out), 0
            window.image_navigation.show()
            window._update_image_position()
            window.prompt_input.setText("person")
            window.start_text_stream()
            wait_for_worker()
            if window.video_label_tab1.pixmap().isNull():
                raise RuntimeError("No detection preview appeared in the UI")
            if set(records[-1]["labels"]) != {"person1", "person2"}:
                raise RuntimeError(f"{tag} failed two-person numbering: {records[-1]}")
            if not window.grab().save(str(out / f"ui_{tag}.png")):
                raise RuntimeError("Could not save the Qt screenshot")
            window.next_image()
            wait_for_worker()
            if window.image_index != 1 or window.image_position_label.text() != "2 / 2":
                raise RuntimeError("Folder navigation did not advance")
            image_records = records.copy()
            if tag != "yoloe_return":
                window.stop_video_stream()
                # VideoCapture's narrow-string path API can read the ASCII relative fixture.
                os.chdir(out)
                window.selected_input_kind, window.selected_input_source = "video", video_name
                window.start_text_stream()
                wait_for_worker()
                os.chdir(ROOT)
                video_records = records[len(image_records):]
                if len(video_records) != 4 or any(set(row["labels"]) != {"person1", "person2"} for row in video_records):
                    raise RuntimeError(f"{tag} failed stable video numbering: {video_records}")
            else:
                video_records = []
            runs.append({"backend": tag, "device": window.engine.device,
                         "image_folder": image_records, "video": video_records})
            print(f"PASS {tag}: image folder, numbered persons, {len(video_records)} video frames", flush=True)
            window.engine.process_frame = original_process
            window.stop_video_stream()
        window.close()
        app.processEvents()

    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "dino_offline": True, "fixture": "Two copies of one portrait; a software test, not identity recognition",
              "runs": runs, "messages": messages}
    (out / "ui_worker_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved validation results in {out}", flush=True)


if __name__ == "__main__":
    main()
