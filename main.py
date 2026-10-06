"""Application entry point and controller for the AI detection desktop app."""
import os
import sys
import gc
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    dll_dir = os.path.join(sys.prefix, "Library", "bin")
    if dll_dir and os.path.exists(dll_dir):
        _dll_directory = os.add_dll_directory(dll_dir)

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from app_utils import parse_prompts
from engines import GroundingDINOEngine, OpenVocabEngine
from visual_engine import VisualPromptEngine
from enhancement import AdaptiveLowLightEngine
from ui import build_ui
from video_worker import IMAGE_SUFFIXES, VideoWorker
from live_video import DisplayMetrics


class AIApp(QMainWindow):
    """Coordinates the UI with model loading, video input, and inference."""

    MODELS = {
        "YOLOE-26S（实时方案起点）": "yoloe-26s-seg.pt",
        "YOLOE-26L（精度对照）": "yoloe-26l-seg.pt",
        "YOLOE-26X（精度优先，较慢）": "yoloe-26x-seg.pt",
        "YOLOE-11L（旧模型对照）": "yoloe-11l-seg.pt",
        "Grounding DINO Tiny（开放词汇对照）": "IDEA-Research/grounding-dino-tiny",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 多模态检测系统 · 小目标与自适应暗光增强")
        self.resize(1400, 850)
        self.engine, self.dce_engine, self.worker = OpenVocabEngine(), AdaptiveLowLightEngine(), None
        self.selected_input_source, self.selected_input_kind = 0, "camera"
        self.image_paths, self.image_index, self.image_folder = [], -1, None
        self.current_video_label, self.active_detection_request = None, None
        build_ui(self)
        import torch
        self.device_combo.setCurrentText("CUDA" if torch.cuda.is_available() else "CPU")

    def on_source_changed(self, index):
        if not self.stop_video_stream():
            return
        kind = self.source_combo.itemData(index)
        self.image_paths, self.image_index, self.image_folder = [], -1, None
        self.image_navigation.hide()

        if kind == "camera":
            self.selected_input_source, self.selected_input_kind = 0, "camera"
            self.file_path_label.hide()
            self._update_recording_controls()
            return
        if kind == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            if path and os.path.isfile(path):
                self.selected_input_source, self.selected_input_kind = path, "video"
                self.file_path_label.setText(f"已选择视频：{os.path.basename(path)}")
                self.file_path_label.show()
                self._update_recording_controls()
                return
        elif kind == "image":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
            )
            if path and os.path.isfile(path):
                self.image_paths, self.image_index = [path], 0
                self.selected_input_source, self.selected_input_kind = path, "image"
                self.file_path_label.setText(f"已选择图片：{os.path.basename(path)}")
                self.file_path_label.show()
                self._update_recording_controls()
                return
        elif kind == "image_folder":
            folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if folder:
                images = sorted(
                    str(path) for path in Path(folder).rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                if images:
                    self.image_paths, self.image_index, self.image_folder = images, 0, folder
                    self.selected_input_source, self.selected_input_kind = images[0], "image_folder"
                    self.image_navigation.show()
                    self._update_image_position()
                    self._update_recording_controls()
                    return
                QMessageBox.warning(self, "没有可用图片", "所选文件夹中没有支持的 JPG、PNG、BMP、WEBP 或 TIFF 图片。")
        self._reset_source_to_camera()

    def _reset_source_to_camera(self):
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(0)
        self.source_combo.blockSignals(False)
        self.selected_input_source, self.selected_input_kind = 0, "camera"
        self.image_paths, self.image_index, self.image_folder = [], -1, None
        self.file_path_label.hide()
        self.image_navigation.hide()
        self._update_recording_controls()

    def _update_recording_controls(self):
        available = self.selected_input_kind in {"camera", "video"}
        running = self.worker is not None and self.worker.isRunning()
        for widget in (self.record_checkbox, self.record_path_input, self.record_path_button):
            widget.setEnabled(available and not running)

    def choose_recording_path(self):
        default = self.record_path_input.text().strip() or str(
            Path(__file__).resolve().parent / "recordings" /
            f"detection_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )
        path, _ = QFileDialog.getSaveFileName(self, "保存检测结果录像", default, "MP4 视频 (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.record_path_input.setText(path)
            self.record_checkbox.setChecked(True)

    def _update_image_position(self):
        if not self.image_paths or not 0 <= self.image_index < len(self.image_paths):
            return
        self.selected_input_source = self.image_paths[self.image_index]
        self.image_position_label.setText(f"{self.image_index + 1} / {len(self.image_paths)}")
        self.file_path_label.setText(
            f"图片文件夹：{self.image_folder}\n当前：{Path(self.selected_input_source).name}"
        )
        self.file_path_label.show()

    def previous_image(self):
        self._move_image(-1)

    def next_image(self):
        self._move_image(1)

    def _move_image(self, offset):
        if len(self.image_paths) < 2:
            return
        request = self.active_detection_request
        if self.worker and not self._stop_input(clear_request=False):
            return
        self.image_index = (self.image_index + offset) % len(self.image_paths)
        self._update_image_position()
        if request:
            self._start_stream(
                request["target_label"], request["prompts"], request["template"],
                visual_reference=request.get("visual_reference"),
                visual_box=request.get("visual_box"),
                visual_label=request.get("visual_label", "object"),
            )

    def apply_settings(self):
        if not self.stop_video_stream():
            return
        model_path = self.MODELS[self.model_combo.currentText()]
        device = self.device_combo.currentText()
        engine = GroundingDINOEngine() if model_path.startswith("IDEA-Research/") else OpenVocabEngine()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Free the previous model before allocating the next one on the GPU.
            self.engine.unload()
            self.engine = engine
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            status = engine.load_model(model_path, device)
        except Exception as exc:
            status = f"模型加载失败：{type(exc).__name__}: {exc}"
        finally:
            QApplication.restoreOverrideCursor()

        if status == "SUCCESS":
            self.engine = engine
            self.engine.imgsz = self.resolution_combo.currentData()
            detail = "\nDINO 文本阈值固定为 0.25，请使用英文目标提示词。" if isinstance(engine, GroundingDINOEngine) else ""
            QMessageBox.information(self, "设置已应用", f"模型已加载：{model_path}\n推理尺寸：{self.engine.imgsz}{detail}\n速度和精度以本机测试结果为准。")
        elif status == "CUDA_UNAVAILABLE":
            QMessageBox.warning(self, "CUDA 不可用", "当前 PyTorch 无法使用 CUDA，请先选择 CPU，或重新安装 CUDA 版 PyTorch。")
        else:
            engine.unload()
            QMessageBox.critical(self, "模型加载失败", status)

    def update_worker_classes(self, text):
        prompts = parse_prompts(text)
        if self.worker and self.current_video_label is self.video_label_tab1:
            self.worker.class_names = prompts
        if self.active_detection_request and self.active_detection_request["target_label"] is self.video_label_tab1:
            self.active_detection_request["prompts"] = prompts

    def update_enhancement_state(self, state):
        if self.worker:
            self.worker.use_low_light_enhancement = state == Qt.Checked.value

    def update_detection_options(self, *_):
        # Both optional paths retain full-image inference; enabling both only
        # increases latency and memory pressure without a clean SAHI comparison.
        if self.sender() is self.sahi_checkbox and self.sahi_checkbox.isChecked():
            self.tile_checkbox.setChecked(False)
        elif self.sender() is self.tile_checkbox and self.tile_checkbox.isChecked():
            self.sahi_checkbox.setChecked(False)
        self.conf_value.setText(f"{self.conf_slider.value() / 100:.2f}")
        self.strength_value.setText(f"{self.strength_slider.value()}%")
        if self.worker:
            self.worker.conf = self.conf_slider.value() / 100.0
            self.worker.max_det = self.max_det_spin.value()
            self.worker.imgsz = self.resolution_combo.currentData()
            self.worker.tiled = self.tile_checkbox.isChecked()
            self.worker.sliced = self.sahi_checkbox.isChecked()
            self.worker.slice_size = self.slice_size_combo.currentData()
            self.worker.slice_overlap, self.worker.max_slices = 0.2, 9
            self.worker.enhancement_strength = self.strength_slider.value() / 100.0
            self.worker.supplement_dark = self.dark_mode_combo.currentData()

    def start_text_stream(self):
        prompts = parse_prompts(self.prompt_input.text())
        if not prompts:
            QMessageBox.warning(self, "需要提示词", "请输入至少一个目标，例如：person，mobile phone，red backpack")
            return
        self._start_stream(self.video_label_tab1, prompts)

    def start_visual_stream(self):
        if isinstance(self.engine, GroundingDINOEngine):
            QMessageBox.warning(self, "模型不支持视觉提示", "Grounding DINO 只支持文本提示。请选择 YOLOE-26S 并应用设置，再上传参考图片框选目标。")
            return
        prompt = self.ref_canvas.get_visual_prompt()
        if prompt is None:
            QMessageBox.warning(self, "需要参考目标", "请上传参考图片并拖拽方框圈出目标。")
            return
        reference, box = prompt
        label = self.ref_class_input.text().strip() or "object"
        self._start_stream(self.video_label_tab2, [label], visual_reference=reference,
                           visual_box=box, visual_label=label)

    def _switch_prompt_engine(self, visual):
        """Move ownership of one loaded YOLOE model between prompt modes."""
        if isinstance(self.engine, GroundingDINOEngine):
            if visual:
                raise ValueError("Grounding DINO 不支持视觉提示，请先加载 YOLOE 模型。")
            return
        was_visual = isinstance(self.engine, VisualPromptEngine)
        if visual == was_visual:
            return
        if was_visual:
            # No encoding on the GUI thread. Force fresh text embeddings when
            # the next worker detects, including when its label is unchanged.
            self.engine.release_visual_prompt()
        replacement = VisualPromptEngine() if visual else OpenVocabEngine()
        # Preserve resource leases/policies together with the model reference.
        replacement.__dict__.update(self.engine.__dict__)
        replacement.current_classes = []
        self.engine.model = None
        self.engine = replacement

    def _start_stream(self, target_label, prompts, template=None, *, visual_reference=None,
                      visual_box=None, visual_label="object"):
        if self.engine.model is None:
            QMessageBox.warning(self, "尚未加载模型", "请先点击“应用设置 / 加载模型”。")
            return
        if self.selected_input_kind in {"image", "image_folder"} and not os.path.isfile(str(self.selected_input_source)):
            QMessageBox.warning(self, "图片不可用", "请选择一个存在的图片文件或图片文件夹。")
            return
        recording_path = None
        if self.selected_input_kind in {"camera", "video"} and self.record_checkbox.isChecked():
            if not self.record_path_input.text().strip():
                self.choose_recording_path()
            recording_path = self.record_path_input.text().strip()
            if not recording_path:
                return
            if Path(recording_path).exists():
                QMessageBox.warning(self, "录像文件已存在", "请重新选择一个未使用的 MP4 文件名，以保留已有录像。")
                return
        if not self._stop_input(clear_request=False):
            return
        self._switch_prompt_engine(visual_reference is not None)
        self.dce_engine.reset()
        worker_kind = "image" if self.selected_input_kind in {"image", "image_folder"} else self.selected_input_kind
        self.worker = VideoWorker(
            self.engine, self.dce_engine, self.selected_input_source, source_kind=worker_kind,
            recording_path=recording_path,
        )
        self.update_detection_options()
        self.worker.class_names, self.worker.visual_template = prompts, template
        self.worker.visual_reference, self.worker.visual_box = visual_reference, visual_box
        self.worker.visual_label = visual_label
        self.worker.use_low_light_enhancement = self.dce_checkbox.isChecked()
        self.current_video_label = target_label
        self.active_detection_request = {
            "target_label": target_label,
            "prompts": list(prompts),
            "template": template,
            "visual_reference": visual_reference,
            "visual_box": visual_box,
            "visual_label": visual_label,
        }
        self.worker.frame_ready.connect(self.update_video_label)
        self._live_metrics = DisplayMetrics()
        self.worker.live_frame_ready.connect(self.update_live_video_label)
        self.worker.error_signal.connect(self.show_video_error)
        self.worker.info_signal.connect(self.show_video_info)
        self.worker.status_signal.connect(self.statusBar().showMessage)
        worker = self.worker
        worker.finished.connect(lambda: self.on_worker_finished(worker))
        self.worker.start()
        self._update_recording_controls()

    def on_worker_finished(self, worker):
        if self.worker is worker:
            self.worker = None
            self._update_recording_controls()
        worker.deleteLater()

    def upload_ref_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ref_canvas.set_image(path)

    @Slot(np.ndarray)
    def update_video_label(self, frame):
        if self.sender() is not self.worker or self.current_video_label is None:
            return
        self._render_video_frame(frame)

    @Slot()
    def update_live_video_label(self):
        worker = self.sender()
        if worker is not self.worker or self.current_video_label is None:
            return
        packet = worker.take_live_frame()
        if packet is None:
            return
        self._render_video_frame(packet.image)
        metrics = self._live_metrics.record(packet, time.perf_counter())
        message = (
            f"显示 {metrics['display_fps']:.1f} FPS | "
            f"读帧→界面 {metrics['latency_ms']:.0f} ms "
            f"(p50 {metrics['p50_ms']:.0f} / p95 {metrics['p95_ms']:.0f}) | "
            f"待检 {metrics['queue_ms']:.0f} ms / 处理 {metrics['processing_ms']:.0f} ms | "
            f"旧帧替换：采集 {packet.capture_dropped} / 显示 {worker.display_dropped} | {packet.status}"
        )
        self.statusBar().showMessage(message)
        self.statusBar().setToolTip(message)

    def _render_video_frame(self, frame):
        height, width, channels = frame.shape
        image = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888)
        self.current_video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.current_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str)
    def show_video_error(self, message):
        if self.sender() is not self.worker:
            return
        QMessageBox.critical(self, "输入源/模型错误", message)
        self.stop_video_stream()

    @Slot(str)
    def show_video_info(self, message):
        if self.sender() is not self.worker:
            return
        QMessageBox.information(self, "提示", message)
        self.stop_video_stream()

    def _stop_input(self, clear_request=True):
        if self.worker:
            if not self.worker.stop():
                self.statusBar().showMessage("正在等待当前模型推理完成，请稍后再操作…")
                return False
            self.worker = None
            self._update_recording_controls()
        if self.current_video_label:
            self.current_video_label.clear()
            self.current_video_label.setText("输入画面预览（已停止）")
            self.current_video_label = None
        if clear_request:
            self.active_detection_request = None
        return True

    def stop_video_stream(self):
        return self._stop_input(clear_request=True)

    def release_model(self):
        if not self.stop_video_stream():
            return False
        self.engine.unload()
        self.statusBar().showMessage("检测已停止，模型和 GPU 使用锁已释放；重新检测前请加载模型。")
        return True

    def closeEvent(self, event):
        if self.release_model():
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    # Ultralytics keeps the text encoder in cwd; use the project for repeatable launches.
    os.chdir(Path(__file__).resolve().parent)
    app = QApplication(sys.argv)
    window = AIApp()
    window.show()
    sys.exit(app.exec())
