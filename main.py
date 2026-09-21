"""Application entry point and controller for the AI detection desktop app."""
import os
import sys
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
from engines import OpenVocabEngine
from enhancement import AdaptiveLowLightEngine
from ui import build_ui
from video_worker import IMAGE_SUFFIXES, VideoWorker


class AIApp(QMainWindow):
    """Coordinates the UI with model loading, video input, and inference."""

    MODELS = {
        "YOLOE-26L（实时推荐）": "yoloe-26l-seg.pt",
        "YOLOE-26X（精度优先，较慢）": "yoloe-26x-seg.pt",
        "YOLOE-26S（速度优先）": "yoloe-26s-seg.pt",
        "YOLOE-11L（旧模型对照）": "yoloe-11l-seg.pt",
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
            return
        if kind == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            if path and os.path.isfile(path):
                self.selected_input_source, self.selected_input_kind = path, "video"
                self.file_path_label.setText(f"已选择视频：{os.path.basename(path)}")
                self.file_path_label.show()
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
                request["target_label"], request["prompts"], request["template"]
            )

    def apply_settings(self):
        if not self.stop_video_stream():
            return
        model_path = self.MODELS[self.model_combo.currentText()]
        device = self.device_combo.currentText()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            status = self.engine.load_model(model_path, device)
        finally:
            QApplication.restoreOverrideCursor()

        if status == "SUCCESS":
            self.engine.imgsz = self.resolution_combo.currentData()
            QMessageBox.information(self, "设置已应用", f"模型已加载：{model_path}\n推理尺寸：{self.engine.imgsz}\n暗光增强无需额外模型文件。")
        elif status == "CUDA_UNAVAILABLE":
            QMessageBox.warning(self, "CUDA 不可用", "当前 PyTorch 无法使用 CUDA，请先选择 CPU，或重新安装 CUDA 版 PyTorch。")
        else:
            QMessageBox.critical(self, "模型加载失败", status)

    def update_worker_classes(self, text):
        prompts = parse_prompts(text)
        if self.worker:
            self.worker.class_names = prompts
        if self.active_detection_request and self.active_detection_request["target_label"] is self.video_label_tab1:
            self.active_detection_request["prompts"] = prompts

    def update_enhancement_state(self, state):
        if self.worker:
            self.worker.use_low_light_enhancement = state == Qt.Checked.value

    def update_detection_options(self, *_):
        self.conf_value.setText(f"{self.conf_slider.value() / 100:.2f}")
        self.strength_value.setText(f"{self.strength_slider.value()}%")
        if self.worker:
            self.worker.conf = self.conf_slider.value() / 100.0
            self.worker.imgsz = self.resolution_combo.currentData()
            self.worker.tiled = self.tile_checkbox.isChecked()
            self.worker.enhancement_strength = self.strength_slider.value() / 100.0
            self.worker.supplement_dark = self.dark_mode_combo.currentData()

    def start_text_stream(self):
        prompts = parse_prompts(self.prompt_input.text())
        if not prompts:
            QMessageBox.warning(self, "需要提示词", "请输入至少一个目标，例如：person，mobile phone，red backpack")
            return
        self._start_stream(self.video_label_tab1, prompts)

    def start_visual_stream(self):
        template = self.ref_canvas.get_cropped_template()
        if template is None or template.size == 0:
            QMessageBox.warning(self, "需要参考目标", "请上传参考图片并拖拽方框圈出目标。")
            return
        self._start_stream(self.video_label_tab2, parse_prompts(self.ref_class_input.text()), template)

    def _start_stream(self, target_label, prompts, template=None):
        if self.engine.model is None:
            QMessageBox.warning(self, "尚未加载模型", "请先点击“应用设置 / 加载模型”。")
            return
        if self.selected_input_kind in {"image", "image_folder"} and not os.path.isfile(str(self.selected_input_source)):
            QMessageBox.warning(self, "图片不可用", "请选择一个存在的图片文件或图片文件夹。")
            return
        if not self._stop_input(clear_request=False):
            return
        self.dce_engine.reset()
        worker_kind = "image" if self.selected_input_kind in {"image", "image_folder"} else self.selected_input_kind
        self.worker = VideoWorker(
            self.engine, self.dce_engine, self.selected_input_source, source_kind=worker_kind
        )
        self.update_detection_options()
        self.worker.class_names, self.worker.visual_template = prompts, template
        self.worker.use_low_light_enhancement = self.dce_checkbox.isChecked()
        self.current_video_label = target_label
        self.active_detection_request = {
            "target_label": target_label,
            "prompts": list(prompts),
            "template": template,
        }
        self.worker.frame_ready.connect(self.update_video_label)
        self.worker.error_signal.connect(self.show_video_error)
        self.worker.info_signal.connect(self.show_video_info)
        self.worker.status_signal.connect(self.statusBar().showMessage)
        worker = self.worker
        worker.finished.connect(lambda: self.on_worker_finished(worker))
        self.worker.start()

    def on_worker_finished(self, worker):
        if self.worker is worker:
            self.worker = None
        worker.deleteLater()

    def upload_ref_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ref_canvas.set_image(path)

    @Slot(np.ndarray)
    def update_video_label(self, frame):
        if self.sender() is not self.worker or self.current_video_label is None:
            return
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
        if self.current_video_label:
            self.current_video_label.clear()
            self.current_video_label.setText("输入画面预览（已停止）")
            self.current_video_label = None
        if clear_request:
            self.active_detection_request = None
        return True

    def stop_video_stream(self):
        return self._stop_input(clear_request=True)

    def closeEvent(self, event):
        if self.stop_video_stream():
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
