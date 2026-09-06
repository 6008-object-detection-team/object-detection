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
from video_worker import VideoWorker


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
        self.selected_video_source, self.current_video_label = 0, None
        build_ui(self)
        import torch
        self.device_combo.setCurrentText("CUDA" if torch.cuda.is_available() else "CPU")

    def on_source_changed(self, index):
        if index == 0:
            self.selected_video_source = 0
            self.file_path_label.hide()
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if path and os.path.exists(path):
            self.selected_video_source = path
            self.file_path_label.setText(f"已选择：{os.path.basename(path)}")
            self.file_path_label.show()
        else:
            self.source_combo.setCurrentIndex(0)

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
        if self.worker:
            self.worker.class_names = parse_prompts(text)

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
        if not self.stop_video_stream():
            return
        self.dce_engine.reset()
        self.worker = VideoWorker(self.engine, self.dce_engine, self.selected_video_source)
        self.update_detection_options()
        self.worker.class_names, self.worker.visual_template = prompts, template
        self.worker.use_low_light_enhancement = self.dce_checkbox.isChecked()
        self.current_video_label = target_label
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
        QMessageBox.critical(self, "视频源/模型错误", message)
        self.stop_video_stream()

    @Slot(str)
    def show_video_info(self, message):
        if self.sender() is not self.worker:
            return
        QMessageBox.information(self, "提示", message)
        self.stop_video_stream()

    def stop_video_stream(self):
        if self.worker:
            if not self.worker.stop():
                self.statusBar().showMessage("正在等待当前模型推理完成，请稍后再操作…")
                return False
            self.worker = None
        if self.current_video_label:
            self.current_video_label.clear()
            self.current_video_label.setText("视频画面预览（已停止）")
        return True

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
