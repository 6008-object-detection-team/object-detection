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
from PySide6.QtCore import Qt, Slot, QSignalBlocker
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from app_utils import parse_prompts
from engines import GroundingDINOEngine, OpenVocabEngine, PromptFreeEngine
from prompt_modes import PROMPTED_PAIRS, resolve_detection_model
from model_families import MODEL_FAMILIES, model_family
from visual_engine import VisualPromptEngine
from enhancement import AdaptiveLowLightEngine
from ui import build_ui
from video_worker import IMAGE_SUFFIXES, VideoWorker, read_image_file
from live_video import DisplayMetrics
from i18n import message, exception_message
from ui_language import UiLanguage, QFileDialog, QMessageBox


class AIApp(QMainWindow):
    """Coordinates the UI with model loading, video input, and inference."""

    MODELS = {
        "YOLOE-26S（实时方案起点）": "yoloe-26s-seg.pt",
        "YOLOE-26L（精度对照）": "yoloe-26l-seg.pt",
        "YOLOE-26X（精度优先，较慢）": "yoloe-26x-seg.pt",
        "YOLOE-11L（旧模型对照）": "yoloe-11l-seg.pt",
        "Grounding DINO Tiny（开放词汇对照）": "IDEA-Research/grounding-dino-tiny",
        "YOLOE-26S 无提示版（4585 类）": "yoloe-26s-seg-pf.pt",
        "YOLOE-26L 无提示版（4585 类）": "yoloe-26l-seg-pf.pt",
        "YOLOE-26X 无提示版（4585 类，较慢）": "yoloe-26x-seg-pf.pt",
        "YOLOE-11L 无提示版（4585 类）": "yoloe-11l-seg-pf.pt",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 多模态检测系统 · 小目标与自适应暗光增强")
        self.resize(1400, 850)
        self.engine, self.dce_engine, self.worker = OpenVocabEngine(), AdaptiveLowLightEngine(), None
        self.selected_input_source, self.selected_input_kind = 0, "camera"
        self.image_paths, self.image_index, self.image_folder = [], -1, None
        self.current_video_label, self.active_detection_request = None, None
        self.MODELS = dict(type(self).MODELS)
        self.active_model_path = ""
        self.active_model_sha256 = ""
        self.training_active = False
        self._feedback_retest = None
        self._feedback_cached = None
        self._feedback_versions = {}
        self.ui_language = UiLanguage(self)
        build_ui(self)
        self.ui_language.capture(self)
        self._update_model_indicator()
        self.ui_language.changed.connect(self._retranslate_dynamic_content)
        import torch
        self.device_combo.setCurrentText("CUDA" if torch.cuda.is_available() else "CPU")

    def on_language_changed(self, index):
        self.ui_language.set_language(self.language_combo.itemData(index))

    def _retranslate_dynamic_content(self, *_):
        # Display names never identify a model internally. Item data remains its path.
        self.MODELS = {self.model_combo.itemText(index): self.model_combo.itemData(index)
                       for index in range(self.model_combo.count())}
        self.ref_canvas.update()
        self.feedback_panel.retranslate()
        self.annotation_panel.retranslate()

    def show_status(self, value):
        self.ui_language.bind(self.statusBar(), value, "status")

    def detection_error(self, title, detail, *, critical=False):
        if self._feedback_retest is not None:
            self.feedback_panel.notify(message("自动重测未完成：{error}", error=detail))
        else:
            (QMessageBox.critical if critical else QMessageBox.warning)(self, title, detail)

    @Slot(object)
    def on_worker_status(self, value):
        if self.sender() is self.worker:
            self.show_status(value)

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
                self.ui_language.bind(self.file_path_label, message("已选择视频：{name}", name=os.path.basename(path)))
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
                self.ui_language.bind(self.file_path_label, message("已选择图片：{name}", name=os.path.basename(path)))
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
        self.ui_language.bind(self.file_path_label, message(
            "图片文件夹：{folder}\n当前：{name}", folder=self.image_folder,
            name=Path(self.selected_input_source).name))
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

    def annotation_busy(self):
        panel = getattr(self, "annotation_panel", None)
        return panel is not None and panel.running

    def apply_settings(self):
        if self.training_active:
            QMessageBox.warning(self, "训练进行中", "请先等待纠错训练完成或请求停止，不同时加载推理模型。")
            return
        if self.annotation_busy():
            QMessageBox.warning(self, "自动标注进行中", "自动标注与检测共用一个模型、串行运行；请先在第 4 页停止自动标注。")
            return
        if not self.stop_video_stream():
            return
        self._load_model_path(self.model_combo.currentData())

    def _load_model_path(self, model_path):
        """Called only after a worker stops; release before loading a sibling."""
        if model_path in self._feedback_versions:
            from feedback import sha256_file
            try:
                self.refresh_feedback_models()
            except Exception as exc:
                self.show_status(exception_message(exc))
                return False
            if model_path not in self._feedback_versions:
                self.show_status(message("修改版当前不可用；不会替换成原版，请检查训练日志。"))
                return False
            if sha256_file(model_path) != self._feedback_versions[model_path]["checkpoint_sha256"]:
                self.detection_error("版本校验失败", "修改版权重校验失败，不加载；请检查日志或重新更新修改版。")
                return False
        device = self.device_combo.currentText()
        engine = (PromptFreeEngine() if model_path in PROMPTED_PAIRS else
                  GroundingDINOEngine() if model_path.startswith("IDEA-Research/") else OpenVocabEngine())
        self.show_status(message("正在加载模型：{path}（先释放旧模型，不改变运行设备）…", path=model_path))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Free the previous model before allocating the next one on the GPU.
            self.engine.unload()
            self.active_model_path = ""
            self.active_model_sha256 = ""
            self.engine = engine
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            status = engine.load_model(model_path, device)
        except Exception as exc:
            status = message("模型加载失败：{kind}: {error}", kind=type(exc).__name__, error=exception_message(exc))
        finally:
            QApplication.restoreOverrideCursor()

        if status == "SUCCESS":
            self.active_model_path = model_path
            self.active_model_sha256 = self._feedback_versions.get(model_path, {}).get("checkpoint_sha256", "")
            self.engine = engine
            self.engine.imgsz = self.resolution_combo.currentData()
            index = self.model_combo.findData(model_path)
            if index >= 0:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(index)
                self.model_combo.blockSignals(False)
            self._update_model_indicator()
            detail = message("\nDINO 文本阈值固定为 0.25，请使用英文目标提示词。") if isinstance(engine, GroundingDINOEngine) else ""
            self.show_status(message(
                "模型已加载：{path}\n推理尺寸：{size}{detail}\n速度和精度以本机测试结果为准。",
                path=model_path, size=self.engine.imgsz, detail=detail))
            return True
        elif status == "CUDA_UNAVAILABLE":
            self.detection_error("CUDA 不可用", "当前 PyTorch 无法使用 CUDA，请先选择 CPU，或重新安装 CUDA 版 PyTorch。")
        else:
            engine.unload()
            self.detection_error("模型加载失败", status, critical=True)
        self._update_model_indicator()
        return False

    def _update_model_indicator(self):
        if not self.active_model_path or self.engine.model is None:
            self.ui_language.bind(self.detection_mode_label, "当前模型：未加载；开始检测时按提示内容选择模式。")
            return
        mode = (message("无提示（{count} 类）", count=self.engine.vocabulary_size)
                if isinstance(self.engine, PromptFreeEngine) else
                message("视觉提示") if isinstance(self.engine, VisualPromptEngine) else message("文字提示"))
        self.ui_language.bind(self.detection_mode_label, message("当前模式：{mode}\n实际模型：{path}",
                              mode=mode, path=Path(self.active_model_path).name))

    def _ensure_detection_model(self, prompts, *, visual=False):
        """Resolve before allocating anything; never turn a PF head into SAVPE."""
        try:
            path = resolve_detection_model(self.model_combo.currentData(), prompt_free=not prompts and not visual)
            if visual and path.startswith("IDEA-Research/"):
                raise ValueError(message("Grounding DINO 不支持视觉提示，请先加载 YOLOE 模型。"))
        except ValueError as exc:
            self.show_status(exception_message(exc))
            self.ui_language.bind(self.prompt_mode_hint, exception_message(exc))
            return False
        self.ui_language.bind(self.prompt_mode_hint, "留空：真正无提示检测；输入文字：指定目标。切换权重前会停止检测，只加载一个模型。")
        target_device = "cpu" if self.device_combo.currentText() == "CPU" else "cuda:0"
        if path in self._feedback_versions:
            # A stable filename can now have new bytes: refresh its committed
            # identity so selecting the same modified model reloads new weights.
            try:
                self.refresh_feedback_models()
            except Exception as exc:
                self.show_status(exception_message(exc))
                return False
            if path not in self._feedback_versions:
                self.show_status(message("修改版当前不可用；不会替换成原版，请检查训练日志。"))
                return False
        expected_sha = self._feedback_versions.get(path, {}).get("checkpoint_sha256", "")
        if self.engine.model is None or self.active_model_path != path or self.engine.device != target_device \
                or (expected_sha and expected_sha != self.active_model_sha256):
            if not self._load_model_path(path):
                return False
        self.engine.imgsz = self.resolution_combo.currentData()
        return True

    def update_worker_classes(self, text):
        prompts = parse_prompts(text)
        text_active = self.current_video_label is getattr(self, "video_label_tab1", None)
        if text_active and (self.worker or self.active_detection_request):
            # Never pass newly typed text into a PF head, or let a cleared
            # prompt fall back to the text engine's legacy seven-class preset.
            if isinstance(self.engine, PromptFreeEngine) != (not prompts):
                if self.stop_video_stream():
                    self.show_status(message("提示模式已改变；点击“开始检测”切换对应权重。不会同时加载两个模型。"))
                return
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
        self._start_stream(self.video_label_tab1, prompts)

    def start_visual_stream(self):
        if isinstance(self.engine, GroundingDINOEngine) or self.model_combo.currentData().startswith("IDEA-Research/"):
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
        if isinstance(self.engine, PromptFreeEngine):
            if visual:
                raise ValueError("无提示模型不能接收参考图；请先加载对应的文字/视觉提示权重。")
            return
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
        # The worker has stopped: free the previous mode's predictor now rather
        # than when Python's cyclic GC next runs (the visual release does so too).
        self.engine._discard_predictor()
        self._update_model_indicator()

    def _start_stream(self, target_label, prompts, template=None, *, visual_reference=None,
                      visual_box=None, visual_label="object"):
        if self.training_active:
            QMessageBox.warning(self, "训练进行中", "纠错训练和检测串行运行，请先等待训练结束。")
            return
        if self.annotation_busy():
            QMessageBox.warning(self, "自动标注进行中", "自动标注与检测共用一个模型、串行运行；请先在第 4 页停止自动标注。")
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
        if not self._ensure_detection_model(prompts, visual=visual_reference is not None):
            self.active_detection_request = None
            return
        self._switch_prompt_engine(visual_reference is not None)
        self._update_model_indicator()
        self.dce_engine.reset()
        worker_kind = "image" if self.selected_input_kind in {"image", "image_folder"} else self.selected_input_kind
        self.worker = VideoWorker(
            self.engine, self.dce_engine, self.selected_input_source, source_kind=worker_kind,
            recording_path=recording_path,
        )
        self.update_detection_options()
        self.worker.class_names, self.worker.visual_template = prompts, template
        self.worker.feedback_metadata = {"model": self.active_model_path,
                                         "model_family": self._feedback_versions.get(self.active_model_path, {}).get("model_family") or model_family(self.active_model_path),
                                         "model_sha256": self.active_model_sha256,
                                         "engine": type(self.engine).__name__,
                                         "detection_mode": "prompt_free" if isinstance(self.engine, PromptFreeEngine) else "visual" if visual_reference is not None else "text",
                                         "vocabulary_size": self.engine.vocabulary_size if isinstance(self.engine, PromptFreeEngine) else len(prompts),
                                         "inference_options": self.engine.inference_options}
        self._feedback_cached = None
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
        self.worker.status_signal.connect(self.on_worker_status)
        worker = self.worker
        worker.finished.connect(lambda: self.on_worker_finished(worker))
        self.worker.start()
        self._update_recording_controls()

    def on_worker_finished(self, worker):
        if self.worker is worker:
            self._feedback_cached = worker.get_feedback_snapshot()
            self.worker = None
            self._update_recording_controls()
            if self._feedback_retest is not None:
                self.finish_feedback_retest()
        worker.deleteLater()

    def retest_feedback(self, quick):
        """Real inference only: accepts an image and labels, never human boxes."""
        from feedback import image_id, sha256_file, safe_path
        from modified_models import ModifiedModels
        try:
            if not self.stop_video_stream():
                raise RuntimeError("正在等待检测停止，尚未更新；请稍后再次点击。")
            registry = ModifiedModels(self.feedback_panel.store.root)
            row = registry.current(quick["family"])
            if row is None:
                raise ValueError("修改版当前不可用；不会替换成原版，请检查训练日志。")
            image_path = safe_path(self.feedback_panel.store.root, Path(quick["image_path"]).relative_to(self.feedback_panel.store.root))
            source = str(image_path)
            # Keep folder navigation when the original still has identical pixels.
            original = str(quick["record"]["provenance"].get("source", ""))
            if Path(original).is_file():
                original_image = read_image_file(original)
                if original_image is not None and image_id(original_image) == quick["record"]["image_id"]:
                    source = original
            self._feedback_retest = dict(quick, expected_sha256=row["checkpoint_sha256"],
                code_sha256={name: sha256_file(Path(__file__).parent/name) for name in
                             ("main.py", "feedback_ui.py", "feedback.py", "engines.py", "video_worker.py")})
            self.feedback_panel.quick_button.setEnabled(False)
            self.refresh_feedback_models()
            self.model_combo.setCurrentIndex(self.model_combo.findData(str(registry.slot(quick["family"])/"modified.pt")))
            self.selected_input_source = source
            if source in self.image_paths:
                self.selected_input_kind = "image_folder"
                self.image_index = self.image_paths.index(source)
                self._update_image_position()
                source_kind = "image_folder"
            else:
                self.selected_input_kind = "image"
                self.image_paths, self.image_index, self.image_folder = [], -1, None
                self.image_navigation.hide()
                self.ui_language.bind(self.file_path_label, message("已选择图片：{name}", name=Path(source).name))
                self.file_path_label.show()
                source_kind = "image"
            with QSignalBlocker(self.source_combo):
                self.source_combo.setCurrentIndex(self.source_combo.findData(source_kind))
            self.prompt_input.setText(", ".join(quick["classes"]))
            self.tabs.setCurrentIndex(0)
            self.start_text_stream()
            if self.worker is None:
                self._feedback_retest = None
                self.feedback_panel.quick_button.setEnabled(True)
                self.feedback_panel.notify(message("模型已更新，但自动重测未启动；请查看状态提示。"))
        except Exception as exc:
            self._feedback_retest = None
            self.feedback_panel.quick_button.setEnabled(True)
            self.feedback_panel.notify(message("自动重测未完成：{error}", error=exception_message(exc)))

    def finish_feedback_retest(self):
        from feedback import atomic_json, image_id, safe_path
        quick, self._feedback_retest = self._feedback_retest, None
        self.feedback_panel.quick_button.setEnabled(True)
        try:
            if self._feedback_cached is None:
                raise RuntimeError("模型已更新，但没有得到重测画面；请查看日志。")
            raw, boxes, metadata = self._feedback_cached
            if image_id(raw) != quick["record"]["image_id"] or metadata.get("model_sha256") != quick["expected_sha256"]:
                raise ValueError("重测图片或权重指纹不一致，不能作为本次更新结果。")
            if quick.get("audit_dir"):
                directory = safe_path(self.feedback_panel.store.root, Path(quick["audit_dir"]).relative_to(self.feedback_panel.store.root))
                atomic_json(directory/"retest.json", {"status": "completed", "scope": "same-image demonstration, not independent evaluation",
                    "image_id": quick["record"]["image_id"], "model_sha256": quick["expected_sha256"],
                    "code_sha256": quick["code_sha256"], "metadata": metadata, "predictions": boxes,
                    "human_boxes_used_in_inference": False})
            self.feedback_panel.notify(message("{name}修改版已更新，本图已自动重测。画面来自新权重推理；同图演示不代表泛化准确率。", name=MODEL_FAMILIES[quick["family"]]))
        except Exception as exc:
            self.feedback_panel.notify(message("自动重测未完成：{error}", error=exception_message(exc)))

    def refresh_feedback_models(self):
        from feedback import FeedbackStore
        current = self.model_combo.currentData()
        store = self.feedback_panel.store if hasattr(self, "feedback_panel") else FeedbackStore()
        versions = store.model_versions()
        self._feedback_versions = {path: row for _, path, row in versions}
        entries = list(type(self).MODELS.items())
        for title, path, row in versions:
            family = row.get("model_family") or model_family(row.get("parent", ""))
            if family:
                title = message("{name}修改版", name=MODEL_FAMILIES[family])
            entries.append((title, path))
        self.ui_language.forget(self.model_combo)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for title, path in entries:
            index = self.model_combo.count()
            self.model_combo.addItem(str(title), path)
            self.ui_language.bind(self.model_combo, title, "itemText", index)
        selected = self.model_combo.findData(current)
        if selected >= 0:
            self.model_combo.setCurrentIndex(selected)
        self.model_combo.blockSignals(False)
        self.MODELS = {self.model_combo.itemText(index): self.model_combo.itemData(index)
                       for index in range(self.model_combo.count())}

    def open_feedback(self):
        snapshot = self.worker.get_feedback_snapshot() if self.worker else self._feedback_cached
        if snapshot is None:
            self.tabs.setCurrentWidget(self.feedback_panel)
            self.feedback_panel.notify(message("先完成一次图片检测或启动视频 / 摄像头，也可在纠错页直接打开图片。"))
            return
        if self.feedback_panel.set_snapshot(snapshot):
            self.tabs.setCurrentWidget(self.feedback_panel)

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
        value = message(
            "显示 {fps:.1f} FPS | 读帧→界面 {latency:.0f} ms (p50 {p50:.0f} / p95 {p95:.0f}) | 待检 {queue:.0f} ms / 处理 {processing:.0f} ms | 旧帧替换：采集 {capture_dropped} / 显示 {display_dropped} | {status}",
            fps=metrics["display_fps"], latency=metrics["latency_ms"], p50=metrics["p50_ms"],
            p95=metrics["p95_ms"], queue=metrics["queue_ms"], processing=metrics["processing_ms"],
            capture_dropped=packet.capture_dropped, display_dropped=worker.display_dropped,
            status=packet.status)
        self.show_status(value)
        self.ui_language.bind(self.statusBar(), value, "toolTip")

    def _render_video_frame(self, frame):
        height, width, channels = frame.shape
        image = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888)
        self.current_video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.current_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(object)
    def show_video_error(self, message):
        if self.sender() is not self.worker:
            return
        self.detection_error("输入源/模型错误", message, critical=True)
        self.stop_video_stream()

    @Slot(object)
    def show_video_info(self, message):
        if self.sender() is not self.worker:
            return
        self.stop_video_stream()
        self.show_status(message)

    def _stop_input(self, clear_request=True):
        if self.worker:
            if not self.worker.stop():
                self.show_status(message("正在等待当前模型推理完成，请稍后再操作…"))
                return False
            self._feedback_cached = self.worker.get_feedback_snapshot()
            self.worker = None
            self._update_recording_controls()
        if self.current_video_label:
            self.current_video_label.clear()
            self.ui_language.bind(self.current_video_label, "输入画面预览（已停止）")
            self.current_video_label = None
        if clear_request:
            self.active_detection_request = None
            if self._feedback_retest is not None:
                self._feedback_retest = None
                self.feedback_panel.quick_button.setEnabled(True)
        return True

    def stop_video_stream(self):
        return self._stop_input(clear_request=True)

    def release_model(self):
        # Stop-and-release covers every inference worker, including auto-annotation.
        if not self.annotation_panel.stop_worker() or not self.stop_video_stream():
            return False
        self.engine.unload()
        self.active_model_path = ""
        self.active_model_sha256 = ""
        self._update_model_indicator()
        self.show_status(message("检测已停止，模型和 GPU 使用锁已释放；重新检测前请加载模型。"))
        return True

    def closeEvent(self, event):
        if self.training_active:
            self.feedback_panel.notify(message("请在纠错页请求停止，并等待进程结束后再关闭。"))
            event.ignore()
            return
        if self.feedback_panel.dirty and not self.feedback_panel.may_replace():
            event.ignore()
            return
        if not self.annotation_panel.may_discard():
            event.ignore()
            return
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
