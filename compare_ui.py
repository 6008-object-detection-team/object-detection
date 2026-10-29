"""Tab 1: pin-and-compare two models on the current still image, side by side.

Reuses the app's single-model invariant: the two selected checkpoints are
loaded, run and released one at a time through the existing load/preflight
path (``AIApp._load_model_path``) and the shared guarded detection path
(``auto_annotation.detect_candidates``), never held in memory together.
"""
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

from app_utils import parse_prompts
from auto_annotation import detect_candidates, read_image_with_sha, weights_sha256
from engines import GroundingDINOEngine, PromptFreeEngine
from overlay_labels import draw_label
from prompt_modes import resolve_detection_model
from i18n import message, exception_message, render
from ui_language import QFileDialog, language_for

MATCH_IOU = 0.5
AGREE_COLOR = (90, 220, 90)     # BGR: both models agree on this box
DIFFER_COLOR = (40, 40, 235)    # BGR: only one model found it, or the classes disagree


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_counts(boxes):
    counts = {}
    for box in boxes:
        counts[box["label"]] = counts.get(box["label"], 0) + 1
    return counts


def match_boxes(boxes_a, boxes_b, iou_threshold=MATCH_IOU):
    """Greedy same-class, best-IoU matching.

    A box with no same-class counterpart at IoU >= *iou_threshold* counts as
    a difference (an unmatched box, or a class mismatch) on its own side.
    """
    remaining = list(range(len(boxes_b)))
    matched, only_a = [], []
    for index, box in enumerate(boxes_a):
        best_j, best_iou = -1, 0.0
        for candidate in remaining:
            if boxes_b[candidate]["label"] != box["label"]:
                continue
            iou = box_iou(box["xyxy"], boxes_b[candidate]["xyxy"])
            if iou > best_iou:
                best_iou, best_j = iou, candidate
        if best_j >= 0 and best_iou >= iou_threshold:
            matched.append((index, best_j, best_iou))
            remaining.remove(best_j)
        else:
            only_a.append(index)
    return matched, only_a, remaining


def header_strip(width, text, height=32):
    """BGR title strip drawn with Qt, so Chinese text renders with any OpenCV build.

    OpenCV's Hershey fonts turn non-ASCII characters into '?'. Needs a
    QGuiApplication, which the dialog always has.
    """
    width = max(1, int(width))
    image = QImage(width, height, QImage.Format_RGB888)
    image.fill(QColor(32, 32, 32))
    painter = QPainter(image)
    try:
        font = painter.font()
        font.setPixelSize(15)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        room = max(1, width - 16)
        painter.drawText(QRect(8, 0, room, height), int(Qt.AlignLeft | Qt.AlignVCenter),
                         painter.fontMetrics().elidedText(str(text), Qt.ElideRight, room))
    finally:
        painter.end()
    rows = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(height, image.bytesPerLine())
    return rows[:, :width * 3].reshape(height, width, 3)[..., ::-1].copy()


def render_panel(frame, boxes, differing, title):
    """BGR image (with a title strip) with every box drawn; *differing* indexes are highlighted."""
    output = frame.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(value)) for value in box["xyxy"])
        color = DIFFER_COLOR if index in differing else AGREE_COLOR
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        score = box.get("score")
        text = box["label"] if score is None else f"{box['label']}: {score:.2f}"
        draw_label(output, text, x1, y1, color)
    return np.vstack([header_strip(output.shape[1], title), output])


def compose_comparison(frame, result_a, result_b, language="zh"):
    """(composite BGR image, summary dict) for two {"title", "boxes"} results on the same frame."""
    matched, only_a, only_b = match_boxes(result_a["boxes"], result_b["boxes"])
    counts_a, counts_b = class_counts(result_a["boxes"]), class_counts(result_b["boxes"])

    def title(result):
        return render(message("{title} | {count} 框", title=result["title"], count=len(result["boxes"])), language)

    panel_a = render_panel(frame, result_a["boxes"], only_a, title(result_a))
    panel_b = render_panel(frame, result_b["boxes"], only_b, title(result_b))
    height = max(panel_a.shape[0], panel_b.shape[0])

    def pad(panel):
        if panel.shape[0] == height:
            return panel
        filler = np.full((height - panel.shape[0], panel.shape[1], 3), 32, np.uint8)
        return np.vstack([panel, filler])

    gap = np.full((height, 4, 3), 255, np.uint8)
    composite = np.hstack([pad(panel_a), gap, pad(panel_b)])
    summary = {"matched": len(matched), "only_a": len(only_a), "only_b": len(only_b),
              "counts_a": counts_a, "counts_b": counts_b}
    return composite, summary


def _to_pixmap(bgr_image):
    rgb = bgr_image[..., ::-1].copy()
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy())


class CompareModelsDialog(QDialog):
    """Side-by-side comparison of two models on the current Tab 1 / Tab 2 image."""

    def __init__(self, app):
        super().__init__()  # app stays a plain reference, like the other tab panels (never the Qt parent)
        self.app = app
        self.results = self.frame = self._composite = None
        self.setWindowTitle("模型对比")
        self.resize(1100, 720)
        root = QVBoxLayout(self)

        picker = QHBoxLayout()
        self.model_a = QComboBox()
        self.model_b = QComboBox()
        for combo in (self.model_a, self.model_b):
            for index in range(app.model_combo.count()):
                combo.addItem(app.model_combo.itemText(index), app.model_combo.itemData(index))
        if self.model_b.count() > 1:
            self.model_b.setCurrentIndex(1)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("留空 = 无提示检测（同规格 PF 权重）；或输入 bottle, cup, cell phone")
        picker.addWidget(QLabel("模型 A:")); picker.addWidget(self.model_a)
        picker.addWidget(QLabel("模型 B:")); picker.addWidget(self.model_b)
        picker.addWidget(QLabel("提示词:")); picker.addWidget(self.prompt_input, 1)
        root.addLayout(picker)

        hint = QLabel(
            "两个模型使用左侧相同的置信度、框数、推理尺寸、SAHI / 分块与暗光设置，在当前图片上逐个加载、推理、释放，"
            "从不同时占用显存。红框为两模型不一致的框（IoU < 0.5 或类别不同）。")
        hint.setWordWrap(True)
        root.addWidget(hint)

        actions = QHBoxLayout()
        self.run_button = QPushButton("开始对比")
        self.run_button.clicked.connect(self.run_comparison)
        self.export_button = QPushButton("导出 PNG…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_png)
        actions.addWidget(self.run_button); actions.addWidget(self.export_button); actions.addStretch(1)
        root.addLayout(actions)

        images = QHBoxLayout()
        self.label_a, self.label_b = QLabel(), QLabel()
        for label in (self.label_a, self.label_b):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(420, 320)
            label.setStyleSheet("background:#202020; color:white;")
            images.addWidget(label)
        root.addLayout(images)

        self.summary = QLabel(); self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        self.status = QLabel(); self.status.setWordWrap(True)
        root.addWidget(self.status)
        # Apply the app's current language immediately and keep switching with it,
        # the same way AIApp.__init__ captures the rest of the window.
        ui_language = getattr(app, "ui_language", None)
        if ui_language is not None:
            ui_language.capture(self)

    def release(self):
        """Free a closed dialog: UiLanguage holds strong references to every captured widget.

        The dialog and all of its children are forgotten together (forgetting only the
        dialog would leave bindings to children that Qt then deletes), and the frame,
        composite and pixmaps (about 110 MB for a 4032x3024 photo) are dropped.
        """
        ui_language = getattr(self.app, "ui_language", None)
        if ui_language is not None:
            for widget in [self, *self.findChildren(QWidget)]:
                ui_language.forget(widget)
        self.results = self.frame = self._composite = None
        self.label_a.clear()
        self.label_b.clear()
        self.deleteLater()

    # Helpers ---------------------------------------------------------------
    def notify(self, value):
        translator = language_for(self.app)
        self.status.setText(render(value, translator.language if translator else "zh"))

    def _current_frame(self):
        app = self.app
        if app.selected_input_kind not in ("image", "image_folder"):
            return None, message("请先在左侧选择单张图片或图片文件夹作为输入源。")
        source = str(app.selected_input_source)
        if not os.path.isfile(source):
            return None, message("图片不可用：{path}", path=source)
        frame, _ = read_image_with_sha(source)
        if frame is None:
            return None, message("无法读取图片：{path}", path=source)
        return frame, None

    def _run_options(self):
        app = self.app
        return {"conf": app.conf_slider.value() / 100.0, "iou": 0.45, "max_det": app.max_det_spin.value(),
                "imgsz": app.resolution_combo.currentData(), "tiled": app.tile_checkbox.isChecked(),
                "sliced": app.sahi_checkbox.isChecked(), "slice_size": app.slice_size_combo.currentData(),
                "slice_overlap": 0.2, "max_slices": 9, "low_light": app.dce_checkbox.isChecked(),
                "supplement_dark": bool(app.dark_mode_combo.currentData()),
                "enhancement_strength": app.strength_slider.value() / 100.0}

    @staticmethod
    def _resolve_path(selected, prompts):
        """The checkpoint that actually runs; DINO and modified weights have no prompt-free
        version, so an empty prompt raises instead of falling back to fixed classes."""
        return resolve_detection_model(selected, prompt_free=not prompts)

    def _run_one(self, path, prompts, options):
        app = self.app
        self.notify(message("正在加载 {path}…", path=path))
        QApplication.processEvents()
        if not app._load_model_path(path):
            raise RuntimeError(message("模型加载失败：{path}", path=path))
        try:
            engine = app.engine
            class_names = [] if isinstance(engine, PromptFreeEngine) else list(prompts)
            self.notify(message("正在对图片推理：{path}…", path=path))
            QApplication.processEvents()
            boxes, status = detect_candidates(engine, self.frame, dict(options, class_names=class_names), app.dce_engine)
            # Titled by the checkpoint that ran: a prompt-free entry with a prompt runs its text sibling.
            return {"path": path, "title": Path(path).name,
                   "sha256": app.active_model_sha256 or weights_sha256(path), "boxes": boxes, "status": status}
        finally:
            app.engine.unload()
            app.active_model_path, app.active_model_sha256 = "", ""
            app._update_model_indicator()

    # Actions -----------------------------------------------------------------
    def run_comparison(self):
        app = self.app
        if app.training_active or app.annotation_busy():
            self.notify(message("训练或自动标注进行中，请先等待结束后再对比模型。"))
            return
        frame, error = self._current_frame()
        if error:
            self.notify(error)
            return
        prompts = parse_prompts(self.prompt_input.text())
        try:
            path_a = self._resolve_path(self.model_a.currentData(), prompts)
            path_b = self._resolve_path(self.model_b.currentData(), prompts)
        except ValueError as exc:
            self.notify(message("对比未完成：{error}", error=exception_message(exc)))
            return
        if path_a == path_b:
            # Compared by the checkpoint that runs, not by the list entry.
            self.notify(message("两个选项实际运行同一个权重（{path}），请选择两个不同的模型再开始对比。", path=path_a))
            return
        if not app.stop_video_stream():
            self.notify(message("正在等待当前模型推理完成，请稍后再操作…"))
            return
        self.frame = frame
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        original_index = app.model_combo.currentIndex()
        options = self._run_options()
        try:
            result_a = self._run_one(path_a, prompts, options)
            result_b = self._run_one(path_b, prompts, options)
        except Exception as exc:
            self.notify(message("对比未完成：{error}", error=exception_message(exc)))
            self.run_button.setEnabled(True)
            if original_index >= 0:
                app.model_combo.setCurrentIndex(original_index)
            return
        self.results = (result_a, result_b)
        self.show_results()
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)
        if original_index >= 0:
            app.model_combo.setCurrentIndex(original_index)
        self.notify(message("对比完成：{a} 共 {boxes_a} 框，{b} 共 {boxes_b} 框。",
                            a=result_a["title"], boxes_a=len(result_a["boxes"]),
                            b=result_b["title"], boxes_b=len(result_b["boxes"])))

    def show_results(self):
        result_a, result_b = self.results
        translator = language_for(self.app)
        language = translator.language if translator else "zh"
        composite, summary = compose_comparison(self.frame, result_a, result_b, language)
        self._composite = composite
        pixmap = _to_pixmap(composite)
        half = pixmap.width() // 2
        self.label_a.setPixmap(pixmap.copy(0, 0, half, pixmap.height()).scaled(
            self.label_a.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label_b.setPixmap(pixmap.copy(half, 0, pixmap.width() - half, pixmap.height()).scaled(
            self.label_b.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.summary.setText(render(message(
            "一致框 {matched} | {a} 独有 {only_a} | {b} 独有 {only_b}（IoU < 0.5 或类别不同视为不一致）\n"
            "{a} 类别统计：{counts_a}\n{b} 类别统计：{counts_b}",
            matched=summary["matched"], a=result_a["title"], only_a=summary["only_a"],
            b=result_b["title"], only_b=summary["only_b"],
            counts_a=", ".join(f"{k}:{v}" for k, v in summary["counts_a"].items()) or "-",
            counts_b=", ".join(f"{k}:{v}" for k, v in summary["counts_b"].items()) or "-"), language))

    def export_png(self):
        if self.results is None or self._composite is None:
            return
        default = str(Path.home() / f"model_compare_{datetime.now():%Y%m%d_%H%M%S}.png")
        path, _ = QFileDialog.getSaveFileName(self, "导出对比图", default, "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            ok, encoded = cv2.imencode(".png", self._composite)
            if not ok:
                raise RuntimeError(message("PNG 编码失败"))
            encoded.tofile(path)
            self.notify(message("已导出对比图：{path}", path=path))
        except Exception as exc:
            self.notify(message("导出未完成：{error}", error=exception_message(exc)))


def open_compare_dialog(app):
    """Minimal ui.py hook target: build and show the dialog modally, then free it."""
    dialog = CompareModelsDialog(app)
    try:
        dialog.exec()
    finally:
        dialog.release()
    return dialog
