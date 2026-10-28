"""Tab 4: folder auto-annotation with serial detection and human review."""
import json
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QThread, Qt, Signal, QSignalBlocker
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QCheckBox, QListWidget, QProgressBar, QDoubleSpinBox, QSplitter)

from app_utils import parse_prompts
from auto_annotation import (AnnotationSession, delete_session_file, detect_candidates, export_coco,
                             export_yolo, read_image_with_sha, read_session_file, session_cache_path,
                             weights_sha256, write_session)
from engines import PromptFreeEngine
from feedback_ui import CorrectionCanvas
from prompt_modes import resolve_detection_model
from i18n import message, exception_message, render
from resource_safety import MAX_SOURCE_PIXELS, ResourceLimitError
from sliced_inference import validate_slice_plan
from ui_language import QFileDialog, QMessageBox, language_for

STATE_TEXT = {"pending": "待审", "accepted": "已接受", "edited": "已修改", "rejected": "已拒绝"}
STATE_MARK = {"pending": "○", "accepted": "✓", "edited": "✎", "rejected": "✗"}
STATE_COLOR = {"undetected": "#888888", "accepted": "#2e9e4f", "edited": "#2f6fd0",
               "rejected": "#c0392b", "failed": "#d4801a"}
# Start of the per-image message for a SAHI plan above the slice cap (the image is skipped).
SLICE_SKIP_PREFIX = "已跳过（SAHI 切片过多）"
# Run-only presets, recorded in the manifest. "accuracy_first" was measured on the held-out
# COCO-240 subset with text prompts (README, docs/evaluation.md); left-side settings stay as they are.
PRESETS = {
    "accuracy_first": {
        "model": "yoloe-26l-seg.pt",
        "options": {"imgsz": 640, "conf": 0.40, "sliced": True, "slice_size": 320, "slice_overlap": 0.2,
                    "max_slices": 9, "tiled": False, "low_light": False},
    },
}


def local_weights_present(name):
    """A real weights file next to the app; a Git LFS pointer is only a few hundred bytes.

    Checked before loading so that Ultralytics never downloads a substitute file.
    """
    path = Path(__file__).resolve().parent / name
    return path.is_file() and path.stat().st_size >= 1024 * 1024


class AutoAnnotationWorker(QThread):
    """Serial, batch=1 detection over a folder with the app's single loaded engine."""
    image_done = Signal(int, object)
    image_failed = Signal(int, object)
    progress = Signal(int, int)
    # Message templates cross the thread boundary untranslated.
    error_signal = Signal(object)
    status_signal = Signal(object)

    def __init__(self, engine, enhancer, jobs, options, visual=None, model_sha256="", run_index=None):
        super().__init__()
        self.engine, self.enhancer = engine, enhancer
        self.jobs, self.options, self.visual = list(jobs), dict(options), visual
        self.model_sha256 = model_sha256
        # Each result names its run, so a result queued behind Stop keeps its provenance.
        self.run_index = run_index
        self.running = True
        self.processed = 0
        self.failed = None
        self.skipped = []  # indices skipped because their SAHI plan exceeds the slice cap

    def run(self):
        total = len(self.jobs)
        try:
            if not self.model_sha256:
                self.model_sha256 = weights_sha256(self.options.get("model", ""))
            if self.visual is not None:
                # Encode the reference in this worker, never in the Qt UI thread.
                self.status_signal.emit(message("正在提取参考物体的视觉特征（后台，单模型）…"))
                reference, box, label = self.visual
                status = self.engine.set_visual_prompt(reference, box, label)
                if status != "SUCCESS":
                    raise RuntimeError(message("视觉参考初始化失败：{status}", status=status))
        except Exception as exc:
            self._fail(exc)
            return
        for done, (index, path) in enumerate(self.jobs, 1):
            if not self.running:
                break
            self.status_signal.emit(message("自动标注 {done}/{total}：{name}", done=done, total=total, name=Path(path).name))
            frame, sha256 = read_image_with_sha(path)
            if frame is None:
                self.image_failed.emit(index, message("无法读取图片：{path}\n请检查文件是否损坏或格式是否受支持。", path=str(path)))
            elif frame.shape[0] * frame.shape[1] > MAX_SOURCE_PIXELS:
                # Same limit as the engine preflight, applied per file so one
                # oversized photo is skipped instead of ending the whole batch.
                self.image_failed.emit(index, message("输入图像超过 1600 万像素，请先缩小图像。"))
            elif (reason := self._slice_skip_reason(frame)) is not None:
                # Also a deterministic per-file input check: the 9-slice cap stays, this image is
                # skipped and reported, and the rest of the folder is still detected.
                self.skipped.append(index)
                self.image_failed.emit(index, reason)
            else:
                try:
                    started = time.perf_counter()
                    boxes, status = detect_candidates(self.engine, frame, self.options, self.enhancer)
                except Exception as exc:
                    # Resource limits and OOM stop the batch: no retry, no device switch.
                    self._fail(exc)
                    break
                self.processed += 1
                self.image_done.emit(index, {"boxes": boxes, "width": frame.shape[1], "height": frame.shape[0],
                                             "sha256": sha256, "elapsed": time.perf_counter() - started, "status": status,
                                             "run_index": self.run_index})
            self.progress.emit(done, total)

    def _slice_skip_reason(self, frame):
        if not self.options.get("sliced"):
            return None
        try:
            validate_slice_plan(frame.shape, self.options.get("slice_size", 512),
                                self.options.get("slice_overlap", 0.2), self.options.get("max_slices", 9))
        except ResourceLimitError as exc:
            return message("已跳过（SAHI 切片过多）：{error}\n本图未检测，本批其余图片继续。", error=exception_message(exc))
        except ValueError:
            return None  # invalid settings: the shared detection path reports them and stops the batch
        return None

    def _fail(self, exc):
        self.failed = f"{type(exc).__name__}: {exc}"
        self.error_signal.emit(message(
            "自动标注已停止，不会自动重试或切换设备：\n{kind}: {error}\n已完成的图片会保留。可改用 YOLOE-26S + 640，关闭 SAHI、旧分块和暗光补检后再开始。",
            kind=type(exc).__name__, error=exception_message(exc)))

    def stop(self):
        self.running = False
        return self.wait(3000)


class AnnotationCanvas(CorrectionCanvas):
    """The correction canvas, keeping each candidate's model score for review."""
    empty_text = "选择图片文件夹并开始自动标注，再在左侧列表中选择图片审核"

    def set_sample(self, image, boxes):
        super().set_sample(image, [])
        self.boxes = [dict(box, xyxy=list(box["xyxy"])) for box in boxes]
        self.update()

    def clear(self):
        self.image = self.pixmap = None
        self.boxes, self.index = [], -1
        self.update()

    def box_text(self, index, box):
        score = box.get("score")
        return f"{index + 1}. {box['label']}" + ("" if score is None else f" {score:.2f}")


class AutoAnnotationPanel(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.session, self.worker, self.run_index = None, None, None
        self.current = -1
        self._export_parent = ""
        root = QVBoxLayout(self)
        intro = QLabel("对图片文件夹逐张运行当前模型，人工接受 / 修改 / 拒绝候选框，再导出 YOLO txt 或 COCO json。\n导出标签是“模型候选 + 人工审核”，不是独立人工真值；只导出已接受或已修改的图片。")
        intro.setWordWrap(True)
        root.addWidget(intro)

        folder_row = QHBoxLayout()
        self.folder_button = QPushButton("选择图片文件夹…")
        self.folder_button.clicked.connect(self.choose_folder)
        self.recursive = QCheckBox("包含子文件夹")
        self.recursive.setToolTip("重新扫描当前文件夹；已有导出结果所在的子文件夹会被跳过。")
        self.recursive.toggled.connect(self.on_recursive_toggled)
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_button)
        folder_row.addWidget(self.recursive)
        folder_row.addWidget(self.folder_label, 1)
        root.addLayout(folder_row)

        prompt_row = QHBoxLayout()
        self.prompt_mode = QComboBox()
        self.prompt_mode.addItem("文字 / 无提示（本页提示词）", "text")
        self.prompt_mode.addItem("视觉提示（第 2 页参考框）", "visual")
        self.prompt_mode.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.prompt_mode.currentIndexChanged.connect(lambda *_: self.update_controls())
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("留空 = 无提示检测（同规格 PF 权重）；或输入 bottle, cup, cell phone")
        self.start_button = QPushButton("开始自动标注")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("停止")
        self.stop_button.setToolTip("当前图片推理结束后停止；已完成的候选框保留。释放模型请用左侧按钮。")
        self.stop_button.clicked.connect(self.stop_worker)
        prompt_row.addWidget(QLabel("提示方式:"))
        prompt_row.addWidget(self.prompt_mode)
        prompt_row.addWidget(QLabel("提示词:"))
        prompt_row.addWidget(self.prompt_input, 1)
        prompt_row.addWidget(self.start_button)
        prompt_row.addWidget(self.stop_button)
        root.addLayout(prompt_row)
        settings_note = QLabel("使用左侧当前模型、设备、置信度、框数、推理尺寸、SAHI / 分块与暗光设置，开始时记入导出清单，运行中修改不影响本批。逐张串行、batch=1，沿用显存 / 内存保护，不使用跟踪编号。再次开始只处理未检测、失败或参数已变的待审图片，不覆盖人工审核。")
        settings_note.setWordWrap(True)
        root.addWidget(settings_note)
        preset_row = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItem("使用左侧设置（默认）", "")
        self.preset.addItem("精度优先：YOLOE-26L + SAHI 320（较慢）", "accuracy_first")
        self.preset.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.preset.setToolTip("精度优先只作用于本页的这一批：加载 YOLOE-26L（左侧模型栏随之显示 26L，仍只保留一个模型），推理尺寸 640、置信度 0.40、SAHI 320 切片（丢弃被内部切片边截断的框），关闭旧分块和暗光补检；设备和框数沿用左侧，其余左侧设置不变。参数记入导出清单。只支持文字提示。在 240 图人工标注留出集上 AP 更高，但更慢、候选框更多，见 README。")
        preset_note = QLabel("预设只改变本批参数，记入导出清单；需要本地 YOLOE-26L 权重，只支持文字提示。")
        preset_note.setWordWrap(True)
        preset_row.addWidget(QLabel("本页预设:"))
        preset_row.addWidget(self.preset)
        preset_row.addWidget(preset_note, 1)
        root.addLayout(preset_row)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.status_label, 2)
        root.addLayout(progress_row)

        body = QSplitter()
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(230)
        self.image_list.currentRowChanged.connect(self.select_image)
        body.addWidget(self.image_list)
        self.canvas = AnnotationCanvas()
        body.addWidget(self.canvas)
        right = QWidget()
        right.setMinimumWidth(250)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.category = QComboBox()
        self.category.setEditable(True)
        self.category.setToolTip("选择或输入类别：修改所选框的类别，也用于新画的框。")
        self.category.currentTextChanged.connect(lambda text: setattr(self.canvas, "label", text.strip() or "object"))
        self.category.activated.connect(lambda *_: self.relabel())
        self.category.lineEdit().editingFinished.connect(self.relabel)
        right_layout.addWidget(QLabel("框的类别（选择或输入）:"))
        right_layout.addWidget(self.category)
        self.mode = QComboBox()
        self.mode.addItems(["选择 / 拖动现有框", "拖拽补框（或删除旧框后重画）"])
        self.mode.setToolTip("D 键切换选择 / 拖动现有框与拖拽补框两种模式。")
        self.mode.currentIndexChanged.connect(lambda index: setattr(self.canvas, "mode", "draw" if index else "select"))
        right_layout.addWidget(self.mode)
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.select_box)
        right_layout.addWidget(self.box_list, 1)
        self.delete_button = QPushButton("删除所选误检框")
        self.delete_button.setToolTip("删除左侧列表中选中的框。快捷键 Delete。")
        self.delete_button.clicked.connect(self.canvas.delete_selected)
        right_layout.addWidget(self.delete_button)
        self.accept_button = QPushButton("接受本图")
        self.accept_button.setToolTip("采用当前全部框（含未修改的候选）；有改动时记为“已修改”。随后跳到下一张待审图片。快捷键 A。")
        self.accept_button.clicked.connect(self.accept_current)
        self.reject_button = QPushButton("拒绝本图（不导出）")
        self.reject_button.setToolTip("本图不导出，随后跳到下一张待审图片。快捷键 R。")
        self.reject_button.clicked.connect(self.reject_current)
        self.reset_button = QPushButton("恢复模型候选")
        self.reset_button.setToolTip("放弃本图的人工修改，恢复模型候选框并改回待审。快捷键 U。")
        self.reset_button.clicked.connect(self.reset_current)
        for button in (self.accept_button, self.reject_button, self.reset_button):
            right_layout.addWidget(button)
        body.addWidget(right)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 4)
        body.setStretchFactor(2, 1)
        root.addWidget(body, 1)

        bulk_row = QHBoxLayout()
        self.bulk_score = QDoubleSpinBox()
        self.bulk_score.setRange(0.0, 1.0)
        self.bulk_score.setSingleStep(0.05)
        self.bulk_score.setDecimals(2)
        self.bulk_score.setValue(0.5)
        self.bulk_button = QPushButton("接受其余待审图片（只保留分数 ≥ 阈值的候选）")
        self.bulk_button.setToolTip("只处理已检测、未改动的待审图片；没有候选达到阈值的图片保持待审。清单记录为批量接受。")
        self.bulk_button.clicked.connect(self.bulk_accept)
        bulk_row.addWidget(QLabel("批量接受阈值:"))
        bulk_row.addWidget(self.bulk_score)
        bulk_row.addWidget(self.bulk_button, 1)
        root.addLayout(bulk_row)

        export_row = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("例如：bottle, cup, cell phone（留空按出现顺序）")
        self.class_input.setToolTip("导出类别 ID 的顺序；框中其它类别自动追加在后面。")
        self.class_input.editingFinished.connect(self.apply_class_order)
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setSingleStep(0.05)
        self.min_score.setDecimals(2)
        self.min_score.setValue(0.0)
        self.min_score.setToolTip("只过滤“按分数批量接受”（未逐张查看）的图片中低于该分数的框；逐张接受或修改的图片按人工结果导出。批量接受的图片若没有框达到该分数则不导出。")
        self.export_yolo_button = QPushButton("导出 YOLO…")
        self.export_yolo_button.setToolTip("快捷键 Ctrl+E（导出 YOLO）。")
        self.export_yolo_button.clicked.connect(lambda: self.export("yolo"))
        self.export_coco_button = QPushButton("导出 COCO…")
        self.export_coco_button.clicked.connect(lambda: self.export("coco"))
        export_row.addWidget(QLabel("类别顺序:"))
        export_row.addWidget(self.class_input, 1)
        export_row.addWidget(QLabel("导出最低分数:"))
        export_row.addWidget(self.min_score)
        export_row.addWidget(self.export_yolo_button)
        export_row.addWidget(self.export_coco_button)
        root.addLayout(export_row)
        self.shortcuts_hint = QLabel(
            "快捷键：A 接受本图 · R 拒绝本图 · U 恢复模型候选 · Delete 删除所选框 · "
            "N / P 下一张 / 上一张图片 · D 切换画框 / 选择模式 · Ctrl+E 导出 YOLO。"
            "输入框获得焦点时快捷键不生效。")
        self.shortcuts_hint.setWordWrap(True)
        root.addWidget(self.shortcuts_hint)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self.canvas.changed.connect(self.on_canvas_changed)
        self.canvas.selected.connect(self.box_list.setCurrentRow)
        self._build_shortcuts()
        self.update_summary()
        self.update_controls()

    # Shared helpers ------------------------------------------------------
    def _language(self):
        translator = language_for(self)
        return translator.language if translator else "zh"

    def _bind(self, widget, value):
        translator = language_for(self)
        if translator:
            translator.bind(widget, value)
        else:
            widget.setText(str(value))

    def notify(self, value):
        """Routine progress stays in the page and status bar, without dialogs."""
        self._bind(self.status_label, value)
        self.app.show_status(value)

    # Keyboard shortcuts (C1) ----------------------------------------------
    @staticmethod
    def _shortcut_blocked():
        """True while the focused widget is a text field: its own keys win."""
        widget = QApplication.focusWidget()
        if widget is None:
            return False
        if isinstance(widget, QLineEdit):
            return True
        return isinstance(widget, QComboBox) and widget.isEditable()

    def _guarded(self, handler):
        def activated():
            if not self._shortcut_blocked():
                handler()
        return activated

    def _build_shortcuts(self):
        """QShortcut on the page itself: scoped to this tab, never a text field."""
        bindings = (("A", self.accept_current), ("R", self.reject_current), ("U", self.reset_current),
                   (QKeySequence.Delete, self.canvas.delete_selected), ("N", lambda: self._move_selection(1)),
                   ("P", lambda: self._move_selection(-1)), ("D", self._toggle_draw_mode),
                   ("Ctrl+E", self._export_shortcut))
        self._shortcuts = []
        for key, handler in bindings:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._guarded(handler))
            self._shortcuts.append(shortcut)

    def _move_selection(self, offset):
        if self.session is None or not len(self.session):
            return
        count = len(self.session)
        row = self.image_list.currentRow()
        self.image_list.setCurrentRow((row + offset) % count if row >= 0 else 0)

    def _toggle_draw_mode(self):
        if self.entry() is None:
            return
        self.mode.setCurrentIndex(0 if self.mode.currentIndex() else 1)

    def _export_shortcut(self):
        if self.export_yolo_button.isEnabled():
            self.export_yolo_button.click()
        elif self.export_coco_button.isEnabled():
            self.export_coco_button.click()

    @property
    def running(self):
        return self.worker is not None

    def entry(self, index=None):
        index = self.current if index is None else index
        if self.session is None or not 0 <= index < len(self.session):
            return None
        return self.session.entries[index]

    def item_text(self, index):
        entry = self.session.entries[index]
        if entry.error:
            state = message("已跳过（切片过多）" if entry.error.startswith(SLICE_SKIP_PREFIX) else "失败")
            mark = "!"
        elif entry.state == "pending":
            state = message("待审（已改动）" if entry.modified else "待审" if entry.detected else "未检测")
            mark = STATE_MARK["pending"] if entry.detected or entry.modified else "·"
        else:
            state, mark = message(STATE_TEXT[entry.state]), STATE_MARK[entry.state]
        return message("{mark} {name} | {state} | {count} 框", mark=mark, name=entry.relative,
                       state=state, count=len(entry.boxes))

    def update_item(self, index):
        item = self.image_list.item(index)
        entry = self.entry(index)
        if item is None or entry is None:
            return
        item.setText(render(self.item_text(index), self._language()))
        key = "failed" if entry.error else entry.state if entry.state != "pending" else \
            ("undetected" if not entry.detected and not entry.modified else "")
        item.setForeground(QColor(STATE_COLOR[key]) if key else self.image_list.palette().text().color())

    def refresh_items(self):
        for index in range(self.image_list.count()):
            self.update_item(index)

    def update_summary(self):
        if self.session is None:
            self._bind(self.summary, message("未选择文件夹"))
            return
        counts = self.session.counts()
        self._bind(self.summary, message(
            "共 {total} 张 | 已检测 {detected} | 待审 {pending} | 已接受 {accepted} | 已修改 {edited} | 已拒绝 {rejected} | 失败 {failed} | 可导出 {exportable} 张 / {boxes} 框",
            **counts))

    def update_controls(self):
        running = self.running
        has_images = self.session is not None and len(self.session) > 0
        entry = self.entry()
        selected = entry is not None and self.canvas.image is not None
        counts = self.session.counts() if self.session is not None else {"exportable": 0}
        self.folder_button.setEnabled(not running)
        self.recursive.setEnabled(not running)
        self.prompt_mode.setEnabled(not running)
        self.preset.setEnabled(not running)
        self.prompt_input.setEnabled(not running and self.prompt_mode.currentData() == "text")
        self.start_button.setEnabled(has_images and not running)
        self.stop_button.setEnabled(running)
        for widget in (self.accept_button, self.reject_button, self.delete_button, self.category, self.mode):
            widget.setEnabled(selected)
        self.reset_button.setEnabled(selected and (entry.detected or entry.state != "pending"))
        self.bulk_button.setEnabled(not running and has_images and self.session.bulk_accept(self.bulk_score.value(), dry_run=True) > 0)
        for button in (self.export_yolo_button, self.export_coco_button):
            button.setEnabled(not running and counts["exportable"] > 0)

    def retranslate(self):
        """No rescanning, re-detection or review reset."""
        if self.session is not None:
            self.refresh_items()
            self.rebuild_boxes()
        self.canvas.update()

    # Folder --------------------------------------------------------------
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择待标注图片文件夹")
        if folder:
            self.load_folder(folder)

    def on_recursive_toggled(self, checked):
        if self.session is not None and not self.running and not self.load_folder(self.session.folder):
            with QSignalBlocker(self.recursive):
                self.recursive.setChecked(not checked)

    def apply_class_order(self):
        if self.session is not None:
            self.session.set_classes(parse_prompts(self.class_input.text()))
            self.rebuild_boxes()

    def may_discard(self):
        """Ask before dropping unexported review or unreviewed detection results."""
        if self.session is None:
            return True
        if self.session.dirty:
            return QMessageBox.question(self, "未导出的标注", "自动标注页有尚未导出的人工审核结果，继续将丢失这些结果。是否继续？") == QMessageBox.Yes
        unreviewed = sum(entry.detected and entry.state == "pending" for entry in self.session.entries)
        if unreviewed:
            return QMessageBox.question(self, "未导出的标注", message(
                "自动标注页有 {count} 张图片的检测候选尚未审核，继续将丢失这些候选，需要重新检测。是否继续？",
                count=unreviewed)) == QMessageBox.Yes
        return True

    # Session auto-save / resume (C2) --------------------------------------
    def _maybe_resume(self, session, folder):
        """Offer to restore a previously saved review session for *folder*."""
        try:
            data = read_session_file(session_cache_path(folder))
        except ValueError as exc:
            self.notify(message("已忽略损坏的标注进度文件：{error}", error=exception_message(exc)))
            return session
        if not data:
            return session
        reviewed = sum(1 for raw in data.get("entries", [])
                       if isinstance(raw, dict) and (raw.get("state") not in (None, "pending") or raw.get("detected")))
        if not reviewed:
            return session
        if QMessageBox.question(self, "恢复标注进度", message(
                "检测到本文件夹上次自动标注保存的审核进度（约 {count} 张已审核或已检测），是否恢复？"
                "图片内容发生变化的会重新标为待审，不会沿用旧候选框。", count=reviewed)) != QMessageBox.Yes:
            return session
        try:
            resumed = AnnotationSession.from_dict(data, folder, recursive=session.recursive)
        except ValueError as exc:
            self.notify(message("恢复失败，已忽略进度文件：{error}", error=exception_message(exc)))
            return session
        if resumed.changed_count:
            self.notify(message("已恢复 {restored} 张图片的审核进度；{changed} 张图片内容已变化，重新标为待审。",
                                restored=resumed.restored_count, changed=resumed.changed_count))
        else:
            self.notify(message("已恢复 {restored} 张图片的审核进度。", restored=resumed.restored_count))
        return resumed

    def _autosave(self):
        """Persist the review session after each human decision; never blocks review."""
        if self.session is None:
            return
        try:
            write_session(self.session)
        except OSError as exc:
            self.notify(message("标注进度未能保存：{error}", error=exception_message(exc)))

    def load_folder(self, folder):
        if self.running or not self.may_discard():
            return False
        try:
            session = AnnotationSession(folder, recursive=self.recursive.isChecked())
        except Exception as exc:
            self.notify(message("读取文件夹失败：{error}", error=exception_message(exc)))
            return False
        if not len(session):
            self.notify(message("所选文件夹中没有 JPG、JPEG、PNG 或 BMP 图片。"))
            return False
        session = self._maybe_resume(session, folder)
        self.session, self.current = session, -1
        session.set_classes(parse_prompts(self.class_input.text()))
        self._bind(self.folder_label, message("{folder}（{count} 张图片）", folder=str(session.folder), count=len(session)))
        self.canvas.clear()
        self.box_list.clear()
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for index in range(len(session)):
            self.image_list.addItem("")
            self.update_item(index)
        self.image_list.blockSignals(False)
        self.progress.setRange(0, len(session))
        self.progress.setValue(0)
        self.update_summary()
        self.update_controls()
        self.notify(message("已载入 {count} 张图片。点击“开始自动标注”生成候选框。", count=len(session)))
        return True

    # Detection -----------------------------------------------------------
    def start(self):
        app = self.app
        if self.running or self.session is None or not len(self.session):
            return
        if app.training_active:
            self.notify(message("纠错训练和检测串行运行，请先等待训练结束。"))
            return
        visual = self.prompt_mode.currentData() == "visual"
        prompts, reference = parse_prompts(self.prompt_input.text()), None
        preset = PRESETS.get(self.preset.currentData() or "")
        if preset is not None:
            problem = self._preset_problem(preset, prompts, visual)
            if problem is not None:
                self.notify(problem)
                return
        if visual:
            prompt = app.ref_canvas.get_visual_prompt()
            if prompt is None:
                self.notify(message("请先在第 2 页上传参考图片并框选目标。"))
                return
            label = app.ref_class_input.text().strip() or "object"
            reference, prompts = (prompt[0], prompt[1], label), [label]
        visual_record = None
        if reference is not None:
            from feedback import image_id
            visual_record = {"reference_image_sha256": image_id(reference[0]),
                             "box_xyxy": [float(value) for value in reference[1]], "label": reference[2]}
        # Nothing to detect: say so before stopping the live stream or loading weights.
        planned = self._planned_identity(prompts, visual, visual_record)
        provisional = None if planned is None else self._run_config(self._preset_options(
            self._run_options(prompts, "visual" if visual else "text" if prompts else "prompt_free")), planned)
        if not self.session.targets(provisional):
            self.notify(message("没有需要检测的待审图片；已审核或已改动的图片不会被覆盖。"))
            return
        # One model, serial tasks: the live detection worker stops first.
        if not app.stop_video_stream():
            self.notify(message("正在等待当前模型推理完成，请稍后再操作…"))
            return
        if preset is not None:
            # The single loaded model must be the preset's; the combo has no load-on-change handler.
            with QSignalBlocker(app.model_combo):
                app.model_combo.setCurrentIndex(app.model_combo.findData(preset["model"]))
        if not app._ensure_detection_model(prompts, visual=visual):
            self.notify(message("自动标注未启动：模型未就绪，请查看状态栏提示。"))
            return
        try:
            app._switch_prompt_engine(visual)
        except ValueError as exc:
            self.notify(exception_message(exc))
            return
        app._update_model_indicator()
        engine = app.engine
        prompt_free = isinstance(engine, PromptFreeEngine)
        if not visual and not prompts and not prompt_free:
            # Never let an empty prompt fall back to the text engine's preset classes.
            self.notify(message("无提示检测需要 YOLOE 无提示权重；不会用固定类别代替。"))
            return
        mode = "visual" if visual else "prompt_free" if prompt_free else "text"
        options = self._preset_options(self._run_options(prompts, mode))
        identity = {"model": app.active_model_path, "model_sha256": app.active_model_sha256,
                    "device": str(engine.device), "visual_reference": visual_record,
                    "inference_options": engine.inference_options}
        config = self._run_config(options, identity)
        jobs = self.session.targets(config)
        if not jobs:
            self.notify(message("没有需要检测的待审图片；已审核或已改动的图片不会被覆盖。"))
            return
        if not self.class_input.text().strip() and not prompt_free:
            self.class_input.setText(", ".join(prompts))
        self.session.set_classes(parse_prompts(self.class_input.text()))
        titles = {path: title for title, path in type(app).MODELS.items()}
        metadata = dict(options, **identity, config=config, prompts=list(prompts),
                        model_title=titles.get(app.active_model_path, Path(app.active_model_path).name),
                        engine=type(engine).__name__,
                        vocabulary_size=engine.vocabulary_size if prompt_free else len(prompts),
                        tracker=None, images_requested=len(jobs),
                        label_provenance="model candidates; not ground truth until a human accepts them")
        self.run_index = self.session.begin_run(metadata)
        app.dce_engine.reset()
        options["model"] = app.active_model_path
        worker = self.worker = AutoAnnotationWorker(engine, app.dce_engine, jobs, options, visual=reference,
                                                     model_sha256=app.active_model_sha256, run_index=self.run_index)
        worker.image_done.connect(self.on_image_done)
        worker.image_failed.connect(self.on_image_failed)
        worker.progress.connect(self.on_progress)
        worker.error_signal.connect(self.on_error)
        worker.status_signal.connect(self.notify)
        worker.finished.connect(lambda: self.on_finished(worker))
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.update_controls()
        if preset is not None:
            self.notify(message("自动标注已开始（精度优先预设：YOLOE-26L + SAHI 320）：{count} 张图片，逐张串行。", count=len(jobs)))
        else:
            self.notify(message("自动标注已开始：{count} 张图片，逐张串行。", count=len(jobs)))
        worker.start()

    def _preset_options(self, options):
        """Run options with the selected preset applied; the left-side widgets are not changed."""
        key = self.preset.currentData() or ""
        preset = PRESETS.get(key)
        return dict(options, **(preset["options"] if preset else {}), preset=key or None)

    def _selected_model(self):
        preset = PRESETS.get(self.preset.currentData() or "")
        return preset["model"] if preset else self.app.model_combo.currentData()

    def _preset_problem(self, preset, prompts, visual):
        """Checked before live detection stops or any weights load; None when the preset can run."""
        if visual or not prompts:
            return message("“精度优先”预设只用于文字提示。请输入提示词，或把本页预设改回“使用左侧设置”。")
        if self.app.model_combo.findData(preset["model"]) < 0 or not local_weights_present(preset["model"]):
            return message("未找到本地 YOLOE-26L 权重（{name}）；预设不会下载或替换模型。请先准备权重，或把本页预设改回“使用左侧设置”。",
                           name=preset["model"])
        return None

    def _run_options(self, prompts, mode):
        app = self.app
        # Visual prompts use the Tab 2 threshold, so the manifest records the value actually applied.
        conf = app.visual_conf_spin.value() if mode == "visual" else app.conf_slider.value() / 100.0
        return {
            "mode": mode, "class_names": [] if mode == "prompt_free" else prompts,
            "conf": conf, "iou": 0.45, "max_det": app.max_det_spin.value(),
            "imgsz": app.resolution_combo.currentData(), "tiled": app.tile_checkbox.isChecked(),
            "sliced": app.sahi_checkbox.isChecked(), "slice_size": app.slice_size_combo.currentData(),
            "slice_overlap": 0.2, "max_slices": 9, "low_light": app.dce_checkbox.isChecked(),
            "supplement_dark": bool(app.dark_mode_combo.currentData()),
            "enhancement_strength": app.strength_slider.value() / 100.0,
        }

    @staticmethod
    def _run_config(options, identity):
        return json.dumps(dict(options, **identity), sort_keys=True, ensure_ascii=False)

    def _planned_identity(self, prompts, visual, visual_record):
        """The identity Start would record when the loaded model is reused as is;
        None when it would load or switch weights (the identity is not known yet)."""
        app = self.app
        try:
            path = resolve_detection_model(self._selected_model(), prompt_free=not prompts and not visual)
        except ValueError:
            return None
        device = "cpu" if app.device_combo.currentText() == "CPU" else "cuda:0"
        engine = app.engine
        # Modified weights can change under a fixed file name; leave those to the full check.
        if engine.model is None or app.active_model_path != path or str(engine.device) != device \
                or path in getattr(app, "_feedback_versions", {}):
            return None
        return {"model": path, "model_sha256": app.active_model_sha256, "device": device,
                "visual_reference": visual_record, "inference_options": engine.inference_options}

    def stop_worker(self):
        """Stop after the current image; True once no annotation worker is running."""
        worker = self.worker
        if worker is None:
            return True
        if not worker.stop():
            self.notify(message("正在等待当前模型推理完成，请稍后再操作…"))
            return False
        # The image in flight during wait() has its result queued: store it before the run is finalised.
        QCoreApplication.sendPostedEvents(self)
        self.on_finished(worker)
        return True

    def on_image_done(self, index, result):
        if self.session is None:
            return
        try:
            refreshed = self.session.set_candidates(index, result["boxes"], result["width"], result["height"],
                                                    result["sha256"], result.get("run_index"))
        except Exception as exc:
            refreshed = False
            self.session.set_error(index, exception_message(exc))
        self.update_item(index)
        if self.current < 0:
            self.image_list.setCurrentRow(index)
        elif index == self.current and refreshed and self.canvas.image is not None:
            self.canvas.set_sample(self.canvas.image, self.session.entries[index].boxes)
            self.rebuild_boxes()
        self.update_summary()

    def on_image_failed(self, index, value):
        if self.session is not None:
            self.session.set_error(index, value)
            self.update_item(index)
            self.update_summary()

    def on_progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def on_error(self, value):
        self.notify(value)

    def on_finished(self, worker):
        if worker is not self.worker:
            return
        self.worker = None
        if self.session is not None and self.run_index is not None:
            skipped = [self.session.entries[index].relative for index in worker.skipped]
            self.session.finish_run(self.run_index, processed=worker.processed, error=worker.failed,
                                    stopped_by_user=not worker.running and worker.failed is None,
                                    model_sha256=worker.model_sha256, skipped_too_many_slices=skipped)
            if worker.failed is None and skipped:
                self.notify(message("自动标注结束：本批检测 {count} 张；{skipped} 张因 SAHI 切片超过上限被跳过（列表中标为“已跳过”）。可改用更大的切片或关闭 SAHI 后再次开始；已审核的图片不会被覆盖。",
                                    count=worker.processed, skipped=len(skipped)))
            elif worker.failed is None:
                self.notify(message("自动标注结束：本批检测 {count} 张。请逐张审核，接受或修改后再导出。", count=worker.processed))
        self.run_index = None
        self.update_summary()
        self.update_controls()
        worker.deleteLater()

    # Review --------------------------------------------------------------
    def select_image(self, row):
        entry = self.entry(row)
        if entry is None:
            self.current = -1
            self.canvas.clear()
            self.rebuild_boxes()
            self.update_controls()
            return
        image, sha256 = read_image_with_sha(entry.path)
        try:
            if image is None:
                raise ValueError(message("无法读取图片：{path}\n请检查文件是否损坏或格式是否受支持。", path=str(entry.path)))
            self.session.set_image_info(row, image.shape[1], image.shape[0], sha256)
        except ValueError as exc:
            self.session.set_error(row, exception_message(exc))
            self.current = -1
            self.canvas.clear()
            self.update_item(row)
            self.notify(exception_message(exc))
            self.update_controls()
            return
        self.current = row
        self.canvas.set_sample(image, entry.boxes)
        self.rebuild_boxes()
        self.update_controls()

    def box_text(self, index, box):
        coords = ", ".join(f"{value:.0f}" for value in box["xyxy"])
        if box.get("score") is None:
            return message("{index}. {label} | 人工框 | {coords}", index=index + 1, label=box["label"], coords=coords)
        return message("{index}. {label} | 分数 {score:.2f} | {coords}", index=index + 1, label=box["label"],
                       score=box["score"], coords=coords)

    def rebuild_boxes(self):
        index = self.canvas.index
        language = self._language()
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for number, box in enumerate(self.canvas.boxes):
            self.box_list.addItem(render(self.box_text(number, box), language))
        self.box_list.setCurrentRow(index)
        self.box_list.blockSignals(False)
        labels = list(self.session.classes) if self.session is not None else []
        labels += [box["label"] for box in self.canvas.boxes]
        text = self.category.currentText()
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItems(list(dict.fromkeys(label for label in labels if label)))
        self.category.setCurrentText(text)
        self.category.blockSignals(False)
        self.select_box(index)

    def select_box(self, index):
        self.canvas.index = index
        if 0 <= index < len(self.canvas.boxes):
            self.category.setCurrentText(self.canvas.boxes[index]["label"])
        self.canvas.update()

    def relabel(self):
        index, label = self.canvas.index, self.category.currentText().strip()
        if 0 <= index < len(self.canvas.boxes) and label and self.canvas.boxes[index]["label"] != label:
            self.canvas.boxes[index]["label"] = label
            self.on_canvas_changed()

    def on_canvas_changed(self):
        if self.entry() is None:
            return
        try:
            self.session.update_boxes(self.current, self.canvas.boxes)
        except ValueError as exc:
            self.notify(exception_message(exc))
        self.update_item(self.current)
        self.rebuild_boxes()
        self.update_summary()
        self.update_controls()
        self._autosave()

    def _next_pending(self, start):
        entries = self.session.entries
        for offset in range(1, len(entries) + 1):
            index = (start + offset) % len(entries)
            if entries[index].state == "pending" and not entries[index].error:
                return index
        return -1

    def _after_review(self, index):
        self.update_item(index)
        self.update_summary()
        following = self._next_pending(index)
        if following >= 0 and following != index:
            self.image_list.setCurrentRow(following)
        self.update_controls()
        self._autosave()

    def accept_current(self):
        if self.entry() is None:
            return
        self.relabel()
        index = self.current
        try:
            self.session.update_boxes(index, self.canvas.boxes)
            self.session.accept(index)
        except ValueError as exc:
            self.notify(exception_message(exc))
            return
        self._after_review(index)

    def reject_current(self):
        if self.entry() is None:
            return
        index = self.current
        self.session.reject(index)
        self._after_review(index)

    def reset_current(self):
        entry = self.entry()
        if entry is None:
            return
        self.session.reset(self.current)
        self.canvas.set_sample(self.canvas.image, entry.boxes)
        self.update_item(self.current)
        self.rebuild_boxes()
        self.update_summary()
        self.update_controls()
        self._autosave()

    def bulk_accept(self):
        if self.session is None or self.running:
            return
        threshold = self.bulk_score.value()
        count = self.session.bulk_accept(threshold, dry_run=True)
        if count == 0:
            self.notify(message("没有符合条件的待审图片：需已检测、未改动，且至少一个候选达到阈值。"))
            return
        if QMessageBox.question(self, "批量接受", message(
                "将接受 {count} 张未逐张查看的待审图片，只保留分数 ≥ {score:.2f} 的候选；没有候选达到阈值的图片保持待审。导出清单会记录为批量接受。是否继续？",
                count=count, score=threshold)) != QMessageBox.Yes:
            return
        accepted = self.session.bulk_accept(threshold)
        self.refresh_items()
        entry = self.entry()
        if entry is not None and self.canvas.image is not None:
            self.canvas.set_sample(self.canvas.image, entry.boxes)
            self.rebuild_boxes()
        self.update_summary()
        self.update_controls()
        self._autosave()
        self.notify(message("已批量接受 {count} 张图片（阈值 {score:.2f}）。", count=accepted, score=threshold))

    # Export --------------------------------------------------------------
    def export(self, fmt):
        if self.session is None or self.running:
            return
        exported = False
        try:
            self.apply_class_order()
            parent = QFileDialog.getExistingDirectory(
                self, "选择导出位置（将在其中新建子文件夹）", self._export_parent or str(self.session.folder.parent))
            if not parent:
                return
            self._export_parent = parent
            target = Path(parent) / f"auto_labels_{fmt}_{datetime.now():%Y%m%d_%H%M%S}"
            result = (export_yolo if fmt == "yolo" else export_coco)(self.session, target, self.min_score.value())
            exported = True
            values = dict(format=fmt.upper(), images=result["images"], boxes=result["boxes"],
                          classes=len(result["classes"]), path=str(result["path"]))
            if result["skipped"]:
                self.notify(message("已导出 {format}：{images} 张图片、{boxes} 个框、{classes} 个类别 → {path}。{skipped} 张批量接受的图片没有达到导出最低分数的框，未导出（见清单）。",
                                    skipped=result["skipped"], **values))
            else:
                self.notify(message("已导出 {format}：{images} 张图片、{boxes} 个框、{classes} 个类别 → {path}", **values))
        except Exception as exc:
            self.notify(message("导出未完成：{error}", error=exception_message(exc)))
        if exported:
            self._offer_clear_progress(fmt)
        self.update_summary()
        self.update_controls()

    def _offer_clear_progress(self, fmt):
        """After a successful export, let the user drop the saved review progress."""
        path = session_cache_path(self.session.folder)
        if not path.is_file():
            return
        if QMessageBox.question(self, "清除本地进度", message(
                "已导出 {format}。是否清除本地保存的审核进度文件？清除后关闭或重开文件夹不会再提供恢复。",
                format=fmt.upper())) == QMessageBox.Yes:
            delete_session_file(self.session.folder, path)
