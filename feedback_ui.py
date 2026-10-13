"""One-click same-image correction demos, plus optional strict sample tooling."""
import json
import sys
import uuid
import codecs
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QProcess, QProcessEnvironment, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QCheckBox, QListWidget, QSpinBox, QPlainTextEdit, QSplitter)

from app_utils import parse_prompts
from feedback import FeedbackStore, PROJECT, checked_boxes, checked_classes, source_group, sha256_file
from model_families import MODEL_FAMILIES, model_family, record_family
from video_worker import read_image_file
from i18n import message, exception_message, message_from_payload, render
from ui_language import QFileDialog, QMessageBox, language_for
from overlay_labels import place_label


class CorrectionCanvas(QWidget):
    changed = Signal()
    selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.image = self.pixmap = None
        self.boxes, self.index = [], -1
        self.label, self.mode = "object", "select"
        self.start = self.current = self.original_box = None
        self.setMinimumSize(360, 260)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_sample(self, image, boxes):
        self.image = image.copy()
        h, w = image.shape[:2]
        rgb = image[..., ::-1].copy()
        self.pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())
        self.boxes = checked_boxes(boxes, w, h)
        self.index = -1
        self.start = None
        self.update()

    def image_rect(self):
        if self.pixmap is None:
            return QRectF()
        scaled = self.pixmap.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRectF((self.width()-scaled.width())/2, (self.height()-scaled.height())/2, scaled.width(), scaled.height())

    def to_image(self, point, clip=False):
        rect = self.image_rect()
        if rect.isEmpty() or (not clip and not rect.contains(point)):
            return None
        x = max(0., min(self.image.shape[1], (point.x()-rect.x()) * self.image.shape[1]/rect.width()))
        y = max(0., min(self.image.shape[0], (point.y()-rect.y()) * self.image.shape[0]/rect.height()))
        return QPointF(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151515"))
        if self.pixmap is None:
            painter.setPen(QColor("white"))
            translator = language_for(self)
            text = "先检测图片 / 视频 / 摄像头，再点击“纠错当前结果”\n也可以直接打开图片，手动补框"
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                             translator.text(text) if translator else text)
            return
        rect = self.image_rect()
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        sx, sy = rect.width()/self.image.shape[1], rect.height()/self.image.shape[0]
        labels = []
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box["xyxy"]
            drawn = QRectF(rect.x()+x1*sx, rect.y()+y1*sy, (x2-x1)*sx, (y2-y1)*sy)
            color = QColor("#ffcc33" if index == self.index else "#50dd88")
            painter.setPen(QPen(color, 2))
            painter.drawRect(drawn)
            labels.append((drawn, f"{index+1}. {box['label']}", color))
        painter.save()
        painter.setClipRect(rect)
        metrics, padding = painter.fontMetrics(), 3
        for drawn, text, color in labels:
            text = metrics.elidedText(text, Qt.ElideRight, max(0, int(rect.width()) - 2 * padding))
            left, top, width, height = place_label(
                drawn.left(), drawn.top(), metrics.horizontalAdvance(text) + 2 * padding,
                metrics.height() + 2 * padding,
                (rect.left(), rect.top(), rect.right(), rect.bottom()))
            panel = QRectF(left, top, width, height)
            painter.fillRect(panel, color)
            painter.setPen(QPen(color, 1))
            painter.drawRect(panel.adjusted(.5, .5, -.5, -.5))
            position = QPointF(left + padding, top + padding + metrics.ascent())
            painter.setPen(QColor("#181818"))
            painter.drawText(position + QPointF(1, 1), text)
            painter.setPen(QColor("white"))
            painter.drawText(position, text)
        painter.restore()
        if self.start is not None and self.mode == "draw" and self.current is not None:
            painter.setPen(QPen(QColor("#55bbff"), 2, Qt.DashLine))
            p1 = QPointF(rect.x()+self.start.x()*sx, rect.y()+self.start.y()*sy)
            p2 = QPointF(rect.x()+self.current.x()*sx, rect.y()+self.current.y()*sy)
            painter.drawRect(QRectF(p1, p2).normalized())

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        point = self.to_image(event.position())
        if point is None:
            return
        self.start = self.current = point
        if self.mode == "select":
            self.index = next((i for i in reversed(range(len(self.boxes))) if QRectF(
                QPointF(*self.boxes[i]["xyxy"][:2]), QPointF(*self.boxes[i]["xyxy"][2:])).contains(point)), -1)
            self.original_box = list(self.boxes[self.index]["xyxy"]) if self.index >= 0 else None
            self.selected.emit(self.index)
        self.update()

    def mouseMoveEvent(self, event):
        if self.start is None:
            return
        self.current = self.to_image(event.position(), clip=True)
        if self.mode == "select" and self.original_box is not None:
            x1, y1, x2, y2 = self.original_box
            dx = max(-x1, min(self.image.shape[1]-x2, self.current.x()-self.start.x()))
            dy = max(-y1, min(self.image.shape[0]-y2, self.current.y()-self.start.y()))
            self.boxes[self.index]["xyxy"] = [x1+dx, y1+dy, x2+dx, y2+dy]
        self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None or event.button() != Qt.LeftButton:
            return
        point = self.to_image(event.position(), clip=True)
        if self.mode == "draw" and point is not None:
            rect = QRectF(self.start, point).normalized()
            if rect.width() >= 2 and rect.height() >= 2 and self.label.strip():
                self.boxes.append({"label": self.label.strip(), "xyxy": [rect.left(), rect.top(), rect.right(), rect.bottom()]})
                self.index = len(self.boxes)-1
                self.changed.emit()
                self.selected.emit(self.index)
        elif self.original_box is not None and self.boxes[self.index]["xyxy"] != self.original_box:
            self.changed.emit()
        self.start = self.original_box = None
        self.update()

    def delete_selected(self):
        if 0 <= self.index < len(self.boxes):
            self.boxes.pop(self.index)
            self.index = -1
            self.changed.emit()
            self.update()


class FeedbackPanel(QWidget):
    def __init__(self, app, store=None):
        super().__init__()
        self.app, self.store = app, store or FeedbackStore()
        self.provenance, self.candidates, self.dirty = {}, [], False
        self.exported = self.training_output = None
        self.process = None
        self.quick_retest = None
        self._advanced_widgets = []
        self._log_messages = deque(maxlen=200)
        self._log_partial = ""
        self._log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        root = QVBoxLayout(self)
        hint = QLabel("改类别、补框或删框 → 点击“一键更新并重测”即可。自动保存、训练、选择同名修改版并重测本图，原模型不变。\n点击即采用当前全部框；允许同图训练和演示，可能过拟合，不作为独立准确率评测。修改版仍支持文字和框选参考图。")
        hint.setWordWrap(True); root.addWidget(hint)
        top = QHBoxLayout()
        for title, action in (("纠错当前结果", app.open_feedback), ("打开图片手动标注…", self.open_image), ("刷新样本 / 模型", self.refresh)):
            button = QPushButton(title); button.clicked.connect(action); top.addWidget(button)
            if action == self.refresh:
                self._advanced_widgets.append(button)
        root.addLayout(top)
        self.classes = QLineEdit("bottle, cup, cell phone")
        self.classes.setPlaceholderText("自动纳入框的类别；可保留已删目标的类别作为负样本")
        self.classes.textEdited.connect(self.mark_dirty)
        root.addWidget(QLabel("更新类别（自动包含当前所有框，不限制修改版以后输入其它文字）:")); root.addWidget(self.classes)
        body = QSplitter()
        self.canvas = CorrectionCanvas(); body.addWidget(self.canvas)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.category = QLineEdit("bottle")
        self.category.textChanged.connect(lambda text: setattr(self.canvas, "label", text))
        self.category.editingFinished.connect(self.relabel)
        right_layout.addWidget(QLabel("框的类别（不使用 person1 等跟踪编号）:")); right_layout.addWidget(self.category)
        self.mode = QComboBox(); self.mode.addItems(["选择 / 拖动现有框", "拖拽补框（或删除旧框后重画）"])
        self.mode.currentIndexChanged.connect(lambda index: setattr(self.canvas, "mode", "draw" if index else "select"))
        right_layout.addWidget(self.mode)
        self.box_list = QListWidget(); self.box_list.currentRowChanged.connect(self.select_box)
        right_layout.addWidget(self.box_list)
        for title, action in (("修改所选框类别", self.relabel), ("删除所选误检框", self.canvas.delete_selected)):
            button = QPushButton(title); button.clicked.connect(action); right_layout.addWidget(button)
            if action == self.relabel:
                self._advanced_widgets.append(button)
        self.scene = QLineEdit(); self.scene.setPlaceholderText("例如：宿舍桌面_白天_拍摄1")
        self.scene.textEdited.connect(self.mark_dirty)
        self.split = QComboBox()
        for title, key in (("训练样本", "train"), ("验证样本（独立场景）", "val"), ("保留测试样本（不参与训练）", "test")):
            self.split.addItem(title, key)
        self.split.activated.connect(self.mark_dirty)
        scene_label = QLabel("场景组（同一场景 / 视频不要拆到不同用途）:")
        right_layout.addWidget(scene_label); right_layout.addWidget(self.scene); right_layout.addWidget(self.split)
        self.reviewed = QCheckBox("我已检查整张图：补齐这些类别的所有目标、删除误检（无目标图也须确认）")
        self.reviewed.toggled.connect(lambda _: setattr(self, "dirty", True))
        right_layout.addWidget(self.reviewed)
        save = QPushButton("保存纠错样本 / 新修订"); save.clicked.connect(self.save_sample); right_layout.addWidget(save)
        self.sample_list = QListWidget(); self.sample_list.itemDoubleClicked.connect(self.load_selected)
        samples_label = QLabel("已保存样本（双击继续审核；旧修订不覆盖）:")
        right_layout.addWidget(samples_label); right_layout.addWidget(self.sample_list)
        self._advanced_widgets.extend((scene_label, self.scene, self.split, self.reviewed, save, samples_label, self.sample_list))
        body.addWidget(right); body.setStretchFactor(0, 3); body.setStretchFactor(1, 2)
        root.addWidget(body, 1)
        self.advanced = QCheckBox("高级样本管理（审核、分组和独立验证）")
        self.advanced.toggled.connect(self.show_advanced)
        root.addWidget(self.advanced)
        quick_footer = QHBoxLayout()
        self.quick_button = QPushButton("一键更新并重测当前图片")
        self.quick_button.clicked.connect(self.quick_update)
        quick_footer.addWidget(self.quick_button, 1)
        self.quick_epochs = QSpinBox(); self.quick_epochs.setRange(1, 100); self.quick_epochs.setValue(20)
        self.quick_epochs.setToolTip("演示更新的训练轮数；默认 20，更多轮数可能更慢且更易过拟合")
        quick_footer.addWidget(QLabel("更新轮数:")); quick_footer.addWidget(self.quick_epochs)
        self.cancel_button = QPushButton("请求停止训练"); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_training)
        quick_footer.addWidget(self.cancel_button)
        root.addLayout(quick_footer)
        advanced_footer = QWidget(); footer = QHBoxLayout(advanced_footer); footer.setContentsMargins(0, 0, 0, 0)
        export = QPushButton("导出当前型号全部已审核纠错"); export.clicked.connect(self.export_dataset); footer.addWidget(export)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 100); self.epochs.setValue(3)
        footer.addWidget(QLabel("训练轮数:")); footer.addWidget(self.epochs)
        self.train_button = QPushButton("生成 / 更新唯一修改版…"); self.train_button.clicked.connect(self.start_training); footer.addWidget(self.train_button)
        root.addWidget(advanced_footer); self._advanced_widgets.append(advanced_footer)
        self.summary = QLabel(); self.summary.setWordWrap(True); root.addWidget(self.summary)
        self.target_summary = QLabel(); self.target_summary.setWordWrap(True); root.addWidget(self.target_summary)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(200); self.log.setMaximumHeight(100); root.addWidget(self.log)
        self.canvas.changed.connect(self.on_boxes_changed)
        self.canvas.selected.connect(self.box_list.setCurrentRow)
        self.refresh()
        app.model_combo.currentIndexChanged.connect(self.on_target_changed)
        self.show_advanced(False)

    def show_advanced(self, visible):
        for widget in self._advanced_widgets:
            widget.setVisible(visible)

    def notify(self, value):
        """Routine feedback stays in the page, never a blocking dialog."""
        self.set_summary(value)
        self.append_log(value)
        self.app.show_status(value)

    def selected_family(self):
        selected = self.app.model_combo.currentData()
        if isinstance(selected, str):
            versions = getattr(self.app, "_feedback_versions", {})
            row = versions.get(selected, {}) if isinstance(versions, dict) else {}
            return row.get("model_family") or model_family(selected)
        return ""

    def on_target_changed(self, *_):
        self.exported = None
        self.refresh(refresh_models=False)

    def set_summary(self, value):
        translator = language_for(self)
        if translator:
            translator.bind(self.summary, value)
        else:
            self.summary.setText(str(value))

    def append_log(self, value):
        self._log_messages.append(value)
        translator = language_for(self)
        self.log.appendPlainText(render(value, translator.language if translator else "zh"))

    def sample_text(self, row):
        return message("{status} | {split} | {scene} | {count} 框",
            status=message({"reviewed": "已审核", "demo_accepted": "演示采用", "demo_draft": "演示草稿"}.get(row["status"], "草稿")),
            split=message({"train": "训练", "val": "验证", "test": "测试"}[row["split"]]),
            scene=row["scene"], count=len(row["boxes"]))

    def retranslate(self):
        """No disk refresh, widget reconstruction, or human-review reset."""
        translator = language_for(self)
        language = translator.language if translator else "zh"
        for index in range(self.sample_list.count()):
            item = self.sample_list.item(index)
            item.setText(render(self.sample_text(item.data(Qt.UserRole)), language))
        scrollbar = self.log.verticalScrollBar()
        old_value, at_end = scrollbar.value(), scrollbar.value() == scrollbar.maximum()
        self.log.setPlainText("\n".join(render(value, language) for value in self._log_messages))
        scrollbar.setValue(scrollbar.maximum() if at_end else old_value)
        self.canvas.update()

    def read_training_output(self, final=False):
        if self.process is None:
            return
        self._log_partial += self._log_decoder.decode(bytes(self.process.readAllStandardOutput()), final=final)
        lines = self._log_partial.split("\n")
        self._log_partial = "" if final else lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            value = line
            try:
                event = json.loads(line)
                if isinstance(event, dict) and "ui_message" in event:
                    value = message_from_payload(event["ui_message"])
                elif isinstance(event, dict) and set(event) == {"epoch", "train_loss", "val_loss"}:
                    value = message("第 {epoch} 轮 | 训练损失 {train_loss:.5f} | 验证损失 {val_loss:.5f}", **event)
                elif isinstance(event, dict) and event.get("training_mode") == "correction_demo":
                    if event.get("fit_loss") is not None:
                        value = message("第 {epoch} 轮 | 训练损失 {train_loss:.5f} | 本图拟合损失 {fit_loss:.5f}（无独立验证）",
                                        epoch=event["epoch"], train_loss=event["train_loss"], fit_loss=event["fit_loss"])
                    else:
                        value = message("第 {epoch} 轮 | 演示训练损失 {train_loss:.5f}（无独立验证）",
                                        epoch=event["epoch"], train_loss=event["train_loss"])
            except (ValueError, TypeError, KeyError):
                pass  # External dependency diagnostics remain verbatim.
            self.append_log(value)

    def mark_dirty(self, *_):
        self.dirty = True
        self.reviewed.setChecked(False)
        self.exported = None

    def on_boxes_changed(self):
        self.mark_dirty()
        self.rebuild_boxes()

    def rebuild_boxes(self):
        index = self.canvas.index
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for box in self.canvas.boxes:
            self.box_list.addItem(box["label"] + " | " + ", ".join(f"{x:.0f}" for x in box["xyxy"]))
        self.box_list.setCurrentRow(index)
        self.box_list.blockSignals(False)
        self.select_box(index)

    def select_box(self, index):
        self.canvas.index = index
        if 0 <= index < len(self.canvas.boxes):
            self.category.setText(self.canvas.boxes[index]["label"])
        self.canvas.update()

    def relabel(self):
        index = self.canvas.index
        if 0 <= index < len(self.canvas.boxes) and self.category.text().strip() \
                and self.canvas.boxes[index]["label"] != self.category.text().strip():
            self.canvas.boxes[index]["label"] = self.category.text().strip()
            self.on_boxes_changed()

    def may_replace(self):
        if not self.dirty or self.canvas.image is None:
            return True
        if self.process is not None:
            self.notify(message("正在更新模型，请等待完成或请求停止。"))
            return False
        try:
            self.relabel()
            classes = checked_classes(parse_prompts(self.classes.text()) + [box["label"] for box in self.canvas.boxes])
            self.store.save_demo(self.canvas.image, self.canvas.boxes, classes=classes,
                                 provenance=self.provenance, candidates=self.candidates, accepted=False)
            self.dirty = False
            self.append_log(message("未更新的修改已暂存为演示草稿，没有训练模型。"))
            return True
        except Exception as exc:
            self.notify(message("暂存未完成：{error}", error=exception_message(exc)))
            return False

    def set_snapshot(self, snapshot):
        image, boxes, metadata = snapshot
        if not self.may_replace():
            return False
        self.canvas.set_sample(image, boxes)
        self.provenance, self.candidates = dict(metadata), list(boxes)
        family = record_family({"provenance": metadata}) or self.selected_family()
        if family:
            self.provenance["model_family"] = family
        self.scene.setText(metadata["origin_group"])
        prompts = metadata.get("prompts") or [box["label"] for box in boxes] or parse_prompts(self.classes.text())
        self.classes.setText(", ".join(dict.fromkeys(prompts)))
        self.reviewed.setChecked(False)
        self.dirty = True
        self.exported = None
        self.rebuild_boxes()
        self.set_summary(message("候选框尚未人工审核。保存的是未增强、未画框的原图，不是界面截图。"))
        return True

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开待纠错原图", "", "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if path:
            image = read_image_file(path)
            if image is None:
                self.notify(message("无法读取这张图片"))
                return
            selected = self.app.model_combo.currentData()
            self.set_snapshot((image, [], {"source": path, "source_kind": "image", "origin_group": source_group("image", path), "model": selected, "model_family": self.selected_family()}))

    def save_sample(self):
        if self.canvas.image is None:
            return
        try:
            if not record_family({"provenance": self.provenance}) and self.selected_family():
                self.provenance["model_family"] = self.selected_family()
            self.store.save(self.canvas.image, self.canvas.boxes, classes=parse_prompts(self.classes.text()), scene=self.scene.text(),
                split=self.split.currentData(), reviewed=self.reviewed.isChecked(), provenance=self.provenance, candidates=self.candidates)
            self.dirty = False
            self.exported = None
            self.refresh()
            self.append_log(message("已保存独立纠错记录。模型权重尚未改变。"))
        except Exception as exc:
            self.notify(message("保存失败：{error}", error=exception_message(exc)))

    def refresh(self, *, refresh_models=True):
        self.sample_list.clear()
        try:
            family = self.selected_family()
            target = message("当前纠错目标：{name}修改版（同型号累计，最多一个模型条目）", name=MODEL_FAMILIES[family]) if family else message("请选择支持纠错的 YOLOE 基础型号")
            translator = language_for(self)
            if translator:
                translator.bind(self.target_summary, target)
            else:
                self.target_summary.setText(str(target))
            rows = self.store.records(family or None)
            demo_rows = self.store.demo_records(family or None)
            for row in rows + demo_rows:
                translator = language_for(self)
                self.sample_list.addItem(render(self.sample_text(row), translator.language if translator else "zh"))
                self.sample_list.item(self.sample_list.count()-1).setData(Qt.UserRole, row)
            reviewed = sum(row["status"] == "reviewed" for row in rows)
            self.set_summary(message("本地样本 {count} 张，已审核 {reviewed}，草稿 {draft}。数据、训练权重和日志不会上传 GitHub。",
                                     count=len(rows), reviewed=reviewed, draft=len(rows)-reviewed))
            if demo_rows:
                self.set_summary(message("累计演示图片 {count} 张，已采用 {accepted}，草稿 {draft}；严格审核样本 {strict} 张。点击更新才训练，数据和权重不上传 GitHub。",
                    count=len(demo_rows), accepted=sum(row["status"] == "demo_accepted" for row in demo_rows),
                    draft=sum(row["status"] == "demo_draft" for row in demo_rows), strict=len(rows)))
            if refresh_models:
                self.app.refresh_feedback_models()
        except Exception as exc:
            self.set_summary(message("样本读取失败：{error}", error=exception_message(exc)))

    def load_selected(self, item):
        row = item.data(Qt.UserRole)
        try:
            store = FeedbackStore(self.store.root/"demo") if row.get("purpose") == "correction_demo" else self.store
            if self.set_snapshot((store.load_image(row), row["boxes"], row["provenance"])):
                self.candidates = row["candidates"]
                self.classes.setText(", ".join(row["classes"]))
                self.scene.setText(row["scene"])
                self.split.setCurrentIndex(self.split.findData(row["split"]))
                self.reviewed.setChecked(row["status"] == "reviewed")
                self.dirty = False
        except Exception as exc:
            self.notify(message("读取失败：{error}", error=exception_message(exc)))

    def export_dataset(self):
        try:
            if self.dirty:
                raise ValueError("请先保存当前纠错，导出只包含已保存并人工审核的图片")
            self.exported = self.store.export_cumulative(self.selected_family())
            self.append_log(message("已生成不可变数据版本：{path}", path=str(self.exported)))
        except Exception as exc:
            self.notify(message("导出未完成：{error}", error=exception_message(exc)))

    def quick_update(self):
        if self.process is not None or getattr(self.app, "_feedback_retest", None) is not None:
            return
        try:
            if self.canvas.image is None:
                raise ValueError("请先打开图片并修改框，再点击一键更新。")
            family = self.selected_family()
            if not family:
                raise ValueError("请选择支持纠错的 YOLOE 基础型号")
            # Commit typed text too: no separate Relabel button is required.
            self.relabel()
            classes = checked_classes(parse_prompts(self.classes.text()) + [box["label"] for box in self.canvas.boxes])
            provenance = dict(self.provenance, model_family=family)
            row = self.store.save_demo(self.canvas.image, self.canvas.boxes, classes=classes,
                                       provenance=provenance, candidates=self.candidates)
            self.provenance = provenance
            self.dirty = False
            self.classes.setText(", ".join(classes))
            self.exported = self.store.export_demo(family)
            self.quick_retest = {"family": family, "classes": classes, "record": row,
                                 "image_path": self.store.root/"demo"/"images"/(row["image_id"] + ".png")}
            from modified_models import ModifiedModels
            parent, _ = ModifiedModels(self.store.root).parent_for(family)
            self.notify(message("已自动保存并准备累计纠错；正在更新 {name}修改版，完成后自动重测本图。", name=MODEL_FAMILIES[family]))
            self._launch_training(family, parent, self.app.device_combo.currentText(), self.quick_epochs.value())
        except Exception as exc:
            self.quick_retest = None
            self.notify(message("更新未启动：{error}", error=exception_message(exc)))

    def start_training(self):
        if self.process is not None:
            return
        try:
            if self.dirty or self.exported is None:
                raise ValueError("先保存样本并导出独立训练/验证数据版本")
            manifest = self.exported/"manifest.json"
            if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("training_mode") == "correction_demo":
                raise ValueError("高级训练请重新导出独立训练/验证数据；同图演示使用一键更新。")
            family = self.selected_family()
            if not family:
                raise ValueError("请选择支持纠错的 YOLOE 基础型号")
            from modified_models import ModifiedModels
            parent, _ = ModifiedModels(self.store.root).parent_for(family)
            device = self.app.device_combo.currentText()
            if QMessageBox.question(self, "更新唯一修改版", message(
                "目标：{name}修改版\n从当前修改版继续训练（首次使用原版），重新使用该型号全部已审核纠错。\n设备：{device}，轮数：{epochs}，batch=1，640。\n检测会停止；通过检查后替换同一份修改权重，原版不修改。失败保留已有修改版，不能保证精度必然提升。是否开始？",
                name=MODEL_FAMILIES[family], device=device, epochs=self.epochs.value())) != QMessageBox.Yes:
                return
            self._launch_training(family, parent, device, self.epochs.value())
        except Exception as exc:
            self.notify(message("训练未启动：{error}", error=exception_message(exc)))

    def _launch_training(self, family, parent, device, epochs):
        try:
            if not self.app.release_model():
                self.quick_retest = None
                self.notify(message("正在等待检测停止，尚未更新；请稍后再次点击。"))
                return
            version = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self.training_output = self.store.root/"models"/".staging"/(family + "_" + version)
            self.training_output.parent.mkdir(parents=True, exist_ok=True)
            process = self.process = QProcess(self)
            process.setWorkingDirectory(str(PROJECT))
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUTF8", "1")
            environment.insert("YOLO_CONFIG_DIR", str(PROJECT/".ultralytics"))
            process.setProcessEnvironment(environment)
            process.setProcessChannelMode(QProcess.MergedChannels)
            self._log_partial = ""
            self._log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            process.readyReadStandardOutput.connect(self.read_training_output)
            process.finished.connect(self.training_finished)
            process.errorOccurred.connect(self.training_error)
            self.train_button.setEnabled(False); self.cancel_button.setEnabled(True)
            self.quick_button.setEnabled(False)
            for widget in (self.canvas, self.category, self.classes, self.mode, self.quick_epochs, self.epochs):
                widget.setEnabled(False)
            self.app.model_combo.setEnabled(False)
            self.app.device_combo.setEnabled(False)
            self.app.training_active = True
            process.start(sys.executable, ["-u", str(PROJECT/"feedback_training.py"), "--dataset", str(self.exported),
                "--parent", str(parent), "--output", str(self.training_output), "--device", device, "--epochs", str(epochs),
                "--family", family, "--store", str(self.store.root), "--run", "--ui-events"])
        except Exception as exc:
            if self.process is not None:
                self.training_finished(-1)
            raise

    def cancel_training(self):
        if self.process is not None and self.training_output is not None:
            if not self.training_output.is_dir():
                process, output = self.process, self.training_output
                QTimer.singleShot(200, lambda: self.cancel_training() if self.process is process and self.training_output == output else None)
                return
            (self.training_output/"cancel.request").touch()
            self.append_log(message("已请求在下一张图片前安全停止；不会强制终止其它进程。"))

    def training_finished(self, code, *_):
        if self.process is None:
            return
        self.read_training_output(final=True)
        self.app.training_active = False
        self.app.model_combo.setEnabled(True)
        self.app.device_combo.setEnabled(True)
        self.train_button.setEnabled(True); self.cancel_button.setEnabled(False)
        self.quick_button.setEnabled(True)
        for widget in (self.canvas, self.category, self.classes, self.mode, self.quick_epochs, self.epochs):
            widget.setEnabled(True)
        self.process.deleteLater(); self.process = None
        quick, self.quick_retest = self.quick_retest, None
        self.refresh()
        if code == 0 and quick:
            try:
                from modified_models import ModifiedModels
                row = ModifiedModels(self.store.root).current(quick["family"])
                if row is None or row.get("run_id") != self.training_output.name:
                    raise ValueError("修改版校验失败；不回退到未经修改的模型，请重新检查或更新。")
                self.notify(message("{name}修改版已更新，正在自动重测当前图片（同图演示）。", name=MODEL_FAMILIES[quick["family"]]))
                quick["audit_dir"] = self.training_output
                QTimer.singleShot(0, lambda: self.app.retest_feedback(quick))
            except Exception as exc:
                self.notify(message("自动重测未完成：{error}", error=exception_message(exc)))
        else:
            self.notify(message("唯一修改版已更新。请在左侧选择同名修改版并重新检测；原版不修改，不新增带版本号的模型。" if code == 0 else "更新未完成，原版和上次可用修改版仍保留。详情见日志。"))

    def training_error(self, error):
        if self.process is None:
            return
        self.append_log(message("训练进程错误：{error}", error=self.process.errorString()))
        if error == QProcess.FailedToStart:
            self.training_finished(-1)
