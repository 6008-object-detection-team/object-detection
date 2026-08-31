"""Reusable Qt widgets and layout construction for the detector UI."""
import cv2
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSlider,
                               QTabWidget, QVBoxLayout, QWidget)


class DrawableLabel(QLabel):
    """Image canvas that lets users draw the visual-prompt rectangle."""

    def __init__(self, app):
        super().__init__()
        self.app, self.raw_image, self.original_pixmap = app, None, None
        self.start_pt, self.end_pt, self.box = QPoint(), QPoint(), None
        self.scaled_pixmap, self.offset = None, QPoint()
        self.drawing = False
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#2b2b2b; color:white; border:1px solid #555;")

    def set_image(self, path):
        self.raw_image = cv2.imread(path)
        self.original_pixmap = QPixmap(path) if self.raw_image is not None else None
        self.start_pt, self.end_pt, self.box = QPoint(), QPoint(), None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if not self.original_pixmap:
            painter.drawText(self.rect(), Qt.AlignCenter, "点击上传参考图片，然后拖拽框选目标")
            return
        self.scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.offset = QPoint((self.width() - self.scaled_pixmap.width()) // 2, (self.height() - self.scaled_pixmap.height()) // 2)
        painter.drawPixmap(self.offset, self.scaled_pixmap)
        if not self.start_pt.isNull() and not self.end_pt.isNull():
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(QRect(self.start_pt, self.end_pt))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.original_pixmap:
                self.app.upload_ref_image()
                return
            self.drawing = True
            self.start_pt = self.end_pt = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pt = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pt = event.position().toPoint()
            self.box = QRect(self.start_pt, self.end_pt).normalized()
            self.update()

    def get_cropped_template(self):
        if self.box is None or self.raw_image is None or self.scaled_pixmap is None:
            return None
        scale_x = self.raw_image.shape[1] / self.scaled_pixmap.width()
        scale_y = self.raw_image.shape[0] / self.scaled_pixmap.height()
        x = max(0, int((self.box.x() - self.offset.x()) * scale_x))
        y = max(0, int((self.box.y() - self.offset.y()) * scale_y))
        w, h = max(10, int(self.box.width() * scale_x)), max(10, int(self.box.height() * scale_y))
        return self.raw_image[y:min(y + h, self.raw_image.shape[0]), x:min(x + w, self.raw_image.shape[1])]


def _video_label():
    label = QLabel("视频画面预览")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("background:black; color:white;")
    return label


def build_ui(app):
    """Create widgets and attach them to the controller methods on *app*."""
    central = QWidget()
    app.setCentralWidget(central)
    root = QHBoxLayout(central)

    settings = QGroupBox("全局设置")
    settings.setFixedWidth(300)
    left = QVBoxLayout(settings)
    app.model_combo = QComboBox()
    app.model_combo.addItems(["小型 (YOLOE-11s)", "中型 (YOLOE-11m)", "大型 (YOLOE-11l，高精度)"])
    app.model_combo.setCurrentIndex(2)
    app.device_combo = QComboBox(); app.device_combo.addItems(["CPU", "CUDA"])
    app.source_combo = QComboBox(); app.source_combo.addItems(["本地摄像头 (Camera 0)", "本地视频文件 (MP4/AVI)"])
    app.source_combo.currentIndexChanged.connect(app.on_source_changed)
    app.file_path_label = QLabel("未选择视频文件"); app.file_path_label.setWordWrap(True); app.file_path_label.hide()
    app.conf_slider = QSlider(Qt.Horizontal); app.conf_slider.setRange(1, 100); app.conf_slider.setValue(40)
    app.dce_checkbox = QCheckBox("开启暗光增强 (Zero-DCE/CLAHE)"); app.dce_checkbox.stateChanged.connect(app.update_enhancement_state)
    apply_button = QPushButton("应用设置 / 下载并加载 YOLOE-11 模型"); apply_button.clicked.connect(app.apply_settings)
    for title, widget in [("模型尺寸:", app.model_combo), ("运行设备:", app.device_combo), ("视频输入源:", app.source_combo), ("", app.file_path_label), ("置信度阈值:", app.conf_slider), ("", app.dce_checkbox), ("", apply_button)]:
        if title: left.addWidget(QLabel(title))
        left.addWidget(widget)
    left.addStretch(); root.addWidget(settings)

    tabs = QTabWidget(); root.addWidget(tabs)
    text_tab = QWidget(); text_layout = QVBoxLayout(text_tab); controls = QHBoxLayout()
    app.prompt_input = QLineEdit(); app.prompt_input.setPlaceholderText("例如：person，mobile phone，red backpack")
    app.prompt_input.textChanged.connect(app.update_worker_classes)
    start = QPushButton("开始检测"); start.clicked.connect(app.start_text_stream)
    stop = QPushButton("关闭检测"); stop.clicked.connect(app.stop_video_stream)
    controls.addWidget(QLabel("检测任意目标:")); controls.addWidget(app.prompt_input); controls.addWidget(start); controls.addWidget(stop)
    app.video_label_tab1 = _video_label(); text_layout.addLayout(controls); text_layout.addWidget(app.video_label_tab1, 1)
    tabs.addTab(text_tab, "1. 开放文本提示词")

    visual_tab = QWidget(); visual_layout = QVBoxLayout(visual_tab); reference = QGroupBox("视觉提示词（参考目标）"); reference_layout = QHBoxLayout(reference)
    side = QVBoxLayout(); upload = QPushButton("上传参考图片"); upload.clicked.connect(app.upload_ref_image)
    app.ref_class_input = QLineEdit(); app.ref_class_input.setPlaceholderText("可选：输入目标标签")
    side.addWidget(upload); side.addWidget(QLabel("目标标签:")); side.addWidget(app.ref_class_input); side.addStretch()
    app.ref_canvas = DrawableLabel(app); app.ref_canvas.setFixedSize(450, 320)
    reference_layout.addLayout(side); reference_layout.addWidget(app.ref_canvas)
    visual_controls = QHBoxLayout(); visual_start = QPushButton("开始检测（视觉提示词）"); visual_start.clicked.connect(app.start_visual_stream)
    visual_stop = QPushButton("关闭检测"); visual_stop.clicked.connect(app.stop_video_stream)
    visual_controls.addWidget(visual_start); visual_controls.addWidget(visual_stop)
    app.video_label_tab2 = _video_label(); visual_layout.addWidget(reference); visual_layout.addLayout(visual_controls); visual_layout.addWidget(app.video_label_tab2, 1)
    tabs.addTab(visual_tab, "2. 视觉提示词")
    tabs.addTab(QWidget(), "3. 模型微调（开发中）")
    tabs.addTab(QWidget(), "4. 自动标注（开发中）")
