"""Reusable Qt widgets and layout construction for the detector UI."""
import cv2
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSlider,
                               QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)


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
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                             self.app.ui_language.text("点击上传参考图片，然后拖拽框选目标"))
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
        prompt = self.get_visual_prompt()
        if prompt is None:
            return None
        image, (left, top, right, bottom) = prompt
        return image[top:bottom, left:right]

    def get_visual_prompt(self):
        """Return the full BGR reference and a clipped box in original pixels."""
        if self.box is None or self.raw_image is None or self.original_pixmap is None:
            return None
        # Compute geometry again: resizing can happen before the next paint event.
        scaled = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        if scaled.width() < 1 or scaled.height() < 1:
            return None
        offset = QPoint((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2)
        image_rect = QRect(offset, scaled.size())
        selected = self.box.normalized().intersected(image_rect)
        if selected.width() < 2 or selected.height() < 2:
            return None
        height, width = self.raw_image.shape[:2]
        scale_x, scale_y = width / scaled.width(), height / scaled.height()
        left = max(0, int((selected.x() - offset.x()) * scale_x))
        top = max(0, int((selected.y() - offset.y()) * scale_y))
        right = min(width, int((selected.x() + selected.width() - offset.x()) * scale_x))
        bottom = min(height, int((selected.y() + selected.height() - offset.y()) * scale_y))
        if right <= left or bottom <= top:
            return None
        # A copy keeps a running request stable if a new reference is uploaded.
        return self.raw_image.copy(), (left, top, right, bottom)


def _video_label():
    label = QLabel("输入画面预览")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("background:black; color:white;")
    return label


def build_ui(app):
    """Create widgets and attach them to the controller methods on *app*."""
    central = QWidget()
    app.setCentralWidget(central)
    root = QHBoxLayout(central)

    # Pinned outside the settings scroll area and outside every page.
    sidebar = QWidget()
    sidebar.setFixedWidth(350)
    sidebar_layout = QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(0, 0, 0, 0)
    language_row = QHBoxLayout()
    language_row.addWidget(QLabel("语言 / Language:"))
    app.language_combo = QComboBox()
    app.language_combo.setObjectName("language")
    app.language_combo.addItem("中文", "zh")
    app.language_combo.addItem("English", "en")
    app.language_combo.setToolTip("选择界面语言；不会修改提示词、类别名称、路径或模型。")
    app.language_combo.currentIndexChanged.connect(app.on_language_changed)
    language_row.addWidget(app.language_combo, 1)
    sidebar_layout.addLayout(language_row)
    app.detection_mode_label = QLabel()
    app.detection_mode_label.setWordWrap(True)
    sidebar_layout.addWidget(app.detection_mode_label)

    settings = QGroupBox("全局设置")
    settings.setMinimumWidth(300)
    left = QVBoxLayout(settings)
    app.model_combo = QComboBox()
    for title, path in app.MODELS.items():
        app.model_combo.addItem(title, path)
    app.device_combo = QComboBox(); app.device_combo.addItems(["CPU", "CUDA"])
    app.source_combo = QComboBox()
    for title, kind in [
        ("本地摄像头 (Camera 0)", "camera"),
        ("本地视频文件 (MP4/AVI)", "video"),
        ("单张图片 (JPG/PNG)", "image"),
        ("图片文件夹", "image_folder"),
    ]:
        app.source_combo.addItem(title, kind)
    app.source_combo.currentIndexChanged.connect(app.on_source_changed)
    app.file_path_label = QLabel("未选择文件"); app.file_path_label.setWordWrap(True); app.file_path_label.hide()
    app.image_navigation = QWidget()
    image_navigation_layout = QHBoxLayout(app.image_navigation)
    image_navigation_layout.setContentsMargins(0, 0, 0, 0)
    app.previous_image_button = QPushButton("← 上一张"); app.previous_image_button.clicked.connect(app.previous_image)
    app.image_position_label = QLabel("0 / 0"); app.image_position_label.setAlignment(Qt.AlignCenter)
    app.next_image_button = QPushButton("下一张 →"); app.next_image_button.clicked.connect(app.next_image)
    image_navigation_layout.addWidget(app.previous_image_button)
    image_navigation_layout.addWidget(app.image_position_label, 1)
    image_navigation_layout.addWidget(app.next_image_button)
    app.image_navigation.hide()
    app.conf_slider = QSlider(Qt.Horizontal); app.conf_slider.setRange(1, 100); app.conf_slider.setValue(40)
    app.conf_value = QLabel("0.40")
    app.max_det_spin = QSpinBox()
    app.max_det_spin.setRange(1, 1000)
    app.max_det_spin.setValue(300)
    app.max_det_spin.setToolTip("预测框数上限，与置信度独立设置。比较时一次只改变一个变量。")
    app.resolution_combo = QComboBox()
    for title, size in [("640（默认，显存优先）", 640), ("960（更多细节，较慢）", 960)]:
        app.resolution_combo.addItem(title, size)
    app.resolution_combo.setCurrentIndex(0)
    app.tile_checkbox = QCheckBox("小目标分块补检（较慢）")
    app.tile_checkbox.setToolTip("保留整图检测，再对四个重叠区域放大检测；每帧增加四次推理。")
    app.sahi_checkbox = QCheckBox("SAHI 小目标切片（默认关闭）")
    app.sahi_checkbox.setToolTip("整图 + SAHI 重叠切片融合。最多 9 个切片，同一模型逐张处理；不会并行占用显存。与旧分块补检互斥。")
    app.slice_size_combo = QComboBox()
    for title, size in [("320 像素（小图对照）", 320), ("512 像素（默认）", 512), ("640 像素", 640)]:
        app.slice_size_combo.addItem(title, size)
    app.slice_size_combo.setCurrentIndex(1)
    app.slice_size_combo.setToolTip("切片重叠 20%，最多 9 块。超过上限会停止检测，请增大切片尺寸或缩小输入图；不会跳过区域。")
    app.dce_checkbox = QCheckBox("开启自适应暗光增强"); app.dce_checkbox.stateChanged.connect(app.update_enhancement_state)
    app.dark_mode_combo = QComboBox()
    app.dark_mode_combo.addItem("原图检测 + 暗光补检（较慢）", True)
    app.dark_mode_combo.addItem("仅增强预览（较快）", False)
    app.strength_slider = QSlider(Qt.Horizontal); app.strength_slider.setRange(0, 100); app.strength_slider.setValue(65)
    app.strength_value = QLabel("65%")
    app.record_checkbox = QCheckBox("保存检测结果录像（开始前设置）")
    app.record_path_input = QLineEdit()
    app.record_path_input.setReadOnly(True)
    app.record_path_input.setPlaceholderText("选择 MP4 保存路径")
    app.record_path_input.setToolTip("仅视频和摄像头可录像；每次开始检测需使用未占用的文件名。")
    app.record_path_button = QPushButton("选择录像保存路径…")
    app.record_path_button.clicked.connect(app.choose_recording_path)
    apply_button = QPushButton("应用设置 / 加载模型"); apply_button.clicked.connect(app.apply_settings)
    release_button = QPushButton("停止并释放模型 / 显存"); release_button.clicked.connect(app.release_model)
    release_button.setToolTip("先等待当前推理结束，再释放权重和 GPU 使用锁，允许运行评价脚本。")
    for title, widget in [("检测模型:", app.model_combo), ("运行设备:", app.device_combo), ("推理分辨率:", app.resolution_combo), ("", app.sahi_checkbox), ("SAHI 切片尺寸（重叠 20%，上限 9 块）:", app.slice_size_combo), ("", app.tile_checkbox), ("输入源：", app.source_combo), ("", app.file_path_label), ("", app.image_navigation), ("置信度阈值（筛选门槛）:", app.conf_value), ("", app.conf_slider), ("最大预测框数（独立变量）:", app.max_det_spin), ("", app.record_checkbox), ("", app.record_path_input), ("", app.record_path_button), ("", app.dce_checkbox), ("暗光处理方式:", app.dark_mode_combo), ("增强强度:", app.strength_value), ("", app.strength_slider), ("", apply_button), ("", release_button)]:
        if title: left.addWidget(QLabel(title))
        left.addWidget(widget)
    for signal in [app.conf_slider.valueChanged, app.max_det_spin.valueChanged, app.resolution_combo.currentIndexChanged, app.tile_checkbox.stateChanged, app.sahi_checkbox.stateChanged, app.slice_size_combo.currentIndexChanged, app.dark_mode_combo.currentIndexChanged, app.strength_slider.valueChanged]:
        signal.connect(app.update_detection_options)
    app.memory_safety_label = QLabel("显存保护：PyTorch 显存预算 3 GiB，保留至少 1 GiB 可用显存；只保留一个模型，逐张/逐切片推理。默认 640，SAHI 默认关闭。")
    app.memory_safety_label.setWordWrap(True); left.addWidget(app.memory_safety_label)
    hint = QLabel("默认 26S + 640；小物体建议 bottle、cup、cell phone。\nSAHI 与普通检测比较时，仅改变切片开关，固定阈值、分辨率和框数。\nDINO 使用英文文本提示，文本阈值固定为 0.25，不支持视觉提示。\n录像保存处理后的画面，无音轨；按输入 FPS 保存，不可用时使用 20 FPS。\n独立实验应关闭暗光和跟踪；基准脚本提供相应控制。\nByteTrack 自动保持 bottle1、bottle2 等编号。")
    hint.setWordWrap(True); left.addWidget(hint)
    left.addStretch()
    settings_scroll = QScrollArea()
    settings_scroll.setWidgetResizable(True)
    settings_scroll.setWidget(settings)
    app.settings_scroll = settings_scroll
    sidebar_layout.addWidget(settings_scroll, 1)
    root.addWidget(sidebar)

    tabs = app.tabs = QTabWidget(); root.addWidget(tabs)
    text_tab = QWidget(); text_layout = QVBoxLayout(text_tab); controls = QHBoxLayout()
    app.prompt_input = QLineEdit(); app.prompt_input.setPlaceholderText("留空检测全部词汇；或输入 person, bottle, cup 等目标")
    app.prompt_input.textChanged.connect(app.update_worker_classes)
    start = QPushButton("开始检测"); start.clicked.connect(app.start_text_stream)
    stop = QPushButton("关闭检测"); stop.clicked.connect(app.stop_video_stream)
    controls.addWidget(QLabel("检测任意目标:")); controls.addWidget(app.prompt_input); controls.addWidget(start); controls.addWidget(stop)
    correct = QPushButton("纠错当前结果"); correct.clicked.connect(app.open_feedback); controls.addWidget(correct)
    from compare_ui import open_compare_dialog
    compare = QPushButton("对比模型…")
    compare.setToolTip("在当前图片上串行加载并比较两个选定模型（一次只占用一个模型），结果可导出为 PNG。")
    compare.clicked.connect(lambda: open_compare_dialog(app))
    controls.addWidget(compare)
    app.prompt_mode_hint = QLabel("留空：真正无提示检测；输入文字：指定目标。切换权重前会停止检测，只加载一个模型。")
    app.prompt_mode_hint.setWordWrap(True)
    app.video_label_tab1 = _video_label(); text_layout.addLayout(controls)
    text_layout.addWidget(app.prompt_mode_hint); text_layout.addWidget(app.video_label_tab1, 1)
    tabs.addTab(text_tab, "1. 无提示 / 开放文本检测")

    visual_tab = QWidget(); visual_layout = QVBoxLayout(visual_tab); reference = QGroupBox("视觉提示词（参考目标）"); reference_layout = QHBoxLayout(reference)
    side = QVBoxLayout(); upload = QPushButton("上传参考图片"); upload.clicked.connect(app.upload_ref_image)
    app.ref_class_input = QLineEdit(); app.ref_class_input.setPlaceholderText("可选：bottle / cup / cell phone")
    visual_hint = QLabel("YOLOE 真正视觉提示：上传图片并框选一个实例。\n标签仅用于显示；检测依据参考物体的视觉特征。\n与文本模式复用同一个模型，参考特征在后台提取。")
    visual_hint.setWordWrap(True)
    side.addWidget(upload); side.addWidget(QLabel("显示标签（不作为文本提示）:")); side.addWidget(app.ref_class_input); side.addWidget(visual_hint); side.addStretch()
    app.ref_canvas = DrawableLabel(app); app.ref_canvas.setFixedSize(450, 320)
    reference_layout.addLayout(side); reference_layout.addWidget(app.ref_canvas)
    visual_controls = QHBoxLayout(); visual_start = QPushButton("开始检测（视觉提示词）"); visual_start.clicked.connect(app.start_visual_stream)
    visual_stop = QPushButton("关闭检测"); visual_stop.clicked.connect(app.stop_video_stream)
    visual_controls.addWidget(visual_start); visual_controls.addWidget(visual_stop)
    visual_correct = QPushButton("纠错当前结果"); visual_correct.clicked.connect(app.open_feedback); visual_controls.addWidget(visual_correct)
    app.video_label_tab2 = _video_label(); visual_layout.addWidget(reference); visual_layout.addLayout(visual_controls); visual_layout.addWidget(app.video_label_tab2, 1)
    tabs.addTab(visual_tab, "2. 视觉提示词")
    from feedback_ui import FeedbackPanel
    app.feedback_panel = FeedbackPanel(app)
    tabs.addTab(app.feedback_panel, "3. 用户纠错 / 迭代优化")
    from auto_annotation_ui import AutoAnnotationPanel
    app.annotation_panel = AutoAnnotationPanel(app)
    tabs.addTab(app.annotation_panel, "4. 自动标注")
