import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                               QVBoxLayout, QGroupBox, QComboBox, QSlider,
                               QLabel, QPushButton, QTabWidget, QLineEdit, QFileDialog,
                               QMessageBox, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor


# ---------------------------------------------------------
# 1. Interactive Drawing Canvas
# ---------------------------------------------------------
class DrawableLabel(QLabel):
    def __init__(self, parent=None, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #555;")
        self.drawing = False
        self.start_pt = QPoint()
        self.end_pt = QPoint()
        self.box = None
        self.original_pixmap = None
        self.raw_image = None
        self.scaled_pixmap = None
        self.img_x_offset = 0
        self.img_y_offset = 0
        self.main_window = parent

    def set_image(self, image_path):
        self.raw_image = cv2.imread(image_path)
        if self.raw_image is None:
            return

        self.original_pixmap = QPixmap(image_path)
        self.start_pt = QPoint()
        self.end_pt = QPoint()
        self.box = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self.original_pixmap and not self.original_pixmap.isNull():
            label_size = self.size()
            self.scaled_pixmap = self.original_pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            self.img_x_offset = (label_size.width() - self.scaled_pixmap.width()) // 2
            self.img_y_offset = (label_size.height() - self.scaled_pixmap.height()) // 2

            painter.drawPixmap(self.img_x_offset, self.img_y_offset, self.scaled_pixmap)

            if not self.start_pt.isNull() and not self.end_pt.isNull():
                pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
                painter.setPen(pen)
                rect = QRect(self.start_pt, self.end_pt)
                painter.drawRect(rect)
        else:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "【点击此处选择参考图片】\n然后在图像上按住鼠标左键拖动画框")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.original_pixmap or self.original_pixmap.isNull():
                if self.main_window:
                    self.main_window.upload_ref_image()
                return

            self.drawing = True
            self.start_pt = event.position().toPoint()
            self.end_pt = self.start_pt
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pt = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pt = event.position().toPoint()

            x = min(self.start_pt.x(), self.end_pt.x())
            y = min(self.start_pt.y(), self.end_pt.y())
            w = abs(self.start_pt.x() - self.end_pt.x())
            h = abs(self.start_pt.y() - self.end_pt.y())
            self.box = (x, y, w, h)
            self.update()

    def get_cropped_template(self):
        if self.box is None or self.raw_image is None or self.scaled_pixmap is None:
            return None

        x, y, w, h = self.box
        rel_x = x - self.img_x_offset
        rel_y = y - self.img_y_offset

        scaled_w = self.scaled_pixmap.width()
        scaled_h = self.scaled_pixmap.height()
        orig_h, orig_w = self.raw_image.shape[:2]

        scale_x = orig_w / scaled_w
        scale_y = orig_h / scaled_h

        orig_x = int(rel_x * scale_x)
        orig_y = int(rel_y * scale_y)
        orig_box_w = int(w * scale_x)
        orig_box_h = int(h * scale_y)

        orig_x = max(0, min(orig_x, orig_w - 1))
        orig_y = max(0, min(orig_y, orig_h - 1))
        orig_box_w = max(10, min(orig_box_w, orig_w - orig_x))
        orig_box_h = max(10, min(orig_box_h, orig_h - orig_y))

        template = self.raw_image[orig_y:orig_y + orig_box_h, orig_x:orig_x + orig_box_w]
        return template


# ---------------------------------------------------------
# 2. ONNX Inference Engine (YOLO)
# ---------------------------------------------------------
class ONNXEngine:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = (640, 640)

    def load_model(self, model_path, device="CPU"):
        if device == "CUDA":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        try:
            new_session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = new_session.get_inputs()[0].name
            self.output_name = new_session.get_outputs()[0].name
            shape = new_session.get_inputs()[0].shape
            if len(shape) == 4:
                self.input_shape = (shape[2], shape[3])

            self.session = new_session
            active_providers = self.session.get_providers()
            print(f"成功加载模型 [{model_path}]，当前实际运行 Provider: {active_providers}")

            # 捕捉 CUDA 降级事件
            if device == "CUDA" and 'CUDAExecutionProvider' not in active_providers:
                return "CUDA_FALLBACK"

            return "SUCCESS"
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return str(e)

    def preprocess(self, img):
        h, w = img.shape[:2]
        new_h, new_w = self.input_shape
        r = min(new_h / h, new_w / w)
        pad_w, pad_h = int(round(w * r)), int(round(h * r))

        resized_img = cv2.resize(img, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = (new_w - pad_w) / 2.0, (new_h - pad_h) / 2.0

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))

        blob = padded_img[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        return np.expand_dims(blob, axis=0), (r, r), (dw, dh)

    def calculate_color_similarity(self, img1, img2):
        try:
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, similarity)
        except Exception:
            return 0.0

    def draw_detections(self, img, boxes, scores, class_ids, class_names=None, is_visual=False):
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if is_visual else (255, 0, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"Visual Match: {score:.2f}" if is_visual else f"class_{cls_id}: {score:.2f}"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def infer(self, frame, conf_thresh=0.5, iou_thresh=0.45, class_names=None, visual_template=None):
        if not self.session:
            cv2.putText(frame, "未加载 ONNX 模型，视频流正常运行中。", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            return frame

        blob, ratio, dwdh = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        pred = outputs[0][0].T

        temp_boxes, temp_scores, temp_class_ids = [], [], []

        # 1. 初筛候选框
        for i in range(pred.shape[0]):
            row = pred[i]
            class_scores = row[4:]
            cls_id = np.argmax(class_scores)
            score = class_scores[cls_id]

            if score >= conf_thresh:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x = int((cx - w / 2 - dwdh[0]) / ratio[0])
                y = int((cy - h / 2 - dwdh[1]) / ratio[1])
                box_w = int(w / ratio[0])
                box_h = int(h / ratio[1])

                temp_boxes.append([x, y, box_w, box_h])
                temp_scores.append(float(score))
                temp_class_ids.append(int(cls_id))

        final_boxes, final_scores, final_class_ids = [], [], []

        # 2. 执行 NMS（非极大值抑制）彻底解决性能卡顿
        if len(temp_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(temp_boxes, temp_scores, conf_thresh, iou_thresh)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw, bh = temp_boxes[i]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + bw), min(frame.shape[0], y + bh)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 3. 仅对过滤后的少数有效框进行色彩匹配
                    if visual_template is not None:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            resized_roi = cv2.resize(roi, (visual_template.shape[1], visual_template.shape[0]))
                            sim = self.calculate_color_similarity(resized_roi, visual_template)
                            if sim > 0.30:  # 稍微放宽相似度以适应增强后的图像
                                final_boxes.append([x1, y1, x2, y2])
                                final_scores.append(float(sim))
                                final_class_ids.append(999)
                    else:
                        final_boxes.append([x1, y1, x2, y2])
                        final_scores.append(float(temp_scores[i]))
                        final_class_ids.append(temp_class_ids[i])

        if final_boxes:
            frame = self.draw_detections(frame, final_boxes, final_scores, final_class_ids, class_names,
                                         is_visual=(visual_template is not None))
        return frame


# ---------------------------------------------------------
# 3. Zero-DCE Enhancement Engine
# ---------------------------------------------------------
class ZeroDCEEngine:
    """处理弱光增强，支持 ONNX 推理以及无模型时的纯算法 Fallback"""

    def __init__(self):
        self.session = None
        self.input_name = None
        self.is_loaded = False

    def load_model(self, model_path="zero_dce.onnx", device="CPU"):
        if not os.path.exists(model_path):
            self.is_loaded = False
            return "NO_MODEL"

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "CUDA" else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.is_loaded = True
            return "SUCCESS"
        except Exception as e:
            print(f"Zero-DCE 加载失败: {e}")
            self.is_loaded = False
            return str(e)

    def enhance(self, img):
        if not self.is_loaded:
            return self.fallback_enhance(img)

        # Zero-DCE 前处理 (规范化到 0~1)
        img_normalized = (img / 255.0).astype(np.float32)
        # 转换为 C, H, W
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        # 增加 Batch 维度 (1, C, H, W)
        input_tensor = np.expand_dims(img_transposed, axis=0)

        # ONNX 推理获取曲线参数 A
        outputs = self.session.run(None, {self.input_name: input_tensor})
        A = outputs[0]  # Shape: (1, 24, H, W)

        # 迭代应用曲线公式: I = I + A * (I - I^2)
        enhanced_tensor = input_tensor
        for i in range(8):
            A_i = A[:, i * 3:(i + 1) * 3, :, :]
            enhanced_tensor = enhanced_tensor + A_i * (enhanced_tensor - enhanced_tensor ** 2)

        # 后处理
        enhanced_frame = np.squeeze(enhanced_tensor, axis=0)
        enhanced_frame = np.transpose(enhanced_frame, (1, 2, 0))
        enhanced_frame = np.clip(enhanced_frame * 255.0, 0, 255).astype(np.uint8)

        return enhanced_frame

    def fallback_enhance(self, img):
        """如果在本地没有找到 zero_dce.onnx，则使用 LAB CLAHE 作为平替算法"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced_img


# ---------------------------------------------------------
# 4. Thread-Safe Video & Processing Worker
# ---------------------------------------------------------
class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, engine, dce_engine, source=0):
        super().__init__()
        self.engine = engine
        self.dce_engine = dce_engine
        self.source = source
        self.running = True
        self.conf = 0.5
        self.iou = 0.45
        self.class_names = []
        self.visual_template = None
        self.use_low_light_enhancement = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 弱光增强（如果开启）
            if self.use_low_light_enhancement:
                frame = self.dce_engine.enhance(frame)

            # 2. 目标检测
            processed_frame = self.engine.infer(
                frame,
                self.conf,
                self.iou,
                class_names=self.class_names,
                visual_template=self.visual_template
            )

            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(rgb_frame)
            self.msleep(30)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ---------------------------------------------------------
# 5. Main GUI Application
# ---------------------------------------------------------
class AIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 多模态检测与标注系统 (增强版)")
        self.resize(1400, 850)

        self.engine = ONNXEngine()
        self.dce_engine = ZeroDCEEngine()
        self.worker = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.init_left_panel(main_layout)
        self.init_right_panel(main_layout)

    def init_left_panel(self, parent_layout):
        left_group = QGroupBox("全局设置")
        left_group.setFixedWidth(300)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("模型尺寸:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["小型 (最快)", "中型 (均衡)", "大型 (精准)"])
        layout.addWidget(self.model_combo)

        layout.addWidget(QLabel("运行设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA"])
        layout.addWidget(self.device_combo)

        layout.addWidget(QLabel("置信度阈值:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        layout.addWidget(self.conf_slider)

        # 新增: 暗光增强开关
        self.dce_checkbox = QCheckBox("开启暗光增强 (Zero-DCE/CLAHE)")
        self.dce_checkbox.stateChanged.connect(self.update_enhancement_state)
        layout.addWidget(self.dce_checkbox)

        self.apply_btn = QPushButton("应用设置 / 加载模型")
        self.apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_btn)

        layout.addStretch()
        left_group.setLayout(layout)
        parent_layout.addWidget(left_group)

    def init_right_panel(self, parent_layout):
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_tab1(), "1. 文本提示词")
        self.tabs.addTab(self.create_tab2(), "2. 视觉提示词")
        self.tabs.addTab(QWidget(), "3. ONNX 推理 (开发中)")
        self.tabs.addTab(QWidget(), "4. 自动标注 (开发中)")
        parent_layout.addWidget(self.tabs)

    # --- TAB 1: TEXT PROMPT ---
    def create_tab1(self):
        tab = QWidget()
        layout = QVBoxLayout()

        ctrl_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("输入检测目标，例如：person, car, dog")
        self.prompt_input.textChanged.connect(self.update_worker_classes)

        start_btn = QPushButton("开始检测")
        start_btn.clicked.connect(
            lambda: self.start_video_stream(self.video_label_tab1, self.prompt_input.text(), is_visual=False))

        stop_btn = QPushButton("关闭检测")
        stop_btn.clicked.connect(self.stop_video_stream)

        ctrl_layout.addWidget(QLabel("文本提示词:"))
        ctrl_layout.addWidget(self.prompt_input)
        ctrl_layout.addWidget(start_btn)
        ctrl_layout.addWidget(stop_btn)

        self.video_label_tab1 = QLabel("视频画面预览")
        self.video_label_tab1.setAlignment(Qt.AlignCenter)
        self.video_label_tab1.setStyleSheet("background-color: black; color: white;")

        layout.addLayout(ctrl_layout)
        layout.addWidget(self.video_label_tab1, stretch=1)
        tab.setLayout(layout)
        return tab

    # --- TAB 2: VISUAL PROMPT ---
    def create_tab2(self):
        tab = QWidget()
        layout = QVBoxLayout()

        ref_group = QGroupBox("视觉提示词 (参考目标)")
        ref_layout = QHBoxLayout()

        upload_layout = QVBoxLayout()
        upload_btn = QPushButton("上传参考图片")
        upload_btn.clicked.connect(self.upload_ref_image)

        self.ref_class_input = QLineEdit()
        self.ref_class_input.setPlaceholderText("输入标签名 (例如：target_object)")

        upload_layout.addWidget(upload_btn)
        upload_layout.addWidget(QLabel("目标标签:"))
        upload_layout.addWidget(self.ref_class_input)
        upload_layout.addStretch()

        self.ref_canvas = DrawableLabel(self)
        self.ref_canvas.setFixedSize(450, 320)

        ref_layout.addLayout(upload_layout)
        ref_layout.addWidget(self.ref_canvas)
        ref_group.setLayout(ref_layout)

        det_layout = QHBoxLayout()
        start_btn = QPushButton("开始检测 (视觉提示词)")
        start_btn.clicked.connect(self.start_visual_stream)

        stop_btn = QPushButton("关闭检测")
        stop_btn.clicked.connect(self.stop_video_stream)

        det_layout.addWidget(start_btn)
        det_layout.addWidget(stop_btn)

        self.video_label_tab2 = QLabel("视频画面预览")
        self.video_label_tab2.setAlignment(Qt.AlignCenter)
        self.video_label_tab2.setStyleSheet("background-color: black; color: white;")

        layout.addWidget(ref_group)
        layout.addLayout(det_layout)
        layout.addWidget(self.video_label_tab2, stretch=1)

        tab.setLayout(layout)
        return tab

    def upload_ref_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择参考图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            if os.path.exists(file_name):
                self.ref_canvas.set_image(file_name)
            else:
                QMessageBox.warning(self, "路径错误", f"找不到该文件: {file_name}")

    # --- SHARED UI LOGIC ---
    def update_worker_classes(self, text):
        if self.worker:
            self.worker.class_names = [c.strip() for c in text.split(",") if c.strip()]

    def update_enhancement_state(self, state):
        if self.worker:
            self.worker.use_low_light_enhancement = (state == Qt.Checked.value)

    def apply_settings(self):
        device = self.device_combo.currentText()
        if self.worker:
            self.worker.conf = self.conf_slider.value() / 100.0

        # 加载目标检测模型
        model_map = {
            "小型 (最快)": "yolov8n.onnx",
            "中型 (均衡)": "yolov8s.onnx",
            "大型 (精准)": "yolov8m.onnx"
        }
        selected_size = self.model_combo.currentText()
        target_model_path = model_map.get(selected_size, "yolov8n.onnx")

        if not os.path.exists(target_model_path):
            fallback_model = "yolov8n.onnx"
            if target_model_path != fallback_model and os.path.exists(fallback_model):
                QMessageBox.warning(self, "模型文件丢失", f"未找到对应模型，已自动回退至：{fallback_model}")
                target_model_path = fallback_model
            else:
                QMessageBox.critical(self, "错误", "未找到任何 YOLO 模型文件！")
                return

        status = self.engine.load_model(target_model_path, device)
        msg = ""
        if status == "SUCCESS":
            msg += f"YOLO 模型加载成功: {target_model_path}\n"
        elif status == "CUDA_FALLBACK":
            QMessageBox.warning(self, "CUDA 降级警告",
                                "请求使用 CUDA，但 ONNX Runtime 回退到了 CPU！\n\n"
                                "原因：可能未安装 onnxruntime-gpu，或 CUDA/cuDNN 环境配置有误。\n"
                                "建议：在终端执行 'pip uninstall onnxruntime' 然后重新安装 'pip install onnxruntime-gpu'。")
            msg += f"YOLO 模型加载成功 (CPU降级): {target_model_path}\n"
        else:
            QMessageBox.critical(self, "错误", f"YOLO 加载失败:\n{status}")
            return

        # 尝试加载 Zero-DCE 模型
        if self.dce_checkbox.isChecked():
            dce_status = self.dce_engine.load_model("zero_dce.onnx", device)
            if dce_status == "SUCCESS":
                msg += "Zero-DCE 增强模型加载成功！"
            elif dce_status == "NO_MODEL":
                msg += "未找到 zero_dce.onnx 文件，已启用内置算法 (CLAHE) 平替方案进行暗光增强。"

        if msg:
            QMessageBox.information(self, "设置已应用", msg)

    def start_video_stream(self, target_label, initial_text, is_visual=False):
        if self.worker is not None:
            self.worker.stop()
            self.worker.frame_ready.disconnect()

        self.worker = VideoWorker(self.engine, self.dce_engine, source=0)
        self.worker.conf = self.conf_slider.value() / 100.0
        self.worker.use_low_light_enhancement = self.dce_checkbox.isChecked()

        if not is_visual:
            self.worker.class_names = [c.strip() for c in initial_text.split(",") if c.strip()]
            self.worker.visual_template = None
        else:
            template = self.ref_canvas.get_cropped_template()
            if template is None or template.size == 0:
                QMessageBox.warning(self, "提示", "请先上传参考图片，并在图片上用鼠标左键拖动画出红色方框圈中目标！")
                return
            self.worker.visual_template = template

        self.current_video_label = target_label
        self.worker.frame_ready.connect(self.update_video_label)
        self.worker.start()

    def start_visual_stream(self):
        self.start_video_stream(self.video_label_tab2, self.ref_class_input.text(), is_visual=True)

    def stop_video_stream(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        if hasattr(self, 'current_video_label') and self.current_video_label is not None:
            self.current_video_label.clear()
            self.current_video_label.setText("视频画面预览 (已停止)")
            self.current_video_label.setStyleSheet("background-color: black; color: white;")

    @Slot(np.ndarray)
    def update_video_label(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.current_video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.current_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIApp()
    window.show()
    sys.exit(app.exec())