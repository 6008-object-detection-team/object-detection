# AI 多模态检测与标注系统 (增强版)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PySide6](https://img.shields.io/badge/PySide6-GUI-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-red)
![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-Inference-orange)

基于 Python 和 PySide6 构建的桌面端 AI 视觉应用。默认使用 **YOLOE-11 开放词汇模型**，可按自然语言提示词检测训练集中未固定定义的类别；例如 `red backpack, cupcake, forklift`。它支持实时摄像头流、视觉特征匹配和暗光画质增强。

## ✨ 核心特性

- **🚀 开放词汇、高精度检测**
  - 提示词不再受 COCO 80 类限制；可输入英文目标名称、短语和多个类别（用中英文逗号、分号或换行分隔）。
  - 默认选择 **YOLOE-11l** 大型模型，以精度优先；小型模型适合实时性优先的场景。
  - 支持一键切换 CPU / CUDA 运行设备。若 CUDA 失败或未配置，系统会自动降级至 CPU 运行并发出友好提示。
- **🌙 暗光视频增强 (Zero-DCE)**
  - 针对暗光/弱光环境提供视频实时提亮功能。
  - 优先使用基于深度学习的 `zero_dce.onnx` 模型进行自适应光照增强。
  - 提供智能 Fallback 机制，在缺少模型时自动切换为基于 LAB 色彩空间的 CLAHE（直方图均衡化）传统平替算法。
- **📝 多模态检测：文本提示词 (Tab 1)**
  - 基于 YOLO 的泛化目标检测。
  - 用户可动态输入需检测的目标类别名称，调整置信度滑块，实现实时检测并在画面上渲染包围盒。
- **🎯 多模态检测：视觉提示词 (Tab 2)**
  - **交互式画布**：上传本地参考图片，使用鼠标直观地拖拽截取目标区域作为“视觉模板”。
  - **基于色彩的二次匹配**：在目标检测的基础上，结合 HSV 直方图相似度算法（Correlation），在实时视频流中精准找寻与模板特征相符的特定目标。
- **⚡ 极致的性能优化**
  - **多线程架构**：采用 `QThread` 分离视频帧的抓取推理与 UI 渲染，彻底告别界面假死和卡顿。
  - **高效后处理**：集成 OpenCV 的 NMS（非极大值抑制）功能，剔除冗余重叠框，提升渲染与后续匹配计算的性能。

## 📦 依赖与安装

确保您的系统已安装 Python 3.8 或更高版本。推荐使用 Conda 或 venv 创建虚拟环境。

```bash
# 1. 克隆代码仓库或下载源码
git clone https://github.com/yourusername/ai-multimodal-system.git
cd ai-multimodal-system

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. [可选] 如果需要开启 GPU (CUDA) 推理加速，请安装对应的 ONNX Runtime GPU 版本
# 请确保先卸载纯 CPU 版本，并安装好 CUDA Toolkit 和 cuDNN
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

## 🛠️ 模型准备

开放词汇文本检测基于 Ultralytics 和 PyTorch。首次点击“应用设置”时，Ultralytics 会自动下载所选的官方 YOLOE-11 权重，因此首次加载需要网络。

- **YOLOE-11 开放词汇模型**（推荐 L）:
  - `yoloe-11l-seg.pt`（默认，高精度）
  - `yoloe-11s-seg.pt`（较快）
  - `yoloe-11m-seg.pt`（均衡）
- **弱光增强模型** (可选，若不提供则降级为传统算法):
  - `zero_dce.onnx`

## 🚀 使用指南

```bash
# Windows: double-click run_app.bat, or activate the verified environment first
conda activate pytorch
python new.py
```

请不要直接用系统 Python 双击 `new.py`；本项目已在 `pytorch` Conda 环境中安装并验证了 CUDA PyTorch、Ultralytics、OpenCV、PySide6、ONNX Runtime 和官方 CLIP。

### 1. 启动与全局配置
1. 在左侧面板，默认保留 **大型（高精度）**；实时性要求高时再改为小型。
2. 选择 **运行设备**（推荐有 N 卡的用户选择 CUDA，否则选择 CPU）。
3. 如需在极暗环境下工作，勾选 **开启暗光增强**。
4. 点击 **“应用设置 / 加载模型”**。

### 2. 文本提示词检测 (Tab 1)
1. 在文本框中输入想要检测的任意目标或短语（如 `person, dog`、`red backpack, cupcake`）。提示词用英文通常识别效果更好。
2. 点击 **“开始检测”**，系统将打开本地摄像头并进行实时追踪。
3. 拖动左侧滑块，动态调整置信度阈值。

### 3. 视觉提示词检测 (Tab 2)
1. 点击 **“上传参考图片”** 加载一张包含目标物体的图片。
2. **非常重要**：在图片画面上，按住鼠标左键拖动画出一个红色框，精准框选出目标作为特征模板。
3. （可选）输入目标标签名称。
4. 点击 **“开始检测 (视觉提示词)”**。程序会在画面中寻找与您框选模板色彩特征高度相似的物体。

## 📅 待办与未来规划 (TODO)

- [ ] **Tab 3: 高级 ONNX 推理模块**：开发独立的模型调试和推理输出查看面板。
- [ ] **Tab 4: 自动标注工具**：支持将视频帧中识别出的目标一键导出为 Pascal VOC (XML) 或 YOLO (TXT) 标注格式，辅助构建微调数据集。
- [ ] 增加更多相似度匹配算法（如基于深度学习的特征向量比对/ReID）以替代纯色彩直方图，提升“视觉提示词”的精准度。
- [ ] 支持打开本地视频文件或 RTSP 网络串流作为数据源。

## 📄 许可证

MIT License. 欢迎自由使用、修改和分发。
