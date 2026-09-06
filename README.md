# AI 多模态检测系统

基于 PySide6、Ultralytics YOLOE 和 PyTorch 的桌面检测应用，支持摄像头与本地视频。当前版本重点改善摄像头小目标检测和暗光处理。

## 下载与配置环境

先安装 Git（包含 Git LFS）和 Python 3.11 / Miniconda，然后执行：

~~~powershell
git lfs install
git clone https://github.com/6008-object-detection-team/object-detection.git
cd object-detection
git lfs pull
conda create -n object-detection python=3.11 -y
conda activate object-detection
python -m pip install -r requirements.txt
python check_environment.py
python main.py
~~~

模型使用 Git LFS 保存，必须下载实际权重；网页中的小型文本指针不是模型。建议使用以上 Git 命令下载，GitHub 的 Download ZIP 默认可能只有模型指针。环境安装需要网络；四种 YOLOE 模型和两种文本编码器均已包含，下载完整后加载这些模型不需要再次下载权重。完整文件约 5 GB，包含历史打包中间文件。

Windows 也可以在已激活的环境中执行 **run_app.bat**。脚本优先使用 `OBJECT_DETECTION_PYTHON` 指定的解释器、当前虚拟环境/Conda 环境或项目 `.venv`；本机原有环境仍作为回退，最后尝试 PATH 中的 Python。组员无需修改代码里的盘符路径。`run_app.bat --check` 只检查环境和模型文件。

依赖文件包含 CUDA 12.8 的 PyTorch 下载源；没有 NVIDIA 显卡时应用会使用 CPU。摄像头权限、驱动和推理速度取决于各自设备。当前已验证的环境是 Windows、Python 3.11.15、PyTorch 2.9.1+cu128、Ultralytics 8.4.126。

**当前入口是 main.py。** add picture.py 是旧 ONNX 版本；build 中是历史打包中间文件，不包含后续改进，运行源码不依赖它们。

## 完整文件与大文件还原

仓库包含源码、全部现有模型、文档、示例图片、验证结果、安装日志、IDE 配置、缓存和 build 中间文件。本地 `.git` 历史及认证数据不上传；虚拟环境和 `.env` 凭据文件也不纳入版本控制。

`build/app/AIDetectionApp.pkg` 原文件为 3,406,349,084 字节，超过 [GitHub Free/Pro 的 Git LFS 单文件 2 GB 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)，因此无损拆分为相邻的四个 `.part001`–`.part004` 文件。`large_files.json` 记录每个分片及原文件的 SHA-256。需要完整恢复历史 build 目录时执行：

~~~powershell
python restore_large_files.py
~~~

该命令只使用 Python 标准库，校验分片后恢复原文件；若同名原文件已有不同内容，会停止并保留已有文件。`python restore_large_files.py --check` 只做完整性校验。运行检测应用不需要还原这个历史打包文件。

## 摄像头小目标设置

1. 默认使用 **YOLOE-26L + CUDA + 960**；启动时根据 PyTorch 是否支持 CUDA 选择设备。
2. 点击“应用设置 / 加载模型”，输入目标名称，例如 pen, key, bottle, cell phone，点击“开始检测”。
3. 小物体漏检时，可以将推理分辨率调到 **1280**，或勾选“小目标分块补检”。不要只降低置信度门槛来代替精度改善。
4. 精度优先时可选 **YOLOE-26X**；速度优先可选 26S。也保留 11L 作为旧模型对照。

| 模型 | 用途 |
| --- | --- |
| YOLOE-26L | 默认，兼顾实时性与精度 |
| YOLOE-26X | 更大模型，推理较慢 |
| YOLOE-26S | 更小模型，适合速度优先或 CPU |
| YOLOE-11L | 与原模型对照 |

摄像头会请求 1920×1080，实际取决于设备支持的分辨率，状态栏显示实际输入尺寸。分块模式在完整画面之外，额外检测四个重叠区域，将检测框映射回原图并去重，有助于保留小目标细节；每帧增加四次推理，默认关闭。提高分辨率不能恢复摄像头本来没有拍清的细节。

模型输出分数保持原值。置信度滑块是显示筛选门槛，不是识别准确率。分块补充框标记 +T，暗光补充框标记 +E。

[官方 YOLOE 模型与精度说明](https://docs.ultralytics.com/models/yoloe/)：新版接入 YOLOE-26，并使用非端到端检测头（end2end=False）。公开基准结果不能直接当成本项目摄像头场景的准确率。

## 自适应暗光增强

原来的流程在缺少 zero_dce.onnx 时，对每帧执行固定强度 CLAHE，再仅检测增强图。当前改为无需下载增强权重的自适应算法：

- 根据画面亮度选择提亮强度，正常曝光自动跳过增强。
- 温和双边滤波抑制噪声，平滑光照增益提亮暗部，限制最大增益并保护高光。
- 同比例调整 B/G/R 通道以尽量保留颜色；平滑相邻帧的曝光参数，场景明显变化时重置。
- 默认强度 65%，可以实时调整。全黑图像保持原样，避免制造不存在的细节。

提供两种方式：

| 方式 | 检测行为 |
| --- | --- |
| 原图检测 + 暗光补检 | 原图照常检测；需要提亮时，再检测增强图并补充不重复的目标 |
| 仅增强预览 | 只提亮显示画面，检测始终使用原图；不增加模型推理次数 |

暗光补检保留原图已有的框、类别和置信度，新增框使用 max(用户阈值, 0.45) 筛选，并抑制重叠的冲突类别。这样不会因增强替换原图而删除已有检测，但新增框仍可能误报，不能保证所有场景的准确率提高。暗光补检增加一次模型推理；“仅增强预览”也有图像处理开销。

视觉模板功能仍使用颜色相似度筛选；现在始终从未增强、未画框的原图取样，以减少提亮和画框对匹配的干扰。匹配分数与模型置信度分开显示。

## 模型与运行环境

已在本机下载并验证：

- yoloe-26l-seg.pt
- yoloe-26x-seg.pt
- yoloe-26s-seg.pt
- mobileclip2_b.ts（新版文本编码器）
- 原有 yoloe-11l-seg.pt 与 mobileclip_blt.ts 保留

26S 也已包含在仓库中。新版要求 ultralytics>=8.4.126,<9；本机验证版本为 8.4.126。文本提示词继续使用官方 ultralytics/CLIP 依赖。启动脚本会切换到项目目录，避免重复下载文本编码器。

## 验证与限制

~~~powershell
python -m unittest -v test_detection
python validate_improvements.py
~~~

回归测试覆盖正常光照不变、暗部提亮与颜色保持、噪声、补检去重、原图结果保留、分块坐标、模型切换、视频异常释放和实时设置。

对比脚本使用 Ultralytics 自带样图，以及确定性生成的小目标和暗光场景，对比旧模型、新模型、分辨率、旧 CLAHE 和新暗光流程。结果在 validation/comparison.json，附有可检查的检测图。统计的检测数量、置信度和处理耗时不是准确率、召回率或 mAP；完整精度评估仍需要来自实际摄像头的标注样本。

开放词汇检测依然需要提供目标名称；本次没有增加不输入提示词的自动物体命名。模型微调和自动标注页面仍为开发中。
