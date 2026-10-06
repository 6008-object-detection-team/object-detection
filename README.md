# AI 多模态检测系统

基于 PySide6、Ultralytics YOLOE、Grounding DINO Tiny、ByteTrack 和 PyTorch 的桌面检测应用。支持摄像头、本地视频、单张图片和图片文件夹，重点是开放词汇检测、小目标、目标编号和暗光处理。

## 2026-10-07 合并更新

- 默认 YOLOE-26S、640、FP32；保留 26L、26X、11L 与 DINO Tiny。模型切换前释放旧模型。
- 视觉提示升级为真正的 YOLOE SAVPE：上传完整参考图并框选目标，用参考物体的视觉特征检测，不再是颜色相似度筛选。DINO 仍为文本后端。
- 新增 SAHI 小目标切片：320/512/640、20% 重叠、最多 9 块串行，与旧四分块互斥，默认关闭。
- 摄像头独立采集，只保留最新待检 / 待显示帧；增加显示 FPS、延迟 p50/p95 与替换计数。离线视频仍逐帧处理。
- 新增 MP4 检测录像和独立最大框数控制；摄像头慢推理时重复处理帧补齐录像时间轴，录像无音轨。
- 保留图片翻页、中文图片路径、ByteTrack 的 person1/person2 编号和自适应暗光处理。
- 增加 Windows / Linux 资源保护、DINO 旧评测接口兼容、模型下载校验、小型 COCO 测试资料和评测工具。

详细范围见 [公开整合更新](docs/UPDATE_20261007.md)，贡献与第三方许可见 [来源说明](UPSTREAM.md)。本次只发布整理后的源码、文档和必要测试资料，不携带私人 Git 历史、个人文件或原始实验日志。

## 安装：先选 PyTorch 后端

已有环境可直接更新，虚拟环境不能跨电脑复制。Windows 示例：

~~~powershell
conda create -n object-detection python=3.11 -y
conda activate object-detection
# NVIDIA 显卡使用 CUDA 版；没有兼容 NVIDIA 显卡则选择 CPU 版。二选一。
python -m pip install -r requirements-cuda.txt
# python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-windows.txt
~~~

requirements-windows.txt 包含通用、SAHI / pycocotools 评价和 DINO 依赖，并固定 Ultralytics 8.4.172、Transformers 4.57.6。只装基础 YOLOE 时可用 requirements.txt；SAHI 需要 requirements-evaluation.txt，DINO 需要 requirements-dino.txt。旧 requirements-grounding-dino.txt 已转向相同的兼容依赖。

本机沿用原 Conda 6008，未重建环境；已验证 Python 3.10.21、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 1.26.4、SciPy 1.15.3、OpenCV 4.11.0.86、PySide6 6.11.2、SAHI 0.11.36。其它电脑按 Python、显卡和驱动选择兼容组合，不能直接套用本机性能。

Linux 可建立 .venv，安装一种 PyTorch 后端，再安装通用 / 可选依赖，通过 run_app.sh 启动。Windows 的详细步骤见 [Windows 配置](docs/WINDOWS_SETUP.md)。

## 下载源码与模型

原仓库历史仍包含大型模型和 build，运行新版源码不需要下载历史打包文件。可使用 GitHub Download ZIP 获取源码，或安装 Git / Git LFS 后跳过克隆时自动下载大文件：

~~~powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone https://github.com/6008-object-detection-team/object-detection.git
cd object-detection
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
# 默认只取得 26S 和 MobileCLIP2，合计约 272 MiB；校验大小与 SHA-256。
python tools/fetch_models.py
# 如需检查全部六个 YOLOE / 编码器权重：
# python tools/fetch_models.py --all
~~~

Git LFS 文本指针不是可加载权重；下载脚本会取得真正二进制文件。固定下载来源、大小和 SHA-256 在 model_sources.json。已有且核验通过的权重不重复下载；不需要执行 git lfs pull 来拉取所有历史 build。

DINO 模型为 IDEA-Research/grounding-dino-tiny，首次选择时需要联网取得权重。默认 Hugging Face 缓存保存在项目 .cache/huggingface；缓存和模型不属于本轮新增上传内容。

## 启动与检测

~~~powershell
python main.py
# 或在已有环境下：
.\run_app.bat
~~~

程序启动时不自动加载模型。推荐先用 YOLOE-26S、CUDA、640，关闭 SAHI、旧分块和暗光补检，点击“应用设置 / 加载模型”，选择输入，再开始检测。英文目标示例：person, bottle, cup, cell phone。

run_app.bat 可选择 OBJECT_DETECTION_PYTHON、项目 .venv 或当前激活的环境；6008 是本机回退配置，不是组员必须使用的环境名称。停止后使用“停止并释放模型 / 显存”。

| 模型 | 主要用途 |
| --- | --- |
| YOLOE-26S | 默认实时方案起点，较轻 |
| YOLOE-26L / 26X | 更大候选，速度与精度需按场景实测 |
| YOLOE-11L | 旧模型对照 |
| Grounding DINO Tiny | 英文开放词汇文本检测对照，不支持 SAVPE 页面 |

推理尺寸最高 960、原图最多 1600 万像素。提高分辨率不能恢复摄像头未拍清的细节；SAHI 增加推理次数，可能引入误检，不默认开启。

## 视觉提示与小目标

在视觉页上传完整参考图，框选目标并填写显示名称，再开始检测。名称用于显示，SAVPE 参考编码驱动检测；单张参考的外观覆盖有限，不能保证其效果优于文本。

SAHI 切片 20% 重叠、最多 9 块串行，超过限制会停止并提示调整输入或切片大小。与旧四分块补检互斥，最终结果映射回原图并去重。

## ByteTrack 编号

视频 / 摄像头同一次会话内，同类目标显示为 person1、person2、bottle1 等。低分框只能短暂续接已有轨迹，不创建新编号；停止重启、切换输入或修改提示词会开启新会话。单张图片和文件夹每张图片独立编号。

编号是运动轨迹，不是人脸或真实身份识别。长时间离开、严重遮挡、快速运动和交叉仍可能换号，不能保证永不交换 ID。

## 暗光、录像与实时指标

暗光处理使用亮度自适应提亮、温和降噪、增益限制和跨帧平滑；正常曝光自动跳过，全黑画面不制造细节。

- 原图检测 + 暗光补检：保留原图框，仅在需要时额外检测增强图；增加推理开销，新增框仍可能误报。
- 仅增强预览：增强显示，检测仍用原图，不增加模型检测次数。

摄像头采集与显示各只有一个待处理结果，较慢时新结果替换旧结果；这减少排队，不代表模型前向计算变快。离线视频按顺序处理每帧。显示 FPS 是 Qt 结果提交频率，读帧到界面延迟不包含曝光、驱动内部缓存或屏幕实际刷新。

MP4 无音轨。摄像头录像用重复处理帧保持源时间轴，重复帧无法恢复未采样动作；本地视频按源 FPS 保存处理帧。详细限制和实测方法见 [视频实时性说明](docs/VIDEO_REALTIME.md)。

## 数据和评测工具

小型 COCO 资料保留人工标注、原始 ID、图片来源、逐图许可和 SHA-256：

- datasets/mini_eval：24 图、71 个瓶子 / 杯子 / 手机框，另有 3 张独立参考。
- datasets/mini_development_quick：6 图快速开发资料。
- datasets/mini_train_probe：48 / 12 图训练与验证准备资料，与诊断集和参考图 ID 分离。

它们用于功能与开发诊断，不是未见最终测试或完整 COCO 榜单。工具存在不代表训练已执行；本轮没有运行新训练。数据用途见 [数据说明](docs/dataset.md)。

evaluate_mini_dataset.py 提供文本、文本 + SAHI、SAVPE、SAVPE + SAHI 的人工 GT 评价；benchmark_detection.py 分别改变置信度或最大框数，避免把两个变量联动。没有人工标注时，框数与耗时不能称为准确率或召回率。见 [评价流程](docs/evaluation.md) 和 [参数实验](docs/experiments.md)。

pan_deng/ 保留 Pan Deng 的 240 图评价、阈值 / 框数扫描和绘图工具。旧 token 打分 DINO 与当前 GUI 的后处理配置不同，旧 AP 不能直接当作新 GUI AP。

## 资源保护、验证与未完成事项

保持单模型、batch=1、3GiB PyTorch 推理预算、最多 9 个串行切片。加载前需至少 4GiB 空闲显存，推理前至少 1GiB，系统可用 RAM 至少 1.5GiB；GPU 文件锁跨进程互斥。OOM 时停止，不静默重试或切换设备。训练与导出须独立检查资源。

~~~powershell
python -m pip check
python -m unittest discover -q
# --check 会检查六个权重；仅下载默认两件时报告其它模型未安装是预期的。
.\run_app.bat --check
# DINO 缓存已备好时：
# $env:HF_HUB_OFFLINE='1'
# .\run_app.bat --check --dino
~~~

完整本地合并版已验证 199 项回归：198 项通过，1 项因缺少完整 COCO 原始标注缓存跳过。真实 CUDA 五路径，以及 Qt 图片文件夹、短视频、目标编号和模型切换验证通过。公开发布目录又重跑了全部 199 项，结果同为 198 通过、1 跳过；依赖检查和 Qt 启动也通过，13 个核心运行文件与实测合并版逐字节一致。冷启动计时和单图 0 框结果不用于宣称速度或精度提升。

尚未在本轮验证真实摄像头优化前后性能、长时录像、完整 COCO 精度或训练。模型微调、自动标注页面仍开发中，未实现中英文界面切换。add picture.py 和 build 是旧版内容，不是当前入口；运行新版不需要 restore_large_files.py 还原历史打包文件。
