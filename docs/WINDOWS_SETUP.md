# Windows 安装与启动

## 使用已有环境

项目入口是 `main.py`。在项目目录中激活自己已安装依赖的环境，再运行：

```powershell
conda activate 6008
python main.py
# 或
.\run_app.bat
```

`6008` 是本机已有环境名称，不要求所有组员使用同名环境。PyCharm / VS Code 应选择该环境的 Python，工作目录设为项目根目录；不要复制其他电脑的虚拟环境。

## 从零安装

使用 Python 3.10 或更新的兼容版本。先选择一种 PyTorch 后端，再安装通用、SAHI、评价和可选 DINO 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# NVIDIA 显卡选择 CUDA 版；没有兼容 NVIDIA 显卡则改用 requirements-cpu.txt。
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe tools/fetch_models.py
.\run_app.bat
```

不要同时安装 CPU / CUDA 两份后端。`requirements-windows.txt` 固定已验证的 Ultralytics 8.4.172 和 Transformers 4.57.6；本机 Python 3.10 使用 SciPy 1.15.3，Python 较新环境可由通用依赖范围选择兼容版本。

本机验证组合：Python 3.10.21、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 1.26.4、OpenCV 4.11.0.86、PySide6 6.11.2、SAHI 0.11.36、pycocotools 2.0.11，RTX 4080 Laptop GPU。它是已验证的一种组合，不保证其它硬件具有相同速度。

## 模型与检查

默认 YOLOE-26S 需要 `yoloe-26s-seg.pt` 和 `mobileclip2_b.ts`，下载脚本会校验大小与 SHA-256。`--all` 取得六个 YOLOE / 编码器权重：

```powershell
python tools/fetch_models.py --all
.\run_app.bat --check
# DINO 首次选择时需要联网下载；准备好缓存后可检查离线可用性。
$env:HF_HUB_OFFLINE='1'
.\run_app.bat --check --dino
```

默认缓存是项目 `.cache/huggingface/`。未准备 DINO 时不应启用离线模式来做首次下载。只有百余字节的 Git LFS 文本指针不是权重；下载脚本能够用实际二进制替换指针，不需要还原历史 build。

## 系统差异与限制

Windows GPU 锁用 `msvcrt`，Linux 用 `fcntl`；Windows 内存统计用 `psutil`。已有路径 / 缓存写入兼容处理，Windows 无符号链接权限的测试夹具也已适配。

Windows 使用 `run_app.bat`；Linux 使用 `run_app.sh`，并通过 `.venv/bin/python` 或 `OBJECT_DETECTION_PYTHON` 选择自己的解释器。Linux shell 脚本不能直接作为 PowerShell 命令运行。

启动后先用 YOLOE-26S、CUDA、640，关闭 SAHI、旧分块和暗光补检，加载模型，再开始检测。结束时使用“停止并释放模型 / 显存”。低帧率需区分推理耗时与采集 / 显示延迟，不从显卡利用率单独判断原因。

摄像头权限、设备读取、长时录像及优化前后 FPS 仍需在每台电脑实测。本轮验证范围见 [公开整合更新](UPDATE_20261007.md)。
