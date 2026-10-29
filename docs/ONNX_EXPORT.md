# ONNX 导出与 ONNX Runtime 对照（工具，未接入 GUI）

项目章程计划过 ONNX Runtime 检测核心，8 月 31 日的后续方向也写了“TensorRT/ONNX 导出”。本轮补上的是**导出工具、无 torch 的 ONNX Runtime 引擎类和带人工真值的对照实验**。界面主检测路径**没有改动**，仍是 PyTorch。

| 状态 | 内容 |
| --- | --- |
| 已验证（2026-10-28，本机） | YOLOE-26S 固定 3 类（bottle、cup、cell phone）导出 3 个 ONNX：完整分割图 `seg`、仅检测分支 `det`、高宽动态的 `det --dynamic-hw`；ONNX Runtime CPU 在 24 图诊断集上与同输入几何的 PyTorch 预测一致；CPU 时延、峰值内存、文件大小按下文协议实测。`seg` 和 `det` 三次导出（实验目录 `models/`、`models_reexport_check/`、`models_final/`）去掉元数据后序列化的计算图完全相同（只有 Ultralytics 写入的 `date` 不同）。`det --dynamic-hw` 用同一版导出工具（`6e4cd80`，此后未改）再导出一次：节点、权重、输入输出和 opset 完全相同，只有中间张量的形状注释（`value_info` 中的符号维度表达式）不同；`models/` 中最早的 dyn 文件由更早版本的工具导出，与最终文件不同，没有用于结果。 |
| 未验证 | onnxruntime-gpu / TensorRT（本环境未安装 onnxruntime-gpu，GPU 只测了 PyTorch CUDA 作参考）；Windows；26L / 26X / 11L；FP16 / INT8；摄像头长时运行；其它类别组合的精度。`coco10` 类别预设可以导出，但没有评测。 |
| 开发中 / 未做 | 接入 GUI；无提示（PF，LRPC 4,585 类）导出；视觉提示（SAVPE）导出；SAHI / 暗光 / ByteTrack 与 ONNX 引擎组合的实测（`OnnxRuntimeEngine` 继承了共用流程，但没有单独验证）。 |

## 固定类别意味着什么

导出前用原权重的文字编码器（`mobileclip2_b.ts`）把类别编码一次，走与应用相同的 `set_classes` 路径；Ultralytics 导出时把文字嵌入融合进检测头。ONNX 图里**没有文字编码器**，所以：

- 运行时不能改类别、不能输入新的文字，也不能用参考图视觉提示。`OnnxRuntimeEngine` 收到不同的类别列表会直接报错，不会悄悄忽略。要换类别就重新导出。
- 换来的是运行时不需要 torch、Ultralytics 和 242 MiB 的文字编码器。

## 安装

```bash
python -m pip install -r requirements-onnx.txt   # onnx、onnxslim；onnxruntime(CPU) 已在 requirements.txt
```

导出工具会关闭 Ultralytics 的自动安装（`YOLO_AUTOINSTALL=False`）：缺依赖就停止，不会改动当前环境。本轮使用 onnx 1.23.2、onnxslim 0.1.98、onnxruntime 1.31.1。

## 导出

先在界面中“停止并释放模型”，同一时间只运行一个模型任务。导出只用 CPU，开始前单独检查可用内存（≥ 3 GiB）和输出盘空间（≥ 1 GiB），推理用的 3 GiB 预算不用于导出。已有文件不会被覆盖。`.onnx` 和 manifest 属于构建产物，Git 已忽略 `*.onnx`，请放在仓库外或 `work/` 中。

```bash
# 只检查参数并打印计划，不加载模型
python tools/export_onnx.py --plan
# 默认：mini_eval 的 3 类，seg + det，imgsz 640，opset 17，静态 batch=1
python tools/export_onnx.py --output-dir work/onnx_export
# 高宽动态（batch 仍为 1），与界面相同的最小矩形 letterbox，推荐用于 CPU
python tools/export_onnx.py --output-dir work/onnx_export --variants det --dynamic-hw
# 其它类别：240 图 COCO 对照的 10 类，或自定义
python tools/export_onnx.py --class-set coco10 --variants det --dynamic-hw --output-dir work/onnx_export
python tools/export_onnx.py --classes "person,bottle,cup" --variants det --dynamic-hw --output-dir work/onnx_export
```

| 变体 | 内容 | 本轮文件 |
| --- | --- | --- |
| `seg` | 完整分割图；输出 `output0 [1,39,8400]` 和 mask 原型 `output1 [1,32,160,160]`，引擎只读 `output0` | 41,777,981 B |
| `det` | 仅检测分支转换：`yoloe-26s.yaml` 构建检测结构，822 个张量逐一与原权重比对完全相同，136 个分割分支张量不导出；Ultralytics 统计 35.5 → 21.9 GFLOPs | 38,159,072 B |
| `det --dynamic-hw` | 同 `det`，输入 `[1,3,height,width]`；导出后把 batch 维重新声明为 1，并在 ORT 上检查 640×640 和 480×640 | 38,236,808 B |

所有变体都是 FP32、opset 17（比 Ultralytics 在 torch 2.9 下默认的 18 保守；本机 ORT 1.31.1 已验证）、经 onnxslim 简化、one2many 原始输出（xywh + 类别分数），NMS 在图外由引擎完成，所以置信度、IoU 和框数上限可以在运行时设置。仅检测分支转换只允许 `yoloe-26s-seg.pt`，与此前 GPU 消融相同，避免把权重转移到不匹配的结构。

每个 `.onnx` 旁边有同名 `.manifest.json`：原权重和文字编码器的路径 / 大小 / SHA-256、类别及来源、文字嵌入 SHA-256、导出参数（imgsz、opset、batch、dynamic、stride、nms、head）、输入输出名与形状、前处理约定、ONNX 文件 SHA-256、onnx/ORT 检查结果、软件版本、源码哈希、主机与负载、导出进程峰值 RSS。引擎加载前核对 ONNX 文件哈希，不一致就拒绝加载。结果所用的 `models_final/` 导出每个变体约 1.0–1.4 s，导出进程峰值 RSS 1,337–1,367 MiB，CPU 冷启动文字编码约 0.82–0.85 s（较早、已被替代的 `models/` 导出中最长为 1.76 s 和 1.35 s）。

## 引擎（工具用）

```python
from onnx_engine import OnnxRuntimeEngine
engine = OnnxRuntimeEngine()
status = engine.load_model("work/onnx_export/yoloe-26s-det-mini_eval-640-dyn.onnx", "CPU", threads=8)
detections = engine.detect(frame_bgr, conf=0.25, iou=0.5, max_det=100)  # 与 OpenVocabEngine 相同的 Detection
```

- 不导入 torch。letterbox、xywh→xyxy、取最高类别、NMS（默认跨类别，与应用相同）、坐标还原和裁剪都按 Ultralytics 8.4.172 用 NumPy 实现；单元测试逐像素 / 逐框对照 Ultralytics 的参考函数。
- 只使用 `CPUExecutionProvider`。请求 CUDA 会直接返回错误，不会悄悄改用 CPU。保留 `resource_safety.py` 的内存与输入尺寸检查；只支持 one2many、FP32。
- 继承 `OpenVocabEngine` 的 `process_frame`（切片、暗光、跟踪、画框）流程，但这些组合没有单独实测；GUI 仍未接入。

## 对照实验

```bash
python tools/benchmark_onnx.py \
  --onnx-seg work/onnx_export/yoloe-26s-seg-mini_eval-640.onnx \
  --onnx-det work/onnx_export/yoloe-26s-det-mini_eval-640.onnx \
  --onnx-det-rect work/onnx_export/yoloe-26s-det-mini_eval-640-dyn.onnx \
  --backends pt_cpu_rect,pt_cpu_square,pt_cpu_det_rect,pt_cpu_det_square,ort_cpu_seg,ort_cpu_det,ort_cpu_det_rect,pt_cuda_rect \
  --repeats 5 --threads 2 --output work/onnx_benchmark_t2
```

`--plan` 只验证输入并打印每次重复的执行顺序。正式计时至少重复 3 次；`--limit N` 用于流程检查，报告会标为 smoke。

| 后端 | 运行时 | 计算图 | letterbox |
| --- | --- | --- | --- |
| `pt_cpu_rect` | PyTorch CPU（**当前界面路径**：OpenVocabEngine + 只取框的预测器） | seg | 最小矩形 |
| `pt_cpu_square` | 同上，强制方形输入（与静态 ONNX 相同的几何），**速度比值的参考** | seg | 640×640 |
| `pt_cpu_det_rect` / `pt_cpu_det_square` | PyTorch 仅检测分支转换 | det | 矩形 / 方形 |
| `ort_cpu_seg` / `ort_cpu_det` | ONNX Runtime CPU，静态图 | seg / det | 640×640 |
| `ort_cpu_det_rect` | ONNX Runtime CPU，高宽动态图 | det | 最小矩形 |
| `pt_cuda_rect` | PyTorch CUDA 参考（3 GiB 预算、GPU 锁） | seg | 最小矩形 |

协议：

- 数据：`datasets/mini_eval` 全部 24 图、71 个人工 COCO 框；模型 YOLOE-26S（SHA-256 `48f24206…0654`）；imgsz 640、conf 0.05、NMS IoU 0.5、跨类别 NMS、每图最多 100 框、FP32。与 `evaluate_mini_dataset.py` 的 text 模式相同；pycocotools 计算 AP。
- 每个后端每次重复都在**新进程**中运行，同一时间只有一个模型；每次重复按轮换顺序依次运行全部后端，前后记录 `os.getloadavg()`。峰值 RSS 是该进程自身的生命周期峰值（含加载）。
- 计时为一次 `detect()`：前处理、前向、解码 / NMS、结果转回；不含读图、加载、文字编码和预热（每进程先跑一次建立预测器，再预热 1 次）。CUDA 前后同步。
- 线程：PyTorch 用 `torch.set_num_threads`，ONNX Runtime 用 `intra_op_num_threads`，两者相同；cv2 固定 1 线程。**Ultralytics 8.4.172 的 `select_device("cpu")` 在建立预测器时会把 torch 线程重置为 min(8, CPU 数 − 1)**，工具在预热前恢复所请求的线程数，计时结束后再检查，不一致就停止。
- 一致性：同一图片中同类别且 IoU > 0.9 的框按参考分数贪心一对一匹配，统计匹配率、IoU 和分数差；在 conf ≥ 0.05（AP 用的全部预测）和 ≥ 0.25（界面常用阈值）分别统计。这里比较的是两个模型的预测，不是真值。

## 结果（2026-10-28，Intel Core Ultra 7 251HX，Linux，RTX 5060 Laptop）

代码 `6e4cd80`（report.json 记录的 8 个源码 SHA-256 与该提交逐一相同；当时工具还不记录 git 提交，现已补上）；torch 2.9.1+cu128、Ultralytics 8.4.172、onnxruntime 1.31.1、numpy 2.5.2、opencv 5.0.0。每行 5 次重复 × 24 图 = 120 个计时样本；“各次平均”是 5 次重复各自的平均值范围。

### 精度（两组线程设置结果相同；每个后端 5 次重复的预测完全一致）

| 输入几何 | 后端 | AP | AP50 | AP75 | APsmall | AR100 | 预测数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 最小矩形 | `pt_cpu_rect`、`pt_cpu_det_rect`、`ort_cpu_det_rect`、`pt_cuda_rect` | 0.5953 | 0.7636 | 0.6836 | 0.4174 | 0.7009 | 161 |
| 640×640 方形 | `pt_cpu_square`、`pt_cpu_det_square`、`ort_cpu_seg`、`ort_cpu_det` | 0.5658 | 0.7293 | 0.6462 | 0.4129 | 0.6597 | 157 |

矩形输入的结果与此前公布的 text 基线（AP 0.5953、161 个预测）完全相同。

### 预测一致性（2 线程运行；8 线程结果相同到 1e-5）

| 候选 → 参考 | 阈值 | 参考 / 候选框 | 匹配 | 最小 IoU | 最大 \|Δscore\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ort_cpu_det_rect` → `pt_cpu_rect`（界面路径） | 0.05 / 0.25 | 161 / 161；81 / 81 | 161；81 | 0.99999 | 3.5e-6 |
| `ort_cpu_seg`、`ort_cpu_det` → `pt_cpu_square` | 0.05 / 0.25 | 157 / 157；81 / 81 | 157；81 | 0.99998 | 3.9e-6 |
| `pt_cpu_det_*` → 同几何的 seg 图 | 0.05 | — | 全部 | 1 | 0（逐值相同） |
| `pt_cpu_rect` → `pt_cpu_square`（只换输入几何） | 0.05 / 0.25 | 157 / 161；81 / 81 | 134；73 | 0.91 | 0.18 |
| `pt_cuda_rect` → `pt_cpu_rect` | 0.05 / 0.25 | 161 / 161；81 / 81 | 160；81 | 0.95 | 1.6e-3 |

### CPU 时延（ms / 图）与内存

| 后端 | 2 线程 中位 | 平均 | p95 | 各次平均 | 8 线程 中位 | 平均 | p95 | 各次平均 | 峰值 RSS MiB（两组最大） | 导入 torch |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| `pt_cpu_rect`（界面） | 112.0 | 111.7 | 125.0 | 111.4–111.8 | 51.1 | 52.1 | 62.9 | 51.6–52.9 | 1011 | 是 |
| `pt_cpu_square` | 148.2 | 150.5 | 159.9 | 148.0–159.9 | 66.0 | 67.5 | 74.1 | 65.9–72.9 | 1011 | 是 |
| `pt_cpu_det_rect` | 85.3 | 84.3 | 93.8 | 81.0–85.5 | 42.3 | 42.4 | 48.1 | 40.8–43.8 | 1038 | 是 |
| `pt_cpu_det_square` | 112.9 | 112.8 | 119.2 | 108.2–117.7 | 55.4 | 55.0 | 57.0 | 52.6–56.3 | 1022 | 是 |
| `ort_cpu_seg` | 138.5 | 138.6 | 140.0 | 138.4–138.9 | 54.1 | 54.9 | 61.6 | 54.3–55.4 | 379 | 否 |
| `ort_cpu_det` | 87.3 | 87.4 | 88.6 | 87.3–87.5 | 33.9 | 34.7 | 40.2 | 34.1–35.6 | 323 | 否 |
| `ort_cpu_det_rect` | **64.7** | 63.5 | 69.9 | 63.4–63.6 | **25.4** | 25.4 | 28.8 | 25.2–25.7 | 271 | 否 |

每个 worker 启动前的 1 分钟负载：2 线程 2.40–7.39（第 1 轮从 7.39 逐步降到 5.26，其后在 2.4–5.3 之间），8 线程 2.41–4.88。`pt_cpu_square` 第 3 次重复（2 线程 159.9、8 线程 72.9）明显偏高，其余后端没有同样的偏移，这一点只作记录。p95 取自 24 张不同图片 × 5 次重复的 120 个样本；同一图片的各次重复比较接近（每图 5 次重复极差的中位数：ONNX Runtime 0.8–3.0 ms，PyTorch 1.9–11.6 ms），所以 p95 实际上主要反映最慢的一两张图，不是 120 个独立样本。两组运行中，ONNX Runtime 每个后端 5 次重复的平均值全部低于 `pt_cpu_square` 5 次中的最小值，`ort_cpu_det_rect` 的 5 次平均值也全部低于 `pt_cpu_rect` 的最小值。

PyTorch CUDA 参考（`pt_cuda_rect`）：中位 5.8 ms、平均 8.8 ms、p95 20.0 ms；每次重复中超过 12 ms 的都是同样 5 张图（62808、543581、301718、474167、2592，各自是首次出现的新输入尺寸），与此前 GPU 消融中的慢图相同；进程峰值 RSS 1764 MiB。它只是参考，不能与 CPU 数值相除来宣称加速。

加载（各 5 次平均）：PyTorch CPU 约 0.72–1.02 s，另加 CPU 文字编码约 0.79–0.87 s；ONNX Runtime 约 0.11–0.14 s，无文字编码。磁盘：`yoloe-26s-seg.pt` 31,072,171 B（权重以 FP16 存储）+ 文字提示所需 `mobileclip2_b.ts` 253,794,476 B；ONNX 为 FP32，单文件 38–42 MB，运行时不需要文字编码器。

## 怎样理解

1. **ONNX Runtime 与 PyTorch 数值一致。** 输入几何相同时，全部框一一匹配，分数差不超过 4.5e-6，AP 完全相同。导出没有带来可测的精度变化。
2. **差别来自输入几何，不来自后端。** 静态方形输入让 4:3 图片多处理约 1/3 的像素，并在这 24 图上把 AP 从 0.5953 降到 0.5658、APsmall 从 0.4174 降到 0.4129；高宽动态的 `det --dynamic-hw` 与界面路径逐框一致。这只是 24 张开发诊断图上的结果，方形输入的精度损失不一定能推广。
3. **CPU 时延。** 与当前界面路径 `pt_cpu_rect` 相比，`ort_cpu_det_rect` 中位时延为 0.58 倍（2 线程，112.0 → 64.7 ms）和 0.50 倍（8 线程，51.1 → 25.4 ms），预测相同。可以拆成两部分：仅检测分支转换本身（PyTorch 112.0 → 85.3 / 51.1 → 42.3 ms），以及同一计算图同一输入下 ONNX Runtime 替换 PyTorch（85.3 → 64.7 / 42.3 → 25.4 ms）。按图配对 bootstrap（24 张图有放回重抽 10,000 次，每次取两个后端全部重复的中位数之比，2.5 / 97.5 百分位）的 95% 区间：合计 0.578 [0.547, 0.581]（2 线程）和 0.497 [0.483, 0.502]（8 线程）；仅检测分支转换 0.761 [0.725, 0.771] 和 0.827 [0.810, 0.848]；ONNX Runtime 替换 PyTorch 0.759 [0.745, 0.779] 和 0.600 [0.582, 0.613]。区间只反映这 24 张图之间的差异，不包括主机、负载或其它图片集。8 线程与本机界面的实际设置相同：界面加载时 Ultralytics 会把 torch 线程重置为 min(8, CPU − 1) = 8。完整 seg 图在 ONNX Runtime 上 mask 原型分支代价很高：2 线程逐算子分析（ORT profiler）中 Conv 约占 81%，seg 图约 143 ms，det 图约 89 ms。因此不推荐在 CPU 上用 seg 图。
4. **内存。** ONNX Runtime 进程不导入 torch，峰值 RSS 271–379 MiB；PyTorch CPU 进程约 1.0 GiB。
5. **线程数影响很大。** 结论只适用于这台主机和这两个线程设置。其它 CPU、Windows 或不同负载需要重新测量；不同负载下的计时不能用来证明加速。

旁注：此前的 CPU 消融（`tools/run_additional_experiments.py`，manifest 记录 2 线程）在加载前调用 `torch.set_num_threads`，而 Ultralytics 建立预测器时会把线程重置为 min(8, CPU − 1)。按代码推断，那批计时实际可能用了 8 线程；其 text_baseline 平均 67.2 ms 也更接近本轮 8 线程（约 52 ms）而不是 2 线程（约 112 ms）的结果，但两者主机负载和时间不同，不能据此下结论。本轮没有重新运行旧实验，引用旧 CPU 时延时请注意。界面 CPU 推理同样经过这一步。

## 限制

- 固定类别、one2many、FP32、batch=1；不支持文字 / 视觉提示、无提示 PF、FP16/INT8。
- 只在 CPUExecutionProvider 上验证；GPU 上的 ONNX Runtime（需要 onnxruntime-gpu）和 TensorRT 未测。PyTorch CUDA 只作参考。
- 24 图诊断集已多次用于方法选择，不是未见最终测试集，也不是 COCO 榜单；没有测试摄像头域、长时视频或其它类别组合。
- 计时只代表本机、本次负载和所列线程数；结论按“同一主机、同一次运行、轮换顺序、重复 5 次”的范围理解。
- 原始结果（report.json、summary.md、timings.csv、逐后端预测、导出 manifest、ORT 逐算子分析）保存在实验目录 `experiments_20261028/onnx-export/`，不提交 Git。最终数值来自其中的 `final_threads2/` 和 `final_threads8/`。较早的 `bench_threads2/` 中 PyTorch 实际用了 8 线程，已标为无效；`bench_*_fixed/` 是修复后的较早重复，结论一致。
