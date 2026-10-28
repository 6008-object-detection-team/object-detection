# 组会实验：置信度阈值与预测框数独立消融

`benchmark_detection.py` 对同一组本地图片或视频固定帧，分别改变置信度阈值和 `max_det`。实验直接调用原图 `detect()`，关闭跟踪、分块、暗光增强、画框和界面显示，避免这些流程影响框数或计时。

这里的“预测框数”指**最终返回框数的上限 `max_det`**，不是要求模型一定预测这么多个框，也不是 Grounding DINO 的模型查询数。没有真实标注时，只报告检测调用时延、输出框数和分数，不能从框数增加推断精度、召回率或 mAP 提高。

## 固定条件

| 实验 | 改变的变量 | 固定的变量 |
| --- | --- | --- |
| `--mode confidence` | `--conf-values` | `--fixed-max-det`、IoU、推理尺寸、输入、提示词 |
| `--mode max-det` | `--max-det-values` | `--fixed-conf`、IoU、推理尺寸、输入、提示词 |
| `--mode both` | 依次执行上述两个独立系列 | 每个系列仍只改变一个变量，不做联动或笛卡尔积 |

Grounding DINO 额外固定 `text_threshold=0.25`。两种引擎均使用英文类别名称，类别和顺序在同一次实验中保持一致。

先用少量输入检查能否跑通，再使用有代表性的白天、暗光、拥挤、小目标图片/视频。跨模型比较时使用相同源文件、固定帧号、英文提示词、设备、重复次数和预热次数；模型大小用实际权重名称标明，本项目提供的是 **YOLOE-26S/26L**，结果不要直接标成普通 YOLO26。

## 在数据盘项目目录运行

Linux 使用已经安装好的项目解释器 `.venv/bin/python`。以下命令在项目根目录执行；`--source` 替换成自己的本地输入，文件路径有空格时加引号。脚本不需要 GUI，也不访问摄像头。

先验证依赖和核心实验逻辑：

```bash
.venv/bin/python check_environment.py
.venv/bin/python -m unittest -v test_benchmark_detection
.venv/bin/python benchmark_detection.py --help
```

### 单图：YOLOE-26S 两个独立实验

```bash
.venv/bin/python benchmark_detection.py \
  --engine yoloe --model yoloe-26s-seg.pt \
  --source "/数据盘/实验输入/scene.jpg" \
  --classes "person,bottle,cell phone" \
  --mode both \
  --conf-values 0.10,0.25,0.40,0.60 \
  --max-det-values 10,30,100,300 \
  --fixed-conf 0.40 --fixed-max-det 300 \
  --iou 0.45 --imgsz 640 --device cuda \
  --warmup 3 --repeat 10 --seed 0 \
  --output validation/experiments/yoloe26s_scene \
  --save-detections
```

对应两条曲线：第一条中 `max_det` 始终为 300，置信度从 0.10 变化到 0.60；第二条中置信度始终为 0.40，`max_det` 从 10 变化到 300。`both` 中相同的基准条件可能出现两次，分别属于两个系列，应按 `sweep` 分组绘图。

改成 `--model yoloe-26l-seg.pt` 并指定新的输出目录，即可为高精度候选收集相同条件的性能数据。没有标注时，这个实验仍不验证“26L 精度最高”的结论。

### 视频：所有配置使用相同帧

```bash
.venv/bin/python benchmark_detection.py \
  --engine yoloe --model yoloe-26s-seg.pt \
  --source "/数据盘/实验输入/demo.mp4" \
  --classes "person,bottle,cell phone" \
  --mode both --frames 0,30,60,90 \
  --fixed-conf 0.40 --fixed-max-det 300 \
  --iou 0.45 --imgsz 640 --device cuda \
  --warmup 3 --repeat 10 --seed 0 \
  --output validation/experiments/yoloe26s_video
```

帧号从 0 起。脚本先按顺序解码并保存这些帧，再复用同一组帧；视频解码不计入检测时延。视频太短、文件缺失或帧解码失败时实验会报错，不会静默换帧。公开整合版不附带本机短视频夹具，可用自己的短视频及有效帧号做流程检查。

`--source` 也可以指向图片目录，脚本按文件名排序读取目录内支持的图片，不递归读取子目录，并把全部解码后的输入保存在内存。大数据集请分批运行以控制内存。

### Grounding DINO：相同输入、独立阈值

模型须已通过项目部署流程准备。优先传入完整的本地模型目录；若传 Hugging Face 模型 ID，首次加载可能需要联网下载。

```bash
.venv/bin/python benchmark_detection.py \
  --engine dino --model "/数据盘/模型/grounding-dino-tiny" \
  --source "/数据盘/实验输入/demo.mp4" \
  --classes "person,bottle,cell phone" \
  --mode both --frames 0,30,60,90 \
  --conf-values 0.10,0.25,0.40,0.60 \
  --max-det-values 10,30,100,300 \
  --fixed-conf 0.40 --fixed-max-det 300 \
  --iou 0.45 --imgsz 640 --device cuda \
  --warmup 3 --repeat 10 --seed 0 \
  --output validation/experiments/dino_video \
  --save-detections
```

CPU 运行用 `--device cpu`；`--device auto` 会在 PyTorch 可用 CUDA 时选择 CUDA，否则使用 CPU。指定 `cuda` 但 GPU 不可用时会报错，避免把意外的 CPU 运行混进 GPU 数据。

## 如何读结果

每次运行使用新的目录。显式指定已经存在的 `--output` 时脚本会停止，保留原结果；省略它则自动生成 `validation/experiments/UTC时间_引擎/`。

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 每个配置/输入及全部输入的时延均值、中位数、P95、标准差、框数统计 |
| `samples.csv` | 每次测量的时延、输出框数、各类别框数和分数统计 |
| `results.json` | CSV 对应结果、完整参数、随机执行顺序、源文件/解码帧 SHA-256、模型来源、软件版本及设备信息 |

CSV 中的 `scope=input` 是单张图片或单个固定视频帧，`scope=all_inputs` 是该配置下全部输入的所有测量汇总。CSV 每行带有设置、设备和软件版本 JSON，单独复制表格后仍能核查实验条件。`--save-detections` 在 JSON 中额外保存每个配置/输入首次测量的框、原始分数和文字标签，便于人工查看。

`mean_ms`、`median_ms` 和 `p95_ms` 的单位均为毫秒。`detection_calls_per_second=1000/mean_ms` 是串行检测调用的吞吐换算，**不是界面视频 FPS**。框数少于 `max_det` 时，增大上限可能没有效果；出现这种结果不等于实验失效。不同模型的置信度分数也不保证具有相同的校准含义。

模型加载、文本首次初始化和每个配置/输入的预热不计时。CUDA 在每次测量前后同步，避免只测到异步提交开销。计时包含输入预处理、模型前向、后处理/NMS、框数截断和结果传回 CPU，因此应在组会上称为“原图检测调用时延”，不要称为纯前向时延。

两个模型的 `imgsz` 含义不同：YOLOE 使用 Ultralytics 的自动 letterbox（最大边按 `imgsz`，随后按步长填充）；Grounding DINO 的目标短边为 `imgsz`，长边不超过 `round(imgsz*1333/800)`。例如同为 640 时实际网络输入像素通常不同。结果记录了 `resize_policy`，适合各自管线的实际应用比较；不能据此声称严格等输入尺寸的纯模型对比。

执行配置顺序按 `--seed` 打乱并记录，降低持续升温等顺序效应；同一输入在每个配置中重复使用。正式汇报前建议至少用 3 次独立运行确认趋势，在供电、温度和其他程序占用相近的条件下分别运行模型，避免同时跑 GPU 实验。

文件时间使用 UTC，确保跨电脑记录可比较。马来西亚本地时间为 UTC+8，例如 `20261004T120000Z` 对应 2026-10-04 20:00；给导师展示时可换成本地日期和时间，保留原始运行 ID。

## 组会展示与后续精度评估

展示两幅独立图：置信度—时延/输出框数，以及 `max_det`—时延/输出框数。标注固定条件、输入数量和重复次数。先解释每条曲线只改变一个变量，再演示 Grounding DINO 的英文文本提示检测；将整体报告控制在 5–7 分钟。

本脚本当前不实现 COCO 标注评价，JSON 的 `ground_truth` 和 `accuracy_metrics` 明确为 `null`。需要 mAP 或小目标 AP 时，另行准备真实标注、固定图片集合及类别映射，使用标准 COCO 评价器，并说明 AP 使用 IoU 0.50:0.95 的规则。开放词汇文字标签必须可靠映射到标注类别；没有匹配上的标签如何处理也应固定。AP 评估一般保留较低检测阈值用于形成完整排序，不能把“各置信度下框数变化”直接当作 AP 变化。

如果只有人工检查，可报告具体样本的漏检/误检案例与对应框截图，但不要把主观检查标成 mAP。此次部署和实验流程验证也不能替代上周组会的精度实验复现。

## SAHI 切片的耗时（2026-10-28）

第 4 页逐张检测的实际调用 `auto_annotation.detect_candidates`（`process_frame`，含画框，不含读图），conf 0.40（界面默认）、640、每图最多 300 框，在 240 张留出 COCO 图（最长边 ≤ 640）上计时。同一次模型锁内背靠背运行：先 26S，再 26L（一次只加载一个模型）；每个配置对全部图片跑 3 遍，每遍轮换配置顺序，每个配置先预热一张。`OMP_NUM_THREADS=2`（与 `run_app.sh` 相同；加载后 Ultralytics 把 torch 设为 8 线程）。RTX 5060 Laptop，1 分钟负载 2–5（同机还有其它任务）。表中是 3 遍各自中位数的中位数，括号内为 3 遍中位数的范围；p95 为 3 遍 p95 的中位数。

| 模型 / 配置（GPU） | 中位 ms/图 | p95 ms/图 |
|---|---|---|
| 26S 整图 | 6.3（6.0–6.3） | 7.7 |
| 26S 新 SAHI 512 | 20.8（20.3–20.9） | 36.3 |
| 26S 旧 SAHI 512 | 20.7（20.5–21.5） | 36.0 |
| 26S 新 SAHI 320 | 48.9（48.8–52.4） | 55.5 |
| 26S 旧 SAHI 320 | 49.9（49.3–50.2） | 55.2 |
| 26L 整图 | 14.2（14.1–14.3） | 16.9 |
| 26L 新 SAHI 512 | 47.8（47.7–48.2） | 89.8 |
| 26L 新 SAHI 320（“精度优先”预设） | 124.3（123.9–124.3） | 130.2 |

- 边缘过滤本身不增加可测的耗时（新旧 SAHI 的差异在 3 遍的波动范围内）。耗时主要由切片数决定：640×480 图在 512 切片下是 2 块，在 320 下是 6 块，640×640 图在 512 下是 4 块。
- 离屏界面真实运行第 4 页预设（同一留出集，另一次锁会话）时，工作线程记录的每张用时中位数 0.121 s、p95 0.131 s；加上加载 26L，240 张共 31 s。
- CPU 只做了小样本（前 8 张图、3 遍，同一锁会话，负载约 7–8）：26S 整图 49 ms，26S 新 SAHI 320 479 ms，26L 整图 193 ms，26L 预设 1.65 s/图（3 遍中位数 1.64–1.92 s）。没有 GPU 时预设每张需要 1–2 秒。
- 不同会话、不同负载下的数字不能互相比较；更大的输入（720p、1080p）切片更多，没有在这里测量。原始计时在数据盘 `experiments_20261028/round2/sahi/out/`，不提交。
