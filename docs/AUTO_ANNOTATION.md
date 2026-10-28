# 文件夹自动标注（第 4 页，2026-10-28）

> **English summary.** Tab 4 "Auto-labeling" now implements the charter's *assisted annotation* deliverable. It runs the currently selected detector (text prompt, prompt-free, or the Tab 2 visual prompt) over an image folder one image at a time, lets a person accept / edit / reject the candidate boxes, and exports YOLO txt or COCO json. Only accepted or edited images are exported. Every export carries `auto_annotation_manifest.json`, which states that the labels are *model candidates reviewed by a human* (not independent ground truth) and records the model file and SHA-256, prompts, thresholds, device, run times and per-image review method. Verified: 25 new unit / offscreen UI tests; the full suite (333 tests: 332 pass, 1 skip); a real CPU run with YOLOE-26S on 5 `mini_eval` images, exported files loaded by pycocotools and by the Ultralytics dataset loader; prompt-free and visual-prompt modes on the same images. Not verified: CUDA in this tab, Grounding DINO in this tab, large folders, a real human labelling session, or the accuracy of the candidates.

## 用途与边界

- 对应项目章程的 “Assisted annotation”：对图片文件夹逐张运行当前模型 → 人工接受 / 修改 / 拒绝 → 导出 YOLO / COCO 标签。
- 导出标签是 **“模型候选 + 人工审核”**，不是独立人工真值。不要用它评价产生这些候选的模型，也不要混入 `datasets/mini_eval` 等人工 GT 评价集。评价仍只用人工 GT（见 AGENTS.md 与 [评价流程](evaluation.md)）。
- 页面不计算任何精度指标；框数不是准确率。

## 操作步骤

1. 在左侧选择模型、设备、置信度、最大框数、推理尺寸、SAHI / 旧分块、暗光设置，与第 1 页共用。开始时这些参数被记录到清单，运行中修改不影响本批。
2. 第 4 页点 **选择图片文件夹…**。支持 JPG / JPEG / PNG / BMP，按相对路径排序，中文路径可用。勾选 **包含子文件夹** 会重新扫描；含 `auto_annotation_manifest.json` 的旧导出文件夹会被跳过。
3. 选择提示方式：
   - **文字 / 无提示（本页提示词）**：输入 `bottle, cup, cell phone` 等使用原版权重；留空则与第 1 页一样切换到同规格 `*-seg-pf.pt`（4,585 类），不会用旧的 7 类占位列表代替。修改版和 DINO 没有无提示版，留空时拒绝开始。
   - **视觉提示（第 2 页参考框）**：使用第 2 页上传并框选的参考目标；参考编码在工作线程中只做一次。
4. 点 **开始自动标注**。逐张串行、batch=1，显示进度。**停止** 在当前图片推理结束后生效，已完成的候选保留。再次开始只处理未检测、读取失败或参数已改变的待审图片，从不覆盖已审核或已改动的图片。运行中也可以先审核还在排队的图片：之后到达的该图检测结果会被丢弃，人工结果、当时看到的候选和所属运行都不变，丢弃次数记在该次运行的 `results_discarded_after_review`。
5. 在左侧列表选图审核。画布沿用第 3 页纠错画布：选择 / 拖动框、拖拽补框、删除所选框；右侧可选择或输入类别修改所选框（也用于新画的框）。框标签显示模型分数，人工框标为“人工框”。
   - **接受本图**：采用当前全部框；与模型候选不同则记为“已修改”，随后跳到下一张待审图。
   - **拒绝本图（不导出）**、**恢复模型候选**（放弃本图改动，改回待审）。
   - **接受其余待审图片**：只处理已检测、未改动的待审图，保留分数 ≥ 阈值的候选；没有候选达到阈值的图保持待审（空标签表示“图中没有目标”，需要人看过）。会先确认，清单记录为批量接受。
6. 设置 **类别顺序**（导出 ID 顺序，框中其它类别自动追加）与可选 **导出最低分数**，点 **导出 YOLO…** / **导出 COCO…**，选择位置后在其中新建 `auto_labels_<yolo|coco>_<时间>` 子文件夹。导出目录不能位于源图片文件夹内，已存在的导出不覆盖；先写临时目录，完成后改名，失败不留下半成品。

关闭程序或换文件夹时，如果有尚未导出的审核结果会先询问。本版本不把审核进度保存到磁盘（开发中）。

## 状态

| 列表标记 | 状态 | 是否导出 |
| --- | --- | --- |
| · | 未检测 | 否 |
| ○ | 待审 / 待审（已改动） | 否 |
| ✓ | 已接受（逐张或批量） | 是 |
| ✎ | 已修改（接受时框与候选不同，或接受后又改动） | 是 |
| ✗ | 已拒绝 | 否 |
| ! | 读取 / 检测失败 | 否 |

导出最低分数只过滤“直接接受”图片中**未改动的模型框**；已修改的图片按人工结果原样导出，人工框没有分数、始终保留。

## 导出格式

**YOLO**（`auto_labels_yolo_*`）

- `images/<相对路径>`：已接受 / 已修改图片的副本；复制后校验 SHA-256 与审核时一致，原图不改动。
- `labels/<相对路径>.txt`：每行 `cls cx cy w h`，先裁剪到图像范围再归一化到 [0, 1]，6 位小数；无框的已接受图写空文件。
- 递归扫描时子文件夹结构保留在 `images/` 与 `labels/` 下。Ultralytics 把图片路径中**最后一个** `/images/` 换成 `/labels/` 来找标签，所以相对路径中有名为 `images` 的文件夹（例如数据集根目录下的 `images/子目录/x.jpg`）时，标签会被找错位置、图片被当作无目标背景。遇到这种情况 YOLO 导出直接拒绝（COCO 不受影响）；请直接选择那个 `images` 文件夹作为根目录。单元测试用 Ultralytics 的 `img2label_paths` 逐个核对标签路径。
- `classes.txt`、`images.txt`（`./images/...`）、`data.yaml`（`nc`、`names`）。`train` 与 `val` 指向同一列表，仅为满足 Ultralytics 格式，**不是验证集**，文件内有注释说明。

**COCO**（`auto_labels_coco_*`）

- `images/` 副本与 `annotations.json`：`info`、`licenses`、`images`（`id` 从 1、`width` / `height`）、`annotations`（`bbox` 为 `[x, y, w, h]`、`area`、`iscrowd: 0`、`id` 从 1）、`categories`（`id` 从 1）。每个标注另有 `attributes.origin`（`model` / `model_edited` / `human`）与 `attributes.model_score`。

**清单** `auto_annotation_manifest.json`（两种格式都有）：`label_provenance: "model candidates reviewed by a human"`、`human_ground_truth: false`、审核统计（逐张接受 / 批量接受 / 已修改 / 未导出的拒绝与待审数）、导出过滤、源文件夹、类别、每次检测运行（模型文件、标题、SHA-256、引擎、设备、提示方式与提示词、置信度、IoU、框数、尺寸、SAHI / 分块 / 暗光、推理选项、视觉参考图哈希与框、开始 / 结束时间、处理数、是否被停止或出错）、相关源码 SHA-256、每张图的源文件 SHA-256、宽高、状态、审核方式与框来源计数。DINO 使用 Hugging Face 缓存，`model_sha256` 记为空。

## 资源与串行

- 复用应用当前的单个模型：开始前停止第 1 / 2 页检测，按提示内容走原有 `_ensure_detection_model` 加载 / 切换路径（`prepare_device`、GPU 文件锁、3 GiB 预算、加载前 4 GiB 空闲显存）。没有新增 GPU 预算逻辑。
- 每张图都走原有 `process_frame` → `detect` 路径，逐次执行 `preflight_inference`（RAM ≥ 1.5 GiB、推理前空闲显存 ≥ 1 GiB、尺寸限制）。超过 1600 万像素的单张图在推理前标为失败并跳过；其它资源限制、OOM 或推理异常会停止整批，不重试、不切换设备。
- 不使用 ByteTrack（编号不是类别）；开启暗光时每张图重置平滑状态。
- 运行中第 1 / 2 页“开始检测”和“应用设置 / 加载模型”会提示先停止自动标注；“停止并释放模型 / 显存”先停止自动标注再释放模型和 GPU 锁；纠错训练开始前同样会释放。

## 验证记录（2026-10-28，本机 Linux，CPU）

环境：数据盘虚拟环境 Python 3.12.13、PyTorch 2.9.1+cu128、Ultralytics 8.4.172、PySide6 6.12.0、OpenCV 5.0.0、SAHI 0.12.8、pycocotools 2.0.11（与 README 中 Windows / 6008 环境版本不同）。模型任务都在共享锁下串行运行。实验输出、截图和日志保存在仓库外，不提交。

- **单元 / 界面测试**：新增 `test_auto_annotation.py` 25 项（扫描与中文路径、YOLO 归一化与边缘裁剪、同名不同扩展名图片的 YOLO 标签冲突、COCO 结构与编号、只导出已接受 / 已修改、清单内容、状态转换、模拟引擎的串行 / 取消 / 资源错误 / 视觉参考 / 暗光、离屏界面：英文翻译完整、按钮启用状态、真实 QThread 跨线程信号、运行中阻止并行检测、释放按钮先停自动标注）。`test_i18n.py` 改为检查新页标题与说明，并把新文件加入翻译覆盖检查。
- **完整回归**：`unittest discover` 333 项，332 通过、1 项跳过（缺少完整 COCO 原始标注缓存，与改动前相同）；pytest 结果相同。改动前同一环境下 `main` 为 308 项，307 通过、1 项跳过。
- **真实运行（文字提示）**：提交 `2a8b039`（之后只增加了 YOLO 标签重名检查与接受时的错误提示，均有单元测试，未重跑真实模型）；YOLOE-26S（`yoloe-26s-seg.pt`，SHA-256 `48f24206…df0654`），CPU，640，置信度 0.25，提示 `bottle, cup, cell phone`；`datasets/mini_eval` 前 5 张图片复制到中文路径文件夹。5 张全部完成检测（候选 1 / 0 / 5 / 2 / 3）。随后用**脚本模拟**的审核动作（不是人工审核）：接受 1 张、在 1 张上补一个手工框并接受（已修改）、拒绝 1 张、批量接受 ≥ 0.50 的 2 张。导出 YOLO 与 COCO 各 4 张图 / 6 个框；pycocotools 成功载入（4 图、6 标注、3 类）；YOLO 标签逐行解析通过（5 列、类别号有效、6 位小数、[0, 1]），Ultralytics `check_det_dataset` 与 `YOLODataset` 读到 4 张图 / 6 个框；清单中模型哈希与权重文件实测一致。单次运行逐张推理用时：首张约 1.6 s（含文字嵌入与预热），其余 0.07–0.09 s（负载约 4.3）；仅为功能记录，不作速度结论。
- **真实运行（无提示 / 视觉提示）**：同 5 张图。留空提示自动切换到 `yoloe-26s-seg-pf.pt`（4,585 类，未发送任何文字类别），共 100 个候选，按 0.25 批量接受后导出的 COCO 含 58 个类别并可被 pycocotools 载入。视觉提示使用 `mini_eval/references` 中人工复核的 cup 参考框（绕过第 2 页画布坐标换算，该换算已有单独测试），5 张中 1 张出现 1 个 cup 候选，导出 COCO 可载入。候选数量不代表精度。
- **截图**：离屏界面中文 / 英文截图已逐张查看，英文状态下第 4 页没有未翻译文字（截图不提交）。

## 未验证 / 开发中

- 未在本页运行 CUDA 或 Grounding DINO；未测试上百张以上的大文件夹耗时与内存。
- 未进行真实人工标注会话，也未测量标注效率；未评价候选框精度（需要人工 GT，且不能用本页导出的标签充当）。
- 未用导出数据实际训练模型；只验证了 Ultralytics 能读取 YOLO 导出。
- 审核进度不落盘，关闭前须导出；不支持多边形 / 分割标注。
- 图像宽高按 OpenCV 解码结果记录（会应用 EXIF 方向）；如果其它工具忽略 EXIF 方向，坐标可能不一致。
