# 文件夹自动标注（第 4 页，2026-10-28；快捷键与自动保存 / 恢复 round 2 更新）

> **English summary.** Tab 4 "Auto-labeling" now implements the charter's *assisted annotation* deliverable. It runs the currently selected detector (text prompt, prompt-free, or the Tab 2 visual prompt) over an image folder one image at a time, lets a person accept / edit / reject the candidate boxes, and exports YOLO txt or COCO json. Only accepted or edited images are exported. Every export carries `auto_annotation_manifest.json`, which states that the labels are *model candidates reviewed by a human* (not independent ground truth) and records the model file and SHA-256, prompts, thresholds, device, run times and per-image review method. Verified: 25 new unit / offscreen UI tests (33 after the 2026-10-28 review fixes); the full suite (333 tests: 332 pass, 1 skip; 417 / 416 / 1 after the review fixes); a real CPU run with YOLOE-26S on 5 `mini_eval` images, exported files loaded by pycocotools and by the Ultralytics dataset loader; prompt-free and visual-prompt modes on the same images. Not verified: CUDA in this tab, Grounding DINO in this tab, large folders, a real human labelling session, or the accuracy of the candidates.
>
> **Update (2026-10-28, branch `imp/sahi`).** With SAHI on, an image whose slice plan exceeds the 9-slice cap is now skipped and marked "Skipped (too many slices)" instead of stopping the whole batch (the cap is unchanged). A page preset "Accuracy first" runs one batch with YOLOE-26L + border-filtered SAHI 320 at 640 / conf 0.40 without changing the left-side settings, and is recorded in the manifest (`preset`). On the 240-image held-out COCO subset it scores AP 0.505 vs 0.413 for the app default, P / R at 0.40 of 0.764 / 0.663 (default 0.826 / 0.537), about 0.12 s per image on an RTX 5060 Laptop GPU and 1.6–1.9 s on CPU. A real offscreen run of this page with the preset on CUDA reproduced those P / R exactly after a scripted (not human) bulk acceptance.
>
> **Round 2 (same date).** Tab 3 and Tab 4 review gained keyboard shortcuts (A/R/U/Del/N/P/D/Ctrl+E on Tab 4; Del/D on Tab 3), which a `QShortcut` on the page itself with `WidgetWithChildrenShortcut` context keeps out of text fields. Tab 4 now auto-saves the review session (candidates, boxes, states, class order and the detection runs) to a JSON file in the project's `.cache/` after every accept / reject / restore / bulk-accept / edit, and offers to resume it when the same folder is reopened, validating each image's size and mtime before trusting its saved candidates. This moves "审核进度保存到磁盘" from 开发中 to 已验证 below. See the keyboard-shortcut and auto-save sections below for what is and is not covered.

## 用途与边界

- 对应项目章程的 “Assisted annotation”：对图片文件夹逐张运行当前模型 → 人工接受 / 修改 / 拒绝 → 导出 YOLO / COCO 标签。
- 导出标签是 **“模型候选 + 人工审核”**，不是独立人工真值。不要用它评价产生这些候选的模型，也不要混入 `datasets/mini_eval` 等人工 GT 评价集。评价仍只用人工 GT（见 AGENTS.md 与 [评价流程](evaluation.md)）。
- 页面不计算任何精度指标；框数不是准确率。

## 操作步骤

1. 在左侧选择模型、设备、置信度、最大框数、推理尺寸、SAHI / 旧分块、暗光设置，与第 1 页共用。开始时这些参数被记录到清单，运行中修改不影响本批。
   也可在第 4 页 **本页预设** 选 **精度优先：YOLOE-26L + SAHI 320（较慢）**：本批改用 YOLOE-26L、640、置信度 0.40、SAHI 320（边缘过滤）、关闭旧分块和暗光补检；设备和最大框数沿用左侧。左侧其它控件不变，只有模型栏会显示实际加载的 26L（仍只保留一个模型）。预设名称和参数记入清单（`preset: "accuracy_first"`）。只支持文字提示；本地没有真实的 `yoloe-26l-seg.pt`（或只有 Git LFS 指针）时拒绝开始，不会下载或换成别的模型。240 图留出集上的精度与耗时见 [评价流程](evaluation.md#sahi-边缘过滤2026-10-28)：AP 更高、召回更高，但精确率略低（要审的误检更多），GPU 约 0.12 s/图，CPU 约 1.6–1.9 s/图。
2. 第 4 页点 **选择图片文件夹…**。支持 JPG / JPEG / PNG / BMP，按相对路径排序，中文路径可用。勾选 **包含子文件夹** 会重新扫描；含 `auto_annotation_manifest.json` 的旧导出文件夹会被跳过。
3. 选择提示方式：
   - **文字 / 无提示（本页提示词）**：输入 `bottle, cup, cell phone` 等使用原版权重；留空则与第 1 页一样切换到同规格 `*-seg-pf.pt`（4,585 类），不会用旧的 7 类占位列表代替。修改版和 DINO 没有无提示版，留空时拒绝开始。
   - **视觉提示（第 2 页参考框）**：使用第 2 页上传并框选的参考目标；参考编码在工作线程中只做一次。置信度取第 2 页的“视觉提示阈值”（默认 0.03），不取左侧滑块；运行记录和清单的 `conf` 是实际使用的值。单张参考的视觉提示分数远低于文字提示，沿用 0.40 时几乎没有候选，见 [视觉提示说明](VISUAL_PROMPT.md)。
4. 点 **开始自动标注**。逐张串行、batch=1，显示进度。**停止** 在当前图片推理结束后生效，已完成的候选保留。再次开始只处理未检测、读取失败或参数已改变的待审图片，从不覆盖已审核或已改动的图片。运行中也可以先审核还在排队的图片：之后到达的该图检测结果会被丢弃，人工结果、当时看到的候选和所属运行都不变，丢弃次数记在该次运行的 `results_discarded_after_review`。
5. 在左侧列表选图审核。画布沿用第 3 页纠错画布：选择 / 拖动框、拖拽补框、删除所选框；右侧可选择或输入类别修改所选框（也用于新画的框）。框标签显示模型分数，人工框标为“人工框”。
   - **接受本图**：采用当前全部框；与模型候选不同则记为“已修改”，随后跳到下一张待审图。
   - **拒绝本图（不导出）**、**恢复模型候选**（放弃本图改动，改回待审）。
   - **接受其余待审图片**：只处理已检测、未改动的待审图，保留分数 ≥ 阈值的候选；没有候选达到阈值的图保持待审（空标签表示“图中没有目标”，需要人看过）。会先确认，清单记录为批量接受。
6. 设置 **类别顺序**（导出 ID 顺序，框中其它类别自动追加）与可选 **导出最低分数**，点 **导出 YOLO…** / **导出 COCO…**，选择位置后在其中新建 `auto_labels_<yolo|coco>_<时间>` 子文件夹。导出目录不能位于源图片文件夹内，已存在的导出不覆盖；先写临时目录，完成后改名，失败不留下半成品。导出成功后会询问是否清除本地保存的审核进度文件（见下）。

关闭程序、换文件夹或切换“包含子文件夹”（会重新扫描）时，如果有尚未导出的审核结果，或有已检测但尚未审核的候选（丢掉后需要重新检测），会先询问。

## 快捷键（第 3 / 4 页）

第 4 页逐张审核与第 3 页纠错画布都加了 `QShortcut`（`WidgetWithChildrenShortcut`，绑定在页面本身），输入框（`QLineEdit` 或可编辑下拉框）获得焦点时不生效，Qt 本身也会让单字母键和 Delete 优先给文本框使用；只有 Ctrl+E 这类带修饰键的快捷键额外加了“焦点在文本框时不执行”的判断。

| 键 | 第 4 页 | 第 3 页 |
| --- | --- | --- |
| A | 接受本图（同“接受本图”按钮），随后跳到下一张待审图 | — |
| R | 拒绝本图（不导出），随后跳到下一张待审图 | — |
| U | 恢复模型候选，改回待审 | — |
| Delete | 删除左侧列表中选中的框 | 删除所选误检框 |
| N / P | 下一张 / 上一张图片（列表顺序，含首尾循环） | — |
| D | 切换“选择 / 拖动现有框”与“拖拽补框”两种模式 | 同左 |
| Ctrl+E | 导出 YOLO（与“导出 YOLO…”按钮相同；没有已接受 / 已修改图片时不生效） | — |

第 3 页没有逐张接受 / 拒绝 / 批量恢复的审核状态机，所以只保留画布上确实存在的删框和切换模式两个快捷键，没有 A/R/U/N/P/Ctrl+E。快捷键的提示文字（已翻译）显示在按钮提示和页面内的“快捷键：…”说明行中。

## 自动保存与恢复（第 4 页）

- 每次人工决定（接受 / 拒绝 / 恢复模型候选 / 批量接受 / 改框或改类别）之后，`AnnotationSession.to_dict()` 把候选框、人工框、每张图的状态、类别顺序和全部检测运行记录（含模型路径、SHA-256 与推理选项）写成 JSON，`auto_annotation.write_session` 原子写入（临时文件 + `fsync` + `os.replace`，不会留下半成品）。
- 保存位置是项目的 `.cache/annotation_sessions/<文件夹路径哈希>.json`（`auto_annotation.session_cache_path`），不写进用户的图片文件夹；路径可用环境变量 `EE6008_ANNOTATION_CACHE_DIR` 整体改到别处（测试用这个环境变量把文件隔离到临时目录，不污染仓库的 `.cache/`）。该目录已在 `.gitignore` 中，不会被提交。
- 再次打开同一文件夹（含切换“包含子文件夹”触发的重新扫描）时，如果存在保存记录且至少有一张图被检测或审核过，会询问是否恢复；选“否”则按全新文件夹处理。
- 恢复时逐张用当前文件的大小和修改时间（`os.stat`）与保存时记录的值核对：不一致的图片视为“已变化”，改回“未检测 / 待审”，不沿用旧候选框（即使它仍在 JSON 里）；审核页之前打开某张图时验证过的 SHA-256 也会一并保留，供下次真正打开该图时再核对一次。单张记录损坏（例如框坐标不是四个数）按“已变化”处理，不影响其它图片；整份 JSON 损坏或版本不匹配（`schema` 字段）会被整体忽略，直接使用全新扫描结果，并在状态栏提示。
- 导出成功后会问是否清除本地进度文件；选“是”才删除，选“否”保留，供下次继续恢复。
- 这是“审核进度保存到磁盘”从开发中变为已验证的实现；范围只覆盖本页（Tab 4），纠错页（Tab 3）没有类似的会话保存。

## 状态

| 列表标记 | 状态 | 是否导出 |
| --- | --- | --- |
| · | 未检测 | 否 |
| ○ | 待审 / 待审（已改动） | 否 |
| ✓ | 已接受（逐张或批量） | 是 |
| ✎ | 已修改（接受时框与候选不同，或接受后又改动） | 是 |
| ✗ | 已拒绝 | 否 |
| ! | 读取 / 检测失败，或“已跳过（切片过多）” | 否 |

导出最低分数只过滤**按分数批量接受**的图片（没有人逐张看过）中低于该分数的框。逐张接受或修改的图片按人工结果原样导出：人看过并接受的低分框是人工确认的目标，删掉会在训练标签里变成漏标。批量接受的图片如果没有框达到该分数，就不导出，而不是写一个空标签（空标签表示“图中没有目标”，没有人确认过）；清单列出这些图片，并记录每张图被分数过滤的框数。

## 导出格式

**YOLO**（`auto_labels_yolo_*`）

- `images/<相对路径>`：已接受 / 已修改图片的副本；复制后校验 SHA-256 与审核时一致，原图不改动。
- `labels/<相对路径>.txt`：每行 `cls cx cy w h`，先裁剪到图像范围再归一化到 [0, 1]，6 位小数；无框的已接受图写空文件。
- 递归扫描时子文件夹结构保留在 `images/` 与 `labels/` 下。Ultralytics 把图片路径中**最后一个** `/images/` 换成 `/labels/` 来找标签，所以相对路径中有名为 `images` 的文件夹（例如数据集根目录下的 `images/子目录/x.jpg`）时，标签会被找错位置、图片被当作无目标背景。遇到这种情况 YOLO 导出直接拒绝（COCO 不受影响）；请直接选择那个 `images` 文件夹作为根目录。单元测试用 Ultralytics 的 `img2label_paths` 逐个核对标签路径。
- `classes.txt`、`images.txt`（`./images/...`）、`data.yaml`（`nc`、`names`）。`train` 与 `val` 指向同一列表，仅为满足 Ultralytics 格式，**不是验证集**，文件内有注释说明。

**COCO**（`auto_labels_coco_*`）

- `images/` 副本与 `annotations.json`：`info`、`licenses`、`images`（`id` 从 1、`width` / `height`）、`annotations`（`bbox` 为 `[x, y, w, h]`、`area`、`iscrowd: 0`、`id` 从 1）、`categories`（`id` 从 1）。每个标注另有 `attributes.origin`（`model` / `model_edited` / `human`）与 `attributes.model_score`。

**清单** `auto_annotation_manifest.json`（两种格式都有）：`label_provenance: "model candidates reviewed by a human"`、`human_ground_truth: false`、审核统计（逐张接受 / 批量接受 / 已修改 / 未导出的拒绝与待审数）、导出过滤、源文件夹、类别、每次检测运行（模型文件、标题、SHA-256、引擎、设备、提示方式与提示词、置信度、IoU、框数、尺寸、SAHI / 分块 / 暗光、推理选项、本页预设 `preset`、视觉参考图哈希与框、开始 / 结束时间、处理数、因切片过多跳过的文件 `skipped_too_many_slices`、是否被停止或出错）、相关源码 SHA-256、每张图的源文件 SHA-256、宽高、状态、审核方式、框来源计数、被导出最低分数过滤的框数（`filtered_by_min_score`）和零面积未写入的框数，以及因过滤后无框而未导出的批量接受图片。DINO 使用 Hugging Face 缓存，`model_sha256` 记为空。

## 资源与串行

- 复用应用当前的单个模型：开始前停止第 1 / 2 页检测，按提示内容走原有 `_ensure_detection_model` 加载 / 切换路径（`prepare_device`、GPU 文件锁、3 GiB 预算、加载前 4 GiB 空闲显存）。没有新增 GPU 预算逻辑。
- 每张图都走原有 `process_frame` → `detect` 路径，逐次执行 `preflight_inference`（RAM ≥ 1.5 GiB、推理前空闲显存 ≥ 1 GiB、尺寸限制）。超过 1600 万像素的单张图在推理前标为失败并跳过；开启 SAHI 时，切片数超过 9 块上限的图片（例如 512 切片下的 1080p 图、任何切片下的 1200 万像素照片）同样在推理前跳过，列表标为“已跳过（切片过多）”，提示中写明哪个切片尺寸可以放下或需要先缩小图片；本批其余图片继续，结束时状态栏汇总跳过数量，文件名记入该次运行的 `skipped_too_many_slices`。9 块上限不变，跳过的图片下次开始时会重试。其它资源限制、OOM 或推理异常会停止整批，不重试、不切换设备。
- 不使用 ByteTrack（编号不是类别）；开启暗光时每张图重置平滑状态。
- 运行中第 1 / 2 页“开始检测”和“应用设置 / 加载模型”会提示先停止自动标注；“停止并释放模型 / 显存”先停止自动标注再释放模型和 GPU 锁；纠错训练开始前同样会释放。

## 验证记录（2026-10-28，本机 Linux，CPU）

环境：数据盘虚拟环境 Python 3.12.13、PyTorch 2.9.1+cu128、Ultralytics 8.4.172、PySide6 6.12.0、OpenCV 5.0.0、SAHI 0.12.8、pycocotools 2.0.11（与 README 中 Windows / 6008 环境版本不同）。模型任务都在共享锁下串行运行。实验输出、截图和日志保存在仓库外，不提交。

- **单元 / 界面测试**：新增 `test_auto_annotation.py` 25 项；2026-10-28 审查修复后为 33 项，另加：递归导出时用 Ultralytics `img2label_paths` 核对标签位置并拒绝名为 `images` 的子文件夹、运行中先审核的图片不被迟到结果改写、停止时正在检测的图片记到本次运行（真实 QThread）、清单只计写入的框、最低分数只过滤批量接受的图片、改类别不改动无关图片的状态、切换子文件夹前询问、无检测任务时不停止实时检测。原 25 项为（扫描与中文路径、YOLO 归一化与边缘裁剪、同名不同扩展名图片的 YOLO 标签冲突、COCO 结构与编号、只导出已接受 / 已修改、清单内容、状态转换、模拟引擎的串行 / 取消 / 资源错误 / 视觉参考 / 暗光、离屏界面：英文翻译完整、按钮启用状态、真实 QThread 跨线程信号、运行中阻止并行检测、释放按钮先停自动标注）。`test_i18n.py` 改为检查新页标题与说明，并把新文件加入翻译覆盖检查。
- **完整回归**：`unittest discover` 333 项，332 通过、1 项跳过（缺少完整 COCO 原始标注缓存，与改动前相同）；pytest 结果相同。改动前同一环境下 `main` 为 308 项，307 通过、1 项跳过。
- **真实运行（文字提示）**：提交 `2a8b039`（之后只增加了 YOLO 标签重名检查与接受时的错误提示，均有单元测试，未重跑真实模型）；YOLOE-26S（`yoloe-26s-seg.pt`，SHA-256 `48f24206…df0654`），CPU，640，置信度 0.25，提示 `bottle, cup, cell phone`；`datasets/mini_eval` 前 5 张图片复制到中文路径文件夹。5 张全部完成检测（候选 1 / 0 / 5 / 2 / 3）。随后用**脚本模拟**的审核动作（不是人工审核）：接受 1 张、在 1 张上补一个手工框并接受（已修改）、拒绝 1 张、批量接受 ≥ 0.50 的 2 张。导出 YOLO 与 COCO 各 4 张图 / 6 个框；pycocotools 成功载入（4 图、6 标注、3 类）；YOLO 标签逐行解析通过（5 列、类别号有效、6 位小数、[0, 1]），Ultralytics `check_det_dataset` 与 `YOLODataset` 读到 4 张图 / 6 个框；清单中模型哈希与权重文件实测一致。单次运行逐张推理用时：首张约 1.6 s（含文字嵌入与预热），其余 0.07–0.09 s（负载约 4.3）；仅为功能记录，不作速度结论。
- **真实运行（无提示 / 视觉提示）**：同 5 张图。留空提示自动切换到 `yoloe-26s-seg-pf.pt`（4,585 类，未发送任何文字类别），共 100 个候选，按 0.25 批量接受后导出的 COCO 含 58 个类别并可被 pycocotools 载入。视觉提示使用 `mini_eval/references` 中人工复核的 cup 参考框（绕过第 2 页画布坐标换算，该换算已有单独测试），5 张中 1 张出现 1 个 cup 候选，导出 COCO 可载入（当时视觉提示沿用左侧滑块）。改用第 2 页视觉提示阈值后（CUDA，参考经第 2 页画布读取与框选），同 5 张图默认 0.03 时清单记录 0.03、共 5 个候选，改为 0.40 后为 1 个。候选数量不代表精度。
- **截图**：离屏界面中文 / 英文截图已逐张查看，英文状态下第 4 页没有未翻译文字（截图不提交）。
- **SAHI 跳过与“精度优先”预设（2026-10-28，分支 `imp/sahi`，CUDA）**：新增 6 项测试（模拟引擎下 640×480 / 1080p / 4032×3024 混合文件夹只跳过超限的图；真实 `process_frame` + 桩检测器下 1080p 图不再结束整批；无效切片设置仍停止整批；预设只改本批参数、写入运行记录、不动左侧控件；非文字提示或缺少 26L 权重时在停止检测和加载前拒绝；跳过状态的中英文显示与运行记录）。真实运行：离屏界面加载 240 张留出 COCO 图，选预设，开始后由应用加载 YOLOE-26L（SHA-256 `a612d2d5…360da9`）到 CUDA，240 张全部完成、无跳过；脚本按分数 0 批量接受（**不是人工审核**）后导出 COCO，与人工 GT 比对得 P / R@0.40 = 0.764 / 0.663，与直接调用检测路径的结果一致；清单记录 `preset: accuracy_first`、`slice_size: 320`、`conf: 0.4`；左侧置信度、尺寸、SAHI 开关和切片尺寸运行前后不变，模型栏变为 26L。工作线程每张中位 0.121 s。

## 未验证 / 开发中

- 除“精度优先”预设的一次 240 图文字提示运行外，未在本页运行 CUDA；未运行 Grounding DINO；未测试上千张的大文件夹耗时与内存，自动保存在大文件夹上的写入耗时也未测量。
- 预设只在最长边 ≤ 640 的 COCO 图上评价过；视觉提示、无提示和其它模型没有预设。
- 未进行真实人工标注会话，也未测量标注效率或快捷键是否确实加快了审核；未评价候选框精度（需要人工 GT，且不能用本页导出的标签充当）。
- 未用导出数据实际训练模型；只验证了 Ultralytics 能读取 YOLO 导出。
- 不支持多边形 / 分割标注。
- 图像宽高按 OpenCV 解码结果记录（会应用 EXIF 方向）；如果其它工具忽略 EXIF 方向，坐标可能不一致。
- 自动保存目前只在决定发生时同步写入；没有测量它对大文件夹审核流畅度的影响，也没有做磁盘写满或只读文件夹的专门测试（原子写失败会提示“标注进度未能保存”，不中断审核）。
- 恢复校验用文件大小 + 修改时间，不是每次都重新算 SHA-256（真正打开该图片审核时仍会用已有的 SHA-256 校验机制）；理论上存在同大小同 mtime 但内容不同的极端情况未被覆盖。
