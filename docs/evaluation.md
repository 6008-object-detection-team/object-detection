# 小型标注集：文本、真正视觉提示与 SAHI 对照

`datasets/mini_eval` 包含 24 张 COCO 2017 人工标注测试图、71 个目标框，以及 3 张完全独立的视觉参考图。目标是 `bottle`、`cup`、`cell phone`。保留原始 COCO 类别 ID、面积和 crowd 标记，不使用模型预测作为真值。文件来源、图像许可和哈希见数据集 `manifest.json`。

先在应用里“释放模型”，再从项目目录运行：

```bash
.venv/bin/python evaluate_mini_dataset.py \
  --dataset datasets/mini_eval --device CUDA \
  --modes text,text_sahi,visual,visual_sahi \
  --imgsz 640 --conf 0.05 --iou 0.5 --max-det 100 \
  --slice-size 320 --slice-overlap 0.2 --max-slices 9 \
  --warmup 1 --output work/mini_evaluation
```

只检查少量图片可加 `--limit 2`，选择 CPU 可将 `--device CUDA` 改为 `--device CPU`。CPU 运行仍然采用同样的图片、提示和参数；显存不足时程序停止，不自动换 CPU 或重试。

| 模式 | 提示 | 推理方式 |
|---|---|---|
| `text` | 固定英文类别名 | 全图 |
| `text_sahi` | 同样类别名 | 全图 + SAHI 切片，类别内 NMS |
| `visual` | 独立参考图 + 人工框，YOLOE SAVPE 编码 | 全图，每类单独推理 |
| `visual_sahi` | 同样独立视觉参考 | 每类全图 + SAHI 切片 |

四组采用同一 YOLOE 26S 权重、阈值、尺寸和全图输出框数上限。视觉编码使用完整参考图和其目标框，每个参考编码一次；之后重复使用该嵌入，不使用颜色直方图筛选，也不重新编码文本。视觉模式按“一个参考类别 → 全部测试图”的顺序执行，汇总三个类别后按分数应用每图最多 100 框。参考图 ID 在任何 `--limit` 之前与完整测试集检查分离。

程序逐张读图，一次只持有一个模型，每个模式完成后卸载。SAHI 切片逐块推理，batch=1，最多 9 块。开始加载权重之前检查全部选定图片的切片数量与系统内存；GPU 进程使用项目统一的 3 GiB 分配上限和互斥锁。阈值过低会增加候选框，切片会增加推理次数，但不会并行加载额外模型。

输出目录包含：

- `summary.md`、`comparison.csv`：四组 AP、AP50、AP75、小目标 AP、AR100、延迟与显存峰值。
- `report.json`：完整类别指标、参数、包版本、数据/权重哈希、参考框和编码时间。
- `modes/<mode>/predictions.json`：可重新评价的标准 COCO 预测。
- `modes/<mode>/timings.csv`：逐图计时。
- `samples/`：左侧人工真值、右侧预测的并排图。仅示例图显示分数 ≥ 0.25 的框；AP 仍使用配置阈值下所有预测。

计时在每次推理前后同步 CUDA，包含预处理、模型、后处理、SAHI 合并与检测结果传回 CPU；不包括加载、读图、参考编码、预热和画图。视觉模式每图时间累计三个类别的推理耗时，不能把单类别耗时当作完整三类检测速度。显存峰值包含参考编码和预热；分配值与缓存保留值分别记录。

COCO 指标使用标准 `maxDets=[1,10,100]`、IoU 0.50–0.95；AP 和 AR 取值 0–1，缺少相应尺寸的真值时写 `null`。一次运行只执行一次逐图测量，速度用于初步比较。需要严谨速度统计时独立重复整个实验，并记录摄像头输入和端到端延迟。

这个人工挑选的小样本适合找错误、验证代码和准备组会，不能当作完整 COCO 榜单成绩，也不能证明在自己的摄像头上泛化。视觉示例的外观与目标差异较大时，结果可能弱于文本提示；SAHI 也可能引入误检。后续应使用项目标注工具加入独立采集的实验室/课堂图片，并保持参考、训练与测试来源分离。

纯 CPU / mock 回归检查：

```bash
.venv/bin/python -m unittest test_visual_engine test_mini_evaluation -v
```

包含完美真值 AP=1、空预测 AP=0、参考图泄漏拒绝、切片/内存预检、各模式参数一致、逐类汇总后总框数上限和视觉缓存切换的覆盖；这些测试不加载模型或运行 GPU。
