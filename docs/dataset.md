# 小型真实标注测试集

数据保存在项目的 `datasets/mini_eval/`，以瓶子、杯子、手机为重点。使用 **COCO 2017 validation** 原始人工标注，不采用模型预测框作为真值，没有下载训练图像，也没有重新训练模型。

| 类别 | 本项目预测 class_id | 原始 COCO category_id | 真值框数 | 其中非 crowd 小目标框 |
|---|---:|---:|---:|---:|
| bottle / 瓶子 | 0 | 44 | 30 | 11 |
| cup / 杯子 | 1 | 47 | 30 | 6 |
| cell phone / 手机 | 2 | 77 | 11 | 4 |
| 合计 | — | — | 71 | 21 |

共有 **24 张评测图片、71 个人工框**，原图最长边不超过 640 像素。默认每类分配 8 张图片名额，其中 4 张按小目标选取、2 张按中等目标选取、2 张按大目标选取。图片中可能同时出现多个类别与多种尺寸，因此名额不等于最终实例数。尺寸分组使用原始 COCO `area`：小目标 `< 32²`，中等目标 `[32², 96²)`，大目标 `>= 96²`。

另外提供 **3 张独立视觉参考图片**，不属于这 24 张评测图。参考对象的原始人工框已目视检查：瓶装饮料、纸杯、智能手机。参考框的来源和图片 ID 写在 `references.json`；原始 GT 没有被修改。

## 文件和来源

- `images/`：24 张原始评测 JPEG。
- `annotations.json`：标准 COCO 格式，保留每张选中图片中三类目标的全部原始框、`area`、`iscrowd`、`segmentation` 和原始标注 ID。其他 COCO 类别不参与这个三类对照。
- `references/`：3 张独立参考 JPEG。
- `references.json`：`classes` 和 `references`；每条包含 `class_id`、`class_name`、`category_id`、`image_file`、`bbox_xyxy`、`bbox_xywh`、原始图片/标注 ID。`image_file` 相对于本数据集根目录。
- `manifest.json`：选取种子、选取原因、原始图片 ID、来源 URL、逐图许可证、逐图 SHA-256、原始 val 标注 SHA-256 和样本限制。
- `preview.jpg`：缩小后的带真值框总览，便于检查样本；评测始终读取未缩小的原图。
- `references_preview.jpg`：三个参考对象的框内裁剪总览，便于检查视觉提示。

来源：[COCO 官方下载页面](https://cocodataset.org/#download)、[官方 S3 标注压缩包](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip)。脚本通过 HTTP Range 只读取 `annotations/instances_val2017.json` 的压缩内容（约 6.53 MB），展开约 19.99 MB；不会下载约 253 MB 全包或约 470 MB 的训练标注 JSON。图片逐张顺序下载，总计约 4.5 MB。

COCO 人工标注属于 CC BY 4.0；图片分别使用原始 Flickr 许可证，逐图记录在清单中。[COCO 官方使用条款](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)。

## 重建与检查

在项目目录运行：

```bash
python tools/prepare_mini_dataset.py
python -m unittest test_dataset -v
```

脚本只使用 Python 标准库下载和处理 COCO JSON，不导入 PyTorch、不占显存、不加载模型。它拒绝忽略 Range 的服务器响应，检查 ZIP 成员长度与 CRC、原始 val JSON 的 SHA-256，限制标注和图片大小，以临时文件写入并 `fsync` 后原子替换完成文件。已有图片和缓存可复用。原始 val JSON 的 SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`，本地 JSON 若已修改，脚本会拒绝将其当成官方真值。

默认固定种子 `6008`。可使用 `--annotation-json /path/instances_val2017.json` 读取本地官方验证标注；也可指定 `--count`、`--classes`、`--seed`、`--output` 重建另一套样本。改动选取规则后，应使用新的输出目录，并记录测试集版本，避免把不同样本上的指标直接比较。

## 怎样用于视觉提示和切片对照

预测类别使用 `bottle,cup,cell phone`，按表中关系映射到原始 COCO ID。所有方法读取同一组评测图和原始 GT；视觉提示使用 `references.json` 的独立参考对象，不从待评测图中挑选答案框。普通检测与切片检测需要固定置信度、IoU、预测框数、输入分辨率及硬件，只改变是否切片，记录精度、召回、小目标表现和总推理耗时。切片逐块顺序推理，不把所有切片拼成 GPU batch。

这个小集用于快速验证功能、发现漏检和比较参数。它偏向小物体，样本量很小，不能当成 COCO 完整验证集的排行榜结果；公开验证集也可能已经影响预训练模型的选择。它没有明确的遮挡程度标注，不能据此声称遮挡专项评价，更不能代替你自己摄像头场景的评价。

## 从自己的视频准备下一批人工标注

```bash
python tools/prepare_mini_dataset.py \
  --video /path/to/local_video.mp4 \
  --video-output datasets/local_to_annotate \
  --interval-seconds 5 --count 24 --max-side 1280
```

这会最多每 5 秒抽取 1 帧、最多 24 帧、最长边限制为 1280 像素，输出 JPEG 和 `unannotated_manifest.json`。状态明确标为 **UNANNOTATED**，不产生伪造的人工标签；完成逐图人工框选、纠错和补漏后，再导出 COCO 格式参加评测。训练、视觉参考和最终测试应按原始视频/场景分组分离，避免相邻帧泄漏。
