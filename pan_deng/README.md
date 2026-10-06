# Pan Deng 的标注评测工具

保留 Pan Deng 对 Grounding DINO Tiny / YOLOE 的标注评价贡献与逐类别 token 打分修正。此公开整合发布脚本，不附带本机原始预测、日志或个人实验存档。

- `download_coco_subset.py`：按十类构建约 240 张 COCO val2017 子集，写入忽略的 `coco_eval_data/`。
- `evaluate_detectors.py`：用人工 COCO 标注和 COCOeval 评价 YOLOE-26L / 26S 与旧 token 打分 DINO，写入本地 `coco_eval_results/`。
- `sweep_threshold_topk.py`：固定预测，分别改变置信度和框数上限。
- `plot_comparison.py` / `plot_sweep.py`：在本地生成上述结果后绘图。
- `requirements-eval.txt`：基础、DINO 和 pycocotools 依赖入口。

```powershell
python -m pip install -r pan_deng\requirements-eval.txt
python pan_deng\download_coco_subset.py --annotations path\to\instances_val2017.json
python pan_deng\evaluate_detectors.py
python pan_deng\sweep_threshold_topk.py
python pan_deng\plot_comparison.py
python pan_deng\plot_sweep.py
```

原始人工标注来自 [COCO 官方下载页](https://cocodataset.org/#download)。运行前先释放界面模型，检查资源；模型串行加载。没有标注的检测框数 / 耗时不等于 mAP。

旧评测 DINO 与当前 GUI 的后处理实现不同，结果须注明配置，不能直接混用。
