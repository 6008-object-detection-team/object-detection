"""CPU uncertainty analysis of saved ablation predictions, plus a static plot.

This imports the offline COCO/bootstrap implementation, never any detector or
torch. Results describe the current small diagnostic subset, not generalization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools import analyze_saved_predictions as offline
except ModuleNotFoundError:  # Direct execution puts only tools/ on sys.path.
    import analyze_saved_predictions as offline

PAIRS = (("text_class_nms", "text_baseline"), ("visual_human_mask", "visual_bbox"),
         ("input960", "text_baseline"), ("teacher_26l", "text_baseline"))
LABELS = {"text_baseline": "Text baseline / 640", "text_class_nms": "Class-aware NMS",
          "text_one2one": "One-to-one head", "prompt_smartphone": "Prompt: smartphone",
          "prompt_mug": "Prompt: mug", "prompt_water_bottle": "Prompt: water bottle",
          "distractors": "Distractor labels", "input768": "Input 768",
          "input960": "Input 960", "dark0.6": "Darken x0.6",
          "motion_blur9": "Motion blur / 9 pixels", "jpeg35": "JPEG quality 35",
          "visual_bbox": "Visual / reference bbox", "visual_loose_bbox": "Visual / loose bbox",
          "visual_human_mask": "Visual / human mask", "prompt_cached": "Text / cached embedding",
          "detect_only": "Detection-only graph", "teacher_26l": "26L teacher candidate"}


def summarize_pairs(bootstrap, confidence=0.95):
    points, draws = bootstrap["point"], bootstrap["draws"]
    results = {}
    for enhanced, base in PAIRS:
        key = f"{enhanced}_minus_{base}"
        results[key] = {"enhanced": enhanced, "base": base, "metrics": {}, "per_class": {}}
        for metric in ("AP", "APsmall"):
            differences = []
            for draw in draws:
                left = draw["modes"][enhanced]["metrics"][metric]
                right = draw["modes"][base]["metrics"][metric]
                differences.append(left - right if left is not None and right is not None else None)
            entry = offline.interval(differences, len(draws), confidence)
            left, right = points[enhanced]["metrics"][metric], points[base]["metrics"][metric]
            entry["point_difference"] = left - right if left is not None and right is not None else None
            results[key]["metrics"][metric] = entry
        for category in points[base]["per_class"]:
            results[key]["per_class"][category] = {}
            for metric in ("AP", "APsmall"):
                differences = []
                for draw in draws:
                    left = draw["modes"][enhanced]["per_class"][category][metric]
                    right = draw["modes"][base]["per_class"][category][metric]
                    differences.append(left - right if left is not None and right is not None else None)
                entry = offline.interval(differences, len(draws), confidence)
                left, right = points[enhanced]["per_class"][category][metric], points[base]["per_class"][category][metric]
                entry["point_difference"] = left - right if left is not None and right is not None else None
                results[key]["per_class"][category][metric] = entry
    return results


def draw_comparison(scenarios, output):
    """Plot the report's point APs without running inference or ranking models."""
    try:
        import matplotlib
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib is not installed; nothing installed"}
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected = [(name, scenario) for name, scenario in scenarios.items() if scenario["status"] == "complete"]
    positions = np.arange(len(selected))
    ap = [scenario["metrics"]["AP"] for _, scenario in selected]
    small = [scenario["metrics"]["APsmall"] for _, scenario in selected]
    figure, axes = plt.subplots(figsize=(9, 8.5), constrained_layout=True)
    axes.barh(positions - 0.18, ap, height=0.34, color="#3678a7", label="AP / IoU 0.50:0.95")
    axes.barh(positions + 0.18, small, height=0.34, color="#e39c36", label="AP small")
    for y, value, small_value in zip(positions, ap, small):
        axes.text(value + 0.009, y - 0.18, f"{value:.3f}", va="center", fontsize=8, color="#285777")
        axes.text(small_value + 0.009, y + 0.18, f"{small_value:.3f}", va="center", fontsize=8, color="#956421")
    axes.set_yticks(positions, [LABELS.get(name, name) for name, _ in selected])
    axes.invert_yaxis()
    axes.set_xlim(0, 0.76)
    axes.set_xlabel("COCO bbox AP (0 to 1)")
    axes.set_title(f"{len(selected)} CPU ablations on 24 labeled images\n"
                   "Diagnostic subset; same-set method selection; point estimates only", fontsize=12)
    axes.grid(axis="x", color="#d7dee3", linewidth=0.6)
    axes.set_axisbelow(True)
    axes.spines[["right", "top"]].set_visible(False)
    axes.legend(loc="lower right", fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {"status": "created", "path": str(output), "scenario_count": len(selected),
            "notice": "Point estimates; no error bars. Different prompts/perturbations do not share the same task or difficulty."}


def display_number(value):
    return "无正例" if value is None else f"{value:+.4f}"


def write_summary(path, result, scenarios):
    rows = ["# 额外方法对照的不确定性", "",
            f"仅使用已有 CPU 预测，200 次按图配对重采样；耗时 {result['elapsed_seconds']:.1f} 秒。"
            "没有加载模型、导入 Torch 或使用 GPU。", "",
            "| 前者减后者 | 指标 | 原始差值 | 95% 条件百分位区间 | 有效抽样 |",
            "|---|---|---:|---|---:|"]
    for pair, comparison in result["differences"].items():
        for metric, entry in comparison["metrics"].items():
            ci = entry["percentile_interval"]
            text = "无正例" if ci is None else f"[{display_number(ci[0])}, {display_number(ci[1])}]"
            rows.append(f"| {pair} | {metric} | {display_number(entry['point_difference'])} | {text} | "
                        f"{entry['valid_samples']}/{entry['requested_samples']} |")
    rows += ["", "所有比较使用相同种子 20261004 和同一组 24 图有放回抽样。重复图、其 GT 和预测按新的图片 ID 复制，"
             "GT annotation ID 也重新编号；不能把重复原 ID 直接交给 COCOeval。保留原人工标注面积，"
             "使用标准 IoU 0.50:0.95、maxDets=100。", "",
             "原测试集有正例的类别若在抽样中缺少正例，其类别 AP 记为 null；整体指标要求原有类别仍有正例，"
             "避免改变类别组成，公开有效抽样比例。区间只使用有效抽样，属于条件百分位区间。", "",
             "## 人工 mask 没有解决所有类别", "",
             "| 类别 | 参考框 AP | 人工 mask AP | 差值 |",
             "|---|---:|---:|---:|"]
    bbox = scenarios["visual_bbox"]["per_class"]
    mask = scenarios["visual_human_mask"]["per_class"]
    for name in bbox:
        base, enhanced = bbox[name]["AP"], mask[name]["AP"]
        rows.append(f"| {name} | {base:.4f} | {enhanced:.4f} | {enhanced-base:+.4f} |")
    rows += ["", "mask 提升主要来自杯子；瓶子 AP 仍很低。此结果检验的是三个固定参考的类别泛化，"
             "不能解释成所有参考物、所有类别或实际摄像头场景都已经解决。", "",
             "## 解释限制", "",
             "本轮在同一个 24 图/71 GT 的小测试集筛选了多种方法，再报告看起来较好的结果，会带来方法筛选偏倚。"
             "这些区间没有对多重比较作校正，也不是泛化显著性证明，不能作为最终模型选择的独立验证。"
             "应锁定候选设置后，在独立采集且按场景划分的数据上重新评价。",
             "手机 small 只有 4 个 GT，小目标结论尤其不稳定；增加 bootstrap 次数不会代替更多独立目标。",
             "模型 26L 的输出仍是未审核预标注候选；本分析始终使用原人工 GT，没有把老师模型输出当真值。", "",
             "CPU 各场景耗时受系统负载影响；这里的 AP 图不证明速度或显存改善。模糊/暗光/JPEG 行表示输入难度变化，"
             "提示词行可能改变语义范围，不能简单按 AP 排成生产推荐榜。", "",
             "## 产物", "",
             "- report.json：输入 SHA-256、参数、点估计、四项配对差值与逐类别区间。",
             "- bootstrap_draws.json：每次抽样 ID 与紧凑的逐模式 AP/APsmall，可复现全局差值。",
             "- comparison.png：18 个已完成 CPU 场景的 AP/APsmall 点估计图；两项 CPU FP16 跳过项未画入。", "",
             "![18 个 CPU 场景](comparison.png)"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("work/additional_experiments/cpu_ablation/report.json"))
    parser.add_argument("--annotations", type=Path, default=Path("datasets/mini_eval/annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("work/additional_experiments/ablation_uncertainty"))
    parser.add_argument("--samples", type=offline.positive_int, default=200)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    upstream = json.loads(args.report.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    needed = list(dict.fromkeys(name for pair in PAIRS for name in pair))
    predictions, hashes = {}, {str(args.report): offline.sha256(args.report),
                               str(args.annotations): offline.sha256(args.annotations)}
    for name in needed:
        scenario = upstream["scenarios"][name]
        if scenario["status"] != "complete":
            raise ValueError(f"Scenario is incomplete: {name}")
        path = args.report.parent / scenario["predictions_file"]
        digest = offline.sha256(path)
        if digest != scenario["predictions_sha256"]:
            raise ValueError(f"Prediction hash does not match report: {name}")
        hashes[str(path)] = digest
        predictions[name] = json.loads(path.read_text(encoding="utf-8"))
    offline.validate_inputs(annotations, predictions)
    args.output.mkdir(parents=True, exist_ok=True)

    def progress(draws):
        print(f"CPU ablation bootstrap {len(draws)}/{args.samples}; {time.perf_counter()-started:.1f}s", flush=True)

    bootstrap = offline.paired_bootstrap(annotations, predictions, args.samples, args.seed, progress=progress)
    for name, point in bootstrap["point"].items():
        for metric in ("AP", "APsmall"):
            if not math.isclose(point["metrics"][metric], upstream["scenarios"][name]["metrics"][metric], abs_tol=1e-12):
                raise ValueError(f"Point metric does not match existing report: {name}/{metric}")
    differences = summarize_pairs(bootstrap)
    compact_draws = [{"sample": draw["sample"], "source_image_ids": draw["source_image_ids"],
                      "metrics": {name: values["metrics"] for name, values in draw["modes"].items()}}
                     for draw in bootstrap["draws"]]
    offline.atomic_json(args.output / "bootstrap_draws.json", {"seed": args.seed, "draws": compact_draws})
    plot = draw_comparison(upstream["scenarios"], args.output / "comparison.png")
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
              "torch_imported": "torch" in sys.modules, "elapsed_seconds": time.perf_counter()-started,
              "image_count": len(annotations["images"]), "annotation_count": len(annotations["annotations"]),
              "settings": {"samples": args.samples, "seed": args.seed, "confidence": 0.95,
                           "sampling_unit": "image", "paired": True, "pairs": PAIRS,
                           "missing_positive_policy": bootstrap["missing_positive_policy"]},
              "input_sha256": hashes, "point": bootstrap["point"], "differences": differences,
              "plot": plot, "limitations": [
                  "Multiple methods selected on the same small subset: selection bias; no multiplicity correction.",
                  "Conditional percentile intervals are diagnostic, not proof of generalization significance.",
                  "Only four small cell-phone GTs; an independent scene-split validation is needed.",
                  "Manual-mask gains are class dependent and do not solve every class."]}
    offline.atomic_json(args.output / "report.json", result)
    write_summary(args.output / "summary.md", result, upstream["scenarios"])
    print(f"CPU ablation uncertainty complete: {args.output.resolve()}; {result['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
