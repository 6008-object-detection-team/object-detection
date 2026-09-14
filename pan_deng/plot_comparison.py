"""Render a three-panel comparison chart from coco_eval_results/report.json.

Reads the actual evaluation numbers (never hardcoded) so the chart always matches
whatever evaluate_detectors.py last produced. Requires matplotlib.

Usage:
    python pan_deng/plot_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "coco_eval_results"

# Reuses the deck's own brand colors so this matches the rest of the report set.
NAVY, BLUE, CRIMSON, TEAL = "#181C62", "#4C5FD5", "#D71440", "#12897E"
INK, MUTED, GRID = "#181C62", "#526277", "#E5E8EF"

# (report.json key, display label, bar/marker color)
SERIES = [
    ("yoloe_26l", "YOLOE-26L", NAVY),
    ("yoloe_26s", "YOLOE-26S", BLUE),
    ("grounding_dino_tiny", "DINO Tiny\n(阈值 0.05)", CRIMSON),
    ("grounding_dino_tiny_t035_default_threshold", "DINO Tiny\n(阈值 0.35\n官方默认)", TEAL),
]


def load_models():
    report = json.loads((RESULTS_DIR / "report.json").read_text(encoding="utf-8"))
    models = dict(report["models"])
    # evaluate_detectors.py only ever writes the three base configurations; the
    # 0.35-threshold fairness check was a separate one-off run (see the eval
    # README) and is merged back in from its own summary file if report.json
    # doesn't already carry it.
    key = "grounding_dino_tiny_t035_default_threshold"
    if key not in models:
        extra_path = RESULTS_DIR / "grounding_dino_tiny_t035_summary.json"
        if extra_path.exists():
            models[key] = json.loads(extra_path.read_text(encoding="utf-8"))
    missing = [k for k, _, _ in SERIES if k not in models]
    if missing:
        raise SystemExit(f"report.json is missing: {missing}. Run evaluate_detectors.py first.")
    return models


def main():
    models = load_models()
    labels = [label for _, label, _ in SERIES]
    colors = [color for _, _, color in SERIES]
    map_5095 = [models[k]["metrics"]["AP@[.5:.95]"] for k, _, _ in SERIES]
    map_50 = [models[k]["metrics"]["AP@.5"] for k, _, _ in SERIES]
    ms_per_image = [models[k]["ms_per_image"] for k, _, _ in SERIES]
    ap_small = [models[k]["metrics"]["AP_small"] for k, _, _ in SERIES]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    fig.patch.set_facecolor("white")

    # ---- Panel 1: headline accuracy, grouped bars ----
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, map_5095, w, label="mAP@[.5:.95]（主指标）", color=colors, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, map_50, w, label="mAP@.5（宽松标准）", color=colors, alpha=0.45, edgecolor="white", linewidth=0.5)
    for rect, val in zip(b1, map_5095):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    for rect, val in zip(b2, map_50):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("平均精度 AP", fontsize=10.5, color=INK)
    ax.set_title("准确率对比（越高越准）", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_ylim(0, max(map_50) * 1.18)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- Panel 2: accuracy vs. speed trade-off ----
    ax = axes[1]
    for label, acc, ms, c in zip(labels, map_5095, ms_per_image, colors):
        name = label.replace("\n", " ")
        ax.scatter(ms, acc, s=420, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        dy = -22 if "0.35" in name else 10
        ax.annotate(name, (ms, acc), xytext=(10, dy), textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlabel("单张图推理耗时 (毫秒，越左越快)", fontsize=10.5, color=INK)
    ax.set_ylabel("mAP@[.5:.95]（越高越准）", fontsize=10.5, color=INK)
    ax.set_title("速度 vs 准确率\n（左上角最理想）", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_xlim(0, max(ms_per_image) * 1.12)
    ax.set_ylim(0, max(map_5095) * 1.25)
    ax.grid(color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- Panel 3: small-object accuracy (relevant to this app's own prompts) ----
    ax = axes[2]
    bars = ax.bar(x, ap_small, 0.55, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    for rect, val in zip(bars, ap_small):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("AP（小目标）", fontsize=10.5, color=INK)
    ax.set_title("小目标检测能力\n(backpack/phone/cup 这类)", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_ylim(0, max(ap_small) * 1.25)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("YOLOE vs. Grounding DINO Tiny — COCO 子集量化对比（240张图，10个类别）",
                 fontsize=15.5, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.02, "数据来源：pan_deng/coco_eval_results/report.json · pycocotools COCOeval",
              ha="center", fontsize=9, color=MUTED)

    plt.tight_layout()
    out = RESULTS_DIR / "comparison_chart.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
