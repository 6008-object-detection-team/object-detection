"""Render the threshold / box-budget sweep charts from coco_eval_results/sweep/sweep.json.

Reads the numbers written by sweep_threshold_topk.py (never hardcoded). Requires
matplotlib.

Usage:
    python pan_deng/plot_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "coco_eval_results" / "sweep"

# Categorical order checked for colour-blind separation (all pairs) on a light surface.
COLORS = {"yoloe_26l": "#2A78D6", "yoloe_26s": "#1BAF7A", "grounding_dino_tiny": "#EB6834"}
INK, MUTED, GRID = "#181C2E", "#5A6072", "#E6E8EE"


def style(ax):
    ax.grid(color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)


def label_line_ends(ax, ends):
    """Direct labels right of each line end, pushed apart (in points) so they never overlap."""
    to_points = 72 / ax.figure.dpi
    points = [(x, y, text, ax.transData.transform((x, y))[1] * to_points) for x, y, text in ends]
    points.sort(key=lambda item: item[3])
    placed = []
    for x, y, text, y_pt in points:
        target = max(y_pt, placed[-1] + 13) if placed else y_pt
        placed.append(target)
        shift = target - y_pt
        arrow = None
        if abs(shift) > 1:
            arrow = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4,
                         relpos=(0, 0.5))
        ax.annotate(text, (x, y), xytext=(10, shift), textcoords="offset points",
                    fontsize=9.5, color=INK, va="center", ha="left",
                    annotation_clip=False, arrowprops=arrow)


def plot_decoupled(report):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.patch.set_facecolor("white")
    left_ends, right_ends = [], []
    for name, result in report["models"].items():
        color, label = COLORS[name], result["label"]
        short = label.replace("Grounding ", "")
        rows = result["threshold_sweep"]
        xs, ys = [r["confidence"] for r in rows], [r["mAP"] for r in rows]
        left.plot(xs, ys, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2,
                  label=label, zorder=3, solid_capstyle="round")
        left_ends.append((xs[-1], ys[-1], f"{short} {ys[-1]:.3f}"))

        rows = result["box_budget_sweep"]
        xs, ys = [r["max_boxes"] for r in rows], [r["mAP"] for r in rows]
        right.plot(xs, ys, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2,
                   label=label, zorder=3, solid_capstyle="round")
        right_ends.append((xs[-1], ys[-1], f"{short} {ys[-1]:.3f}"))

    budget = report["models"][next(iter(report["models"]))]["threshold_sweep"][0]["max_boxes"]
    left.set_title(f"每图最多 {budget} 个框不变，只改置信度阈值", fontsize=12.5,
                   fontweight="bold", color=INK, pad=10, loc="left")
    left.set_xlabel("置信度阈值", fontsize=10.5, color=INK)
    left.set_ylabel("mAP@[.5:.95]", fontsize=10.5, color=INK)
    left.set_xlim(0, 0.72)

    right.set_xscale("log")
    budgets = [r["max_boxes"] for r in next(iter(report["models"].values()))["box_budget_sweep"]]
    right.set_xticks(budgets)
    right.set_xticklabels([str(b) for b in budgets])
    right.minorticks_off()
    right.set_title("阈值≈0 不变，只改每图最多保留的框数", fontsize=12.5,
                    fontweight="bold", color=INK, pad=10, loc="left")
    right.set_xlabel("每张图最多保留的框数（按分数从高到低，对数刻度）", fontsize=10.5, color=INK)

    top = max(r["mAP"] for res in report["models"].values()
              for key in ("threshold_sweep", "box_budget_sweep") for r in res[key])
    for ax in (left, right):
        style(ax)
        ax.set_ylim(0, top * 1.15)
    left.legend(loc="lower left", fontsize=9, frameon=False)

    fig.suptitle("把置信度阈值和框数分开测（COCO 子集 240 张图，10 类）",
                 fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left", y=1.03)
    fig.text(0.01, -0.04, "数据来源：pan_deng/coco_eval_results/sweep/sweep.json · "
             "每个模型只推理一次，所有设置离线套用 · pycocotools COCOeval",
             fontsize=9, color=MUTED)
    plt.tight_layout(w_pad=11)
    # Place end labels after layout so their point offsets match the final axes size.
    label_line_ends(left, left_ends)
    label_line_ends(right, right_ends)
    out = SWEEP_DIR / "decoupled_sweep.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_operating_point(report):
    models = report["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.6), sharey=True)
    fig.patch.set_facecolor("white")
    styles = [("precision", "精确率", (0, (5, 3)), 1.6), ("recall", "召回率", (0, (1, 2.5)), 1.8),
              ("F1", "F1", "solid", 2.4)]
    for ax, (name, result) in zip(axes, models.items()):
        color = COLORS[name]
        rows = result["threshold_sweep"]
        xs = [r["confidence"] for r in rows]
        for key, text, dash, width in styles:
            ax.plot(xs, [r[key] for r in rows], color=color, lw=width, linestyle=dash,
                    label=text, zorder=3, dash_capstyle="round", solid_capstyle="round")
        best = result["best_f1"]
        ax.axvline(best["confidence"], color=MUTED, lw=0.8, zorder=2)
        ax.plot(best["confidence"], best["F1"], "o", ms=9, color=color, mec="white",
                mew=2, zorder=4)
        ax.set_title(f"{result['label']}　F1 最高 {best['F1']:.2f}（阈值 {best['confidence']}）",
                     fontsize=12, fontweight="bold", color=INK, pad=8, loc="left")
        ax.set_xlabel("置信度阈值", fontsize=10.5, color=INK)
        ax.set_xlim(0, 0.72)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("IoU 0.5 下的比例", fontsize=10.5, color=INK)
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("应用里阈值设多少：看固定阈值下的精确率 / 召回率 / F1",
                 fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left", y=1.04)
    fig.text(0.01, -0.05, "每图最多 100 个框 · COCO 匹配规则 · COCO 子集上的最优阈值不能直接搬到摄像头场景，需要用自己的标注样本复核",
             fontsize=9, color=MUTED)
    plt.tight_layout()
    out = SWEEP_DIR / "operating_point.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    path = SWEEP_DIR / "sweep.json"
    if not path.exists():
        raise SystemExit("sweep/sweep.json not found. Run sweep_threshold_topk.py first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    for out in (plot_decoupled(report), plot_operating_point(report)):
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
