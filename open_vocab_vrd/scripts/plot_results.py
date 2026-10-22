"""Generate all research report visualizations for VRD Experiment A vs B.

Reads data from:
  - outputs/run_20260422_213310/  (Experiment A: visual only)
  - outputs/run_20260422_231714/  (Experiment B: visual + prior)

Saves plots to: outputs/plots/

Usage:
    cd open_vocab_vrd/
    python scripts/plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
PLOTS_DIR   = ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXP_A_DIR   = ROOT / "outputs" / "run_20260422_213310"
EXP_B_DIR   = ROOT / "outputs" / "run_20260422_231714"
TRIPLETS_TRAIN = ROOT / "data" / "triplets_train.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_eval(run_dir: Path) -> dict:
    with open(run_dir / "eval_results.json", encoding="utf-8") as f:
        return json.load(f)

def load_log(run_dir: Path) -> dict[str, list]:
    """Return dict of column_name -> list of values (floats)."""
    rows: dict[str, list] = {}
    with open(run_dir / "log.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                rows.setdefault(k, []).append(float(v))
    return rows

def predicate_counts_from_triplets(path: Path) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        triplets = json.load(f)
    counts: dict[str, int] = {}
    for t in triplets:
        p = t["predicate"].lower()
        counts[p] = counts.get(p, 0) + 1
    return counts

# Load everything
eval_a = load_eval(EXP_A_DIR)
eval_b = load_eval(EXP_B_DIR)
log_a  = load_log(EXP_A_DIR)
log_b  = load_log(EXP_B_DIR)
pred_freq = predicate_counts_from_triplets(TRIPLETS_TRAIN)

metrics_a = eval_a["metrics"]
metrics_b = eval_b["metrics"]
per_pred_a: dict[str, float] = eval_a["per_predicate_r50"]
per_pred_b: dict[str, float] = eval_b["per_predicate_r50"]

# Head / Tail split: predicates with >= 800 training occurrences are "head"
HEAD_THRESH = 800
predicates_all = sorted(per_pred_a.keys())
head_preds = [p for p in predicates_all if pred_freq.get(p, 0) >= HEAD_THRESH]
tail_preds = [p for p in predicates_all if pred_freq.get(p, 0) <  HEAD_THRESH]

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
COLOR_A   = "#4878CF"   # blue  - Exp A
COLOR_B   = "#D65F5F"   # red   - Exp B
COLOR_POS = "#2CA02C"   # green - positive delta
COLOR_NEG = "#D62728"   # red   - negative delta

def save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

def bar_label(ax, bars, fmt="{:.1f}%", fontsize=9):
    """Annotate bars with their values."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.003,
            fmt.format(h * 100),
            ha="center", va="bottom", fontsize=fontsize,
        )

# ===========================================================================
# Plot 1 — Overall Metrics Comparison (R@50, mR@50)
# ===========================================================================
print("Generating plot 1: overall_comparison.png")
metric_keys   = ["R@50",  "mR@50"]
vals_a = [metrics_a[k] for k in metric_keys]
vals_b = [metrics_b[k] for k in metric_keys]

x      = np.arange(len(metric_keys))
width  = 0.32

fig, ax = plt.subplots(figsize=(7, 5))
bars_a = ax.bar(x - width / 2, vals_a, width, label="Exp A (visual only)", color=COLOR_A)
bars_b = ax.bar(x + width / 2, vals_b, width, label="Exp B (visual + prior)", color=COLOR_B)

bar_label(ax, bars_a)
bar_label(ax, bars_b)

ax.set_xticks(x)
ax.set_xticklabels(metric_keys, fontsize=13)
ax.set_ylabel("Recall (%)", fontsize=12)
ax.set_title("Overall Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate delta arrows
for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
    delta = (vb - va) * 100
    ax.annotate(
        f"+{delta:.2f} pp",
        xy=(x[i] + width / 2, vb + 0.03),
        ha="center", fontsize=10, color="#1a6e1a", fontweight="bold",
    )

save(fig, "overall_comparison.png")

# ===========================================================================
# Plot 2 — Per-Predicate R@50 Grouped Bar Chart (sorted by improvement)
# ===========================================================================
print("Generating plot 2: predicate_comparison.png")

# Sort by Exp B - Exp A (descending)
sorted_preds = sorted(
    predicates_all,
    key=lambda p: per_pred_b[p] - per_pred_a[p],
    reverse=True,
)
vals_pa = [per_pred_a[p] for p in sorted_preds]
vals_pb = [per_pred_b[p] for p in sorted_preds]

x2    = np.arange(len(sorted_preds))
w2    = 0.38

fig, ax = plt.subplots(figsize=(16, 6))
brs_a = ax.bar(x2 - w2 / 2, vals_pa, w2, label="Exp A (visual only)",    color=COLOR_A, alpha=0.88)
brs_b = ax.bar(x2 + w2 / 2, vals_pb, w2, label="Exp B (visual + prior)", color=COLOR_B, alpha=0.88)

ax.set_xticks(x2)
ax.set_xticklabels(sorted_preds, rotation=40, ha="right", fontsize=10)
ax.set_ylabel("R@50", fontsize=12)
ax.set_title("Per-Predicate Recall@50 Comparison (sorted by improvement)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

save(fig, "predicate_comparison.png")

# ===========================================================================
# Plot 3 — Predicate-wise Improvement Delta (Exp B - Exp A)
# ===========================================================================
print("Generating plot 3: predicate_delta.png")

delta_sorted_preds = sorted(
    predicates_all,
    key=lambda p: per_pred_b[p] - per_pred_a[p],
    reverse=True,
)
deltas = [(per_pred_b[p] - per_pred_a[p]) for p in delta_sorted_preds]
colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]

fig, ax = plt.subplots(figsize=(14, 5))
bars_d = ax.bar(delta_sorted_preds, deltas, color=colors, width=0.6)

# Value labels
for bar, d in zip(bars_d, deltas):
    sign = "+" if d >= 0 else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        d + (0.004 if d >= 0 else -0.012),
        f"{sign}{d * 100:.1f}pp",
        ha="center", va="bottom" if d >= 0 else "top",
        fontsize=9, fontweight="bold",
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(delta_sorted_preds)))
ax.set_xticklabels(delta_sorted_preds, rotation=40, ha="right", fontsize=10)
ax.set_ylabel("Improvement (Exp B - Exp A)", fontsize=12)
ax.set_title("Predicate-wise Improvement (Exp B - Exp A)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(axis="y", linestyle="--", alpha=0.4)

save(fig, "predicate_delta.png")

# ===========================================================================
# Plot 4 — Head vs Tail Analysis
# ===========================================================================
print("Generating plot 4: head_tail.png")

def mean_recall(preds_list: list[str], per_pred: dict[str, float]) -> float:
    vals = [per_pred[p] for p in preds_list]
    return float(np.mean(vals)) if vals else 0.0

head_a = mean_recall(head_preds, per_pred_a)
head_b = mean_recall(head_preds, per_pred_b)
tail_a = mean_recall(tail_preds, per_pred_a)
tail_b = mean_recall(tail_preds, per_pred_b)

categories = ["Head\n(Exp A)", "Head\n(Exp B)", "Tail\n(Exp A)", "Tail\n(Exp B)"]
values     = [head_a, head_b, tail_a, tail_b]
bar_colors = [COLOR_A, COLOR_B, COLOR_A, COLOR_B]

fig, ax = plt.subplots(figsize=(8, 5))
bars_ht = ax.bar(categories, values, color=bar_colors, width=0.5)

for bar, val in zip(bars_ht, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val * 100:.1f}%",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

# Annotate which predicates are head/tail
head_label = f"Head predicates (freq >= {HEAD_THRESH}):\n{', '.join(head_preds)}"
tail_label = f"Tail predicates (freq < {HEAD_THRESH}):\n{', '.join(tail_preds)}"
ax.text(0.01, -0.28, head_label, transform=ax.transAxes, fontsize=7.5, color="#333")
ax.text(0.50, -0.28, tail_label, transform=ax.transAxes, fontsize=7.5, color="#333")

ax.set_ylabel("Mean R@50", fontsize=12)
ax.set_title("Head vs Tail Predicate Performance", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Draw vertical separator between head and tail groups
ax.axvline(x=1.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)

# Improvement arrows for tail (more interesting)
tail_delta = (tail_b - tail_a) * 100
ax.annotate(
    f"+{tail_delta:.1f} pp",
    xy=(3, tail_b + 0.04), ha="center", fontsize=11, color="#1a6e1a", fontweight="bold",
)

fig.subplots_adjust(bottom=0.3)
save(fig, "head_tail.png")

# ===========================================================================
# Plot 5a — Training Loss Curves (Exp B: train + val)
# ===========================================================================
print("Generating plot 5a: training_curve.png")

epochs_b = [int(e) for e in log_b["epoch"]]
epochs_a = [int(e) for e in log_a["epoch"]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs_b, log_b["train_loss"], color=COLOR_B, linewidth=2.0, label="Exp B train loss")
ax.plot(epochs_b, log_b["val_loss"],   color=COLOR_B, linewidth=2.0, linestyle="--", label="Exp B val loss")
ax.plot(epochs_a, log_a["train_loss"], color=COLOR_A, linewidth=1.5, label="Exp A train loss", alpha=0.7)
ax.plot(epochs_a, log_a["val_loss"],   color=COLOR_A, linewidth=1.5, linestyle="--", label="Exp A val loss", alpha=0.7)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training and Validation Loss Curves", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
ax.set_xlim(1, max(epochs_a[-1], epochs_b[-1]))

save(fig, "training_curve.png")

# ===========================================================================
# Plot 5b — Val Top-1 Accuracy Curve (Exp A vs Exp B)
# ===========================================================================
print("Generating plot 5b: val_accuracy_curve.png")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs_a, log_a["val_top1acc"], color=COLOR_A, linewidth=2.0, label="Exp A (visual only)")
ax.plot(epochs_b, log_b["val_top1acc"], color=COLOR_B, linewidth=2.0, label="Exp B (visual + prior)")

# Mark best points
best_a_epoch = epochs_a[int(np.argmax(log_a["val_top1acc"]))]
best_a_val   = max(log_a["val_top1acc"])
best_b_epoch = epochs_b[int(np.argmax(log_b["val_top1acc"]))]
best_b_val   = max(log_b["val_top1acc"])

ax.scatter([best_a_epoch], [best_a_val], color=COLOR_A, s=70, zorder=5)
ax.scatter([best_b_epoch], [best_b_val], color=COLOR_B, s=70, zorder=5)
ax.annotate(f"Best A: {best_a_val*100:.2f}%",
            xy=(best_a_epoch, best_a_val), xytext=(best_a_epoch + 1, best_a_val - 0.01),
            fontsize=9, color=COLOR_A, fontweight="bold")
ax.annotate(f"Best B: {best_b_val*100:.2f}%",
            xy=(best_b_epoch, best_b_val), xytext=(best_b_epoch + 1, best_b_val + 0.002),
            fontsize=9, color=COLOR_B, fontweight="bold")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val Top-1 Accuracy", fontsize=12)
ax.set_title("Validation Accuracy per Epoch (Exp A vs Exp B)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.legend(fontsize=11)
ax.grid(linestyle="--", alpha=0.4)
ax.set_xlim(1, max(epochs_a[-1], epochs_b[-1]))

save(fig, "val_accuracy_curve.png")

# ===========================================================================
# Plot 6 — R@K Comparison (R@20, R@50, R@100 for both experiments)
# ===========================================================================
print("Generating plot 6: rk_comparison.png")

rk_keys  = ["R@20", "R@50", "R@100"]
rk_a     = [metrics_a[k] for k in rk_keys]
rk_b     = [metrics_b[k] for k in rk_keys]

x6   = np.arange(len(rk_keys))
w6   = 0.32

fig, ax = plt.subplots(figsize=(8, 5))
brs6_a = ax.bar(x6 - w6 / 2, rk_a, w6, label="Exp A (visual only)",    color=COLOR_A, alpha=0.88)
brs6_b = ax.bar(x6 + w6 / 2, rk_b, w6, label="Exp B (visual + prior)", color=COLOR_B, alpha=0.88)

bar_label(ax, brs6_a, fontsize=10)
bar_label(ax, brs6_b, fontsize=10)

ax.set_xticks(x6)
ax.set_xticklabels(rk_keys, fontsize=13)
ax.set_ylabel("Recall", fontsize=12)
ax.set_title("Recall@K Comparison (K = 20, 50, 100)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)

save(fig, "rk_comparison.png")

# ===========================================================================
# Plot 7 — Predicate Frequency Distribution (training data)
# ===========================================================================
print("Generating plot 7: predicate_frequency.png")

# Sort by training count descending
freq_preds  = sorted(pred_freq.keys(), key=lambda p: pred_freq.get(p, 0), reverse=True)
freq_vals   = [pred_freq.get(p, 0) for p in freq_preds]

bar_colors_freq = [
    "#4878CF" if pred_freq.get(p, 0) >= HEAD_THRESH else "#CC9933"
    for p in freq_preds
]

fig, ax = plt.subplots(figsize=(14, 5))
bars_f = ax.bar(freq_preds, freq_vals, color=bar_colors_freq, width=0.65)

# Log scale to show tail clearly
ax.set_yscale("log")
ax.set_ylabel("Training count (log scale)", fontsize=12)
ax.set_title("Predicate Frequency in Training Set", fontsize=13, fontweight="bold")
ax.set_xticks(range(len(freq_preds)))
ax.set_xticklabels(freq_preds, rotation=40, ha="right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.axhline(HEAD_THRESH, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(len(freq_preds) - 1, HEAD_THRESH * 1.1, f"Head/Tail split ({HEAD_THRESH})",
        ha="right", fontsize=9, color="black")

# Count labels on bars
for bar, v in zip(bars_f, freq_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.2,
            f"{v:,}", ha="center", fontsize=8, rotation=0)

# Legend patches
import matplotlib.patches as mpatches
head_patch = mpatches.Patch(color="#4878CF", label=f"Head (>= {HEAD_THRESH} samples)")
tail_patch  = mpatches.Patch(color="#CC9933", label=f"Tail (< {HEAD_THRESH} samples)")
ax.legend(handles=[head_patch, tail_patch], fontsize=10)

save(fig, "predicate_frequency.png")

# ===========================================================================
# Summary
# ===========================================================================
print("\nAll plots saved to:", PLOTS_DIR)
print(f"  1. overall_comparison.png    - R@50 & mR@50 bar chart (A vs B)")
print(f"  2. predicate_comparison.png  - Per-predicate grouped bars (sorted by improvement)")
print(f"  3. predicate_delta.png       - Delta (Exp B - Exp A) per predicate")
print(f"  4. head_tail.png             - Head vs tail mean recall (A vs B)")
print(f"  5. training_curve.png        - Train/val loss over 30 epochs")
print(f"  6. val_accuracy_curve.png    - Val top-1 accuracy curve (A vs B)")
print(f"  7. rk_comparison.png         - R@20 / R@50 / R@100 comparison")
print(f"  8. predicate_frequency.png   - Training data class distribution (log scale)")
