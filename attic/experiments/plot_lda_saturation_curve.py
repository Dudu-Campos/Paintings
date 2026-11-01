"""LDA saturation curve: BAcc vs number of features, two views.

1. Block-order step curve: whole representations added one at a time in
   exact-Shapley rank order (from `outputs/full_factorial_ablation_text/`),
   reading the already-computed 5-fold LDA BAcc at each cumulative mask --
   no retraining needed, the factorial sweep already covers these points.
2. Top-k ceiling curve: LDA re-fit on the top-k individually ranked features
   (structure coefficient) from `outputs/lda_topk_curve/`, showing the best
   achievable BAcc at each raw feature budget regardless of which block a
   feature comes from.

Reading both together: how close does adding whole representations in order
of importance get to the ceiling of picking the single best k features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FACTORIAL_DIR = ROOT / "outputs/full_factorial_ablation_text"
TOPK_DIR = ROOT / "outputs/lda_topk_curve"
OUTPUT_CSV = ROOT / "results/lda_block_order_vs_topk_curve.csv"
OUTPUT_PNG = ROOT / "results/lda_block_order_vs_topk_curve.png"

# Cumulative masks along the exact-Shapley rank order (best factor first),
# built from FACTORS = (SLIC=1, Descriptors=2, Hilbert=4, Line graph=8,
# Felzenszwalb=16, Fourier=32, Text=64, Multiscale=128) in
# evaluate_full_factorial_ablation_text.py.
STEPS = [
    (0, "Chance"),
    (2, "+ Descriptors"),
    (18, "+ Felzenszwalb"),
    (19, "+ SLIC"),
    (83, "+ Text"),
    (91, "+ Line graph"),
    (95, "+ Hilbert"),
    (127, "+ Fourier"),
    (255, "+ Multiscale (full pipeline)"),
]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#d8d7d0"
TEXT_SECONDARY = "#52514e"


def main() -> None:
    summary = pd.read_csv(FACTORIAL_DIR / "configuration_summary.csv")
    summary = summary[summary.classifier.eq("lda")].set_index("mask")
    masks = [mask for mask, _ in STEPS]
    labels = [label for _, label in STEPS]
    steps = summary.loc[masks, ["dimensions", "bacc_mean", "bacc_std"]].copy()
    steps["label"] = labels
    steps.to_csv(OUTPUT_CSV, index=False)

    topk = pd.read_csv(TOPK_DIR / "curve_summary.csv").sort_values("k")

    figure, axis = plt.subplots(figsize=(11, 6.5))

    chance = steps.loc[0, "bacc_mean"]
    axis.axhline(chance, color=GRID, linewidth=1.5, linestyle="--", zorder=1)
    axis.text(
        1.15, chance, "chance (1/27)", color=TEXT_SECONDARY, fontsize=8.5,
        va="bottom", ha="left",
    )

    block_steps = steps.iloc[1:]
    axis.plot(
        block_steps["dimensions"], block_steps["bacc_mean"], "o-",
        color=BLUE, linewidth=2, markersize=7, zorder=3, label="Representações inteiras, em ordem Shapley",
    )
    axis.fill_between(
        block_steps["dimensions"],
        block_steps["bacc_mean"] - block_steps["bacc_std"],
        block_steps["bacc_mean"] + block_steps["bacc_std"],
        color=BLUE, alpha=0.12, zorder=2,
    )
    offsets = [(-6, -16), (10, 26), (14, -18), (0, 24), (-4, -20), (6, 22), (0, -20), (-40, 14)]
    aligns = ["right", "center", "left", "center", "right", "center", "center", "right"]
    for (_, row), (dx, dy), align in zip(block_steps.iterrows(), offsets, aligns, strict=True):
        axis.annotate(
            row["label"], (row["dimensions"], row["bacc_mean"]), xytext=(dx, dy),
            textcoords="offset points", ha=align, fontsize=8, color=TEXT_SECONDARY,
        )
    axis.set_ylim(top=block_steps["bacc_mean"].max() + 0.07)

    axis.plot(
        topk["k"], topk["bacc_mean"], "o-",
        color=ORANGE, linewidth=2, markersize=6, zorder=3,
        label="Top-k features individuais (coeficiente de estrutura LDA)",
    )
    axis.fill_between(
        topk["k"], topk["bacc_mean"] - topk["bacc_std"], topk["bacc_mean"] + topk["bacc_std"],
        color=ORANGE, alpha=0.12, zorder=2,
    )

    axis.set_xscale("log")
    axis.set_xlabel("Número de features (escala log)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("WikiArt (27 estilos, LDA) — construção por representação vs. melhores k features")
    axis.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
