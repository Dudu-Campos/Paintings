"""Plot latent-space CKA between PRISM representation families.

Inputs are produced by ``evaluate_family_latent_cka.py``. The figure is meant to
support a narrow claim: different representation families induce different
latent organizations under the same learner. It therefore shows both the
family-by-family CKA matrix and the within-vs-between distribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INK = "#202124"
MUTED = "#6b6f76"
GRID = "#e2e2df"
SURFACE = "#fbfbfa"
WITHIN = "#2a78d6"
BETWEEN = "#e34948"


def regime_label(scope: str, centered: bool) -> str:
    if scope == "all" and not centered:
        return "All samples"
    if scope == "contested" and not centered:
        return "Contested samples"
    if scope == "all" and centered:
        return "All, class-centered"
    return "Contested, class-centered"


def matrix_for(table: pd.DataFrame, scope: str, centered: bool) -> tuple[list[str], np.ndarray]:
    subset = table[(table.scope == scope) & (table.class_centered == centered)]
    families = sorted(set(subset.family_a) | set(subset.family_b))
    matrix = np.eye(len(families), dtype=float)
    index = {name: i for i, name in enumerate(families)}
    for row in subset.itertuples(index=False):
        i, j = index[row.family_a], index[row.family_b]
        matrix[i, j] = matrix[j, i] = float(row.cka_mean)
    return families, matrix


def plot_heatmap(axis, families: list[str], matrix: np.ndarray, title: str) -> None:
    image = axis.imshow(matrix, cmap="viridis", vmin=0.2, vmax=0.95)
    axis.set_xticks(range(len(families)))
    axis.set_yticks(range(len(families)))
    axis.set_xticklabels(families, rotation=35, ha="right", fontsize=9)
    axis.set_yticklabels(families, fontsize=9)
    axis.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8)
    axis.tick_params(length=0, colors=INK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for i in range(len(families)):
        for j in range(len(families)):
            value = matrix[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center",
                      fontsize=8.5, color="white" if value < 0.62 else INK)
    return image


def plot_distribution(axis, pairs: pd.DataFrame, scope: str, centered: bool) -> None:
    subset = pairs[(pairs.scope == scope) & (pairs.class_centered == centered)]
    values = [subset[subset.relation == "within"].cka.to_numpy(float),
              subset[subset.relation == "between"].cka.to_numpy(float)]
    positions = [0, 1]
    rng = np.random.default_rng(0)
    for pos, vals, colour in zip(positions, values, [WITHIN, BETWEEN]):
        jitter = rng.normal(0, 0.035, len(vals))
        axis.scatter(np.full(len(vals), pos) + jitter, vals, s=10, alpha=0.22,
                     color=colour, linewidths=0)
        mean = vals.mean()
        lo, hi = np.quantile(vals, [0.025, 0.975])
        axis.plot([pos - 0.18, pos + 0.18], [mean, mean], color=colour, lw=3)
        axis.vlines(pos, lo, hi, color=colour, lw=2, alpha=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels(["within\nsame family", "between\nfamilies"], fontsize=10)
    axis.set_ylim(0.15, 1.0)
    axis.set_ylabel("Linear CKA", fontsize=10, color=INK)
    axis.set_title(regime_label(scope, centered), fontsize=11, fontweight="bold", color=INK, pad=8)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path,
                        default=ROOT / "results/family_latent_cka_multitask125")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/family_latent_cka_multitask125.png")
    args = parser.parse_args()

    matrix_table = pd.read_csv(args.prefix.with_name(args.prefix.name + "_matrix.csv"))
    pairs = pd.read_csv(args.prefix.with_name(args.prefix.name + "_pairs.csv"))
    summary = pd.read_csv(args.prefix.with_name(args.prefix.name + "_bootstrap.csv"))

    fig = plt.figure(figsize=(13.0, 8.4), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.16, 1.0, 1.0], wspace=0.36, hspace=0.38)

    families, matrix = matrix_for(matrix_table, "contested", True)
    heat = plot_heatmap(fig.add_subplot(grid[:, 0]), families, matrix,
                        "Latent CKA by representation family\ncontested, class-centered")

    regimes = [("all", False), ("contested", False), ("all", True), ("contested", True)]
    for cell, (scope, centered) in zip([grid[0, 1], grid[0, 2], grid[1, 1], grid[1, 2]], regimes):
        plot_distribution(fig.add_subplot(cell), pairs, scope, centered)

    cbar = fig.colorbar(heat, ax=fig.axes[0], fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label("Linear CKA", color=MUTED, fontsize=10)
    cbar.ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle("Representation families induce distinct latent organizations",
                 fontsize=16, fontweight="bold", color=INK, y=0.98)
    delta = summary[(summary.scope == "contested") & (summary.class_centered)].iloc[0]
    fig.text(0.5, 0.018,
             "Same MLP, same bottleneck, same folds and seeds. "
             f"Contested class-centered: within - between = {delta.delta_within_minus_between:.3f} "
             f"[{delta.bootstrap_ci_low:.3f}, {delta.bootstrap_ci_high:.3f}].",
             ha="center", va="bottom", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.savefig(args.output, dpi=240, facecolor="white")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
