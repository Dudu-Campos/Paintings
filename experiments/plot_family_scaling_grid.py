"""Plot the corrected family-count vs feature-count scaling figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fbfbfa"
GRID = "#dedede"
INK = "#0b0b0b"
MUTED = "#565656"
COLOURS = {
    "1 family": "#2a78d6",
    "2 families": "#00a884",
    "3 families": "#8a5ccf",
    "4 families": "#eb6834",
}
MARKERS = {"1 family": "o", "2 families": "s", "3 families": "^", "4 families": "D"}


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.75, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=10.5)


def format_k(value, _) -> str:
    if value < 1000:
        return f"{value:g}"
    return f"{value / 1000:.1f}k".replace(".0k", "k")


def exponent(x, bacc) -> float:
    return -np.polyfit(np.log(np.asarray(x, float)), np.log(100 - np.asarray(bacc, float)), 1)[0]


def plot_curve(axis, x, mean, sd, label, beta, annotate=True) -> None:
    colour = COLOURS[label]
    error = 100 - mean
    axis.plot(x, error, color=colour, linewidth=2.5, marker=MARKERS[label],
              markersize=6.2, markeredgecolor="white", markeredgewidth=1.0,
              label=label, zorder=4)
    axis.fill_between(x, error - sd, error + sd, color=colour, alpha=0.13,
                      linewidth=0, zorder=2)
    if annotate:
        axis.annotate(f"{label}\nβ={beta:.3f}", xy=(x[-1], error.iloc[-1]),
                      xytext=(9, 0), textcoords="offset points",
                      ha="left", va="center", color=colour,
                      fontsize=10.5, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/family_scaling_grid_125.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/family_scaling_grid_125.png")
    parser.add_argument("--data-budget", type=int, default=128)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data["train_size"] = data.train_size.replace({79853: 79852, 66260: 66259})
    order = ["1 family", "2 families", "3 families", "4 families"]
    data = data[data.pool.isin(order)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), sharey=True)
    fig.patch.set_facecolor("white")

    all_errors = 100 - data.bacc
    low = max(1.0, np.floor(all_errors.min() - 1.2))
    high = np.ceil(all_errors.max() + 1.5)
    ylim = (low, high)

    # Left: feature budget fixed, number of representation families changes.
    left_data = data[data.budget.eq(args.data_budget)]
    for label in order:
        sub = left_data[left_data.pool.eq(label)]
        mean = sub.groupby("train_size").bacc.mean().sort_index()
        sd = sub.groupby("train_size").bacc.std(ddof=1).reindex(mean.index).fillna(0.0)
        beta = exponent(mean.index, mean.to_numpy())
        plot_curve(axes[0], mean.index.to_numpy(), mean, sd, label, beta)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(sorted(left_data.train_size.unique()))
    axes[0].set_xlim(left_data.train_size.min() / 1.15, left_data.train_size.max() * 1.85)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    axes[0].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_xlabel("training images", fontsize=12, color=MUTED)
    axes[0].set_ylabel("balanced error (%)", fontsize=12, color=MUTED)
    axes[0].set_title("Data axis — fixed 128-feature budget", fontsize=13.5,
                      fontweight="bold", color=INK, loc="left", pad=10)
    axes[0].text(0.03, 0.05,
                 "only the number of representation families changes",
                 transform=axes[0].transAxes, ha="left", va="bottom",
                 fontsize=9.8, color=MUTED)
    style(axes[0])

    # Right: training set fixed, selected-feature count changes.
    full_size = int(data.train_size.max())
    right_data = data[data.train_size.eq(full_size)]
    for label in order:
        sub = right_data[right_data.pool.eq(label)]
        mean = sub.groupby("budget").bacc.mean().sort_index()
        sd = sub.groupby("budget").bacc.std(ddof=1).reindex(mean.index).fillna(0.0)
        beta = exponent(mean.index, mean.to_numpy())
        plot_curve(axes[1], mean.index.to_numpy(), mean, sd, label, beta)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(sorted(right_data.budget.unique()))
    axes[1].set_xlim(right_data.budget.min() / 1.12, right_data.budget.max() * 1.75)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_xlabel("number of selected features", fontsize=12, color=MUTED)
    axes[1].set_title("Dimensionality axis — full training set", fontsize=13.5,
                      fontweight="bold", color=INK, loc="left", pad=10)
    axes[1].text(0.03, 0.05,
                 "only the selected-feature budget changes",
                 transform=axes[1].transAxes, ha="left", va="bottom",
                 fontsize=9.8, color=MUTED)
    style(axes[1])

    for axis in axes:
        axis.set_ylim(*ylim)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.yaxis.set_minor_formatter(mticker.NullFormatter())

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=10.5)
    fig.suptitle("Representation provenance and feature budget are separated",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.015, ha="left", y=0.995)
    fig.text(0.5, 0.014,
             "Multitask125 · LDA probe · redundancy-first feature selection · 5 folds × 3 selection seeds · text excluded",
             ha="center", va="bottom", fontsize=10.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    fig.savefig(args.output, dpi=240, facecolor="white")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
