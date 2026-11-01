"""Bundle version of the paired provenance/dimensionality scaling figure.

Unlike ``plot_paired_dimensionality_scaling.py``, which highlights one fixed
budget per axis, this plot uses the whole paired grid as a family of scaling
curves. The goal is to make the visual language closer to ordinary scaling-law
figures: many curves, same axes, slope is the object.
"""
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

BANK, ONE = "#2a78d6", "#eb6834"
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
POOLS = [("bank", "eight-family bank", BANK, "o"),
         ("descriptors", "descriptor family", ONE, "s")]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10.5)


def format_k(v, _) -> str:
    if v < 1000:
        return f"{v:g}"
    return f"{v/1000:.1f}k".replace(".0k", "k") if v < 10000 else f"{v/1000:.0f}k"


def exponent(x, bacc) -> float:
    return -np.polyfit(np.log(np.asarray(x, float)), np.log(100 - np.asarray(bacc, float)), 1)[0]


def alpha_rank(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 0.30 + 0.70 * index / (total - 1)


def data_bundle(ax, data: pd.DataFrame, ylim: tuple[float, float]) -> None:
    budgets = sorted(data.budget.unique())
    rows = []
    for pool, label, colour, marker in POOLS:
        for rank, budget in enumerate(budgets):
            sub = data[(data.pool == pool) & (data.budget == budget)]
            mean = sub.groupby("train_size").bacc.mean()
            sd = sub.groupby("train_size").bacc.std(ddof=1)
            error = 100 - mean
            alpha = alpha_rank(rank, len(budgets))
            highlight = budget == max(budgets)
            ax.plot(mean.index, error, color=colour, alpha=alpha,
                    linewidth=2.5 if highlight else 1.35,
                    marker=marker if highlight else None, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
            if highlight:
                ax.fill_between(mean.index, error - sd, error + sd,
                                color=colour, alpha=0.15, linewidth=0, zorder=2)
            beta = exponent(mean.index, mean.to_numpy())
            rows.append((pool, budget, beta))
            if highlight:
                ax.annotate(f"{label}\nβ={beta:.3f}", xy=(mean.index[-1], error.iloc[-1]),
                            xytext=(9, 0), textcoords="offset points",
                            color=colour, fontsize=10.5, fontweight="bold", va="center")
    summary = pd.DataFrame(rows, columns=["pool", "budget", "beta"])
    means = summary.groupby("pool").beta.mean()
    ax.text(0.03, 0.96,
            f"mean β: bank {means['bank']:.3f} · descriptors {means['descriptors']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted(data.train_size.unique()))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([70, 75, 80, 85, 90])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(*ylim)
    ax.set_title("Data axis — a bundle of feature budgets", fontsize=13.5,
                 fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlabel("training images", fontsize=12, color=MUTED)
    ax.set_ylabel("balanced error (%)", fontsize=12, color=MUTED)
    ax.text(0.03, 0.05, "thin curves: 8, 16, 32, 64 features · bold curve: 128 features",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.8, color=MUTED)
    style(ax)


def dimensionality_bundle(ax, data: pd.DataFrame, ylim: tuple[float, float]) -> None:
    # Stop at 64 to avoid the one-family pool exhausting its 130 columns.
    data = data[data.budget <= 64].copy()
    sizes = sorted(data.train_size.unique())
    rows = []
    for pool, label, colour, marker in POOLS:
        for rank, size in enumerate(sizes):
            sub = data[(data.pool == pool) & (data.train_size == size)]
            mean = sub.groupby("budget").bacc.mean()
            sd = sub.groupby("budget").bacc.std(ddof=1)
            error = 100 - mean
            alpha = alpha_rank(rank, len(sizes))
            highlight = size == max(sizes)
            ax.plot(mean.index, error, color=colour, alpha=alpha,
                    linewidth=2.5 if highlight else 1.25,
                    marker=marker if highlight else None, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
            if highlight:
                ax.fill_between(mean.index, error - sd, error + sd,
                                color=colour, alpha=0.15, linewidth=0, zorder=2)
            beta = exponent(mean.index, mean.to_numpy())
            rows.append((pool, size, beta))
            if highlight:
                ax.annotate(f"{label}\nβ={beta:.3f}", xy=(mean.index[-1], error.iloc[-1]),
                            xytext=(9, 0), textcoords="offset points",
                            color=colour, fontsize=10.5, fontweight="bold", va="center")
    summary = pd.DataFrame(rows, columns=["pool", "train_size", "beta"])
    means = summary.groupby("pool").beta.mean()
    ax.text(0.03, 0.96,
            f"mean β: bank {means['bank']:.3f} · descriptors {means['descriptors']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted(data.budget.unique()))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([70, 75, 80, 85, 90])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(*ylim)
    ax.set_title("Dimensionality axis — a bundle of training sizes", fontsize=13.5,
                 fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlabel("number of selected features", fontsize=12, color=MUTED)
    ax.text(0.03, 0.05, "thin curves: 2.5k–40k images · bold curve: full train set",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.8, color=MUTED)
    style(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/paired_dimensionality_scaling_125.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/paired_dimensionality_scaling_bundle.png")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data["train_size"] = data.train_size.replace({79853: 79852, 66260: 66259})

    error = 100 - data.bacc
    low = max(1.0, np.floor(error.min() - 1.0))
    high = np.ceil(error.max() + 1.5)
    ylim = (low, high)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=True)
    fig.patch.set_facecolor("white")
    data_bundle(axes[0], data, ylim)
    dimensionality_bundle(axes[1], data, ylim)

    fig.suptitle("Representation provenance changes data scaling more than dimensional scaling",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.015, ha="left", y=0.99)
    fig.text(0.5, 0.012,
             "Balanced error is plotted in log scale. Each curve is the same LDA probe and the same redundancy-first selection rule; only the feature pool changes.",
             ha="center", va="bottom", fontsize=10.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(args.output, dpi=240, facecolor="white")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
