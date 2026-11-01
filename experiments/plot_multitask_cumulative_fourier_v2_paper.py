"""Paper-style cumulative Multitask LDA curve using Fourier-v2."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fbfbfa"
GRID = "#dedede"
INK = "#101010"
MUTED = "#666666"
MAIN = "#7c3fa8"
OLD = "#b8a4c8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/cumulative_fourier_v2_lda.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multitask_cumulative_fourier_v2_lda_paper.png")
    parser.add_argument("--compare-old", action="store_true",
                        help="also draw the old Fourier endpoint as a faint reference")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    new = data[data.version.eq("Fourier v2")].copy()
    old = data[data.version.eq("Fourier v1")].copy()
    ks = sorted(new.k.unique())
    names = new.drop_duplicates("k").set_index("k").added.to_dict()
    names[8] = "Fourier"

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.7, 6.2), sharex=True, facecolor="white",
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.09})
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c7c7c7")
        ax.tick_params(colors=MUTED)

    mean = new.groupby("k").bacc.mean().reindex(ks)
    sd = new.groupby("k").bacc.std(ddof=1).reindex(ks)
    top.fill_between(ks, mean - sd, mean + sd, color=MAIN, alpha=0.15, linewidth=0)
    top.plot(ks, mean, color=MAIN, marker="s", markersize=6.3,
             markeredgecolor="white", markeredgewidth=1.0, linewidth=2.25,
             label="LDA with Fourier-v2")

    if args.compare_old:
        old_mean = old.groupby("k").bacc.mean().reindex(ks)
        top.plot(ks, old_mean, color=OLD, marker="s", markersize=4.8,
                 markeredgecolor="white", markeredgewidth=0.8, linewidth=1.6,
                 linestyle="--", label="old Fourier endpoint")

    top.set_ylabel("balanced accuracy (%)", color=INK)
    top.set_title("Multitask125 cumulative visual LDA with stable spectral summaries",
                  loc="left", fontsize=13.2, fontweight="bold", color=INK)
    top.legend(frameon=False, loc="lower right")

    pivot = new.pivot_table(index="fold", columns="k", values="bacc")
    deltas = []
    for k in ks[1:]:
        delta = pivot[k] - pivot[k - 1]
        pvalue = ttest_rel(pivot[k], pivot[k - 1]).pvalue
        deltas.append((k, delta.mean(), delta.std(ddof=1), pvalue, int((delta > 0).sum())))

    for k, avg, std, pvalue, positives in deltas:
        strong = pvalue < 0.05
        bottom.bar(k, avg, width=0.60, color=MAIN if strong else "none",
                   edgecolor=MAIN, linewidth=1.5, alpha=0.90, zorder=4)
        bottom.errorbar(k, avg, yerr=std, fmt="none", color=MAIN,
                        capsize=3, linewidth=1.0, zorder=5)
        bottom.annotate(f"{positives}/5", (k, bottom.get_ylim()[0] if bottom.get_ylim()[0] < 0 else 0),
                        textcoords="offset points", xytext=(0, -18),
                        ha="center", va="top", fontsize=8.2, color=MUTED)
    bottom.axhline(0, color="#707070", linewidth=1.0)
    bottom.set_ylabel("LDA step delta (pp)", color=INK)
    bottom.set_xticks(ks)
    bottom.set_xticklabels([names[k] for k in ks], rotation=28, ha="right", color=MUTED)
    bottom.text(0.01, 0.04,
                "Filled bars: paired t-test p < 0.05. Numbers below bars: positive folds.",
                transform=bottom.transAxes, ha="left", va="bottom",
                fontsize=8.5, color=MUTED)

    # Keep lower annotations visible.
    y0, y1 = bottom.get_ylim()
    bottom.set_ylim(y0 - 0.18 * (y1 - y0), y1 + 0.08 * (y1 - y0))

    fig.tight_layout()
    fig.savefig(args.output, dpi=240, facecolor="white")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
