"""Plot cumulative visual LDA with per-step significance."""
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

COLOURS = {"Fourier v1": "#7c3fa8", "Fourier v2": "#eb6834"}
MARKERS = {"Fourier v1": "s", "Fourier v2": "o"}
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#555555"


def label(name: str) -> str:
    return {"L(G_felz)": "L(G_felz)", "Fourier v2": "Fourier v2"}.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/cumulative_fourier_v2_lda.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_fourier_v2_lda_steps.png")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    ks = sorted(data.k.unique())
    names = (data[data.version.eq("Fourier v2")]
             .drop_duplicates("k").set_index("k").added.to_dict())

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.2, 6.4), sharex=True, facecolor="white",
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)

    for version in ["Fourier v1", "Fourier v2"]:
        sub = data[data.version.eq(version)]
        mean = sub.groupby("k").bacc.mean().reindex(ks)
        sd = sub.groupby("k").bacc.std(ddof=1).reindex(ks)
        top.fill_between(ks, mean - sd, mean + sd, color=COLOURS[version],
                         alpha=0.13, linewidth=0)
        top.plot(ks, mean, color=COLOURS[version], marker=MARKERS[version],
                 markersize=6, markeredgecolor="white", markeredgewidth=1,
                 linewidth=2.2, label=version)

    top.set_ylabel("balanced accuracy (%)", color=INK)
    top.set_title("Cumulative visual LDA: Fourier-v2 fixes the last step",
                  loc="left", fontsize=13.5, fontweight="bold", color=INK)
    top.legend(frameon=False, loc="lower right")

    # Bottom: deltas for Fourier-v2, with old Fourier shown as hollow outline
    pivot = {version: data[data.version.eq(version)].pivot_table(
        index="fold", columns="k", values="bacc") for version in COLOURS}
    width = 0.34
    for offset, version in [(-width / 2, "Fourier v1"), (width / 2, "Fourier v2")]:
        for k in ks[1:]:
            delta = pivot[version][k] - pivot[version][k - 1]
            pvalue = ttest_rel(pivot[version][k], pivot[version][k - 1]).pvalue
            strong = pvalue < 0.05
            face = COLOURS[version] if strong else "none"
            bottom.bar(k + offset, delta.mean(), width=width,
                       color=face, edgecolor=COLOURS[version], linewidth=1.4,
                       alpha=0.9, zorder=4)
            bottom.errorbar(k + offset, delta.mean(), yerr=delta.std(ddof=1),
                            fmt="none", color=COLOURS[version], capsize=2.5,
                            linewidth=1.0, zorder=5)
            if version == "Fourier v2":
                bottom.annotate(f"p={pvalue:.3f}", (k + offset, delta.mean()),
                                textcoords="offset points",
                                xytext=(0, 7 if delta.mean() >= 0 else -14),
                                ha="center", fontsize=7.8, color=MUTED)
    bottom.axhline(0, color="#666666", linewidth=1.0)
    bottom.set_ylabel("step delta (pp)", color=INK)
    bottom.set_xticks(ks)
    bottom.set_xticklabels([label(names[k]) for k in ks],
                           rotation=28, ha="right", color=MUTED)
    bottom.text(0.01, 0.03,
                "Filled bars mark p < 0.05; p-values shown for the Fourier-v2 curve.",
                transform=bottom.transAxes, fontsize=8.5, color=MUTED,
                ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
