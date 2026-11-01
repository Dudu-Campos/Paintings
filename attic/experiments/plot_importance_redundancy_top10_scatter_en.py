"""Single-panel English scatter plot for top-10 local redundancy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC = "top10_mean_r2"
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
}
LABELS = {
    "Region graphs": "Region graphs",
    "Image statistics": "Image statistics",
    "Space-filling curve": "Space-filling curve",
    "Spectral": "Spectral",
}
TEXT = "#202124"
MUTED = "#65676b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--importance",
        type=Path,
        default=ROOT / "results/rich_feature_importance_vs_redundancy_pt.csv",
    )
    parser.add_argument(
        "--redundancy",
        type=Path,
        default=ROOT / "results/rich_feature_redundancy_r2/neighborhood_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/rich_importance_redundancy_top10_scatter_en.png",
    )
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--size", type=float, default=15.0)
    args = parser.parse_args()

    importance = pd.read_csv(args.importance)
    redundancy = pd.read_csv(args.redundancy)
    data = importance.merge(
        redundancy[["feature", METRIC]],
        on="feature",
        how="inner",
        validate="one_to_one",
    )
    data.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.8f")

    y_cut = float(data["importance_pp"].quantile(0.75))
    y_max = float(data["importance_pp"].quantile(0.997) * 1.15)
    y_max = max(y_max, float(data["importance_pp"].max() * 1.08))

    x_cut = float(data[METRIC].quantile(0.75))
    x_max = float(data[METRIC].quantile(0.995) * 1.10)
    x_max = min(1.0, max(x_max, float(data[METRIC].max() * 1.04)))

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    handles = []
    for group, rows in data.groupby("group", sort=False):
        colour = COLOURS.get(group, "#777777")
        axis.scatter(
            rows[METRIC],
            rows["importance_pp"],
            s=args.size,
            color=colour,
            alpha=args.alpha,
            linewidth=0,
            zorder=3,
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=colour,
                label=f"{LABELS.get(group, group)} ({len(rows)})",
                markersize=6,
            )
        )

    axis.axvline(x_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    axis.axhline(y_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, y_max)
    axis.set_xlabel("Mean r² of the 10 nearest neighbors", fontsize=11)
    axis.set_ylabel("Normalized supervised importance (%)", fontsize=11)
    axis.set_title(
        "Feature importance vs. local redundancy",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    axis.legend(handles=handles, frameon=False, fontsize=9.2, loc="upper left")

    figure.text(
        0.02,
        0.012,
        "Points are individual rich features. Dashed lines mark the 75th percentile of local redundancy and supervised importance.",
        fontsize=8.7,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=230, facecolor="white")
    paper_output = args.output.with_name(args.output.stem + "_paper.png")
    figure.savefig(paper_output, dpi=230, facecolor="white")
    plt.close(figure)
    print(f"saved: {args.output}")
    print(f"saved: {paper_output}")


if __name__ == "__main__":
    main()
