"""Simple scatter plots for importance against local redundancy metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
}
LABELS = {
    "Region graphs": "Grafos de regiao",
    "Image statistics": "Estatisticas de imagem",
    "Space-filling curve": "Curva de preenchimento",
    "Spectral": "Espectral",
}
METRICS = [
    ("p95_r2", "p95 dos r²"),
    ("top10_mean_r2", "media dos 10 maiores r²"),
    ("neighbors_ge_0_8", "vizinhos com r² >= 0,8"),
]
TEXT = "#202124"
MUTED = "#65676b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance", type=Path, default=ROOT / "results/rich_feature_importance_vs_redundancy_pt.csv")
    parser.add_argument("--redundancy", type=Path, default=ROOT / "results/rich_feature_redundancy_r2/neighborhood_metrics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_importance_redundancy_metric_scatter_pt.png")
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--size", type=float, default=13.0)
    args = parser.parse_args()

    importance = pd.read_csv(args.importance)
    redundancy = pd.read_csv(args.redundancy)
    columns = ["feature"] + [name for name, _label in METRICS]
    data = importance.merge(redundancy[columns], on="feature", how="inner", validate="one_to_one")
    data.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.8f")

    y_max = float(data["importance_pp"].quantile(0.997) * 1.15)
    y_max = max(y_max, float(data["importance_pp"].max() * 1.08))
    y_cut = float(data["importance_pp"].quantile(0.75))

    figure, axes = plt.subplots(1, len(METRICS), figsize=(15.0, 5.4), sharey=True)
    handles = []
    for axis, (metric, label) in zip(axes, METRICS):
        axis.set_facecolor("#fbfbfa")
        axis.grid(color="#dedede", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

        for group, rows in data.groupby("group", sort=False):
            colour = COLOURS.get(group, "#777777")
            axis.scatter(
                rows[metric],
                rows["importance_pp"],
                s=args.size,
                color=colour,
                alpha=args.alpha,
                linewidth=0,
                zorder=3,
            )
            if metric == METRICS[0][0]:
                handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=colour, label=f"{LABELS.get(group, group)} ({len(rows)})", markersize=6))

        x_cut = float(data[metric].quantile(0.75))
        axis.axvline(x_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
        axis.axhline(y_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
        axis.set_xlabel(label, fontsize=10.5)
        axis.set_ylim(0, y_max)
        x_min = min(0.0, float(data[metric].min()))
        x_max = float(data[metric].quantile(0.995) * 1.10)
        if metric != "neighbors_ge_0_8":
            x_max = min(1.0, max(x_max, float(data[metric].max() * 1.04)))
        axis.set_xlim(x_min, x_max)
        axis.set_title(label, fontsize=11, fontweight="bold", color=TEXT)

    axes[0].set_ylabel("Importancia supervisionada normalizada (%)", fontsize=11)
    axes[0].legend(handles=handles, frameon=False, fontsize=8.8, loc="upper left")
    figure.suptitle(
        "Importancia por feature vs. redundancia local",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.015,
        0.012,
        "Pontos sao features ricas individuais. Linhas tracejadas: corte de redundancia local e percentil 75 de importancia.",
        fontsize=8.8,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=230, facecolor="white")
    figure.savefig(args.output.with_name(args.output.stem + "_paper.png"), dpi=230, facecolor="white")
    plt.close(figure)
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_name(args.output.stem + '_paper.png')}")


if __name__ == "__main__":
    main()
