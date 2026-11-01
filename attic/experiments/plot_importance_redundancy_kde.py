"""2D KDE of feature redundancy vs supervised importance by family."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
    "SLIC multigraph": "#eb6834",
    "Descriptors": "#2a78d6",
    "Hilbert": "#1baf7a",
}
LABELS = {
    "Region graphs": "Grafos de regiao",
    "Image statistics": "Estatisticas de imagem",
    "Space-filling curve": "Curva de preenchimento",
    "Spectral": "Espectral",
    "SLIC multigraph": "Grafo SLIC",
    "Descriptors": "Descritores",
    "Hilbert": "Hilbert",
}
TEXT = "#202124"
MUTED = "#65676b"


def kde_grid(rows: pd.DataFrame, x_grid: np.ndarray, y_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    points = rows[["max_r2", "importance_pp"]].to_numpy(dtype=float).T
    kde = gaussian_kde(points, bw_method=bandwidth)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    if density.max() > 0:
        density = density / density.max()
    return density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/rich_feature_importance_vs_redundancy_pt.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_importance_redundancy_kde_pt.png")
    parser.add_argument("--bandwidth", type=float, default=0.22)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--points", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    needed = {"group", "max_r2", "importance_pp"}
    if not needed <= set(data.columns):
        raise ValueError(f"Input must contain {sorted(needed)}")
    data = data[np.isfinite(data["max_r2"]) & np.isfinite(data["importance_pp"])].copy()
    y_max = float(data["importance_pp"].quantile(0.995) * 1.18)
    y_max = max(y_max, float(data["importance_pp"].max() * 1.08))
    x_grid = np.linspace(0.0, 1.0, 260)
    y_grid = np.linspace(0.0, y_max, 240)

    figure, axis = plt.subplots(figsize=(9.8, 7.0))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    handles = []
    for group, rows in data.groupby("group", sort=False):
        if len(rows) < 5:
            continue
        colour = COLOURS.get(group, "#777777")
        density = kde_grid(rows, x_grid, y_grid, args.bandwidth)
        contour_levels = np.linspace(0.18, 1.0, args.levels)
        axis.contourf(
            x_grid,
            y_grid,
            density,
            levels=np.r_[0.0, contour_levels],
            colors=[colour],
            alpha=0.055,
            antialiased=True,
        )
        axis.contour(
            x_grid,
            y_grid,
            density,
            levels=contour_levels,
            colors=[colour],
            linewidths=1.15,
            alpha=0.88,
        )
        if args.points:
            axis.scatter(
                rows["max_r2"],
                rows["importance_pp"],
                s=8,
                color=colour,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        handles.append(plt.Line2D([0], [0], color=colour, lw=2.2, label=f"{LABELS.get(group, group)} ({len(rows)})"))

    important_cut = float(data["importance_pp"].quantile(0.75))
    axis.axvline(0.80, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.1)
    axis.axhline(important_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.1)
    axis.text(0.785, y_max * 0.965, "menos redundante", ha="right", va="top", fontsize=9, color=MUTED)
    axis.text(0.815, y_max * 0.965, "mais redundante", ha="left", va="top", fontsize=9, color=MUTED)
    axis.text(0.025, important_cut + y_max * 0.025, "maior importancia", ha="left", va="bottom", fontsize=9, color=MUTED)

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, y_max)
    axis.set_xlabel("Redundancia da feature (maior r² com qualquer outra feature)", fontsize=11)
    axis.set_ylabel("Importancia supervisionada normalizada (%)", fontsize=11)
    axis.set_title(
        "Densidade das features no plano importancia-redundancia\n"
        "KDE 2D por familia; todas as features ricas",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )
    axis.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    figure.text(
        0.02,
        0.012,
        "Contornos mostram onde cada familia concentra features. "
        "A linha vertical marca r² = 0,80; a horizontal marca o percentil 75 de importancia.",
        fontsize=8.7,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=230, facecolor="white")
    figure.savefig(args.output.with_name(args.output.stem + "_paper.png"), dpi=230, facecolor="white")
    plt.close(figure)
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_name(args.output.stem + '_paper.png')}")


if __name__ == "__main__":
    main()
