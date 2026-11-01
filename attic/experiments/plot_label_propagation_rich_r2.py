"""Plot label-propagation diagnostics for the rich r^2 feature graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["Region graphs", "Image statistics", "Space-filling curve", "Spectral"]
LABELS = {
    "Region graphs": "Grafos de regiao",
    "Image statistics": "Estatisticas de imagem",
    "Space-filling curve": "Curva de preenchimento",
    "Spectral": "Espectral",
}
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/rich_feature_label_propagation_r2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_feature_label_propagation_r2/label_propagation_summary_pt.png")
    args = parser.parse_args()

    features = pd.read_csv(args.input_dir / "feature_probabilities.csv")
    confusion = pd.read_csv(args.input_dir / "confusion.csv", index_col=0).reindex(index=FAMILIES, columns=FAMILIES)
    row_norm = confusion.div(confusion.sum(axis=1), axis=0).fillna(0.0)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.05, 1.25]})

    axis = axes[0]
    image = axis.imshow(row_norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=35, ha="right", fontsize=9)
    axis.set_yticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], fontsize=9)
    axis.set_xlabel("Familia inferida pelo grafo")
    axis.set_ylabel("Familia de origem")
    axis.set_title("Propagacao de rotulos no grafo de r2", fontweight="bold")
    for row in range(len(FAMILIES)):
        for col in range(len(FAMILIES)):
            count = int(confusion.iloc[row, col])
            pct = 100 * row_norm.iloc[row, col]
            colour = "white" if pct > 55 else "#202124"
            axis.text(col, row, f"{count}\n{pct:.0f}%", ha="center", va="center", fontsize=8.5, color=colour)
    figure.colorbar(image, ax=axis, shrink=0.75, label="proporcao por linha")

    axis = axes[1]
    image_features = features[features["family"].eq("Image statistics")].copy()
    counts = image_features["predicted_family"].value_counts().reindex(FAMILIES, fill_value=0)
    bars = axis.bar(
        [LABELS[f] for f in FAMILIES],
        counts.to_numpy(),
        color=[COLOURS[f] for f in FAMILIES],
        width=0.72,
    )
    axis.set_title("Features de imagem: identidade inferida", fontweight="bold")
    axis.set_ylabel("Numero de features")
    axis.set_ylim(0, max(counts.max() * 1.18, 1))
    axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelrotation=25)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    total = counts.sum()
    for bar, value in zip(bars, counts.to_numpy()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.max() * 0.025, 1),
            f"{int(value)}\n{100 * value / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.text(
        0.02,
        0.96,
        "rotulo da propria feature oculto durante a predicao",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.suptitle("Identidade representacional estimada a partir da redundancia", fontsize=13.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=230, facecolor="white")
    figure.savefig(args.output.with_name(args.output.stem + "_paper.png"), dpi=230, facecolor="white")
    plt.close(figure)
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_name(args.output.stem + '_paper.png')}")


if __name__ == "__main__":
    main()
