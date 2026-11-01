"""Distributions of importance after penalizing feature redundancy.

Takes the merged scatter CSV produced by
``plot_feature_importance_vs_redundancy.py`` and collapses importance and
redundancy into one-dimensional scores:

* raw importance: I
* soft non-redundant importance: I * sqrt(1 - max_r2)
* strict non-redundant importance: I * (1 - max_r2)
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
METRICS = [
    ("importance_pp", "Importancia bruta", "I"),
    ("soft_unique_pp", "Importancia parcialmente unica", "I * sqrt(1 - r²max)"),
    ("strict_unique_pp", "Importancia estritamente unica", "I * (1 - r²max)"),
]
TEXT = "#202124"
MUTED = "#65676b"


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values.astype(float))
    y = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/rich_feature_importance_vs_redundancy_pt.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_nonredundant_importance_distributions_pt.png")
    parser.add_argument("--log-x", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if not {"importance_pp", "max_r2", "group"} <= set(data.columns):
        raise ValueError("Input must contain importance_pp, max_r2 and group")
    uniqueness = np.clip(1.0 - data["max_r2"].to_numpy(dtype=float), 0.0, 1.0)
    data["soft_unique_pp"] = data["importance_pp"] * np.sqrt(uniqueness)
    data["strict_unique_pp"] = data["importance_pp"] * uniqueness
    data.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.8f")

    summary_rows = []
    for group, rows in data.groupby("group"):
        record = {"group": group, "features": len(rows)}
        for column, _title, _formula in METRICS:
            record[f"{column}_sum"] = float(rows[column].sum())
            record[f"{column}_mean"] = float(rows[column].mean())
            record[f"{column}_median"] = float(rows[column].median())
            record[f"{column}_p90"] = float(rows[column].quantile(0.90))
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows).sort_values("soft_unique_pp_sum", ascending=False)
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False, float_format="%.8f")

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    groups = list(data["group"].drop_duplicates())
    for axis, (column, title, formula) in zip(axes, METRICS):
        axis.set_facecolor("#fbfbfa")
        axis.grid(color="#dedede", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for group in groups:
            values = data.loc[data["group"].eq(group), column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if args.log_x:
                values = values[values > 0]
            x, y = ecdf(values)
            axis.plot(
                x,
                y,
                color=COLOURS.get(group, "#777777"),
                linewidth=2.2,
                label=f"{LABELS.get(group, group)} ({len(values)})",
            )
        axis.set_title(title, fontsize=11.5, fontweight="bold", color=TEXT)
        axis.set_xlabel(formula, fontsize=10.5)
        if args.log_x:
            axis.set_xscale("log")
        axis.tick_params(labelsize=9)
    axes[0].set_ylabel("Frequencia acumulada", fontsize=10.5)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.suptitle(
        "Distribuicao de importancia corrigida por redundancia",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.015,
        0.012,
        "Cada curva e a distribuicao acumulada das features de uma familia. "
        "Quanto mais deslocada para a direita, maior o score por feature; "
        "a penalizacao troca importancia bruta por importancia nao redundante.",
        fontsize=8.8,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, facecolor="white")
    figure.savefig(args.output.with_name(args.output.stem + "_paper.png"), dpi=220, facecolor="white")
    plt.close(figure)

    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_name(args.output.stem + '_paper.png')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
