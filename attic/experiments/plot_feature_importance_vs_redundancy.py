"""Plot old compact feature importance against pairwise redundancy.

Uses the already-computed compact visual-feature artifacts:

* ``results/visual_feature_importance_lda_structure_summary.csv``
* ``results/visual_feature_redundancy_r2_feature_summary.csv``

The point of the plot is to separate useful/unique features from useful-but-
redundant features and low-value duplicated columns.
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
    "SLIC multigraph": "#eb6834",
    "Descriptors": "#2a78d6",
    "Hilbert": "#1baf7a",
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
}
LABELS = {
    "SLIC multigraph": "Grafo SLIC",
    "Descriptors": "Descritores",
    "Hilbert": "Hilbert",
    "Region graphs": "Grafos de regiao",
    "Image statistics": "Estatisticas de imagem",
    "Space-filling curve": "Curva de preenchimento",
    "Spectral": "Espectral",
}
TEXT = "#202124"
MUTED = "#65676b"


def classify(row: pd.Series, unique_cut: float, important_cut: float) -> str:
    unique = row["max_r2"] < unique_cut
    important = row["importance_normalized_mean"] >= important_cut
    if important and unique:
        return "importante e pouco redundante"
    if important:
        return "importante, mas redundante"
    if unique:
        return "pouco redundante, baixo sinal"
    return "baixo sinal e redundante"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance", type=Path, default=ROOT / "results/visual_feature_importance_lda_structure_summary.csv")
    parser.add_argument("--redundancy", type=Path, default=ROOT / "results/visual_feature_redundancy_r2_feature_summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/feature_importance_vs_redundancy_pt.png")
    parser.add_argument("--top-labels", type=int, default=14)
    parser.add_argument("--unique-cut", type=float, default=0.80)
    parser.add_argument("--title-note", default="features visuais; importancia LDA corrigida por estrutura")
    args = parser.parse_args()

    importance = pd.read_csv(args.importance)
    redundancy = pd.read_csv(args.redundancy)
    merged = importance.merge(
        redundancy[[column for column in ["feature", "max_r2", "best_partner", "best_partner_display"] if column in redundancy.columns]],
        on="feature",
        how="inner",
        validate="one_to_one",
    )
    dropped = sorted(set(importance["feature"]) - set(merged["feature"]))
    if dropped:
        print(f"dropping {len(dropped)} features without r2 summary (usually constant columns): {', '.join(dropped)}")
    merged["importance_pp"] = 100 * merged["importance_normalized_mean"]
    important_cut = float(merged["importance_normalized_mean"].quantile(0.75))
    merged["quadrant"] = merged.apply(classify, axis=1, unique_cut=args.unique_cut, important_cut=important_cut)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values("importance_normalized_mean", ascending=False).to_csv(
        args.output.with_suffix(".csv"), index=False, float_format="%.8f"
    )

    figure, axis = plt.subplots(figsize=(9.8, 7.2))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    for group, rows in merged.groupby("group", sort=False):
        axis.scatter(
            rows["max_r2"],
            rows["importance_pp"],
            s=34 if len(merged) > 800 else 46,
            color=COLOURS.get(group, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.76,
            label=f"{LABELS.get(group, group)} ({len(rows)})",
            zorder=3,
        )

    axis.axvline(args.unique_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.2)
    axis.axhline(100 * important_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.2)
    y_max = float(merged["importance_pp"].quantile(0.995) * 1.16)
    y_max = max(y_max, float(merged["importance_pp"].max() * 1.08))
    y_pad = y_max * 0.025
    axis.set_ylim(0, y_max)
    axis.text(args.unique_cut - 0.015, y_max * 0.94, "menos redundante",
              ha="right", va="top", fontsize=9, color=MUTED)
    axis.text(args.unique_cut + 0.015, y_max * 0.94, "mais redundante",
              ha="left", va="top", fontsize=9, color=MUTED)
    axis.text(0.025, 100 * important_cut + y_pad, "maior importancia",
              ha="left", va="bottom", fontsize=9, color=MUTED)

    label_rows = pd.concat([
        merged.nlargest(args.top_labels, "importance_normalized_mean"),
        merged[merged["max_r2"].lt(args.unique_cut)].nlargest(max(4, args.top_labels // 2), "importance_normalized_mean"),
    ]).drop_duplicates("feature").head(args.top_labels + 6)
    for _, row in label_rows.iterrows():
        label = str(row["display_name"])
        if len(label) > 24:
            label = label[:21] + "..."
        axis.annotate(
            label,
            (row["max_r2"], row["importance_pp"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=TEXT,
            alpha=0.86,
        )

    axis.set_xlim(-0.025, 1.025)
    axis.set_xlabel("Redundancia da feature (maior r² com qualquer outra feature)", fontsize=11)
    axis.set_ylabel("Importancia supervisionada normalizada (%)", fontsize=11)
    axis.set_title(
        "Importancia por feature vs. redundancia linear\n"
        + args.title_note,
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    figure.text(
        0.02,
        0.012,
        "Linhas tracejadas: r² = 0,80 e percentil 75 de importancia. "
        "Pontos no alto-esquerda sao sinais fortes e pouco duplicados; "
        "alto-direita sao sinais fortes, mas compartilhados por proxies.",
        fontsize=8.4,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(args.output, dpi=220, facecolor="white")
    figure.savefig(args.output.with_name(args.output.stem + "_paper.png"), dpi=220, facecolor="white")
    plt.close(figure)

    summary = (
        merged.groupby(["group", "quadrant"], as_index=False)
        .agg(features=("feature", "count"), importance_pp_sum=("importance_pp", "sum"))
        .sort_values(["group", "quadrant"])
    )
    summary.to_csv(args.output.with_name(args.output.stem + "_quadrants.csv"), index=False, float_format="%.6f")
    print(f"merged features: {len(merged)}")
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_name(args.output.stem + '_paper.png')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
