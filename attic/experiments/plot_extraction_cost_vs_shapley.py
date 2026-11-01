"""Extraction cost (CPU-seconds/image) vs. Shapley contribution (MLP,
5-fold, `outputs/full_factorial_ablation_text/shapley_summary.csv`).

Costs are grounded in real logs, normalized to a single-worker-equivalent
CPU-seconds/image so multiprocess worker counts do not distort the
comparison (workers x wall_clock_seconds / n_images for wall-clock logs;
SLIC/Felzenszwalb already log a per-image `elapsed_seconds_all_graphs`
column, used directly). Two points are flagged specially, not colored
differently (an 8-point scatter cannot carry 8 reliably distinct hues, so
identity comes from the direct labels, not color):
  - Text: cost is the one-time ArtEmis/CATR captioning GPU run
    (`logs/artemis_train_inference.log` + `logs/artemis_inference.log`),
    GPU-seconds, not CPU-seconds -- not directly comparable to the rest.
  - Multiscale: no extraction log exists anywhere; shown as an estimate
    (4 x the SLIC+Descriptors+Hilbert per-image cost, since it is that same
    extraction re-run on 4 quadrant crops), drawn hollow, not a measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = ROOT / "results/extraction_cost_vs_shapley.csv"
OUTPUT_PNG = ROOT / "results/extraction_cost_vs_shapley.png"

BLUE = "#2a78d6"
GRID = "#d8d7d0"
TEXT_SECONDARY = "#52514e"

# cost_s_per_image derived in-session from real logs (see module docstring
# and CLAUDE.md); shapley_mlp_pp from outputs/full_factorial_ablation_text/shapley_summary.csv
ROWS = [
    {"modality": "Line graph", "cost_s_per_image": 0.1107, "shapley_mlp_pp": 7.037, "kind": "measured"},
    {"modality": "Fourier", "cost_s_per_image": 0.1526, "shapley_mlp_pp": 3.722, "kind": "measured"},
    {"modality": "Hilbert", "cost_s_per_image": 0.1559, "shapley_mlp_pp": 5.060, "kind": "measured"},
    {"modality": "Descriptors", "cost_s_per_image": 0.2812, "shapley_mlp_pp": 9.564, "kind": "measured"},
    {"modality": "SLIC", "cost_s_per_image": 1.3985, "shapley_mlp_pp": 8.117, "kind": "measured"},
    {"modality": "Felzenszwalb", "cost_s_per_image": 3.3601, "shapley_mlp_pp": 8.060, "kind": "measured"},
    {"modality": "Text", "cost_s_per_image": 0.1001, "shapley_mlp_pp": 6.207, "kind": "gpu"},
    {"modality": "Multiscale", "cost_s_per_image": 7.3423, "shapley_mlp_pp": 1.411, "kind": "estimated"},
]


def main() -> None:
    data = pd.DataFrame(ROWS)
    data.to_csv(OUTPUT_CSV, index=False)

    figure, axis = plt.subplots(figsize=(10.5, 7.5))
    for _, row in data.iterrows():
        if row["kind"] == "estimated":
            axis.scatter(
                row["cost_s_per_image"], row["shapley_mlp_pp"], s=140,
                facecolors="none", edgecolors=BLUE, linewidths=2, linestyle="--", zorder=3,
            )
        elif row["kind"] == "gpu":
            axis.scatter(
                row["cost_s_per_image"], row["shapley_mlp_pp"], s=140, marker="D",
                facecolors=BLUE, edgecolors="white", linewidths=1, zorder=3,
            )
        else:
            axis.scatter(
                row["cost_s_per_image"], row["shapley_mlp_pp"], s=140,
                facecolors=BLUE, edgecolors="white", linewidths=1, zorder=3,
            )
        offset = (8, 6)
        align = "left"
        if row["modality"] == "Fourier":
            offset = (8, -14)
        elif row["modality"] == "Multiscale":
            offset, align = (-10, 12), "right"
        axis.annotate(
            row["modality"], (row["cost_s_per_image"], row["shapley_mlp_pp"]),
            xytext=offset, textcoords="offset points", fontsize=10, color=TEXT_SECONDARY,
            ha=align,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Custo de extração (s/imagem, CPU-equivalente, escala log)")
    axis.set_ylabel("Contribuição Shapley (MLP, p.p. de BAcc)")
    axis.set_title("WikiArt — custo de extração vs. contribuição de cada representação")
    axis.grid(True, which="both", color=GRID, alpha=0.5, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor="white", markersize=10, label="Medido (log real)"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE,
                   markeredgecolor="white", markersize=9, label="Medido, GPU (captioning) — unidade diferente"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=BLUE, markersize=10, linestyle="--", label="Estimado (não medido)"),
    ]
    legend = axis.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
