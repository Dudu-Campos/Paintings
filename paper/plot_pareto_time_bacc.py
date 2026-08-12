"""No-pretraining Pareto plot: wall time versus BAcc.

The three points deliberately exclude every pre-trained representation.  All
times cover one five-fold evaluation on the current server.  The visual GIST
preparation is estimated from observed SLIC worker times and descriptor/Hilbert
timings on the same data; the other two totals were measured in their runs.
This is an exploratory planning figure, not a hardware benchmark.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "paper/pareto_time_bacc_estimates.csv"
OUTPUT_PNG = ROOT / "paper/pareto_time_bacc.png"


def is_pareto(frame: pd.DataFrame) -> pd.Series:
    """True if no other method is both faster and at least as accurate."""
    values = []
    for row in frame.itertuples(index=False):
        dominates = (
            (frame.wall_minutes <= row.wall_minutes)
            & (frame.bacc >= row.bacc)
            & ((frame.wall_minutes < row.wall_minutes) | (frame.bacc > row.bacc))
        )
        values.append(not dominates.any())
    return pd.Series(values, index=frame.index)


def main() -> None:
    # SLIC: 111,742 cumulative worker-s / 32 workers = 58.2 min.  The
    # descriptor and Hilbert observations were 5.1 and 2.1 min with 16
    # workers, scaled conservatively to the 32-worker original extraction.
    gist_preparation_minutes = 111_741.697 / 32 / 60 + 5.1 / 2 + 2.1 / 2
    rows = [
        {
            "method": "Haralick + HOG + MLP", "family": "Baseline clássico", "bacc": 0.226822,
            "wall_minutes": 12.2 + 229.457 / 60, "time_type": "measured",
            "notes": "12.2 min feature extraction plus measured MLP cross-validation.",
        },
        {
            "method": "Nosso método visual", "family": "Visual interpretável", "bacc": 0.452248,
            "wall_minutes": gist_preparation_minutes + 126.507 / 60, "time_type": "estimated",
            "notes": "SLIC + descriptors + Hilbert preparation, then MLP CV.",
        },
        {
            "method": "ResNet-50 do zero", "family": "CNN do zero", "bacc": 0.423435,
            "wall_minutes": 40_882.6 / 60, "time_type": "measured",
            "notes": "Five-fold end-to-end training from random initialization; data augmentation.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["pareto"] = is_pareto(frame)
    frame.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

    colors = {
        "Baseline clássico": "#6f7c86", "Visual interpretável": "#2878c8",
        "CNN do zero": "#e8782b",
    }
    label_offsets = {
        "Haralick + HOG + MLP": (10, 8),
        "Nosso método visual": (10, 8),
        "ResNet-50 do zero": (-10, 8),
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    for row in frame.itertuples(index=False):
        marker = "o" if row.time_type == "measured" else "D"
        axis.scatter(
            row.wall_minutes, row.bacc, s=90, marker=marker,
            color=colors[row.family], edgecolor="#1b1b1b" if row.pareto else "white",
            linewidth=1.4, zorder=3,
        )
        offset = label_offsets[row.method]
        axis.annotate(row.method, (row.wall_minutes, row.bacc), xytext=offset,
                      textcoords="offset points", fontsize=8.5,
                      ha="right" if offset[0] < 0 else "left")

    frontier = frame.loc[frame.pareto].sort_values("wall_minutes")
    axis.plot(frontier.wall_minutes, frontier.bacc, color="#333333", linewidth=1.3,
              linestyle="--", zorder=2, label="Fronteira de Pareto")
    for family, color in colors.items():
        axis.scatter([], [], color=color, s=55, label=family)
    axis.scatter([], [], color="#666666", marker="o", s=55, label="Tempo medido")
    axis.scatter([], [], color="#666666", marker="D", s=55, label="Tempo estimado")
    axis.set_xscale("log")
    axis.set(
        xlabel="Tempo de execução estimado (minutos; escala log)",
        ylabel="Balanced accuracy (BAcc)",
        title="Sem pré-treinamento: tempo × classificação de estilos",
        ylim=(0.18, 0.48),
    )
    axis.grid(which="both", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=True, fontsize=8)
    figure.text(
        0.01, 0.01,
        "Cinco folds; nenhum encoder pré-treinado. Haralick+HOG e ResNet: tempos medidos.\n"
        "Nosso método: SLIC estimado pelos tempos de worker, mais descritores, Hilbert e MLP.",
        fontsize=8, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    figure.savefig(OUTPUT_PNG, dpi=190)
    print(frame[["method", "bacc", "wall_minutes", "time_type", "pareto"]].to_string(index=False))
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
