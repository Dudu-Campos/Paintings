"""Curva de desempenho contra número de features, por POOL de origem.

Substitui a curva de PCA do §5.3 no papel de "o que acontece no eixo de
dimensionalidade", e é mais limpa por uma razão específica: na PCA o eixo x são
k **misturas lineares** de todas as 639 features, então acrescentar uma
representação muda as próprias componentes e o eixo deixa de ser comparável
entre condições. Aqui as features de um mesmo k são **as mesmas quantidades
medidas**, e só o pool de onde saem muda.

Duas leituras que a figura tem de suportar ao mesmo tempo:

1. **Nível.** No mesmo k e com o mesmo critério, sair de oito famílias bate sair
   de uma em +2,9 a +4,0pp, estável ao longo do eixo.
2. **Teto.** A curva de Descriptors não pode passar de 130 features -- ela
   termina, não satura por saturação. A linha vertical marca isso, porque sem
   ela o salto para +7,9pp em k=128 se lê como divergência das curvas quando é
   apenas o pool acabando.

O expoente do erro contra k é praticamente o mesmo nos dois pools (0,121 e
0,109, medidos onde ambos têm folga), o que **preserva o controle negativo do
§5.3**: acrescentar representações desloca o nível, não muda a forma da curva
no eixo de dimensionalidade. É no eixo de DADOS que o expoente se move (§5.1).
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
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
ARMS = [
    ("bank", "max-r2-strat", "eight families, redundancy-first", "#2a78d6", "o", "-"),
    ("bank", "random", "eight families, random", "#89b8ee", "o", "--"),
    ("descriptors", "max-r2-strat", "one family, redundancy-first", "#eb6834", "s", "-"),
    ("descriptors", "random", "one family, random", "#f2a683", "s", "--"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "results/redundancy_selection_curve.csv")
    parser.add_argument("--descriptors", type=Path,
                        default=ROOT / "results/redundancy_selection_descriptors.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/selection_curve_pools.png")
    args = parser.parse_args()

    tables = {"bank": pd.read_csv(args.bank), "descriptors": pd.read_csv(args.descriptors)}
    shared = sorted(set(tables["descriptors"].n_features))

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)

    # O pool de uma família tem 130 features e acaba; sem marcar isso o salto do
    # último ponto se lê como divergência das curvas.
    axis.axvline(130, color="#c4c4c4", linestyle=":", linewidth=1.3, zorder=1)
    axis.text(130 * 0.94, 41.0, "one family exhausted (130 features)", rotation=90,
              fontsize=10.5, color="#9a9a9a", ha="right", va="bottom")

    for pool, order, label, colour, marker, style in ARMS:
        frame = tables[pool]
        frame = frame[frame.order.eq(order) & frame.n_features.isin(
            shared if pool == "descriptors" else sorted(set(frame.n_features)))]
        grouped = frame.groupby("n_features")["bacc"]
        mean, error = grouped.mean(), grouped.std()
        axis.plot(mean.index, mean.values, color=colour, marker=marker, markersize=7.5,
                  linewidth=2.4, linestyle=style, markeredgecolor=SURFACE,
                  markeredgewidth=1.2, zorder=4, label=label, solid_capstyle="round")
        axis.fill_between(mean.index, mean - error, mean + error, color=colour,
                          alpha=0.13, linewidth=0, zorder=2)

    axis.set_xscale("log", base=2)
    axis.set_xticks([8, 16, 32, 64, 128, 256, 512])
    axis.set_xticklabels(["8", "16", "32", "64", "128", "256", "512"])
    axis.minorticks_off()
    axis.set_xlabel("number of features given to the classifier", fontsize=14.5, color=MUTED)
    axis.set_ylabel("balanced accuracy (%)", fontsize=14.5, color=MUTED)
    axis.set_title("Where the features come from beats how many there are",
                   fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)
    legend = axis.legend(loc="lower right", fontsize=11.5, frameon=True,
                         facecolor="white", edgecolor=GRID)
    legend.get_frame().set_linewidth(0.8)

    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
