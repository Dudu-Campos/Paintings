"""Curvas cumulativas por LOO: o que a capacidade do classificador faz com views ruins.

Dois painéis, um por critério de ordenação. Em cada um, os conjuntos são
IDÊNTICOS nos dois braços a cada k -- então a distância vertical entre as curvas
é puramente o classificador, sem confundir com escolha de views.

O que a figura mostra: o LDA sobe monotonicamente até k=10, incluindo as duas
views que a eliminação para trás descarta primeiro. O MLP satura em k=5 e depois
**desce**. Capacidade baixa transforma qualquer coisa em ganho; capacidade alta
transforma o mesmo material em custo. O ponto de saturação vai marcado porque é
onde a leitura muda de sinal.

Cores e ordem seguem `plot_standalone_by_classifier`: rampa sequencial validada
para daltonismo (protanopia 12,1 · deuteranopia 14,6 · tritanopia 12,8 de ΔE
OKLab) e monotônica em luminância, legível em preto e branco sem hachura.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import display  # noqa: E402
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
STYLE = {"LDA": ("#7c3fa8", "s", "--"), "MLP": ("#0b7a5a", "o", "-")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/cumulative_by_loo.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cumulative_by_loo.png")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    orderings = list(data.ordering.unique())
    figure, axes = plt.subplots(1, len(orderings), figsize=(13.8, 6.4), sharey=True)

    for axis, ordering in zip(np.atleast_1d(axes), orderings):
        cut = data[data.ordering.eq(ordering)]
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=14)

        for classifier, (colour, marker, dash) in STYLE.items():
            group = cut[cut.classifier.eq(classifier)]
            mean = group.groupby("k").bacc.mean()
            error = group.groupby("k").bacc.std()
            axis.plot(mean.index, mean.values, color=colour, marker=marker,
                      markersize=9, linewidth=2.8, linestyle=dash, zorder=4,
                      markeredgecolor=SURFACE, markeredgewidth=1.4, label=classifier)
            axis.fill_between(mean.index, mean - error, mean + error,
                              color=colour, alpha=0.15, linewidth=0, zorder=2)
            # O pico só é informativo onde a curva depois DESCE; marcá-lo no LDA,
            # que é monotônico, sugeriria um máximo que não existe.
            peak = int(mean.idxmax())
            if peak < mean.index.max():
                axis.plot([peak], [mean[peak]], marker="o", markersize=17,
                          markerfacecolor="none", markeredgecolor=colour,
                          markeredgewidth=2.4, zorder=6)
                axis.annotate(f"peak at k={peak}\n"
                              f"{mean[mean.index.max()] - mean[peak]:+.2f}pp by k={mean.index.max()}",
                              (peak, mean[peak]), xytext=(6, -46),
                              textcoords="offset points", fontsize=12.5,
                              color=colour, fontweight="bold")

        names = cut.groupby("k").added.first()
        axis.set_xticks(names.index)
        axis.set_xticklabels([f"{k}\n{display(names[k])}" for k in names.index],
                             fontsize=10.5, color=MUTED, rotation=52, ha="right")
        axis.set_title(ordering.replace("by ", "views added in order of "),
                       fontsize=15.5, fontweight="bold", color=INK, loc="left", pad=10)
        axis.set_xlabel("views in the model", fontsize=15, color=MUTED, labelpad=6)

    axes[0].set_ylabel("balanced accuracy (%)", fontsize=15.5, color=MUTED)
    legend = axes[0].legend(loc="lower right", fontsize=14, frameon=True,
                            facecolor="white", edgecolor=GRID)
    legend.get_frame().set_linewidth(0.8)
    figure.suptitle("Low capacity turns any view into gain; high capacity turns it into cost",
                    fontsize=17, fontweight="bold", color=INK, x=0.006, ha="left", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")

    for ordering in orderings:
        cut = data[data.ordering.eq(ordering)]
        table = cut.pivot_table(index="k", columns="classifier", values="bacc")
        for classifier in STYLE:
            series = table[classifier]
            print(f"  {ordering:<12} {classifier:<4} pico k={int(series.idxmax()):>2} "
                  f"({series.max():.2f}), k=10 {series.iloc[-1]:.2f}, "
                  f"queda {series.iloc[-1] - series.max():+.2f}pp")


if __name__ == "__main__":
    main()
