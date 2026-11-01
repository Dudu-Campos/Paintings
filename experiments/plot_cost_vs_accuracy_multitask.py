"""Acurácia do MLP contra custo de extração acumulado, MultitaskPainting100k.

Custos medidos nos logs da extração de 21-22/ago (99.816 imagens, 14 workers
de CPU nesta máquina). Atribuição por bloco:

* `SLIC` e `Felzenszwalb` = `segmentation_multigraph_features.py`, que faz a
  própria segmentação.
* `Line graph` e `L(G_felz)` = exportar o RAG (`extract_*_rag_graphs.py`) mais
  consumir o RAG (`extract_line_graph_features.py`). O RAG exportado **não**
  é compartilhado com o bloco de multigrafo correspondente -- os dois caminhos
  segmentam separadamente. É desperdício real do pipeline atual, mas é o custo
  medido, e inventar o custo "se fosse compartilhado" seria número que ninguém
  pagou.

O grafo de similaridade não é extração: ele é calculado por fold no treino,
então entra como custo de **treino**, marcado à parte na figura.

Sem linha de referência de CNN: esta figura é de ablação, e o custo de uma
backbone pré-treinada não é comparável ao nosso -- os minutos de inferência
dela escondem 1,28 milhão de imagens rotuladas que já foram pagas. Pôr as
duas no mesmo eixo compararia grandezas diferentes.
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

RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
ACCENT = "#e34948"
REFERENCE = "#52514e"
SURFACE = "#fbfbfa"
GRID = "#dedede"
INK = "#0b0b0b"
MUTED = "#52514e"

# Minutos de PAREDE com 14 workers, lidos dos timestamps dos logs. NAO sao
# CPU-minutos: em CPU-hora os valores seriam ~14x maiores (146 CPU-h no total
# contra 10,4 h de parede). O eixo diz "wall-clock" por isso.
COST_MINUTES = {
    "Felzenszwalb": 193.25,          # multigrafo felz  19:16:00 -> 22:29:15
    "Descriptors": 48.2,
    "SLIC": 81.6,                    # multigrafo slic  17:54:25 -> 19:16:00
    "L(G_felz)": 186.8 + 7.7,        # RAG felz 14:27:09->17:33:59 + L(G) 7m42
    "Line graph": 82.3 + 12.7,       # RAG slic 13:04:51->14:27:09 + L(G) 12m40
    "Hilbert": 3.2,
    "Fourier": 7.0,                  # 422s
    "Gabor": 3.6,
}
GRAPH_MINUTES = 100 * 5 / 60          # ~100s por fold, 5 folds (custo de treino)

SHORT = {"Felzenszwalb": "Felz", "Descriptors": "Desc", "SLIC": "SLIC",
         "L(G_felz)": "L(G_felz)", "Line graph": "L(G_slic)", "Hilbert": "Hilbert",
         "Fourier": "Fourier", "Gabor": "Gabor", "Similarity graph": "graph"}


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/nested_mlp_multitask.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cost_vs_accuracy_multitask.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    steps = sorted(frame["n_modalities"].unique())
    last = max(steps)
    names = {m: frame.loc[frame.n_modalities.eq(m), "modalities"].iloc[0].split("+")
             for m in steps}

    points = []
    for m in steps:
        block = frame[frame.n_modalities.eq(m)]
        visual = [n for n in names[m] if n != "Similarity graph"]
        cost = sum(COST_MINUTES[n] for n in visual)
        if "Similarity graph" in names[m]:
            cost += GRAPH_MINUTES
        points.append({"m": m, "cost": cost / 60,
                       "bacc": block["bacc"].mean() * 100,
                       "std": block["bacc"].std() * 100,
                       "added": SHORT.get(names[m][-1], names[m][-1])})
    points = pd.DataFrame(points)

    figure, axis = plt.subplots(figsize=(10.4, 6.4))
    style(axis)

    axis.plot(points["cost"], points["bacc"], color=RAMP[4], linewidth=2.2,
              zorder=3, solid_capstyle="round")
    for row in points.itertuples():
        colour = ACCENT if row.m == last else RAMP[min(row.m - 1, len(RAMP) - 1)]
        axis.errorbar(row.cost, row.bacc, yerr=row.std, color=colour,
                      marker="o", markersize=9 if row.m == last else 7.5,
                      capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5,
                      zorder=5 if row.m == last else 4)

    # Tabela em vez de rótulos no ponto: os cinco últimos passos custam quase
    # nada e se amontoam no mesmo x, então linha-guia vira emaranhado. A tabela
    # ainda ganha em informação -- traz custo e acurácia, que o rótulo não traz.
    x0, y0, step = 0.505, 0.455, 0.0455
    axis.text(x0, y0 + step * 0.95, "step", transform=axis.transAxes,
              fontsize=9.4, color=MUTED, fontweight="bold")
    axis.text(x0 + 0.075, y0 + step * 0.95, "added", transform=axis.transAxes,
              fontsize=9.4, color=MUTED, fontweight="bold")
    axis.text(x0 + 0.285, y0 + step * 0.95, "hours", transform=axis.transAxes,
              fontsize=9.4, color=MUTED, fontweight="bold", ha="right")
    axis.text(x0 + 0.395, y0 + step * 0.95, "BAcc", transform=axis.transAxes,
              fontsize=9.4, color=MUTED, fontweight="bold", ha="right")
    for index, row in enumerate(points.itertuples()):
        colour = ACCENT if row.m == last else RAMP[min(row.m - 1, len(RAMP) - 1)]
        y = y0 - index * step
        weight = "bold" if row.m == last else "normal"
        axis.plot([x0 + 0.012], [y + 0.011], marker="o", markersize=6.5,
                  color=colour, transform=axis.transAxes, clip_on=False, zorder=6)
        axis.text(x0 + 0.038, y, str(row.m), transform=axis.transAxes,
                  fontsize=9.6, color=colour, fontweight=weight)
        axis.text(x0 + 0.075, y, row.added, transform=axis.transAxes,
                  fontsize=9.6, color=colour, fontweight=weight)
        axis.text(x0 + 0.285, y, f"{row.cost:.1f}", transform=axis.transAxes,
                  fontsize=9.6, color=colour, fontweight=weight, ha="right")
        axis.text(x0 + 0.395, y, f"{row.bacc:.1f}", transform=axis.transAxes,
                  fontsize=9.6, color=colour, fontweight=weight, ha="right")

    axis.set_xlabel("cumulative extraction cost (wall-clock hours, 14 workers, "
                    "99,816 images)", fontsize=13.5, color=MUTED)
    axis.set_ylabel("balanced accuracy, production MLP (%)", fontsize=14.5, color=MUTED)
    axis.set_title("Balanced accuracy versus cumulative extraction cost\n"
                   "representations added in Shapley order",
                   fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    axis.set_xlim(points["cost"].min() - 0.55, points["cost"].max() + 0.55)

    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    print(points.to_string(index=False))


if __name__ == "__main__":
    main()
