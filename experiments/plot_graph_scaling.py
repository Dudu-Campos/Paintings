"""Escalonamento do grafo de similaridade contra o tamanho do corpus.

O grafo é a **única** peça do pipeline cujo custo não é por imagem: as demais
oito modalidades são O(n) trivialmente paralelas, e o classificador é O(E n d h).
Esta figura existe para mostrar o expoente desse termo isolado, e para deixar
visível que ele **não é assintoticamente benigno**.

Duas linhas de referência, em cinza, ancoradas no último ponto medido: a de
inclinação 1 (linear) e a de inclinação 2 (varredura por força bruta, O(n^2 d)).
Elas são o que torna a figura legível como análise de complexidade em vez de um
gráfico de tempo -- sem elas o leitor não tem contra o que comparar a curva.

O achado: a consulta corre com beta = 1,94 na ponta, praticamente encostada na
referência de 2. Em d = 639 nenhuma árvore de indexação ajuda e o sklearn cai
para varredura; o expoente cresce com n porque a varredura vai dominando o
overhead constante. Logo **1,70 é o número do nosso regime, não a assíntota** --
e reportar 1,70 sem a referência de 2 ao lado seria vender uma folga que não
existe em escala.
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

SERIES = {
    "total_seconds": ("total", "#0b0b0b", "o", 2.8),
    "fit_seconds": ("kNN graph + diffusion", "#eda100", "s", 2.2),
    "query_seconds": ("neighbour query", "#2a78d6", "D", 2.2),
    "aggregate_seconds": ("feature aggregation", "#e87ba4", "P", 2.2),
}
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"


def slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    beta, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(beta), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/graph_scaling.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/graph_scaling.png")
    parser.add_argument("--extrapolate-to", type=float, default=1e6)
    args = parser.parse_args()

    table = pd.read_csv(args.input).groupby("n")[list(SERIES)].mean()
    n = table.index.values.astype(float)

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, which="major", zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)

    # Referências ancoradas no último ponto do total: é contra elas que a
    # inclinação medida se lê, e ancorar no fim mantém a comparação na região
    # onde o overhead constante já não distorce.
    anchor_x, anchor_y = n[-1], table["total_seconds"].values[-1]
    # Só à direita da âncora: as referências existem para ler a extrapolação, e
    # estendê-las para a esquerda cruzava as quatro séries medidas sem informar.
    span = np.array([anchor_x, args.extrapolate_to])
    for exponent, label, offset in ((1.0, "slope 1  (linear)", -20),
                                    (2.0, "slope 2  (brute-force kNN)", 8)):
        axis.plot(span, anchor_y * (span / anchor_x) ** exponent,
                  color="#c4c4c4", linestyle="--", linewidth=1.2, zorder=1)
        axis.annotate(label,
                      xy=(args.extrapolate_to, anchor_y * (args.extrapolate_to / anchor_x) ** exponent),
                      xytext=(-6, offset), textcoords="offset points",
                      fontsize=10.5, color="#9a9a9a", ha="right")

    for column, (label, colour, marker, width) in SERIES.items():
        values = table[column].values
        axis.plot(n, values, color=colour, marker=marker, markersize=8,
                  linewidth=width, markeredgecolor=SURFACE, markeredgewidth=1.3,
                  zorder=4, solid_capstyle="round", label=label)
        # Extrapolação só do total, e pontilhada, porque é previsão e não medida.
        if column == "total_seconds":
            beta, intercept = slope(n[-3:], values[-3:])
            future = np.array([n[-1], args.extrapolate_to])
            axis.plot(future, np.exp(intercept) * future ** beta, color=colour,
                      linestyle=":", linewidth=2.0, zorder=3)
            predicted = np.exp(intercept) * args.extrapolate_to ** beta
            axis.plot([args.extrapolate_to], [predicted], marker="o", markersize=7,
                      markerfacecolor="white", markeredgecolor=colour,
                      markeredgewidth=2.0, zorder=5)
            axis.annotate(f"{predicted / 60:.0f} min at 1M images",
                          xy=(args.extrapolate_to, predicted),
                          xytext=(-18, 0), textcoords="offset points",
                          fontsize=11.5, color=INK, ha="right", va="center",
                          fontweight="bold")

    x0, y0, step = 0.045, 0.905, 0.052
    axis.text(x0 + 0.030, y0 + step, "stage", transform=axis.transAxes,
              fontsize=11, color=MUTED, fontweight="bold")
    axis.text(x0 + 0.400, y0 + step, "β", transform=axis.transAxes,
              fontsize=11, color=MUTED, fontweight="bold", ha="right")
    for index, (column, (label, colour, marker, _)) in enumerate(SERIES.items()):
        beta, _ = slope(n[-3:], table[column].values[-3:])
        y = y0 - index * step
        axis.plot([x0], [y + 0.013], marker=marker, markersize=8, color=colour,
                  transform=axis.transAxes, clip_on=False, zorder=6)
        axis.text(x0 + 0.030, y, label, transform=axis.transAxes,
                  fontsize=11.5, color=colour)
        axis.text(x0 + 0.400, y, f"{beta:.2f}", transform=axis.transAxes,
                  fontsize=11.5, color=colour, ha="right")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(n[0] * 0.8, args.extrapolate_to * 1.5)
    axis.set_xlabel("corpus size (images)", fontsize=14.5, color=MUTED)
    axis.set_ylabel("similarity-graph cost (seconds, one fold)",
                    fontsize=14.5, color=MUTED)
    axis.set_title("The similarity graph is the only super-linear stage",
                   fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)

    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    for column in SERIES:
        g, _ = slope(n, table[column].values)
        t, _ = slope(n[-3:], table[column].values[-3:])
        print(f"  {column:<20} beta global {g:5.2f}   últimos 3 pontos {t:5.2f}")


if __name__ == "__main__":
    main()
