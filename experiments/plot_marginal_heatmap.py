"""Marginal de cada view por TAMANHO da coalizao: onde cada uma deixa de pagar.

As barras de `view_pareto.png` respondem "quais views ajudam" com um numero por
view. Este mapa abre a mesma quantidade no eixo que importa: quantas OUTRAS
views ja estao presentes. Cada celula e a media do marginal sobre todas as
coalizoes daquele tamanho (`sum(C(11,s))` = 2.048 no total, por view).

Duas coisas so aparecem aqui:

1. **Toda view satura**, e o mapa mostra a taxa. O marginal cai de ~20pp
   sozinha para <0,3pp com onze vizinhas. Isso e a curva de saturacao do §5.1
   vista por view em vez de agregada.
2. **Tres views cruzam o zero**, e em pontos diferentes: `HVG` ja em s=1,
   `Random walk` em s~4, `Fourier` em s~6, e o `Similarity graph` so em s=11.
   A coluna da direita e, por construcao, o LOO -- entao o mapa contem o LOO e o
   Shapley (media das colunas, ponderada) na mesma imagem, e mostra POR QUE eles
   discordam.

**Escala de cor simetrica em log** (`SymLogNorm`): a faixa vai de +30 a -1,6, e
numa escala linear tudo de s>=4 colapsaria no branco -- que e exatamente a
regiao onde a decisao de incluir ou nao a view e tomada. Divergente com meio
neutro, nunca arco-iris; vermelho positivo e azul negativo sobrevivem aos tres
dicromatismos.
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
from matplotlib.colors import SymLogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_shapley_twelve_views import VIEWS  # noqa: E402
from view_names import display  # noqa: E402

SURFACE, INK, MUTED = "#fbfbfa", "#0b0b0b", "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/view_pareto.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/view_marginal_heatmap.png")
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    value = dict(zip(table["mask"], table.bacc))
    rows = []
    for i, view in enumerate(VIEWS):
        bit = 1 << i
        for mask in value:
            if mask & bit:
                continue
            rows.append((view, bin(mask).count("1"), value[mask | bit] - value[mask]))
    long = pd.DataFrame(rows, columns=["view", "size", "delta"])
    grid = long.pivot_table(index="view", columns="size", values="delta")
    order = long.groupby("view").delta.mean().sort_values(ascending=False).index
    grid = grid.loc[order]

    figure, axis = plt.subplots(figsize=(8.4, 4.9), facecolor=SURFACE)
    norm = SymLogNorm(linthresh=0.2, vmin=-32, vmax=32, base=10)
    image = axis.imshow(grid.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            cell = grid.iloc[y, x]
            if np.isnan(cell):
                continue
            # Tinta clara so onde o fundo e escuro; nas celulas quase neutras a
            # tinta escura e a legivel.
            light = abs(cell) > 6
            axis.text(x, y, f"{cell:.2f}" if abs(cell) < 10 else f"{cell:.1f}",
                      ha="center", va="center", fontsize=7.2,
                      color="white" if light else "#2a2a2a")

    axis.set_xticks(range(grid.shape[1]))
    axis.set_xticklabels(grid.columns, fontsize=9, color=MUTED)
    axis.set_yticks(range(grid.shape[0]))
    axis.set_yticklabels([display(v) for v in grid.index], fontsize=9.5, color=MUTED)
    axis.set_xlabel("number of other views already in the model", fontsize=10,
                    color=INK, labelpad=6)
    axis.set_title("Every view saturates; three of them turn negative",
                   fontsize=12, color=INK, pad=10, loc="left", fontweight="bold")

    bar = figure.colorbar(image, ax=axis, pad=0.02, aspect=28)
    bar.set_label("marginal gain when the view is added (pp)", fontsize=9,
                  color=INK)
    bar.set_ticks([-30, -3, -0.3, 0, 0.3, 3, 30])
    bar.ax.set_yticklabels(["−30", "−3", "−0.3", "0", "0.3", "3", "30"], fontsize=8)
    axis.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.6)
    axis.tick_params(which="minor", length=0)

    figure.tight_layout()
    figure.savefig(args.output, dpi=200, facecolor=SURFACE)
    print(grid.round(2).to_string())
    print(f"figura: {args.output}")


if __name__ == "__main__":
    main()
