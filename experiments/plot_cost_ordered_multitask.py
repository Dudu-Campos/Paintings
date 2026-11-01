"""Acurácia contra custo de extração, blocos adicionados do MAIS BARATO ao mais caro.

Substitui `plot_cost_vs_accuracy_multitask.py`, que adicionava em ordem de
Shapley -- o que fazia a curva COMEÇAR pelo bloco mais caro (Felzenszwalb) e
deixava vazio todo o eixo abaixo dele. O leitor não conseguia responder "e se eu
tiver 1 CPU-hora?".

Eixos lineares: a curva sobe quase vertical até 1 h e achata por cinco, então a
forma **mostra** a afirmação ("modelo bom com custo baixo") em vez de anotá-la.
Log-log e mono-log foram testados e descartados (§9.7).

Custo corrigido para RAG compartilhado: o `COST_MINUTES` antigo somava, em
`L(G_felz)` e `Line graph`, uma re-segmentação que o pipeline atual não paga
(`--export-rag-dir` no passo de multigrafo). Total 5,95 h, não 10,44.

Dois pontos ficam fora do eixo de custo, e a tabela marca com †:
* grafo (`logit-add`, §5.6): custo de TREINO, e depende dos outros oito blocos.
* texto: 35 min de GPU, moeda diferente do eixo.
São **ramos** a partir do hand-8, não cadeia -- `logit-add + texto` não foi medido.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import display  # noqa: E402
RAMP = ["#9dc4f2", "#7fb0ed", "#619ce8", "#4389e3",
        "#2f78d4", "#2769bd", "#1f5ba6", "#184f95"]
GRAPH_C, TEXT_C = "#e08a2e", "#7a4fbf"
SURFACE, GRID, MUTED, INK = "#fbfbfa", "#e2e2e2", "#6b6a67", "#1b1b1a"

COST_FIXED = {"Hilbert": 3.2, "Gabor": 3.6, "Fourier": 7.0, "Desc": 48.2,
              "SLIC": 81.6, "L(G_slic)": 12.7, "Felz": 193.25, "L(G_felz)": 7.7}
# O CSV guarda a abreviacao que o script de avaliacao gravou; a figura mostra o
# nome de exibicao de `view_names.py`, que e o padrao do artigo.
KEY = {"Desc": "Descriptors", "Felz": "Felzenszwalb", "L(G_slic)": "Line graph",
       "L(G_felz)": "L(G_felz)", "SLIC": "SLIC", "Hilbert": "Hilbert",
       "Fourier": "Fourier", "Gabor": "Gabor"}
OUTSIDE = [("graph", 11.0, 39.44, 0.76, GRAPH_C, "s"),
           ("text", 35.0, 39.57, 0.95, TEXT_C, "D")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "results/cost_ordered_multitask.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "results/cost_ordered_multitask.png")
    args = ap.parse_args()

    agg = (pd.read_csv(args.input)
           .groupby(["step", "added"])
           .agg(bacc=("bacc", "mean"), sd=("bacc", "std"))
           .reset_index().sort_values("step"))
    agg["cost_hours"] = np.cumsum([COST_FIXED[a] for a in agg.added]) / 60
    x, y, sd = agg.cost_hours.to_numpy(), agg.bacc.to_numpy(), agg.sd.to_numpy()
    ceiling = float(y[-1])

    fig, axis = plt.subplots(figsize=(11.0, 7.0), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    axis.plot(x, y, color=RAMP[5], linewidth=2.2, zorder=3)
    for i, row in enumerate(agg.itertuples()):
        axis.errorbar(row.cost_hours, row.bacc, yerr=row.sd, color=RAMP[i],
                      marker="o", markersize=10, capsize=4, elinewidth=1.5,
                      linestyle="none", zorder=4)
    for _, minutes, bacc, bsd, colour, marker in OUTSIDE:
        cum = x[-1] + minutes / 60
        axis.plot([x[-1], cum], [y[-1], bacc], color=colour, linewidth=1.3,
                  linestyle=(0, (3, 3)), zorder=3)
        axis.errorbar(cum, bacc, yerr=bsd, color=colour, marker=marker,
                      markersize=10, capsize=4, elinewidth=1.5,
                      linestyle="none", zorder=4)

    # --- linhas de referencia: quanto custa cada fracao do teto --------------
    # 25% fica FORA: o bloco mais barato sozinho (Hilbert) ja e 36% do teto,
    # entao aquela linha cairia em espaco vazio.
    for frac in (50, 75, 90, 100):
        level = ceiling * frac / 100
        axis.axhline(level, color="#b9b8b5", linewidth=1.2, linestyle=(0, (6, 4)), zorder=1)
        axis.text(7.32, level, f"{frac}%", fontsize=13.5, color="#4a4947",
                  ha="right", va="bottom", zorder=2, fontweight="bold")
    # Rotulo de cada bloco. Deslocamentos em pontos, ajustados a mao: no eixo
    # linear os tres baratos ficam em x~0,1 e os quatro caros em x~6, entao
    # posicao automatica colide.
    OFFSET = {1: (13, 0, "left"), 2: (14, -6, "left"), 3: (14, 4, "left"),
              4: (0, 17, "center"), 5: (0, -19, "center"), 6: (-13, 15, "right"),
              7: (0, -19, "center"), 8: (-16, 15, "right")}
    for i, row in enumerate(agg.itertuples()):
        dx, dy, ha = OFFSET[int(row.step)]
        axis.annotate(display(KEY[row.added]), xy=(row.cost_hours, row.bacc), xytext=(dx, dy),
                      textcoords="offset points", fontsize=11, color=RAMP[i],
                      ha=ha, va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
                      zorder=5)
    for j, (label, minutes, bacc, _sd, colour, _m) in enumerate(OUTSIDE):
        axis.annotate(label, xy=(x[-1] + minutes / 60, bacc),
                      xytext=(10, -17) if j == 0 else (0, 16),
                      textcoords="offset points", fontsize=11, color=colour,
                      ha="left" if j == 0 else "center",
                      va="top" if j == 0 else "bottom", zorder=5)

    axis.set_xlim(-0.3, 7.4)
    axis.set_ylim(11.0, 43.5)
    axis.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    axis.set_yticks([15, 20, 25, 30, 35, 40])
    axis.grid(False)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=15)
    axis.set_xlabel("cumulative extraction cost (wall-clock hours, 14 workers, "
                    "99,816 images)", fontsize=15, color=MUTED)
    axis.set_ylabel("balanced accuracy, production MLP (%)", fontsize=15, color=MUTED)
    axis.set_title("Balanced accuracy versus cumulative extraction cost\n"
                   "representations added from cheapest to most expensive "
                   "— Multitask, 125 classes",
                   fontsize=15, color=INK, loc="left", pad=14)

    fig.subplots_adjust(left=0.095, right=0.975, top=0.855, bottom=0.125)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"figura: {args.output}")


if __name__ == "__main__":
    main()
