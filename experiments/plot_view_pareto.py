"""Fronteira de Pareto sobre as 4.096 combinacoes de views, e quem fica de fora.

Dois paineis, porque sao duas perguntas e uma so nao responde a outra:

ESQUERDA -- a forma da fronteira. Nuvem das 4.096 combinacoes em cinza, com a
fronteira nao-dominada por cima. O que ela mostra: a curva satura, e depois de
~350d o ganho por coluna e quase nulo.

DIREITA -- o SHAPLEY por view (decisao do usuario, 06/out), que e o numero que
o §5.9c reporta e o texto cita. Barra = Shapley, bigode = desvio entre os 5
folds, fonte `results/shapley_twelve_summary.csv`.

**A fracao de coalizoes FICA, como rotulo secundario.** A versao anterior deste
painel usava essa fracao como a barra, com um argumento que continua valido: o
Shapley e uma MEDIA, e media positiva pequena pode vir de "ajuda pouco sempre"
ou de "ajuda muito as vezes e atrapalha as outras". Mantendo os dois no mesmo
painel nada se perde, e os dois casos ficam visiveis de uma vez --
`Similarity graph` tem o MAIOR Shapley (6,11) e ajuda em so 91% das coalizoes
(p5 = -0,31, ou seja atrapalha em algumas), enquanto `Text` tem Shapley menor
(5,22) e ajuda em 100%. Trocar a barra reordena o topo justamente ai.

CUIDADO DE FONTE: o `view_marginal_distribution.csv` tem uma coluna `media` que
NAO e o Shapley -- e a media uniforme sobre os 2.048 subconjuntos (Banzhaf), e
difere bastante (Text 3,73 contra 5,22). A barra usa o Shapley com pesos de
coalizao do `shapley_twelve_summary.csv`.

A linha vertical fica em ZERO, que e a referencia de "acrescenta alguma coisa".
Cores e hachuras vem de
`view_names.py` (§7.18): identidade nunca depende so de cor -- as barras vao
rotuladas no eixo e a hachura marca a familia.

Ressalva herdada da fonte: sonda LDA. Isto ORDENA subconjuntos; o §8 tem seis
casos de "LDA extrai, MLP destroi", entao nao e previsao do BranchMLP.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import colour, display, edge, hatch  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
FRONT = "#0b3d91"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", type=Path, default=ROOT / "results/view_pareto.csv")
    parser.add_argument("--marginal", type=Path,
                        default=ROOT / "results/view_marginal_distribution.csv")
    parser.add_argument("--shapley", type=Path,
                        default=ROOT / "results/shapley_twelve_summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/view_pareto.png")
    args = parser.parse_args()

    table = pd.read_csv(args.pareto)
    table = table[table.n_views > 0]
    shap = (pd.read_csv(args.shapley).rename(columns={"Unnamed: 0": "view"})
            [["view", "shapley", "sd"]])
    marginal = pd.read_csv(args.marginal).merge(shap, on="view", how="inner")
    if len(marginal) != len(shap):
        raise SystemExit("views do Shapley nao casam com as da distribuicao marginal")
    marginal = marginal.sort_values("shapley")

    # Fronteira: nenhum outro ponto tem dimensao menor-ou-igual e BAcc maior.
    ordered = table.sort_values(["dimensions", "bacc"], ascending=[True, False])
    keep, best = [], -np.inf
    for row in ordered.itertuples():
        if row.bacc > best:
            keep.append(row.Index)
            best = row.bacc
    front = table.loc[keep].sort_values("dimensions")

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 4.8), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------- esquerda: nuvem + fronteira ----------------
    left.set_facecolor(SURFACE)
    left.set_axisbelow(True)
    left.grid(True, color=GRID, linewidth=0.7)
    # Hexbin em vez de scatter: com 4.096 pontos a nuvem satura de sobreposicao
    # exatamente na faixa de 300-700d, que e onde a densidade importa. Contagem
    # em log porque as celulas vao de 1 a ~80. Sequencial de uma matiz so
    # (amarelo -> marrom), que e a matiz mais distante do azul da fronteira e do
    # vermelho do ponto de 12 views -- amarelo/marrom contra azul e o eixo que
    # melhor sobrevive aos tres dicromatismos.
    hexes = left.hexbin(table.dimensions, table.bacc, gridsize=34, bins="log",
                        cmap="YlOrBr", mincnt=1, linewidths=0.15,
                        edgecolors="#fbfbfa", zorder=2)
    # Barra HORIZONTAL dentro do canto inferior esquerdo, que esta vazio: uma
    # barra vertical entre os paineis empurrava as anotacoes e cortava o rotulo
    # do ponto de 12 views.
    cax = left.inset_axes([0.06, 0.10, 0.30, 0.035])
    bar = figure.colorbar(hexes, cax=cax, orientation="horizontal")
    bar.set_label("subsets per cell", fontsize=8.5, color=INK, labelpad=2)
    bar.ax.tick_params(labelsize=7.5, length=2)
    left.plot(front.dimensions, front.bacc, "-o", color=FRONT, markersize=5,
              linewidth=2.2, zorder=5, label="Pareto frontier",
              path_effects=[pe.Stroke(linewidth=4.0, foreground="white"),
                            pe.Normal()])

    # O ponto de TODAS as 12 views e a afirmacao mais forte do painel: ele esta
    # FORA da fronteira, entao usar tudo e dominado por um subconjunto menor.
    full = table[table.n_views == 12].iloc[0]
    best = front.iloc[-1]
    left.scatter([full.dimensions], [full.bacc], s=110, marker="X",
                 color="#b03030", edgecolors="white", linewidths=1.4, zorder=7)
    left.plot([best.dimensions, full.dimensions], [best.bacc, full.bacc],
              linestyle=":", linewidth=1.2, color="#6a6a6a", zorder=6)
    # Os dois rotulos vao para a faixa vazia embaixo-a-direita, com linha de
    # chamada: sobre os hexagonos escuros nenhuma cor de texto e legivel.
    arrow = dict(arrowstyle="-", linewidth=0.9, color="#8a8a8a",
                 shrinkA=2, shrinkB=6)
    left.annotate(f"best subset · {int(best.dimensions)}d · {best.bacc:.2f}",
                  (best.dimensions, best.bacc), xytext=(660, 18.5),
                  textcoords="data", fontsize=9, color=FRONT, ha="left",
                  va="center", arrowprops=arrow, zorder=8)
    left.annotate(f"all 12 views · {int(full.dimensions)}d · {full.bacc:.2f}",
                  (full.dimensions, full.bacc), xytext=(660, 12.0),
                  textcoords="data", fontsize=9, color="#b03030", ha="left",
                  va="center", arrowprops=arrow, zorder=8)
    left.set_xlabel("dimensions of the concatenated subset", fontsize=10, color=INK)
    left.set_ylabel("balanced accuracy (%)", fontsize=10, color=INK)
    left.set_title("Using all twelve views is dominated", fontsize=11, color=INK,
                   loc="left", fontweight="bold")

    # ---------------- direita: Shapley por view ----------------
    right.set_facecolor(SURFACE)
    right.set_axisbelow(True)
    right.grid(True, axis="x", color=GRID, linewidth=0.7)
    positions = np.arange(len(marginal))
    for y, row in zip(positions, marginal.itertuples()):
        right.barh(y, row.shapley, height=0.72, color=colour(row.view),
                   edgecolor=edge(row.view), hatch=hatch(row.view), linewidth=0.8)
        # Bigode = desvio ENTRE FOLDS. No `Similarity graph` ele e 1,10 contra
        # 0,09-0,27 das outras, porque o bloco e recalculado por fold com OOF
        # aninhado e herda a variancia do procedimento -- sem o bigode, o valor
        # dele parece ter a precisao dos demais.
        right.plot([row.shapley - row.sd, row.shapley + row.sd], [y, y],
                   color=edge(row.view), linewidth=1.3, solid_capstyle="butt", zorder=6)
    right.axvline(0, color="#b03030", linewidth=1.2, linestyle="--", zorder=5)
    right.set_yticks(positions)
    right.set_yticklabels([display(v) for v in marginal.view], fontsize=9.5, color=MUTED)
    lo = min(0.0, float((marginal.shapley - marginal.sd).min()))
    hi = float((marginal.shapley + marginal.sd).max())
    # Margem a direita para os DOIS rotulos (valor + "helps N%"), que saem em
    # offset de PONTOS e portanto nao escalam com os dados.
    right.set_xlim(lo - 0.9, hi + 3.4)
    right.set_xlabel("Shapley value (p.p. of BAcc) · exact over 4,096 masks × 5 folds"
                     "\nwhisker: s.d. between folds", fontsize=9.5, color=INK)
    right.set_title("How much each view is worth", fontsize=11, color=INK,
                    loc="left", fontweight="bold")
    for y, row in zip(positions, marginal.itertuples()):
        # Shapley em destaque; a fracao de coalizoes em que a view AJUDA fica
        # como rotulo secundario -- e a informacao que a media apaga.
        #
        # Deslocamento em PONTOS, nao em unidades de dado: em unidades de dado o
        # par de rotulos encosta um no outro nas barras grandes e, na barra
        # NEGATIVA do HVG, cai em cima da linha de zero. A ancora tambem vira de
        # lado quando a barra e negativa.
        # Ancora SEMPRE do lado positivo, mesmo na barra negativa do HVG: a
        # esquerda nao ha espaco (os rotulos de view colam no eixo) e os dois
        # textos cairiam em cima do nome da view.
        ancora = max(row.shapley + row.sd, 0.0)
        right.annotate(f"{row.shapley:+.2f}", (ancora, y),
                       textcoords="offset points", xytext=(5, 0),
                       va="center", ha="left", fontsize=8.5, color=INK)
        right.annotate(f"helps {row.frac_positiva:.0f}%", (ancora, y),
                       textcoords="offset points", xytext=(44, 0),
                       va="center", ha="left", fontsize=7.5, color="#8a8a8a")

    for axis in (left, right):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#9a9a9a")

    figure.tight_layout()
    figure.savefig(args.output, dpi=200, facecolor=SURFACE)
    print(f"fronteira: {len(front)} pontos, de {front.dimensions.min():.0f}d "
          f"a {front.dimensions.max():.0f}d")
    print(f"figura: {args.output}")


if __name__ == "__main__":
    main()
