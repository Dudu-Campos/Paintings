"""Curva cumulativa em ordem de Shapley e o passo a passo da queda final.

Dois paineis compartilhando o eixo x:

CIMA -- as curvas LDA e MLP. Os conjuntos sao IDENTICOS nos dois a cada k, entao
a distancia vertical e puramente capacidade, sem confundir com escolha de views.

BAIXO -- o delta de CADA passo no MLP. E ele que mostra a afirmacao: as tres
ultimas views subtraem, e a maior perda individual nao e a dormente obvia (HVG)
e sim o **Fourier**, que esta no banco de producao.

**Por que o MLP aqui e o raso, e nao o `BranchMLP` de producao.** Decisao do
usuario em 03/set: nenhuma cabeca e proxy de todas as configuracoes, e um MLP
denso sem encoder por bloco e o instrumento *generico* -- o especializado
responderia sobre si mesmo. O nivel entao NAO e comparavel com as tabelas do
artigo (o `hand-8` da 36,32 aqui contra 38,74 no §9.2, e a diferenca inteira e o
cabecote); a figura afirma FORMA, nao nivel, e a legenda tem de dizer isso.

A ordem vem do Shapley de 5 folds. Ela coincide, nas tres ultimas posicoes, com
a fracao de coalizoes em que a view ajuda (`view_marginal_distribution.csv`), que
e a estatistica do painel direito de `view_pareto.png`: 6 de 6 views acima de 95%
dao delta positivo no MLP e 3 de 3 abaixo de 70% dao negativo. Os rotulos de %
no painel de baixo mostram esse pareamento.
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
LOSS = "#b03030"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/cumulative_by_shapley.csv")
    parser.add_argument("--marginal", type=Path,
                        default=ROOT / "results/view_marginal_distribution.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_by_shapley.png")
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    marginal = pd.read_csv(args.marginal).set_index("view")
    names = table.drop_duplicates("k").set_index("k").added.to_dict()
    ks = sorted(table.k.unique())

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.10})

    # ---------------- cima: as duas curvas ----------------
    top.set_facecolor(SURFACE)
    top.set_axisbelow(True)
    top.grid(True, axis="y", color=GRID, linewidth=0.7)
    peaks, means = {}, {}
    for clf, (colour, marker, dash) in STYLE.items():
        group = table[table.classifier == clf].groupby("k").bacc
        mean, sd = group.mean().reindex(ks), group.std().reindex(ks)
        means[clf] = mean
        top.fill_between(ks, mean - sd, mean + sd, color=colour, alpha=0.13, linewidth=0)
        top.plot(ks, mean, linestyle=dash, marker=marker, markersize=6,
                 linewidth=2.0, color=colour, label=clf, zorder=4)
        peaks[clf] = int(mean.idxmax())

    # As tres ultimas views: a regiao que a figura existe para mostrar.
    start = len(ks) - 3
    for axis in (top, bottom):
        axis.axvspan(start + 0.5, ks[-1] + 0.5, color=LOSS, alpha=0.06,
                     linewidth=0, zorder=0)

    for clf, k in peaks.items():
        colour = STYLE[clf][0]
        top.scatter([k], [means[clf][k]], s=170, facecolors="none",
                    edgecolors=colour, linewidths=1.8, zorder=5)
        loss = means[clf][ks[-1]] - means[clf][k]
        top.annotate(f"{clf}  {loss:+.2f} pp", (k, means[clf][k]),
                     textcoords="offset points",
                     xytext=(12, 14 if clf == "MLP" else -20),
                     fontsize=9, color=colour, fontweight="bold")
    top.annotate("last three views", ((start + 0.5 + ks[-1] + 0.5) / 2,
                                      top.get_ylim()[1]),
                 ha="center", va="top", fontsize=9, color=LOSS)
    top.set_ylabel("balanced accuracy (%)", fontsize=10, color=INK)
    top.set_title("The last three views subtract, and cost the MLP most",
                  fontsize=12, color=INK, pad=10, loc="left", fontweight="bold")
    top.legend(frameon=False, fontsize=10, loc="lower right",
               bbox_to_anchor=(1.0, 0.02))

    # ---------------- baixo: delta por passo, no MLP ----------------
    bottom.set_facecolor(SURFACE)
    bottom.set_axisbelow(True)
    bottom.grid(True, axis="y", color=GRID, linewidth=0.7)
    bottom.axhline(0, color="#5a5a5a", linewidth=1.0, zorder=3)
    deltas = means["MLP"].diff()
    # Faixa reservada embaixo para os %, senao eles caem em cima dos valores das
    # barras negativas (o rodape e uma LINHA, nao o fundo do eixo).
    low, high = deltas.min(), deltas.max()
    footer = low - 0.62 * (high - low)
    bottom.set_ylim(footer - 0.18 * (high - low), high + 0.30 * (high - low))
    for k in ks[1:]:
        value = deltas[k]
        colour = STYLE["MLP"][0] if value > 0 else LOSS
        bottom.bar(k, value, width=0.62, color=colour, alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=4)
        bottom.annotate(f"{value:+.2f}", (k, value), textcoords="offset points",
                        xytext=(0, 4 if value > 0 else -12), ha="center",
                        fontsize=8, color=colour)
        # A previsao vinda da enumeracao em LDA, para o leitor parear.
        share = marginal.loc[names[k], "frac_positiva"]
        bottom.annotate(f"{share:.0f}%", (k, footer), ha="center", va="center",
                        fontsize=8, color=MUTED)

    bottom.set_ylabel("MLP step Δ (pp)", fontsize=10, color=INK)
    bottom.set_xticks(ks)
    bottom.set_xticklabels([display(names[k]) for k in ks], fontsize=9,
                           color=MUTED, rotation=32, ha="right",
                           rotation_mode="anchor")
    # Sem rotulo de eixo x: os nomes das views ja identificam o eixo, e a
    # explicacao do % cinza pertence a legenda do artigo, nao a figura.

    for axis in (top, bottom):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#9a9a9a")

    figure.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.16)
    figure.savefig(args.output, dpi=200, facecolor=SURFACE)
    print(f"picos: {peaks}")
    print("deltas MLP:", deltas.round(2).to_dict())
    print(f"figura: {args.output}")


if __name__ == "__main__":
    main()
