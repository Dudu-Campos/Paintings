"""Cadeia cumulativa das doze views, LDA x MLP raso, Multitask 125 classes.

**O QUE ESTA FIGURA AFIRMA, e o que ela deliberadamente NAO afirma.** Ela mostra
que o ganho satura e que o canal de texto e a unica adicao tardia que ainda paga.
Ela NAO afirma nada sobre views individuais na ponta: com 5 folds e desvio entre
folds de 1,45pp no MLP, o teste pareado so detecta efeitos acima de ~1,8pp, e
quase nenhum passo tardio e desse tamanho -- nem os bons. Sete dos onze passos do
MLP nao sao significativos, e entre eles estao Gabor, Hilbert e o grafo, que a
enumeracao mostra ajudarem em 100%, 95% e 91% das coalizoes.

Por isso as barras do painel de baixo sao **preenchidas quando p<0,05 e vazadas
quando nao**: a figura mostra o proprio poder estatistico em vez de esconde-lo.
A alegacao sobre quais views nao se pagam pertence a `view_pareto.png`, que tem
2.048 medicoes por view em vez de 5 -- e a linha cinza de % embaixo e o ponteiro
para la.

Ordem: sete de Shapley, depois grafo e texto, depois as tres de menor consistencia
(ver `evaluate_cumulative_twelve.py` para por que grafo e texto nao entram na
posicao de Shapley deles).

MLP raso de proposito: nenhuma cabeca e proxy de todas as configuracoes. O NIVEL
nao e comparavel com as tabelas do artigo -- a figura afirma forma.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import display  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
# Tres cabecas, da mais fraca a mais forte. O `BranchMLP` e o de producao (um
# encoder por view); o `MLP` e o denso raso, que e o instrumento GENERICO --
# nenhuma cabeca e proxy de todas as configuracoes, entao a figura mostra as
# duas e deixa o leitor ver que a FORMA nao depende disso.
STYLE = {"LDA": ("#7c3fa8", "s", "--"),
         "MLP": ("#0b7a5a", "o", "-"),
         "BranchMLP": ("#c2571a", "D", "-")}
DELTA_OF = "BranchMLP"

# Textos por idioma, num dicionario so. A traducao e de EXIBICAO -- as chaves
# internas das views vem do registro e nao podem ser renomeadas (§7.18).
#
# `virgula`: em portugues o separador decimal e a virgula, e o CLAUDE.md ja
# escreve assim. Deixar "42.5" no eixo e "+0,79pp" no texto do artigo seriam
# duas convencoes para o mesmo numero na mesma pagina.
TEXTOS = {
    "en": {
        "y_topo": "balanced accuracy (%)",
        "y_baixo": "{} step \u0394 (pp)",
        "titulo": "Three heads, one shape: the text channel is the only late gain",
        "variante": "BranchMLP, graph concatenated",
        "codificacao": "encoding",
        "significativo": "p < 0.05",
        "nao_significativo": "not significant",
        "virgula": False,
    },
    "pt": {
        "y_topo": "acur\u00e1cia balanceada (%)",
        "y_baixo": "\u0394 do passo no {} (pp)",
        "titulo": "Tr\u00eas cabe\u00e7as, uma forma: o canal de texto \u00e9 o \u00fanico ganho tardio",
        "variante": "BranchMLP, grafo concatenado",
        "codificacao": "codifica\u00e7\u00e3o",
        "significativo": "p < 0,05",
        "nao_significativo": "n\u00e3o significativo",
        "virgula": True,
    },
}


def numero(valor: float, casas: int, virgula: bool, sinal: bool = False) -> str:
    texto = f"{valor:+.{casas}f}" if sinal else f"{valor:.{casas}f}"
    return texto.replace(".", ",") if virgula else texto
ALPHA = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/cumulative_twelve.csv")
    parser.add_argument("--show-variant", action="store_true",
                        help="desenha tambem o grafo CONCATENADO, tracejado. "
                             "Fora por padrao desde 05/set: a figura tem tres "
                             "cabecas e a quarta curva competia com elas. O "
                             "contraste entre codificacoes esta medido no §5.6 "
                             "e nao precisa desta figura para ser afirmado.")
    parser.add_argument("--additive", type=Path,
                        default=ROOT / "results/cumulative_logit_add.csv")
    parser.add_argument("--marginal", type=Path,
                        default=ROOT / "results/view_marginal_distribution.csv")
    parser.add_argument("--lang", choices=["en", "pt"], default="en",
                        help="idioma dos rotulos; as CHAVES internas nao mudam (§7.18)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_twelve.png")
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    marginal = pd.read_csv(args.marginal).set_index("view")

    # A codificacao de PRODUCAO do grafo em 125 classes e o `logit-add` (§5.6),
    # nao a concatenacao. O braco aditivo cobre k>=8 (onde o grafo entra); ate
    # k=7 ele e identico ao `BranchMLP`, entao a curva de producao e a emenda
    # dos dois. A variante concatenada fica desenhada por baixo, tracejada: o
    # contraste entre as duas E o achado do §5.6 e vale mostrar.
    additive = pd.read_csv(args.additive)
    concat_tail = table[table.classifier.eq("BranchMLP")
                        & table.k.ge(additive.k.min())].copy()
    concat_tail["classifier"] = "BranchMLP (graph concatenated)"
    table = pd.concat([
        table[~table.classifier.eq("BranchMLP") | table.k.lt(additive.k.min())],
        table[table.classifier.eq("BranchMLP") & table.k.lt(additive.k.min())],
        additive.assign(classifier="BranchMLP"),
        concat_tail], ignore_index=True).drop_duplicates(
            subset=["k", "classifier", "fold"])
    names = table.drop_duplicates("k").set_index("k").added.to_dict()
    ks = sorted(table.k.unique())

    def series(clf, k):
        return table[(table.k == k) & (table.classifier == clf)] \
            .sort_values("fold").bacc.to_numpy()

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 6.6), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})

    top.set_facecolor(SURFACE)
    top.set_axisbelow(True)
    top.grid(True, axis="y", color=GRID, linewidth=0.7)
    for clf, (colour, marker, dash) in STYLE.items():
        group = table[table.classifier == clf].groupby("k").bacc
        mean, sd = group.mean().reindex(ks), group.std().reindex(ks)
        top.fill_between(ks, mean - sd, mean + sd, color=colour, alpha=0.13, linewidth=0)
        top.plot(ks, mean, linestyle=dash, marker=marker, markersize=6,
                 linewidth=2.0, color=colour, label=clf, zorder=4)
    T = TEXTOS[args.lang]
    tail = table[table.classifier.eq("BranchMLP (graph concatenated)")]
    if args.show_variant and not tail.empty:
        group = tail.groupby("k").bacc.mean()
        bridge = table[table.classifier.eq("BranchMLP") & table.k.eq(7)].bacc.mean()
        xs = [7] + list(group.index)
        ys = [bridge] + list(group.to_numpy())
        top.plot(xs, ys, linestyle=":", linewidth=1.6, marker="d",
                 markersize=4.5, color=STYLE["BranchMLP"][0], alpha=0.55,
                 zorder=3, label=T["variante"])
        k0 = int(group.index.min())
        gap = table[table.classifier.eq("BranchMLP") & table.k.eq(k0)].bacc.mean() - group[k0]
        # Acima do ponto e a esquerda: abaixo ela cai sobre a curva do LDA.
        top.annotate(f"{T['codificacao']}\n{numero(gap, 2, T['virgula'], True)} pp",
                     (k0, group[k0]),
                     textcoords="offset points", xytext=(-10, 8), fontsize=8.5,
                     ha="right", va="bottom", color=STYLE["BranchMLP"][0],
                     fontweight="bold")
        top.annotate("", xy=(k0, group[k0]),
                     xytext=(k0, table[table.classifier.eq("BranchMLP")
                                       & table.k.eq(k0)].bacc.mean()),
                     arrowprops=dict(arrowstyle="<->", linewidth=1.1,
                                     color=STYLE["BranchMLP"][0]))

    top.set_ylabel(T["y_topo"], fontsize=10, color=INK)
    top.set_title(T["titulo"],
                  fontsize=12, color=INK, pad=10, loc="left", fontweight="bold")
    top.legend(frameon=False, fontsize=8.8, loc="lower right", ncol=2)

    # ---------------- deltas por passo, com significancia explicita ----------
    bottom.set_facecolor(SURFACE)
    bottom.set_axisbelow(True)
    bottom.grid(True, axis="y", color=GRID, linewidth=0.7)
    bottom.axhline(0, color="#5a5a5a", linewidth=1.0, zorder=3)
    colour = STYLE[DELTA_OF][0]
    steps = []
    for k in ks[1:]:
        delta = series(DELTA_OF, k) - series(DELTA_OF, k - 1)
        pvalue = stats.ttest_rel(series(DELTA_OF, k),
                                 series(DELTA_OF, k - 1))[1]
        steps.append((k, delta.mean(), delta.std(), pvalue))
    low = min(m - s for _, m, s, _ in steps)
    high = max(m + s for _, m, s, _ in steps)
    footer = low - 0.42 * (high - low)
    bottom.set_ylim(footer - 0.20 * (high - low), high + 0.16 * (high - low))
    for k, mean, sd, pvalue in steps:
        strong = pvalue < ALPHA
        bottom.bar(k, mean, width=0.62, zorder=4, linewidth=1.4,
                   color=colour if strong else "none",
                   edgecolor=colour, alpha=0.9 if strong else 1.0)
        bottom.errorbar(k, mean, yerr=sd, fmt="none", ecolor=colour,
                        elinewidth=1.1, capsize=3, zorder=5)
        share = marginal.loc[names[k], "frac_positiva"]
        bottom.annotate(f"{share:.0f}%", (k, footer), ha="center", va="center",
                        fontsize=8, color=MUTED)
    bottom.set_ylabel(T["y_baixo"].format(DELTA_OF), fontsize=10, color=INK)
    bottom.set_xticks(ks)
    bottom.set_xticklabels([display(names[k], args.lang) for k in ks], fontsize=9,
                           color=MUTED, rotation=32, ha="right",
                           rotation_mode="anchor")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=colour,
                             alpha=0.9, label=T["significativo"]),
               plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=colour,
                             linewidth=1.4, label=T["nao_significativo"])]
    bottom.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
                  loc="upper right")

    if T["virgula"]:
        # Vale para os DOIS paineis: o de cima esta em BAcc (42.5) e o de baixo
        # em pp (2.5), e misturar ponto num e virgula no outro seria pior que
        # deixar os dois em ingles.
        formatador = mticker.FuncFormatter(
            lambda v, _pos: f"{v:g}".replace(".", ","))
        top.yaxis.set_major_formatter(formatador)
        bottom.yaxis.set_major_formatter(formatador)

    for axis in (top, bottom):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#9a9a9a")

    figure.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.16)
    figure.savefig(args.output, dpi=200, facecolor=SURFACE)
    for k, mean, sd, pvalue in steps:
        mark = "*" if pvalue < ALPHA else " "
        print(f"  {names[k]:<18}{mean:+6.2f} ± {sd:4.2f}  p={pvalue:.3f} {mark}")
    print(f"figura: {args.output}")


if __name__ == "__main__":
    main()
