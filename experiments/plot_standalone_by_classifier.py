"""Cada view sozinha, sob quatro classificadores: a ordenação é compartilhada.

A pergunta do `evaluate_standalone_by_classifier.py` é se a ordenação das views
depende do estimador. Medido em 125 classes, 5 folds: **não depende**. Spearman
entre as ordenações vai de +0,955 a +0,991 nos seis pares.

O que difere é o NÍVEL, por um fator quase constante por bloco (excluindo o HVG,
que está no acaso e cuja razão é ruído dividido por ruído):

    LDA / MLP           0,775 ± 0,090
    XGBoost / MLP       0,696 ± 0,035
    RandomForest / MLP  0,442 ± 0,037

Daí o desenho: um ponto por (view, classificador), views ligadas por linha. Se a
razão fosse exatamente constante as linhas seriam paralelas; o quanto elas se
cruzam é a parte da diferença que NÃO é fator de escala.

Isso fecha uma pendência metodológica: o Shapley do §5.9c usa sonda LDA, e o §8
avisa que "LDA extrai, MLP destrói" em vetores multi-bloco. Aqui cada bloco é
único e homogêneo, e a ordenação é preservada -- então usar LDA para ORDENAR
views é defensável com dado, não por conveniência.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Hachura fina: o padrão (1,0) fecha as barras estreitas num bloco sólido.
matplotlib.rcParams["hatch.linewidth"] = 0.6
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import colour, display, edge, hatch  # noqa: E402

SURFACE, GRID, MUTED, INK = "#fbfbfa", "#e2e2e2", "#6b6a67", "#1b1b1a"
ORDER = ["MLP", "LDA", "XGBoost", "kNN", "RandomForest"]

# Textos por idioma, num dicionario so -- acrescentar um idioma nao deve
# exigir cacar literais espalhados pelo arquivo. A traducao e de EXIBICAO: as
# chaves internas das views vem do registro e nao mudam (§7.18).
TEXTOS = {
    "en": {
        "y": "balanced accuracy (%)",
        "titulo": "Each view alone, across classifiers",
        "sub_grade": "same ordering everywhere; only the level changes",
        "sub_slope": "the ordering is shared; the level is a near-constant factor",
        "corpus": "— Multitask, 125 classes",
        "ordem": "bar order, left to right",
        "acaso": "chance",
        "spearman": "Spearman between the orderings",
        "pares": "across all {n} pairs of classifiers, HVG excluded",
    },
    "pt": {
        "y": "acurácia balanceada (%)",
        "titulo": "Cada view sozinha, em cinco classificadores",
        "sub_grade": "a mesma ordenação em todos; só o nível muda",
        "sub_slope": "a ordenação é compartilhada; o nível é um fator quase constante",
        "corpus": "— Multitask, 125 classes",
        "ordem": "ordem das barras, da esquerda para a direita",
        "acaso": "acaso",
        "spearman": "Spearman entre as ordenações",
        "pares": "nos {n} pares de classificadores, HVG excluído",
    },
}



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, nargs="+",
                    default=[ROOT / "results/standalone_by_classifier_125.csv",
                             ROOT / "results/standalone_knn_125.csv"],
                    help="o kNN roda em arquivo separado porque foi medido na "
                         "absinto enquanto a mojito escrevia o primeiro (§1: "
                         "nunca duas maquinas no mesmo output)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/standalone_by_classifier_125.png")
    ap.add_argument("--chance", type=float, default=100 / 125)
    ap.add_argument("--exclude", nargs="*",
                    default=["Caption embedding", "Similarity graph (class scores)"],
                    help="blocos a omitir por padrao. O embedding da legenda sai "
                         "porque colide na leitura com o canal de texto TF-IDF, "
                         "que e outra coisa e tem sinal oposto (§5.9b). O "
                         "`Similarity graph (class scores)` sai por ser duplicata "
                         "VISUAL do `Similarity graph`: 31,15 contra 31,17, ou "
                         "seja as 32 estatisticas valem 0,02pp -- isso e um "
                         "resultado (§6: o grafo da identidade de classe, nao "
                         "confianca de vizinhanca), mas como barra nao mostra nada.")
    ap.add_argument("--folds", type=int, default=5,
                    help="folds exigidos por célula; blocos incompletos saem da "
                         "figura em vez de entrar como média de menos folds")
    ap.add_argument("--legend", action="store_true", default=True,
                    help="chave na célula vaga da grade, em vez dos rótulos "
                         "rotacionados de x repetidos em dois painéis")
    ap.add_argument("--no-legend", dest="legend", action="store_false")
    ap.add_argument("--lang", choices=["en", "pt"], default="en",
                    help="idioma dos rotulos; as CHAVES internas nao mudam (§7.18)")
    ap.add_argument("--legend-cols", type=int, default=2)
    ap.add_argument("--ncols", type=int, default=2,
                    help="colunas da grade de painéis (2 por padrão)")
    # Desligado por decisão do usuário em 1/set ("só quero os bars na imagem").
    # O Spearman continua no §5.9d e no CSV; a legenda do artigo o carrega. A
    # célula que sobra fica vazia, o que é preferível a texto competindo com o
    # traço numa figura cuja afirmação é a silhueta repetida.
    ap.add_argument("--summary", action="store_true", default=False)
    ap.add_argument("--no-summary", dest="summary", action="store_false")
    ap.add_argument("--style", choices=["bars", "slope"], default="bars",
                    help="bars: um painel por classificador, views na MESMA ordem "
                         "em todos -- a afirmacao vira 'os cinco paineis tem a "
                         "mesma silhueta'. slope: a versao anterior, linhas "
                         "ligando a mesma view entre classificadores.")
    ap.add_argument("--label-gap", type=float, default=0.043,
                    help="folga minima entre rotulos, em fracao do eixo y")
    args = ap.parse_args()

    # O SVM fica de fora: e o unico estimador que so roda em subamostra de 15k
    # (custo O(n^2)), entao o numero dele nao e comparavel de face com os cinco
    # que rodam na coorte inteira. Ele esta no CSV e tem leitura propria.
    frame = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    table = (frame[~frame.subsampled]
             .pivot_table(index="block", columns="classifier", values="bacc")
             .dropna())
    table = table.drop(index=args.exclude, errors="ignore")
    table = _require_full_folds(frame, table, args.folds)
    table = table[[c for c in ORDER if c in table.columns]]
    table = table.sort_values(table.columns[0], ascending=False)
    xs = np.arange(len(table.columns), dtype=float)

    if args.style == "bars":
        draw_bars(table, args)
    else:
        draw_slope(table, args)
    print(f"figura: {args.output}")
    print(table.round(2).to_string())


def _require_full_folds(frame: pd.DataFrame, table: pd.DataFrame,
                        folds: int) -> pd.DataFrame:
    """Tira da figura todo bloco que não tenha `folds` folds em TODA coluna.

    Existe por um erro real, de 1/set: os dois jobs ainda estavam calculando o
    `Similarity graph` (que é `PER_FOLD`, o mais caro) quando a figura foi
    regerada. O `pivot_table` faz média silenciosa sobre os folds que existirem,
    então o bloco entrou com **1 fold** -- e foi parar no topo de todos os
    painéis, que é a posição que mais pesa na leitura. Nada no traço denunciava.

    Um bloco parcial não é "quase certo": ele é uma média sobre um subconjunto de
    folds diferente do dos outros, num corpus onde o desvio entre folds chega a
    1,4pp (§9.2). Melhor faltar da figura do que entrar comparável por acidente.
    """
    counts = (frame[~frame.subsampled]
              .pivot_table(index="block", columns="classifier", values="bacc",
                           aggfunc="count")
              .reindex(index=table.index, columns=table.columns))
    complete = (counts == folds).all(axis=1)
    if not complete.all():
        for block in table.index[~complete]:
            got = counts.loc[block].fillna(0).astype(int).to_dict()
            print(f"  FORA (incompleto, exige {folds} folds): {block}  {got}")
    return table[complete]


def draw_bars(table: pd.DataFrame, args) -> None:
    """Painéis por classificador, views sempre na MESMA ordem.

    O eixo y de cada painel vai ate o proprio maximo, nao ate um maximo comum.
    Isso e deliberado: a afirmacao da figura e sobre ORDENACAO, e com escala
    livre os paineis ficam com a mesma silhueta se -- e so se -- a ordenacao for
    a mesma. Um eixo compartilhado esmagaria o RandomForest (0,43x) e tornaria
    ilegivel justamente o painel mais informativo. O nivel nao se perde: vai
    anotado como razao em cada painel.

    Grade de `--ncols` colunas (2 por padrao, decidido em 1/set: empilhado em
    coluna unica ocupava quase uma pagina). A celula que sobra recebe o resumo
    numerico, para nao ficar buraco -- e ela carrega o Spearman, que e a
    afirmacao que as barras so mostram qualitativamente.
    """
    order = list(table.index)
    names = [display(b, args.lang) for b in order]
    colours = [colour(b) for b in order]
    edges = [edge(b) for b in order]
    hatches = [hatch(b) for b in order]
    xs = np.arange(len(order))
    panels = list(table.columns)

    reference = panels[0]
    ratios = table.drop(index="HVG", errors="ignore")
    ratios = ratios.div(ratios[reference], axis=0)

    ncols = max(1, args.ncols)
    nrows = int(np.ceil((len(panels) + (1 if args.summary else 0)) / ncols))
    fig, axes = plt.subplots(nrows, ncols, dpi=130,
                             figsize=(5.4 * ncols, 2.5 * nrows + 0.9))
    flat = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(SURFACE)

    for slot, (axis, name) in enumerate(zip(flat, panels)):
        values = table[name].to_numpy(dtype=float)
        axis.set_facecolor(SURFACE)
        bars = axis.bar(xs, values, width=0.72, color=colours, zorder=3,
                        edgecolor=edges, linewidth=0.8)
        for bar, pattern in zip(bars, hatches):
            bar.set_hatch(pattern)
        axis.axhline(args.chance, color="#b9b8b5", linewidth=1.0,
                     linestyle=(0, (4, 4)), zorder=2)

        # As razões entre classificadores saíram da figura em 1/set (poluição):
        # elas estão no CSV e no §5.9d, e o nível já se lê pelos máximos de eixo
        # diferentes em cada painel. O que a figura precisa carregar é a
        # ORDENAÇÃO, e para isso basta o nome do painel.
        axis.text(0.99, 0.94, name, transform=axis.transAxes, fontsize=13,
                  color=INK, ha="right", va="top")

        axis.set_ylim(0, values.max() * 1.32)
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
        axis.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=10.5)

        # Com legenda, NENHUM painel leva rotulo: a ordem das barras e identica
        # em todos, entao a chave e o proprio eixo x virado na vertical. Sem ela,
        # rotula todo painel que nao tenha outro logo abaixo -- com celula vazia
        # na grade, a ultima coluna termina uma linha antes.
        axis.set_xticks(xs)
        if args.legend or slot + ncols < len(panels):
            axis.set_xticklabels([])
        else:
            axis.set_xticklabels(names, rotation=38, ha="right", fontsize=9.5,
                                 color=INK)
        axis.tick_params(axis="x", length=0, pad=3)

    for axis in flat[len(panels):]:
        axis.set_axis_off()
    if len(flat) > len(panels):
        spare = flat[len(panels)]
        if args.legend:
            _legend(spare, order, names, colours, edges, hatches,
                    cols=args.legend_cols, lang=args.lang)
        elif args.summary:
            _summary(spare, table, ratios, reference, lang=args.lang)

    T = TEXTOS[args.lang]
    fig.supylabel(T["y"], fontsize=12.5, color=MUTED, x=0.012)
    fig.suptitle(f"{T['titulo']}\n{T['sub_grade']} {T['corpus']}",
                 fontsize=14, color=INK, x=0.045, ha="left")
    # Com legenda não há rótulo rotacionado nenhum, então o `hspace` pode ser
    # apertado. Sem ela precisa de folga: os rótulos do painel do meio da coluna
    # direita descem sobre a célula vaga -- foi a primeira colisão da grade.
    fig.subplots_adjust(left=0.075, right=0.99, top=0.905,
                        bottom=0.055 if args.legend else 0.115,
                        hspace=0.16 if args.legend else 0.42, wspace=0.16)
    fig.savefig(args.output, facecolor=SURFACE)


def _legend(axis, order, names, colours, edges, hatches, cols=2, lang="en") -> None:
    """A chave, na célula vaga -- o eixo x virado na vertical.

    Vale a troca porque a ordem das barras é **idêntica** nos cinco painéis: a
    chave não é uma indireção cromática, é posicional. Sem ela os rótulos
    rotacionados apareciam duas vezes (o painel do meio da coluna direita e o da
    esquerda embaixo), comiam altura e deixavam a composição torta.

    Em DUAS colunas por padrão, lidas de cima para baixo e depois à direita. Em
    coluna única as onze entradas ocupavam a altura inteira da célula com
    entrelinha apertada enquanto metade da largura ficava vazia.

    A ordem daqui é a MESMA das barras e não pode ser reordenada por
    conveniência: o §7.18 registra que dentro de uma família a separação é de
    dE 7,5, então identificar view **só** por cor não é seguro -- é a posição que
    carrega a identidade, e a cor só agrupa.
    """
    axis.set_axis_off()
    axis.text(0.02, 1.00, TEXTOS[lang]["ordem"], transform=axis.transAxes,
              fontsize=10.5, color=MUTED, va="top")

    per_col = int(np.ceil(len(names) / cols))
    step, width = 0.115, 1.0 / cols
    for rank, (name, fill, line, pattern) in enumerate(
            zip(names, colours, edges, hatches)):
        col, row = divmod(rank, per_col)
        x = 0.02 + col * width
        y = 0.86 - row * step
        patch = plt.Rectangle((x, y - 0.06), 0.062, 0.064, facecolor=fill,
                              edgecolor=line, linewidth=0.8,
                              transform=axis.transAxes, clip_on=False)
        patch.set_hatch(pattern)
        axis.add_patch(patch)
        axis.text(x + 0.082, y - 0.028, name, transform=axis.transAxes,
                  fontsize=11, color=INK, va="center")


def _summary(axis, table: pd.DataFrame, ratios: pd.DataFrame, reference: str,
             lang: str = "en") -> None:
    """A célula vaga da grade, com o número que as barras não dizem.

    Só o Spearman. As barras mostram a ordenação preservada qualitativamente; o
    coeficiente é a versão quantitativa da mesma frase, e é a única coisa da
    figura que o leitor não consegue ler do traço.
    """
    from scipy.stats import spearmanr

    ranked = table.drop(index="HVG", errors="ignore")
    pairs = [spearmanr(ranked[a], ranked[b]).statistic
             for i, a in enumerate(table.columns) for b in table.columns[i + 1:]]

    axis.set_axis_off()
    axis.text(0.06, 0.66, TEXTOS[lang]["spearman"],
              transform=axis.transAxes, fontsize=12, color=MUTED, va="top")
    axis.text(0.06, 0.53, f"+{min(pairs):.3f} to +{max(pairs):.3f}",
              transform=axis.transAxes, fontsize=17, color=INK, va="top",
              fontweight="bold")
    axis.text(0.06, 0.36, TEXTOS[lang]["pares"].format(n=len(pairs)),
              transform=axis.transAxes, fontsize=10.5, color=MUTED, va="top")


def draw_slope(table: pd.DataFrame, args) -> None:
    xs = np.arange(len(table.columns), dtype=float)
    fig, axis = plt.subplots(figsize=(10.6, 7.2), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    for block in table.index:
        ys = table.loc[block].to_numpy(dtype=float)
        colour = colour(block)
        axis.plot(xs, ys, color=colour, linewidth=1.6, alpha=0.55, zorder=2)
        axis.scatter(xs, ys, s=64, color=colour, zorder=4,
                     edgecolor="white", linewidth=0.9)

    # Os quatro blocos de segmentacao ficam a menos de 3pp uns dos outros na
    # coluna do MLP, entao os rotulos colidem. Empurra de cima para baixo com
    # folga fixa e liga o rotulo deslocado ao ponto por um filete.
    anchor = table.iloc[:, 0].to_numpy(dtype=float)
    top = anchor.max() * 1.06
    gap = top * args.label_gap
    placed = anchor.copy()
    for i in range(1, len(placed)):
        placed[i] = min(placed[i], placed[i - 1] - gap)
    for block, y_point, y_label in zip(table.index, anchor, placed):
        colour = colour(block)
        axis.annotate(display(block, args.lang), xy=(xs[0] - 0.09, y_label),
                      fontsize=11, color=colour, ha="right", va="center",
                      zorder=5, annotation_clip=False)
        if abs(y_label - y_point) > gap * 0.15:
            axis.plot([xs[0] - 0.075, xs[0] - 0.02], [y_label, y_point],
                      color=colour, linewidth=0.9, alpha=0.5, zorder=3,
                      clip_on=False)

    axis.axhline(args.chance, color="#b9b8b5", linewidth=1.1,
                 linestyle=(0, (4, 4)), zorder=1)
    axis.text(xs[-1] + 0.10, args.chance, TEXTOS[args.lang]["acaso"], fontsize=10.5,
              color=MUTED, ha="left", va="bottom")

    ratios = table.drop(index="HVG", errors="ignore")
    ratios = ratios.div(ratios[table.columns[0]], axis=0)
    for x, name in zip(xs, table.columns):
        mean, spread = ratios[name].mean(), ratios[name].std(ddof=1)
        tag = "reference" if x == 0 else f"×{mean:.2f} ± {spread:.2f}"
        axis.annotate(tag, xy=(x, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -34), textcoords="offset points",
                      fontsize=10.5, color=MUTED, ha="center")

    axis.set_xlim(xs[0] - 1.25, xs[-1] + 0.55)
    axis.set_ylim(0, top)
    axis.set_xticks(xs)
    axis.set_xticklabels(list(table.columns), fontsize=13, color=INK)
    axis.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.tick_params(axis="y", colors=MUTED, labelsize=12)
    axis.tick_params(axis="x", length=0, pad=10)
    T = TEXTOS[args.lang]
    axis.set_ylabel(T["y"], fontsize=13.5, color=MUTED)
    axis.set_title(f"{T['titulo']}\n{T['sub_slope']} {T['corpus']}",
                   fontsize=14.5, color=INK, loc="left", pad=16)

    fig.subplots_adjust(left=0.175, right=0.965, top=0.845, bottom=0.135)
    fig.savefig(args.output, facecolor=SURFACE)


if __name__ == "__main__":
    main()
