"""UpSet dos acertos por view: quem acerta o quê, e junto com quem.

Venn não serve acima de três conjuntos; UpSet é a forma canônica (Lex et al.,
IEEE TVCG 2014). Cada coluna é um **padrão de acerto** -- o subconjunto exato de
views que classifica aquela imagem corretamente -- e a barra é quantas imagens
têm esse padrão.

Três referências que a figura precisa carregar, porque o tamanho bruto das
barras engana:

  união observada       quantas imagens ao menos uma view acerta
  união se INDEPENDENTES  1 - prod(1 - acc_i). Se as views errassem de forma
                        independente, a união seria MUITO maior; a distância
                        até ela é a medida de erro correlacionado.
  união se ACASO        1 - (1 - 1/C)^n. Com muitas views fracas e poucas
                        classes, o acaso já cobre boa parte -- no WikiArt com
                        19 views e 27 classes, 51%.

Sem essas duas o oráculo da união se lê como complementaridade quando pode ser
aritmética.

A coluna "nenhuma view acerta" entra explicitamente: ela é a fração do corpus
opaca a todo o banco, e some numa figura que só mostre interseções não-vazias.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import displays  # noqa: E402
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
ACCENT, PALE = "#0b7a5a", "#c9cdd1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/view_predictions_multitask.npz")
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--max-views", type=int, default=10,
                        help="mantém as N views mais acuradas; acima disso a "
                             "matriz de pontos vira ruído ilegível")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / f"results/{args.input.stem}_upset.png"

    data = np.load(args.input, allow_pickle=True)
    labels = data["labels"]
    # A fatia de 50d do grafo é decomposição interna de UMA view, não uma
    # segunda view: as duas aparecem sempre no mesmo padrão e ocupariam colunas
    # sem informação. O número comparativo fica no CSV da ablação.
    hidden = {"Similarity graph (class scores)"}
    views = [k for k in data.files if k not in ("labels", "stems") and k not in hidden]
    hit = {v: data[v] == labels for v in views}
    accuracy = {v: hit[v].mean() for v in views}
    views = sorted(views, key=lambda v: -accuracy[v])[:args.max_views]
    matrix = np.column_stack([hit[v] for v in views])

    union = matrix.any(axis=1)
    accs = np.array([accuracy[v] for v in views])
    n_classes = len(np.unique(labels))
    independent = 1 - np.prod(1 - accs)
    by_chance = 1 - (1 - 1 / n_classes) ** len(views)

    # Assinatura de cada imagem = tupla de quais views acertam.
    keys = [tuple(row) for row in matrix]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    empty = counts.pop(tuple([False] * len(views)), 0)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:args.top]

    figure = plt.figure(figsize=(14.2, 8.8))
    # wspace largo: os rótulos das views vivem ENTRE o painel de acurácia e a
    # matriz de pontos, que é onde o UpSet canônico os põe. Com wspace apertado
    # as barras de acurácia cobriam os nomes.
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 4.0), height_ratios=(2.5, 2),
                               wspace=0.30, hspace=0.06)
    bars = figure.add_subplot(grid[0, 1])
    dots = figure.add_subplot(grid[1, 1], sharex=bars)
    sets = figure.add_subplot(grid[1, 0])

    for axis in (bars, dots, sets):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=12)

    sizes = [c for _, c in top]
    bars.bar(range(len(top)), sizes, color=ACCENT, zorder=3, width=0.62)
    for index, value in enumerate(sizes):
        bars.text(index, value * 1.02, f"{100 * value / len(labels):.1f}",
                  ha="center", fontsize=10.5, color=MUTED)
    bars.grid(color=GRID, linewidth=0.7, axis="y", zorder=0)
    bars.set_axisbelow(True)
    bars.set_ylabel("images with this exact pattern", fontsize=13.5, color=MUTED)
    bars.tick_params(labelbottom=False)
    bars.spines["left"].set_color(GRID)
    bars.spines["bottom"].set_visible(False)

    for row in range(len(views)):
        dots.axhspan(row - 0.5, row + 0.5, color="white" if row % 2 else "#f1f1ef",
                     zorder=0, linewidth=0)
    for column, (pattern, _) in enumerate(top):
        active = [r for r, on in enumerate(pattern) if on]
        dots.scatter([column] * len(views), range(len(views)), s=88, color=PALE, zorder=2)
        if active:
            dots.scatter([column] * len(active), active, s=88, color=ACCENT, zorder=3)
            dots.plot([column, column], [min(active), max(active)],
                      color=ACCENT, linewidth=2.2, zorder=2)
    dots.set_yticks(range(len(views)))
    dots.set_yticklabels([])
    dots.set_xticks([])
    dots.set_ylim(len(views) - 0.5, -0.5)
    dots.set_xlabel(f"the {len(top)} most common hit patterns", fontsize=13.5,
                    color=MUTED, labelpad=8)
    for side in ("left", "bottom"):
        dots.spines[side].set_visible(False)

    sets.barh(range(len(views)), [100 * accuracy[v] for v in views],
              color=PALE, zorder=3, height=0.62)
    sets.set_xlim(max(100 * accs) * 1.15, 0)
    sets.invert_xaxis()
    sets.set_xlabel("accuracy (%)", fontsize=12.5, color=MUTED)
    sets.grid(color=GRID, linewidth=0.7, axis="x", zorder=0)
    sets.set_axisbelow(True)
    sets.set_ylim(len(views) - 0.5, -0.5)
    sets.set_yticks(range(len(views)))
    sets.set_yticklabels(displays(views), fontsize=13, color=INK)
    sets.tick_params(axis="y", labelleft=False, labelright=True, length=0, pad=8)
    sets.spines["left"].set_visible(False)
    sets.spines["bottom"].set_color(GRID)

    summary = (f"at least one view correct: {100 * union.mean():.1f}%      "
               f"if views erred independently: {100 * independent:.1f}%      "
               f"if all were chance (1/{n_classes}): {100 * by_chance:.1f}%\n"
               f"no view correct: {100 * empty / len(labels):.1f}% of the corpus"
               f"      best single view: {100 * max(accs):.1f}%")
    figure.text(0.012, 0.955, summary, fontsize=12.5, color=INK, va="top")
    figure.suptitle("Which views get which images right", fontsize=17,
                    fontweight="bold", color=INK, x=0.008, ha="left", y=0.995)
    figure.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.09)
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    print(f"  união {100*union.mean():.2f}%  independência {100*independent:.2f}%  "
          f"acaso {100*by_chance:.2f}%  nenhuma {100*empty/len(labels):.2f}%")


if __name__ == "__main__":
    main()
