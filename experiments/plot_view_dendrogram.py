"""Taxonomia de views a partir do padrão de acerto, com suporte por bootstrap.

Refaz `evaluate_channel_taxonomy` sobre dados melhores: aquele rodou no WikiArt
com as 19 views antigas e usou excesso de falha dupla; aqui são os acertos por
imagem do Multitask com as 12 views atuais, e a similaridade é a **concordância
relativa à independência**.

A normalização é o ponto. Concordância crua (IoU) correlaciona **r = +0,938**
com a acurácia mínima do par -- agrupar por ela produziria uma árvore de
"views boas / médias / ruins", não de famílias. Dividindo pela concordância
esperada sob independência, `a*b/(a+b-a*b)`, o confundimento cai para +0,248.

**Todo dendrograma impõe uma árvore, mesmo em dados que não têm uma.** Com 12
folhas e ligação média, um agrupamento sai sempre; a pergunta é se ele é
reprodutível. Cada nó traz portanto o **suporte por bootstrap**: a fração de
reamostragens das imagens em que aquele mesmo conjunto de folhas reaparece como
um clado. Suporte baixo significa que o ramo é artefato da amostra e não deve
ser lido como família.
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import displays  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
HIDDEN = {"Similarity graph (class scores)"}


def agreement(matrix: np.ndarray) -> np.ndarray:
    """Concordância par a par dividida pela esperada sob independência."""
    size = matrix.shape[1]
    accuracy = matrix.mean(axis=0)
    out = np.ones((size, size))
    for i, j in itertools.combinations(range(size), 2):
        both = (matrix[:, i] & matrix[:, j]).mean()
        either = (matrix[:, i] | matrix[:, j]).mean()
        a, b = accuracy[i], accuracy[j]
        out[i, j] = out[j, i] = (both / either) / ((a * b) / (a + b - a * b))
    return out


def tree_of(matrix: np.ndarray, method: str):
    ratio = agreement(matrix)
    # Distância decrescente na concordância: 1/razão põe pares muito
    # concordantes perto e independentes (razão 1) a distância 1.
    distance = 1.0 / ratio
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method=method), ratio


def clades(tree, size: int) -> set[frozenset]:
    members = {i: frozenset([i]) for i in range(size)}
    found = set()
    for step, (left, right, _, _) in enumerate(tree):
        joined = members[int(left)] | members[int(right)]
        members[size + step] = joined
        if 1 < len(joined) < size:
            found.add(joined)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/view_predictions_multitask.npz")
    parser.add_argument("--method", default="average")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "results/view_dendrogram.png")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    labels = data["labels"]
    views = [k for k in data.files if k not in ("labels", "stems") and k not in HIDDEN]
    views = sorted(views, key=lambda v: -(data[v] == labels).mean())
    hits = np.column_stack([data[v] == labels for v in views])
    size = len(views)

    tree, ratio = tree_of(hits, args.method)
    observed = clades(tree, size)

    rng = np.random.default_rng(0)
    support = {clade: 0 for clade in observed}
    for _ in range(args.bootstrap):
        take = rng.integers(0, len(hits), len(hits))
        resampled, _ = tree_of(hits[take], args.method)
        for clade in clades(resampled, size):
            if clade in support:
                support[clade] += 1
    support = {k: v / args.bootstrap for k, v in support.items()}

    figure, axis = plt.subplots(figsize=(11.0, 7.2))
    axis.set_facecolor(SURFACE)
    drawing = dendrogram(tree, labels=displays(views), orientation="left",
                         color_threshold=0, above_threshold_color="#3a3a3a", ax=axis)
    axis.tick_params(colors=MUTED, labelsize=14)
    axis.set_xlabel("distance  (1 / agreement relative to independence)",
                    fontsize=14, color=MUTED, labelpad=8)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)

    # Anotar o suporte no nó: sem isso o dendrograma afirma uma árvore que os
    # dados podem não sustentar.
    order = drawing["leaves"]
    position = {leaf: 5 + 10 * index for index, leaf in enumerate(order)}
    members = {i: frozenset([i]) for i in range(size)}
    for step, (left, right, height, _) in enumerate(tree):
        left, right = int(left), int(right)
        members[size + step] = members[left] | members[right]
        position[size + step] = (position[left] + position[right]) / 2
        value = support.get(members[size + step])
        if value is not None:
            axis.annotate(f"{value:.2f}", (height, position[size + step]),
                          xytext=(3, 5), textcoords="offset points",
                          fontsize=11.5, color="#d13c3c", fontweight="bold")
    figure.suptitle("View taxonomy from agreement on correct predictions",
                    fontsize=16, fontweight="bold", color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")

    print(f"\nsuporte por bootstrap ({args.bootstrap} reamostragens):")
    for clade, value in sorted(support.items(), key=lambda kv: -kv[1]):
        names = ", ".join(displays([views[i] for i in sorted(clade)]))
        print(f"  {value:.2f}  {names}")
    for k in (2, 3, 4):
        assignment = fcluster(tree, k, criterion="maxclust")
        groups = [", ".join(displays([views[i] for i in np.flatnonzero(assignment == c)]))
                  for c in range(1, k + 1)]
        print(f"\ncorte em {k} grupos:")
        for group in groups:
            print(f"    {group}")


if __name__ == "__main__":
    main()
