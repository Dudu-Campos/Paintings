"""Curva de saturação por número de modalidades, livre de ordenação.

Substitui a figura antiga de "seleção progressiva de grupos de features", que
percorria **uma** ordem gulosa e por isso mede retorno decrescente de forma
quase tautológica: se você adiciona o melhor bloco primeiro, a curva é côncava
por construção.

Aqui cada ponto é a média sobre **todos** os C(8,m) subconjuntos de tamanho m,
extraída do fatorial 2^8 completo. Todas as ordens entram com peso igual, então
a concavidade não pode ser artefato de ordenação -- é a mesma lógica que já
justifica o Shapley exato usado no resto do projeto.

A linha tracejada mostra o caminho guloso (melhor bloco a cada passo) para
comparação: a distância entre ela e a média é o viés de seleção que a figura
antiga não separava.

Saída: results/modality_saturation.png
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

# Slots 1 e 3 da paleta categórica do projeto.
COLORS = {"lda": "#2a78d6", "mlp": "#1baf7a"}
LABELS = {"lda": "LDA", "mlp": "MLP"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d7dbe0"
FACTORS = ("SLIC", "Descriptors", "Hilbert", "Line graph", "Felzenszwalb",
           "Fourier", "Text", "Multiscale")


def greedy_path(mask_bacc: dict[int, float]) -> list[float]:
    """Caminho guloso: a cada passo acrescenta o bloco que mais soma."""
    current, path = 0, [mask_bacc[0]]
    for _ in range(len(FACTORS)):
        candidates = [(mask_bacc[current | (1 << i)], current | (1 << i))
                      for i in range(len(FACTORS)) if not current & (1 << i)]
        best, current = max(candidates)
        path.append(best)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "outputs/full_factorial_ablation_text/factorial_fold_results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/modality_saturation.png")
    parser.add_argument("--paper", action="store_true",
                        help="remove título e rodapé: em paper a legenda "
                             "carrega essa informação e a duplicata rouba espaço")
    args = parser.parse_args()

    detail = pd.read_csv(args.results)
    figure, axis = plt.subplots(figsize=(9.5, 5.4))

    for classifier in ("lda", "mlp"):
        subset = detail[detail.classifier == classifier]
        # Média por máscara entre folds, depois estatísticas entre máscaras.
        by_mask = subset.groupby("mask")["bacc"].mean() * 100
        sizes = by_mask.index.to_series().map(lambda m: bin(m).count("1"))
        grouped = by_mask.groupby(sizes)
        mean = grouped.mean()
        low, high = grouped.quantile(0.10), grouped.quantile(0.90)

        color = COLORS[classifier]
        axis.fill_between(mean.index, low, high, color=color, alpha=0.16, linewidth=0)
        axis.plot(mean.index, mean, color=color, linewidth=2, marker="o", markersize=7,
                  markeredgecolor="white", markeredgewidth=1.4,
                  label=f"{LABELS[classifier]} — mean over all subsets", zorder=3)
        greedy = greedy_path(by_mask.to_dict())
        axis.plot(range(len(greedy)), greedy, color=color, linewidth=1.4,
                  linestyle=(0, (5, 3)), alpha=0.75, zorder=2,
                  label=f"{LABELS[classifier]} — greedy path")
        axis.annotate(f"{mean.iloc[-1]:.1f}", (mean.index[-1], mean.iloc[-1]),
                      textcoords="offset points", xytext=(10, -3), fontsize=10,
                      color=color, fontweight="bold")

    axis.grid(True, alpha=0.28, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xticks(range(len(FACTORS) + 1))
    axis.set_xlabel("Number of active views", fontsize=10, color=MUTED)
    axis.set_ylabel("Balanced accuracy (%)", fontsize=10, color=MUTED)
    if not args.paper:
        axis.set_title("Saturation by view count, independent of ordering",
                   fontsize=12.5, color=INK, loc="left", pad=12)
    legend = axis.legend(loc="lower right", frameon=True, fontsize=9)
    legend.get_frame().set_edgecolor(GRID)
    legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_color(MUTED)
    if not args.paper:
        figure.text(0.011, 0.015,
                f"WikiArt, 79,207 images, 27 styles · full 2^8 factorial (256 masks, 5 folds) "
                "· band = 10–90th percentile across subsets of the same size",
                fontsize=8.5, color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white",
                   bbox_inches="tight" if args.paper else None)
    print(f"salvo: {args.output}")

    for classifier in ("lda", "mlp"):
        subset = detail[detail.classifier == classifier]
        by_mask = subset.groupby("mask")["bacc"].mean() * 100
        sizes = by_mask.index.to_series().map(lambda m: bin(m).count("1"))
        mean = by_mask.groupby(sizes).mean()
        greedy = greedy_path(by_mask.to_dict())
        gap = np.array(greedy) - mean.to_numpy()
        print(f"{classifier}: viés guloso por tamanho (pp) = "
              f"{np.round(gap, 2).tolist()}")


if __name__ == "__main__":
    main()
