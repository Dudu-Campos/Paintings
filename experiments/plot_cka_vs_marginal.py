"""Análogo da figura do projeto pai, no caso 1: CKA vs ganho medido, sonda LDA.

Um ponto por (dataset × bloco), n=32. O eixo x é o CKA do bloco contra a união
dos outros -- o análogo do "quanto a conversão reescreve o que já havia", com a
diferença de que aqui não existe representação nativa (nenhum bloco nosso parte
do pixel cru), então a referência é o pool.

O eixo y é o marginal de deixar-um-de-fora, que é a contribuição que casa
conceitualmente com esse CKA: `BAcc(todos) − BAcc(todos \\ i)`.

O tamanho do ponto codifica a dimensão do bloco de propósito, e não por enfeite:
o LOO correlaciona +0,476 (p=0,006) com log(dimensões), mais forte do que com a
qualidade isolada (+0,415). Esconder isso deixaria a figura sugerindo mais do
que ela sustenta -- Multiscale tem 1.032d contra 57 do Hilbert.

Texto em inglês, como todas as figuras do projeto.

Saída: results/cka_vs_marginal.png e _paper.png (sem título, para a legenda
carregar o texto).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
# Paleta categórica fixa do projeto (slots 1, 3, 4, 8 do skill `dataviz`).
COLOURS = {"wikiart": "#2a78d6", "pandora": "#e34948",
           "rijksdraw": "#1baf7a", "rijksmuseum": "#eda100"}
TITLES = {"wikiart": "WikiArt (79k, 27 styles)",
          "pandora": "Pandora (18k, 18 styles)",
          "rijksdraw": "Rijksmuseum drawings (48k, 11 periods)",
          "rijksmuseum": "Rijksmuseum paintings (4.9k, 10 periods)"}


def partial_spearman(x, y, controls):
    stack = np.column_stack([np.asarray(c, float) for c in controls]
                            + [np.ones(len(x))])
    residual = lambda v: v - stack @ np.linalg.lstsq(stack, v, rcond=None)[0]
    return spearmanr(residual(np.asarray(x, float)), residual(np.asarray(y, float)))


def panel(axis, group, column, dataset) -> None:
    """Um dataset, oito blocos. Facetar existe porque os 32 pontos juntos
    empilham rótulo em cima de rótulo justamente na região densa, que é onde a
    ausência de relação precisa ser visível."""
    rho, p_value = spearmanr(group[column], group.loo_marginal)
    axis.axhline(0, color="#999999", linewidth=0.9, zorder=1)
    axis.scatter(group[column], group.loo_marginal,
                 s=12 * np.sqrt(group.dims), alpha=0.85,
                 color=COLOURS[dataset], edgecolor="white", linewidth=0.9,
                 zorder=3)
    for row in group.itertuples(index=False):
        axis.annotate(row.block, (getattr(row, column), row.loo_marginal),
                      textcoords="offset points",
                      xytext=(0, 7 + 0.42 * np.sqrt(row.dims)),
                      ha="center", fontsize=7.4, color="#444444", zorder=4)
    axis.set_xscale("log")
    axis.set_title(f"{TITLES[dataset]}   ·   Spearman {rho:+.2f} (p={p_value:.2f})",
                   fontsize=10, color="#333333")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.margins(x=0.16, y=0.20)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def build(table: pd.DataFrame, output: Path, column: str, label: str,
          with_title: bool) -> None:
    order = ["wikiart", "rijksdraw", "pandora", "rijksmuseum"]
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))
    for axis, dataset in zip(axes.ravel(), order):
        panel(axis, table[table.dataset == dataset], column, dataset)
    for axis in axes[1]:
        axis.set_xlabel(f"{label}  —  how much the block restates the rest")
    for axis in axes[:, 0]:
        axis.set_ylabel("Δ balanced accuracy when removed (pp, LDA)")

    rho, p_value = spearmanr(table[column], table.loo_marginal)
    controlled = partial_spearman(table[column], table.loo_marginal,
                                  [table.solo_bacc, np.log(table.dims)])
    quality = spearmanr(table[column], table.solo_bacc)
    if with_title:
        figure.suptitle(
            "Representational similarity does not predict measured gain\n"
            f"pooled n={len(table)}:  Spearman {rho:+.3f} (p={p_value:.3f})  ·  "
            f"with solo accuracy {quality[0]:+.3f} (p={quality[1]:.4f}), the confound  ·  "
            f"controlling for it and for block size {controlled[0]:+.3f} "
            f"(p={controlled[1]:.2f})\nmarker area ∝ block size",
            fontsize=11.5)
        figure.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"salvo: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", type=Path,
                        default=ROOT / "results/cka_vs_marginal/pooled.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cka_vs_marginal.png")
    args = parser.parse_args()

    table = pd.read_csv(args.pooled)
    build(table, args.output, "cka_raw", "linear CKA with the rest", True)
    build(table, args.output.with_name(args.output.stem + "_paper.png"),
          "cka_raw", "linear CKA with the rest", False)
    build(table, args.output.with_name(args.output.stem + "_whitened.png"),
          "cka_whitened", "whitened CKA with the rest", True)


if __name__ == "__main__":
    main()
