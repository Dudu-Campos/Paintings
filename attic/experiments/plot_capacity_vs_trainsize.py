"""Figura de `evaluate_capacity_vs_trainsize.py`: pico de BAcc e k* contra dados.

Dois painéis, um eixo cada (nunca eixo duplo):
  esquerda  pico de BAcc contra tamanho do treino -- a lei de escala em dados
  direita   k* (nº de componentes onde o pico ocorre) contra tamanho do treino,
            com o tamanho do pool como linha de referência

O painel da direita é o que carrega a mensagem: quando k* encosta no tamanho do
pool, o gargalo deixou de ser dados e passou a ser representação.

Pontos por fold aparecem como marcas pequenas atrás da média, e a contagem de
folds é anotada -- enquanto a grade não fecha, alguns tamanhos têm menos folds
que outros e isso não pode ficar escondido.
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

# Slots 1 e 8 da paleta categórica do skill dataviz (convenção do projeto).
BLUE, RED = "#2a78d6", "#e34948"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d7dbe0"
POOL_DIMENSIONS = 1576          # pool vision-only (máscara 191)


def thousands(value: float) -> str:
    text = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
    return f"{text}k"


def style(axis, ticks) -> None:
    axis.grid(True, alpha=0.28, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xscale("log")
    # Ticks só nos tamanhos medidos: os menores do log colidiam e não eram dados.
    axis.set_xticks(ticks)
    axis.set_xticklabels([thousands(t) for t in ticks])
    axis.minorticks_off()
    axis.set_xlabel("Training images", fontsize=10, color=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "outputs/capacity_vs_trainsize/fold_results.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/capacity_vs_trainsize.png")
    parser.add_argument("--linear-y", action="store_true",
                        help="y linear nos dois painéis (padrão: log-log)")
    parser.add_argument("--paper", action="store_true",
                        help="remove título e rodapé: em paper a legenda "
                             "carrega essa informação e a duplicata rouba espaço")
    args = parser.parse_args()
    log_y = not args.linear_y

    detail = pd.read_csv(args.results)
    # O caso "treino completo" grava o tamanho realizado, que difere alguns
    # exemplos entre folds -- agrupa por tamanho pedido, senão o mesmo ponto
    # vira dois.
    requested = np.array([2500, 5000, 10000, 20000, 40000])
    def group_size(n: int) -> int:
        near = requested[np.abs(requested - n) <= 0.02 * n]
        return int(near[0]) if len(near) else int(round(n, -2))
    detail["n_train"] = detail["n_train"].map(group_size)

    peaks = detail.loc[detail.groupby(["fold", "n_train"])["bacc"].idxmax()].copy()
    peaks["bacc"] *= 100
    summary = (peaks.groupby("n_train", as_index=False)
               .agg(bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
                    k_mean=("k", "mean"), k_std=("k", "std"),
                    folds=("fold", "nunique"))
               .sort_values("n_train"))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- esquerda: pico de BAcc -------------------------------------------
    left = axes[0]
    left.scatter(peaks["n_train"], peaks["bacc"], s=14, color=BLUE, alpha=0.28,
                 linewidths=0, zorder=2)
    left.errorbar(summary["n_train"], summary["bacc_mean"],
                  yerr=summary["bacc_std"].fillna(0), color=BLUE, linewidth=2,
                  marker="o", markersize=7, markeredgecolor="white",
                  markeredgewidth=1.4, capsize=3, elinewidth=1.2, zorder=3)
    ticks = summary["n_train"].tolist()
    style(left, ticks)
    if log_y:
        # Log-log: a inclinação é o expoente da lei de potência.
        left.set_yscale("log")
        left.set_yticks([28, 32, 36, 40, 44])
        left.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        left.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    # Lei de potência BAcc = A*n^b: em log-log é reta, e a inclinação é b.
    slope, intercept = np.polyfit(np.log(summary["n_train"]), np.log(summary["bacc_mean"]), 1)
    fitted = np.exp(intercept) * summary["n_train"].astype(float) ** slope
    residual = 1 - ((summary["bacc_mean"] - fitted) ** 2).sum() / (
        (summary["bacc_mean"] - summary["bacc_mean"].mean()) ** 2).sum()
    left.plot(summary["n_train"], fitted, color=INK, linewidth=1, linestyle=(0, (5, 3)),
              alpha=0.5, zorder=1)
    left.annotate(f"BAcc ∝ n$^{{{slope:.3f}}}$   R² = {residual:.3f}",
                  (0.04, 0.90), xycoords="axes fraction", fontsize=9.5, color=INK)

    left.set_ylabel("Peak balanced accuracy (%)", fontsize=10, color=MUTED)
    left.set_title("Accuracy at the best k", fontsize=11.5, color=INK, loc="left", pad=10)
    for row in summary.itertuples(index=False):
        # Contagem de folds só onde a grade ainda não fechou, junto do valor.
        text = f"{row.bacc_mean:.1f}" + ("" if row.folds == 5 else f"  ({row.folds}f)")
        left.annotate(text, (row.n_train, row.bacc_mean),
                      textcoords="offset points", xytext=(0, 12), ha="center",
                      fontsize=8.5, color=MUTED)

    # --- direita: k* com o pool como referência ---------------------------
    right = axes[1]
    right.axhline(POOL_DIMENSIONS, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    right.annotate(f"pool = {POOL_DIMENSIONS}d", (summary["n_train"].min(), POOL_DIMENSIONS),
                   textcoords="offset points", xytext=(2, 6), fontsize=8.5, color=MUTED)
    right.scatter(peaks["n_train"], peaks["k"], s=14, color=RED, alpha=0.28,
                  linewidths=0, zorder=2)
    right.errorbar(summary["n_train"], summary["k_mean"],
                   yerr=summary["k_std"].fillna(0), color=RED, linewidth=2,
                   marker="o", markersize=7, markeredgecolor="white",
                   markeredgewidth=1.4, capsize=3, elinewidth=1.2, zorder=3)
    style(right, ticks)
    right.set_ylabel("k* (components at peak)", fontsize=10, color=MUTED)
    right.set_title("Where the optimum sits", fontsize=11.5, color=INK, loc="left", pad=10)
    if log_y:
        right.set_yscale("log")
        right.set_yticks([64, 128, 256, 512, 1024, POOL_DIMENSIONS])
        right.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        right.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        right.set_ylim(56, POOL_DIMENSIONS * 1.35)
    else:
        right.set_ylim(0, POOL_DIMENSIONS * 1.12)

    partial = summary["folds"].min() < 5
    if not args.paper:
        figure.suptitle(
        "The bottleneck shifts from data to representation as training grows"
        + ("   ·   partial: (Nf) marks sizes with fewer than 5 folds" if partial else ""),
        fontsize=12.5, color=INK, x=0.011, ha="left", y=0.985)
    if not args.paper:
        figure.text(0.011, 0.015,
                "WikiArt, 27 styles · vision-only pool of 1,576d · LDA probe over PCA components "
                "· test fold never reduced",
                fontsize=8.5, color=MUTED, ha="left")
    figure.tight_layout(rect=(0, 0.035, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white",
                   bbox_inches="tight" if args.paper else None)
    print(f"salvo: {args.output}")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
