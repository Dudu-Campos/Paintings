"""Duas leis de escala em dados: CNN do zero contra features hand-designed.

A CNN parte perto do acaso com pouco dado e sobe íngreme; as features
projetadas partem alto e sobem devagar. As duas retas em log-log se cruzam, e o
ponto de cruzamento é a escala de dados em que deixa de valer a pena projetar
features à mão -- que é a versão quantitativa do argumento de regime do paper.

Dados:
  CNN         `outputs/cnn_scratch_n{N}/fold_metrics.csv` mais o ponto de
              79.936 já existente em `outputs/cnn_scratch/`. Treino
              subamostrado, fold de teste sempre intacto.
  hand-des.   `outputs/capacity_vs_trainsize/` -- pico de BAcc sobre k, sonda
              LDA no pool vision-only de 1.576d.

Nota de leitura que a figura carrega no rodapé: a curva hand-designed é a da
**sonda LDA**, mais fraca que o pipeline de produção (MLP, 1.154d, 54,42% no
mesmo treino). O cruzamento com o pipeline de produção fica bem mais à direita.

Saída: results/cnn_vs_handdesigned_scaling.png
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BLUE, RED = "#2a78d6", "#e34948"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d7dbe0"
N_CLASSES = 27


def thousands(value: float) -> str:
    text = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
    return f"{text}k"


def load_cnn() -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(ROOT / "outputs/cnn_scratch_n*/fold_metrics.csv")):
        size = int(re.search(r"_n(\d+)/", path).group(1))
        metrics = pd.read_csv(path)
        rows.append((size, 100 * metrics.balanced_accuracy.mean(),
                     100 * metrics.balanced_accuracy.std(), len(metrics)))
    metrics = pd.read_csv(ROOT / "outputs/cnn_scratch/fold_metrics.csv")
    rows.append((round(79936 * 0.8), 100 * metrics.balanced_accuracy.mean(),
                 100 * metrics.balanced_accuracy.std(), len(metrics)))
    frame = pd.DataFrame(rows, columns=["n", "bacc", "std", "folds"])
    return frame[frame.folds >= 5].sort_values("n").reset_index(drop=True)


def load_handdesigned() -> pd.DataFrame:
    detail = pd.read_csv(ROOT / "outputs/capacity_vs_trainsize/fold_results.csv")
    requested = np.array([2500, 5000, 10000, 20000, 40000])
    def group(value: int) -> int:
        near = requested[np.abs(requested - value) <= 0.02 * value]
        return int(near[0]) if len(near) else int(round(value, -2))
    detail["n_train"] = detail.n_train.map(group)
    peaks = detail.loc[detail.groupby(["fold", "n_train"]).bacc.idxmax()]
    frame = (peaks.groupby("n_train", as_index=False)
             .agg(bacc=("bacc", "mean"), std=("bacc", "std"), folds=("fold", "nunique")))
    frame[["bacc", "std"]] *= 100
    return frame.rename(columns={"n_train": "n"}).sort_values("n").reset_index(drop=True)


def fit(frame: pd.DataFrame) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(frame.n.astype(float)), np.log(frame.bacc), 1)
    return slope, intercept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--error", action="store_true",
                        help="eixo y em erro balanceado (100-BAcc), a forma padrão "
                             "das scaling laws -- a curva cai em vez de subir")
    parser.add_argument("--paper", action="store_true",
                        help="remove título e rodapé: em paper a legenda "
                             "carrega essa informação e a duplicata rouba espaço")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("results/cnn_vs_handdesigned_scaling"
                              + ("_error" if args.error else "") + ".png")

    cnn, hand = load_cnn(), load_handdesigned()
    if args.error:
        # Erro balanceado: as scaling laws clássicas ajustam loss/erro, não
        # acurácia. Acurácia é limitada em [0,1] e satura, o que infla o
        # expoente da curva mais baixa.
        for frame in (cnn, hand):
            frame["bacc"] = 100 - frame["bacc"]
    slope_cnn, intercept_cnn = fit(cnn)
    slope_hand, intercept_hand = fit(hand)
    crossing = float(np.exp((intercept_hand - intercept_cnn) / (slope_cnn - slope_hand)))

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    grid = np.logspace(np.log10(2000), np.log10(1.4e5), 200)

    # Faixa entre o último ponto medido e o cruzamento ajustado: é onde a
    # previsão vive sem dado por baixo.
    axis.axvspan(min(crossing, 6.4e4), max(crossing, 6.4e4),
                 color=MUTED, alpha=0.07, linewidth=0, zorder=0)
    axis.plot(grid, np.exp(intercept_hand) * grid ** slope_hand, color=BLUE,
              linewidth=1.1, linestyle=(0, (5, 3)), alpha=0.55, zorder=1)
    axis.plot(grid, np.exp(intercept_cnn) * grid ** slope_cnn, color=RED,
              linewidth=1.1, linestyle=(0, (5, 3)), alpha=0.55, zorder=1)

    for frame, color, label in ((hand, BLUE, "Hand-designed features (LDA probe)"),
                                (cnn, RED, "ResNet-50 from scratch")):
        axis.errorbar(frame.n, frame.bacc, yerr=frame["std"].fillna(0), color=color,
                      linewidth=2, marker="o", markersize=7.5, markeredgecolor="white",
                      markeredgewidth=1.4, capsize=3, elinewidth=1.2, label=label, zorder=3)

    chance = 100 - 100 / N_CLASSES if args.error else 100 / N_CLASSES
    axis.axhline(chance, color=MUTED, linestyle=":", linewidth=1, zorder=1)
    axis.annotate(f"chance ({chance:.1f}%)", (2100, chance),
                  textcoords="offset points", xytext=(2, 4 if not args.error else -12),
                  fontsize=8, color=MUTED)

    # Rótulos de expoente sobre as próprias retas, em espaço vazio -- ao lado do
    # último ponto eles colidiam entre si e com os marcadores.
    def on_line(x, slope, intercept, dy):
        return x, np.exp(intercept) * x ** slope * dy

    lift = 0.90 if args.error else 1.13
    x_hand, y_hand = on_line(26000, slope_hand, intercept_hand, lift)
    axis.annotate(f"∝ n$^{{{slope_hand:.3f}}}$", (x_hand, y_hand), ha="center",
                  fontsize=11, color=BLUE, fontweight="bold")
    x_cnn, y_cnn = on_line(11000, slope_cnn, intercept_cnn, 1.12 if args.error else 0.70)
    axis.annotate(f"∝ n$^{{{slope_cnn:.3f}}}$", (x_cnn, y_cnn), ha="center",
                  fontsize=11, color=RED, fontweight="bold")
    axis.annotate(f"crossover ≈ {crossing / 1000:.0f}k",
                  (crossing * 1.02, 96.5 if args.error else 4.3),
                  ha="center", fontsize=9, color=MUTED)

    axis.set_xscale("log"); axis.set_yscale("log")
    ticks = [2500, 5000, 10000, 20000, 40000, 80000]
    axis.set_xticks(ticks); axis.set_xticklabels([thousands(t) for t in ticks])
    axis.set_yticks([55, 65, 75, 85, 95] if args.error else [4, 8, 16, 32, 48])
    axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.minorticks_off()
    axis.set_xlim(2100, 1.35e5)
    axis.set_ylim(50, 100) if args.error else axis.set_ylim(3.2, 62)
    axis.grid(True, alpha=0.28, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xlabel("Training images", fontsize=10, color=MUTED)
    axis.set_ylabel("Balanced error (%)" if args.error else "Balanced accuracy (%)",
                    fontsize=10, color=MUTED)
    ratio = abs(slope_cnn / slope_hand)
    if not args.paper:
        axis.set_title((f"The CNN reduces error {ratio:.1f}× faster with data — "
                    f"fitted crossover at ~{crossing / 1000:.0f}k")
                   if args.error else
                   "Two data scaling laws: CNN from scratch vs hand-designed features",
                   fontsize=12.5, color=INK, loc="left", pad=12)
    legend = axis.legend(loc="lower left" if args.error else "upper left",
                         frameon=True, fontsize=9.5)
    legend.get_frame().set_edgecolor(GRID); legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_color(MUTED)

    if not args.paper:

        figure.text(0.011, 0.040,
                "WikiArt, 27 styles · 5 folds per point · training subsampled, test fold never reduced",
                fontsize=8.5, color=MUTED)
    if not args.paper:
        figure.text(0.011, 0.012,
                "Hand-designed = LDA probe over 1,576d · y in balanced error, the standard scaling-law form "
                "(accuracy saturates and distorts the exponent).",
                fontsize=8.5, color=MUTED)
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white",
                   bbox_inches="tight" if args.paper else None)
    print(f"salvo: {args.output}")
    print(f"CNN  ∝ n^{slope_cnn:.4f} | hand ∝ n^{slope_hand:.4f} | cruzamento {crossing:,.0f}")


if __name__ == "__main__":
    main()
