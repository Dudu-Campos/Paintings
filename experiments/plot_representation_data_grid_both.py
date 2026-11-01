"""Grade representação × dados nos dois classificadores, lado a lado.

Duas mensagens numa figura só:

  o padrão replica    o expoente do erro cresce com o número de modalidades
                      tanto na sonda LDA quanto no MLP de produção -- não é
                      artefato do classificador linear
  o MLP é mais rápido a taxa dele é ~3x a do LDA em todo m, e com 6 modalidades
                      chega perto da CNN treinada do zero (−0,149)

Ajuste do expoente: **só de 10k para cima**. Abaixo disso o MLP é instável --
com uma modalidade ele chega a piorar de 2.500 para 5.000 imagens, porque
sobreajusta antes de convergir. Os pontos aparecem todos no gráfico (esconder
seria maquiar), mas a reta ajustada começa onde o regime é estável.

Saída: results/representation_data_grid_both.png
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
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d7dbe0"
BLUE, GREEN = "#2a78d6", "#1baf7a"
FIT_FROM = 10000


def thousands(value: float) -> str:
    return f"{value / 1000:.1f}".rstrip("0").rstrip(".") + "k"


def load(path: Path) -> pd.DataFrame:
    detail = pd.read_csv(path)
    requested = np.array([2500, 5000, 10000, 20000, 40000])
    def group(value: int) -> int:
        near = requested[np.abs(requested - value) <= 0.02 * value]
        return int(near[0]) if len(near) else 63365
    detail["n_train"] = detail.n_train.map(group)
    if detail.k.nunique() > 1:      # curva do LDA: pico sobre componentes
        detail = detail.loc[detail.groupby(
            ["n_modalities", "fold", "n_train"]).bacc.idxmax()]
    frame = (detail.groupby(["n_modalities", "n_train"], as_index=False)
             .agg(bacc=("bacc", "mean"), std=("bacc", "std")))
    frame[["bacc", "std"]] *= 100
    return frame


def draw(axis, curves, ramp, counts):
    exponents, endpoints = {}, []
    for index, count in enumerate(counts):
        block = curves[curves.n_modalities == count].sort_values("n_train")
        n = block.n_train.to_numpy(dtype=float)
        error = 100 - block.bacc.to_numpy()
        colour = ramp(0.30 + 0.62 * index / max(len(counts) - 1, 1))
        axis.errorbar(n, error, yerr=block["std"].fillna(0), color=colour,
                      linewidth=1.9, marker="o", markersize=5.5,
                      markeredgecolor="white", markeredgewidth=1,
                      capsize=2.5, elinewidth=1, zorder=3 + index)
        stable = n >= FIT_FROM
        slope, intercept = np.polyfit(np.log(n[stable]), np.log(error[stable]), 1)
        exponents[count] = slope
        endpoints.append((error[-1], n[-1], count, colour))

    # Separa rótulos que ficariam colados: no MLP os finais de m=4,5,6 estão a
    # 0,3pp um do outro e os números se sobrepõem.
    endpoints.sort()
    minimum = 0.022                      # separação mínima em log10 do eixo y
    placed = []
    for error_end, n_end, count, colour in endpoints:
        target = np.log10(error_end)
        if placed and target - placed[-1] < minimum:
            target = placed[-1] + minimum
        placed.append(target)
        axis.annotate(f"{count}", (n_end, 10 ** target), textcoords="offset points",
                      xytext=(8, -3), fontsize=9, color=colour, fontweight="bold",
                      annotation_clip=False)
    return exponents


def style(axis, ticks, ylabel=None):
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xticks(ticks); axis.set_xticklabels([thousands(t) for t in ticks])
    axis.set_yticks([50, 57, 65, 72, 80])
    axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.minorticks_off()
    axis.grid(True, alpha=0.28, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xlabel("Training images", fontsize=10, color=MUTED)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=10, color=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lda", type=Path,
                        default=ROOT / "outputs/representation_data_grid/fold_results.csv")
    parser.add_argument("--mlp", type=Path,
                        default=ROOT / "outputs/representation_data_grid_mlp/fold_results.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/representation_data_grid_both.png")
    parser.add_argument("--max-modalities", type=int, default=6)
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    frames = {}
    for name, path in (("LDA", args.lda), ("MLP", args.mlp)):
        frame = load(path)
        frames[name] = frame[frame.n_modalities <= args.max_modalities]
    counts = sorted(frames["LDA"].n_modalities.unique())

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    exponents = {}
    for axis, (name, ramp) in zip(axes, (("LDA", plt.get_cmap("Blues")),
                                         ("MLP", plt.get_cmap("Greens")))):
        exponents[name] = draw(axis, frames[name], ramp, counts)
        ticks = sorted(frames[name].n_train.unique())
        style(axis, ticks, "Balanced error (%)" if name == "LDA" else None)
        axis.set_title(f"{name} probe" if name == "LDA" else "MLP (production)",
                       fontsize=11.5, color=INK, loc="left", pad=10)
    # Sem rótulo "modalities" nos painéis: os números colados às curvas já
    # carregam a cor da série, e o inset rotula o eixo.

    inset = figure.add_axes((0.565, 0.235, 0.155, 0.26))
    for name, colour in (("LDA", BLUE), ("MLP", GREEN)):
        inset.plot(list(exponents[name]), [-v for v in exponents[name].values()],
                   color=colour, linewidth=1.8, marker="o", markersize=4.5,
                   markeredgecolor="white", markeredgewidth=0.9, label=name)
    inset.set_title("error exponent", fontsize=8.5, color=MUTED, pad=4)
    inset.set_xlabel("views", fontsize=7.5, color=MUTED, labelpad=1)
    inset.tick_params(colors=MUTED, labelsize=7)
    inset.grid(True, alpha=0.25, color=GRID, linewidth=0.7)
    inset.set_axisbelow(True)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        inset.spines[side].set_color(GRID)
    legend = inset.legend(fontsize=7.5, frameon=False, loc="lower right")
    for text in legend.get_texts():
        text.set_color(MUTED)

    if not args.paper:
        figure.text(0.008, 0.036,
                    "WikiArt, 27 styles · 5 folds · training subsampled, test fold never reduced · "
                    "LDA: peak over PCA components; MLP: one branch per modality at native width",
                    fontsize=8.5, color=MUTED)
        figure.text(0.008, 0.011,
                    f"Exponents fitted from {thousands(FIT_FROM)} upward: below that the MLP is "
                    "unstable (with one modality it degrades from 2.5k to 5k). Multiscale excluded.",
                    fontsize=8.5, color=MUTED)
    figure.subplots_adjust(left=0.062, right=0.975, top=0.925, bottom=0.20, wspace=0.06)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white",
                   bbox_inches="tight" if args.paper else None)
    print(f"salvo: {args.output}")
    for name in ("LDA", "MLP"):
        print(f"  {name}: " + "  ".join(f"m={m}:{v:+.4f}" for m, v in exponents[name].items()))


if __name__ == "__main__":
    main()
