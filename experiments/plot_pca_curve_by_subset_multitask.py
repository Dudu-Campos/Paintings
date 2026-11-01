"""Figura das curvas BAcc(k) por número de modalidades, MultitaskPainting100k.

Painel esquerdo: a família de curvas (uma por subconjunto aninhado em ordem de
Shapley). Painel direito: a progressão destilada -- pico sobre k contra número
de modalidades, com desvio entre folds.

Cor: rampa azul ORDINAL (passos 250-600 da rampa sequencial de referência),
porque a série é ordenada, não categórica -- passos adjacentes próximos são o
que codifica a ordem, e a identidade vai por posição mais rótulo direto. O
último passo (com grafo de similaridade) é qualitativamente diferente dos
outros -- acrescenta um operador, não mais uma modalidade -- então recebe o
vermelho de destaque, validado contra toda a rampa (pior ΔE OKLab: 29,9
visão normal, 17,7 protanopia, 23,9 deuteranopia, 29,2 tritanopia).
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

RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
ACCENT = "#e34948"
SURFACE = "#fbfbfa"
GRID = "#dedede"
INK = "#0b0b0b"
MUTED = "#52514e"

# Nomes de exibicao vem de view_names.display(), a fonte unica (§7.18). Este
# arquivo tinha um mapa proprio -- o QUINTO do repositorio -- que abreviava
# `L(G_felz)` como "L(G_f)" e `Line graph` como "L(G)", reproduzindo as duas
# convencoes para a mesma operacao que o §7.18 existe para consertar.
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT / "experiments"))
from view_names import display  # noqa: E402


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/pca_curve_by_subset_multitask.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/pca_curve_by_subset_multitask.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    steps = sorted(frame["n_modalities"].unique())
    last = max(steps)
    names = {m: frame.loc[frame.n_modalities.eq(m), "modalities"].iloc[0].split("+")
             for m in steps}
    colours = {m: (ACCENT if m == last else RAMP[min(m - 1, len(RAMP) - 1)]) for m in steps}

    figure, (curve_axis, peak_axis) = plt.subplots(
        1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    for axis in (curve_axis, peak_axis):
        style(axis)

    # --- painel 1: família de curvas -------------------------------------
    for m in steps:
        block = frame[frame.n_modalities.eq(m)].groupby("k")["bacc"].mean() * 100
        added = display(names[m][-1])
        curve_axis.plot(block.index, block.values, color=colours[m],
                        linewidth=2.6 if m == last else 2.0,
                        zorder=5 if m == last else 3,
                        solid_capstyle="round")
        if m in (last - 1, last):
            curve_axis.annotate(
                f"{m} blocks" + (" + graph" if m == last else ""),
                xy=(block.index[-1], block.values[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=colours[m], fontsize=10.5,
                fontweight="bold" if m == last else "normal",
                va="center", zorder=6)

    curve_axis.set_xscale("log")
    curve_axis.set_xlabel("k — PCA components", fontsize=11.5, color=MUTED)
    curve_axis.set_ylabel("Balanced accuracy (%)", fontsize=11.5, color=MUTED)
    curve_axis.set_title("Every added representation lifts the whole curve",
                         fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    curve_axis.set_xlim(1.7, 2400)
    # Legenda incremental: numa progressão aninhada o que informa é o que cada
    # passo ACRESCENTA, não a lista cumulativa (que fica longa e se repete).
    handles = [plt.Line2D([0], [0], color=colours[m], linewidth=2.4,
                          label=f"{m}   {'' if m == 1 else '+ '}"
                                f"{display(names[m][-1])}")
               for m in steps]
    curve_axis.legend(handles=handles, frameon=False, fontsize=8.6,
                      loc="upper left", labelcolor=MUTED, handlelength=1.4,
                      borderpad=0.2, labelspacing=0.35,
                      title="representations added", title_fontsize=8.8)

    # --- painel 2: pico sobre k, por número de modalidades ----------------
    peaks, errors = [], []
    for m in steps:
        block = frame[frame.n_modalities.eq(m)]
        best_k = block.groupby("k")["bacc"].mean().idxmax()
        rows = block[block.k.eq(best_k)]
        peaks.append(rows["bacc"].mean() * 100)
        errors.append(rows["bacc"].std() * 100)

    peak_axis.errorbar(steps[:-1], peaks[:-1], yerr=errors[:-1], color=RAMP[4],
                       marker="o", markersize=7, linewidth=2.2, capsize=3,
                       markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    peak_axis.errorbar(steps[-2:], peaks[-2:], yerr=[0, errors[-1]], color=ACCENT,
                       marker="o", markersize=8, linewidth=2.6, capsize=3,
                       markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    peak_axis.annotate(f"+{peaks[-1] - peaks[-2]:.2f} pp",
                       xy=(steps[-1], peaks[-1]), xytext=(-6, 16),
                       textcoords="offset points", color=ACCENT, fontsize=11.5,
                       fontweight="bold", ha="right")
    peak_axis.set_xticks(steps)
    peak_axis.set_xlim(0.4, last + 0.7)
    peak_axis.set_xlabel("number of representations", fontsize=11.5, color=MUTED)
    peak_axis.set_ylabel("Balanced accuracy at best k (%)", fontsize=11.5, color=MUTED)
    peak_axis.set_title("The graph outruns four blocks",
                        fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)

    figure.text(0.005, 0.015,
                "MultitaskPainting100k · 82,824 paintings · 25 styles · LDA probe on PCA components, "
                "5-fold · nested subsets in measured Shapley order · error bars = SD across folds",
                fontsize=8.4, color=MUTED)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    for m, peak, error in zip(steps, peaks, errors):
        print(f"  m={m}: {peak:.2f} ± {error:.2f}  ({'+'.join(names[m])[:60]})")


if __name__ == "__main__":
    main()
