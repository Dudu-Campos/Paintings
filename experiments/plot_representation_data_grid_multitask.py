"""Grade representação x dados, MultitaskPainting100k.

Erro balanceado contra tamanho de treino, uma curva por número de
representações aninhadas em ordem de Shapley medida. Inset: o expoente da lei
de potência por número de representações.

Duas escolhas de eixo que não são cosméticas:

* **log-log**, não lin-log. Uma lei de potência só vira reta em log-log, então
  é a única parametrização em que o expoente é a inclinação que se vê. Em
  lin-log as curvas parecem deslocadas mesmo quando os expoentes diferem, e a
  afirmação do paper passa a depender de acreditar no inset.
* **erro**, não acurácia. Convenção do achado 18: acurácia satura e deforma o
  expoente (no WikiArt o ajuste em acurácia dava cruzamento em 50k, que os
  próprios dados contradiziam; em erro dava 97k, consistente).

O último passo é o grafo de similaridade -- ainda vision-only, mas um operador
sobre as representações existentes e não uma medição nova da imagem. Por isso
recebe cor de destaque: a pergunta que ele responde é diferente.
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

SHORT = {"Felzenszwalb": "Felz", "Descriptors": "Desc", "SLIC": "SLIC",
         "L(G_felz)": "L(G_f)", "Line graph": "L(G)", "Hilbert": "Hilb",
         "Fourier": "Four", "Gabor": "Gabor", "Similarity graph": "graph"}


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, which="major", zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/representation_data_grid_multitask.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/representation_data_grid_multitask.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    # O treino completo difere por fold (66.259 vs 66.260, resto da divisão em
    # 5 partes): unificar, senão vira duas colunas quase idênticas na grade.
    frame["size"] = np.where(frame.train_size > 60_000, 66_260, frame.train_size)
    frame["error"] = (1 - frame["bacc"]) * 100

    steps = sorted(frame["n_modalities"].unique())
    last = max(steps)
    names = {m: frame.loc[frame.n_modalities.eq(m), "modalities"].iloc[0].split("+")
             for m in steps}
    colours = {m: (ACCENT if m == last else RAMP[min(m - 1, len(RAMP) - 1)]) for m in steps}

    # Expoente por fold: o ajuste é feito em cada fold e depois agregado, para
    # a barra de erro ser incerteza empírica e não do ajuste médio.
    exponents, spreads = {}, {}
    for m in steps:
        per_fold = []
        for fold in sorted(frame.fold.unique()):
            block = frame[frame.n_modalities.eq(m) & frame.fold.eq(fold)].sort_values("size")
            per_fold.append(-np.polyfit(np.log(block["size"]), np.log(block["error"]), 1)[0])
        exponents[m], spreads[m] = float(np.mean(per_fold)), float(np.std(per_fold, ddof=1))

    figure, axis = plt.subplots(figsize=(10.6, 6.6))
    style(axis)

    for m in steps:
        curve = frame[frame.n_modalities.eq(m)].groupby("size")["error"].mean()
        axis.plot(curve.index, curve.values, color=colours[m],
                  linewidth=2.7 if m == last else 2.0,
                  marker="o", markersize=5.5 if m == last else 4.4,
                  markeredgecolor=SURFACE, markeredgewidth=1.1,
                  zorder=5 if m == last else 3, solid_capstyle="round")
        axis.annotate(f"{m}" + ("  + graph" if m == last else ""),
                      xy=(curve.index[-1], curve.values[-1]),
                      xytext=(8, 0), textcoords="offset points",
                      color=colours[m], fontsize=10.5,
                      fontweight="bold" if m == last else "normal",
                      va="center", zorder=6)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([2500, 5000, 10000, 20000, 40000, 66260])
    axis.set_xticklabels(["2.5k", "5k", "10k", "20k", "40k", "66k"])
    axis.set_yticks([60, 65, 70, 75])
    axis.set_yticklabels(["60", "65", "70", "75"])
    axis.minorticks_off()
    axis.set_xlim(2200, 108_000)
    axis.set_xlabel("training images", fontsize=12, color=MUTED)
    axis.set_ylabel("balanced error (%)", fontsize=12, color=MUTED)
    axis.set_title("More representations make data pay off faster",
                   fontsize=15, fontweight="bold", color=INK, loc="left", pad=30)
    axis.text(0, 1.017,
              "log–log, so the power-law exponent is the visible slope — the curves fan out, they are not shifted copies",
              transform=axis.transAxes, fontsize=10.2, color=MUTED)

    # --- inset: expoente por número de representações --------------------
    # Canto superior direito: as curvas descem, então é a região vazia.
    inset = figure.add_axes((0.575, 0.575, 0.275, 0.255), zorder=10)
    style(inset)
    # Fundo opaco: sem isso as curvas 1 e 2 passam por trás do inset.
    inset.patch.set_facecolor("#ffffff")
    inset.patch.set_alpha(1.0)
    for side in ("left", "bottom"):
        inset.spines[side].set_color("#c9c9c9")
    solid = [m for m in steps if m != last]
    inset.errorbar(solid, [exponents[m] for m in solid],
                   yerr=[spreads[m] for m in solid], color=RAMP[5],
                   marker="o", markersize=5, linewidth=2.0, capsize=2.5,
                   markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    inset.errorbar([solid[-1], last], [exponents[solid[-1]], exponents[last]],
                   yerr=[0, spreads[last]], color=ACCENT, marker="o",
                   markersize=6, linewidth=2.4, capsize=2.5,
                   markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
    inset.set_xticks(steps)
    inset.set_xlabel("representations", fontsize=9.5, color=MUTED, labelpad=2)
    inset.set_title("error exponent", fontsize=10, color=MUTED, loc="left", pad=5)
    inset.tick_params(labelsize=8.5)
    inset.text(0.035, 0.94,
               f"graph: +{exponents[last] - exponents[solid[-1]]:.4f}",
               transform=inset.transAxes, color=ACCENT, fontsize=9.5,
               fontweight="bold", va="top", ha="left")
    inset.set_xlim(0.3, last + 0.7)

    figure.text(0.005, 0.022,
                "MultitaskPainting100k · 82,824 paintings · 25 styles · LDA probe, peak over PCA components · 5 folds · "
                "training subsampled, test fold never reduced",
                fontsize=8.6, color=MUTED)
    figure.text(0.005, 0.002,
                "vision-only throughout — no external supervision · nested in measured Shapley order · "
                "exponent fitted per fold, bars = SD across folds",
                fontsize=8.6, color=MUTED)
    figure.tight_layout(rect=(0, 0.042, 1, 1))
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    for m in steps:
        print(f"  m={m}: expoente {exponents[m]:.4f} ± {spreads[m]:.4f}  "
              f"({SHORT.get(names[m][-1], names[m][-1])})")


if __name__ == "__main__":
    main()
