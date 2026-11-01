"""O ganho de agregar representações CRESCE com o tamanho do treino.

Teste direto da alegação do §5.1, e mais transparente que a versão publicada
lá. Aquela ajusta um expoente por número de canais (nove ajustes) e correlaciona
os expoentes com m -- n=8 pontos, r=+0,963. Esta mede a mesma coisa com **um**
ajuste: a diferença `log erro(1 canal) − log erro(todos)` contra `log n`.

A identidade que torna isso exato: se erro_m(n) = a_m * n^(−c_m), então

    log erro_1(n) − log erro_m(n) = log(a_1/a_m) + (c_m − c_1) * log n

ou seja, **a inclinação desta reta é exatamente a diferença de expoentes**. Se
acrescentar representações apenas deslocasse o nível das curvas (mesmo expoente,
intercepto diferente), a reta seria **horizontal** -- e é por isso que a
referência tracejada no nível do primeiro ponto está na figura: ela é a hipótese
nula desenhada, não decoração.

Por que o eixo y é log-erro e não pontos percentuais: as duas curvas decaem, e a
diferença em pp mistura mudança de expoente com o decaimento comum. Em log a
separação é limpa. O gap em pp vai junto no CSV porque é o número legível para
o resumo (no WikiArt ele quase triplica, 4,8 -> 13,1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
CORPORA = [
    ("WikiArt, 27 styles", ROOT / "outputs/representation_data_grid/peaks.csv",
     "n_train", "#2a78d6", "o"),
    ("Multitask, 25 styles", ROOT / "results/representation_data_grid_multitask.csv",
     "train_size", "#eda100", "s"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/channel_gap_vs_data.png")
    parser.add_argument("--csv", type=Path, default=ROOT / "results/channel_gap_vs_data.csv")
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)

    rows = []
    for label, path, column, colour, marker in CORPORA:
        frame = pd.read_csv(path)
        frame["error"] = 100 * (1 - frame.bacc)
        table = frame.pivot_table(index=column, columns="n_modalities", values="error")
        # Duplicatas do maior n (63365/63366) são o mesmo ponto medido duas
        # vezes; colapsar evita peso duplo no ajuste. Agrupa pela chave
        # arredondada mas mantém o n VERDADEIRO no eixo -- arredondar o eixo
        # punha 2.500 em 2.000 (round(2,5) = 2 em numpy).
        key = np.round(table.index / 1000).astype(int)
        sizes = table.index.to_series().groupby(key).mean().values.astype(float)
        table = table.groupby(key).mean()
        low, high = table.columns.min(), table.columns.max()
        gap_log = np.log(table[low].values) - np.log(table[high].values)
        slope, intercept, r, p, se = stats.linregress(np.log(sizes), gap_log)

        axis.plot(sizes, gap_log, color=colour, marker=marker, markersize=8.5,
                  linewidth=2.6, markeredgecolor=SURFACE, markeredgewidth=1.3,
                  zorder=4, label=f"{label}   slope {slope:+.4f} ± {se:.4f}")
        fit = np.array([sizes[0] * 0.85, sizes[-1] * 1.18])
        axis.plot(fit, intercept + slope * np.log(fit), color=colour,
                  linestyle="-", linewidth=1.0, alpha=0.45, zorder=3)
        # A hipótese nula: se acrescentar canais só deslocasse o nível, o gap
        # ficaria no valor do primeiro ponto para sempre.
        axis.plot(fit, [gap_log[0]] * 2, color=colour, linestyle="--",
                  linewidth=1.1, alpha=0.4, zorder=2)
        for size, value in zip(sizes, gap_log):
            rows.append({"corpus": label, "n_train": size, "gap_log_error": value,
                         "slope": slope, "stderr": se, "r": r, "p": p})

    axis.text(0.035, 0.93,
              "dashed: what a pure level shift would look like\n"
              "(same exponent, different intercept)",
              transform=axis.transAxes, fontsize=11, color=MUTED, va="top")

    axis.set_xscale("log")
    axis.set_xlabel("training images", fontsize=14.5, color=MUTED)
    axis.set_ylabel("log error(1 channel) − log error(all channels)",
                    fontsize=14.5, color=MUTED)
    axis.set_title("Adding representations changes the slope, not just the level",
                   fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)
    legend = axis.legend(loc="lower right", fontsize=11.5, frameon=True,
                         facecolor="white", edgecolor=GRID)
    legend.get_frame().set_linewidth(0.8)

    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"saved: {args.output}\nsaved: {args.csv}")


if __name__ == "__main__":
    main()
