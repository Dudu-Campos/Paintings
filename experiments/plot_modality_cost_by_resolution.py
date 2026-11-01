"""Custo por imagem de cada modalidade contra a resolução de entrada.

Log-log, porque o que se quer ler é o **expoente** (custo ∝ N^α) e lei de
potência só vira reta nessa parametrização.

O achado que a figura mostra: duas classes bem separadas. Segmentação cresce
com a área (α ≈ 2) até bater no teto de reamostragem de 1024 px -- marcado pela
vertical pontilhada, sem rótulo porque a legenda o explica -- e a partir daí é
plana.

CORRIGIDO em 1/set. A versão anterior desta figura dizia que Descriptors,
Hilbert e Gabor subiam "porque o benchmark as alimenta com a imagem inteira".
**Não alimentava**: os três reduzem internamente para 128 px
(`TARGET`/`N_SIZE`) e o benchmark os chamava sem flag de tamanho, então a
subida media decode de JPEG maior mais downscale. Os α publicados para eles
(0,07 / 0,74 / 0,46) eram expoentes de I/O.

Com a flag passada, os α reais são **1,19 / 2,02 / 2,08** e a ordenação inverte:
Gabor e Hilbert passam de "as que menos respondem à resolução" para as que
**mais** respondem, acima do Felzenszwalb (1,98). Isso é o fisicamente
plausível -- convolução de K filtros e autocorrelação sobre N² pontos são O(p)
ou pior.

A coluna da legenda é **"run at"** = ponto de operação da PRODUÇÃO (128 px para
Descriptors/Hilbert/Gabor, 1024 para os segmentadores), não o teto do benchmark.
Aqui todas operam na resolução do ponto plotado.

A figura vai até **1024**, o cap de produção. O ponto de 2048 foi descartado por
dois motivos: mede um regime acima do cap, e o benchmark faz **upscale** (toda
imagem vira `(side, side)`), então acima de ~800 px são quadros interpolados --
para o Felzenszwalb isso importa, porque o custo dele depende da estrutura de
gradiente e a interpolação apaga aresta.

O que a figura NÃO mostra, e a prosa antiga afirmava: que "acima de 1024 px o
custo do pipeline é constante, enquanto o de uma CNN acompanha a resolução".
Aquele trecho plano era o cap do benchmark, e uma CNN a 224 px é igualmente
constante na resolução da fonte. O argumento defensável está no §5.4b (a resolução de
operação é um parâmetro livre desacoplado do classificador).

Cores: as cinco fatias saem de uma ordem enumerada e validada (pior par
adjacente ΔE OKLab 24,6 sob simulação de protanopia/deuteranopia/tritanopia,
contra alvo de 8). A ordem default da paleta reprova em tritanopia com cinco
séries -- amarelo e magenta convergem --, então foi reordenada, que é o
procedimento previsto e não uma troca de valores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import colour, display  # noqa: E402

# ordem validada; ver docstring
MARKERS = {"Felzenszwalb": "o", "SLIC": "s", "Descriptors": "^",
           "Hilbert": "D", "Gabor": "P", "Fourier": "v",
           "L(G_felz)": "X", "Line graph": "*"}
# RUN_AT (ponto de operacao da producao) foi REMOVIDO da figura em 1/set: e
# propriedade da configuracao final, e esta figura e de ablacao. Pelo mesmo
# motivo saiu a vertical pontilhada que marcava o cap de 1024.
_RUN_AT_REMOVIDO = {"Felzenszwalb": 1024, "SLIC": 1024, "Descriptors": 128,
        "Hilbert": 128, "Gabor": 128}
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/modality_cost_by_resolution.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/modality_cost_by_resolution.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    table = frame.pivot_table(index="step", columns="side", values="ms_per_image")
    order = table[table.columns.max()].sort_values(ascending=False).index

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, which="major", zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)

    sides = np.array(sorted(table.columns))

    for name in order:
        values = table.loc[name].dropna()
        axis.plot(values.index, values.values, color=colour(name),
                  marker=MARKERS[name], markersize=8, linewidth=2.4,
                  markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=4,
                  solid_capstyle="round")



    # Tabela no lugar de rótulos na ponta: Hilbert e Gabor convergem em 2048
    # (as duas reamostram para 128 px, então só sobra decodificação) e os
    # rótulos se sobrepunham. A tabela ainda traz o expoente, que é o que se
    # quer comparar entre modalidades e que a curva sozinha não dá em número.
    x0, y0, step = 0.045, 0.930, 0.0385
    axis.text(x0 + 0.030, y0 + step, "view", transform=axis.transAxes,
              fontsize=10, color=MUTED, fontweight="bold")
    axis.text(x0 + 0.300, y0 + step, "α", transform=axis.transAxes,
              fontsize=10, color=MUTED, fontweight="bold", ha="right")
    for index, name in enumerate(order):
        row = table.loc[name]
        alpha = np.log(row[1024] / row[128]) / np.log(1024 / 128)
        y = y0 - index * step
        axis.plot([x0], [y + 0.013], marker=MARKERS[name], markersize=7,
                  color=colour(name), transform=axis.transAxes,
                  clip_on=False, zorder=6)
        axis.text(x0 + 0.030, y, display(name), transform=axis.transAxes,
                  fontsize=10.5, color=colour(name))
        axis.text(x0 + 0.300, y, f"{alpha:.2f}", transform=axis.transAxes,
                  fontsize=10.5, color=colour(name), ha="right")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sides)
    axis.set_xticklabels([str(s) for s in sides])
    axis.set_yticks([1, 10, 100, 1000])
    axis.set_yticklabels(["1", "10", "100", "1000"])
    axis.minorticks_off()
    axis.set_xlim(sides[0] * 0.88, sides[-1] * 1.14)
    axis.set_xlabel("input image side (px)", fontsize=14.5, color=MUTED)
    axis.set_ylabel("extraction cost (ms per image, 1 core)", fontsize=14.5, color=MUTED)
    axis.set_title("Extraction cost versus input resolution, per view",
                   fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)

    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")

    print("\nexpoente empírico do custo (128 -> 1024, antes do teto):")
    for name in order:
        values = table.loc[name]
        alpha = np.log(values[1024] / values[128]) / np.log(1024 / 128)
        print(f"  {name:<13} alpha={alpha:5.2f}")


if __name__ == "__main__":
    main()
