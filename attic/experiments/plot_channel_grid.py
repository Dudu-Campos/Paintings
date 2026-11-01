"""Erro contra tamanho de treino, uma curva por número de canais.

Companheira de `evaluate_unsupervised_channel_grid.py`, e desenhada para ser
rodada **enquanto** a grade roda: lê o CSV incremental, usa o que estiver lá e
carimba no rodapé quantos folds cada ponto tem. Um braço com um fold só sai sem
faixa de incerteza, e o título diz "provisional".

Escolhas de codificação:

* O erro, e não a acurácia, no eixo y. É a forma das leis de escala
  (Hestness 2017; Kaplan 2020) e o achado 18 mediu o preço de ignorar isso: o
  cruzamento CNN × hand-designed sai em 50k quando ajustado em acurácia e em
  97k quando ajustado em erro, e os dados já contradiziam o primeiro.
* Rampa **sequencial** de um tom só para o número de canais, porque a variável é
  ordenada. Uma paleta categórica aqui sugeriria que "3 canais" e "7 canais" são
  categorias sem ordem entre si.
* Rótulo direto na ponta direita de cada curva, sem caixa de legenda: com oito
  séries ordenadas a legenda obriga o leitor a fazer o casamento cor-a-cor duas
  vezes por leitura.

Saída: results/channel_grid_{arm}.png (e _paper.png sem título).
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
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TITLES = {"engineered": "Engineering blocks (provenance)",
          "unsupervised": "Measured channels (spectral grouping)"}


def ramp(count: int) -> list[str]:
    """Rampa sequencial clara→escura em torno do azul da paleta do projeto."""
    base = np.array([(0.80, 0.87, 0.95), (0.03, 0.20, 0.42)])
    return [matplotlib.colors.to_hex(base[0] + (base[1] - base[0]) * t)
            for t in np.linspace(0, 1, count)]


def exponent(frame: pd.DataFrame) -> float:
    curve = frame.groupby("size")["error"].mean()
    if len(curve) < 3:
        return float("nan")
    return -np.polyfit(np.log(curve.index.to_numpy(dtype=float)),
                       np.log(curve.to_numpy()), 1)[0]


def draw(table: pd.DataFrame, arm: str, output: Path, with_title: bool) -> None:
    data = table[table.arm == arm]
    counts = sorted(data.n_groups.unique())
    colours = dict(zip(counts, ramp(len(counts))))
    folds = int(data.groupby(["n_groups", "size"]).fold.nunique().max())

    figure, axis = plt.subplots(figsize=(9.6, 6.6))
    ends = []
    for count in counts:
        piece = data[data.n_groups == count]
        curve = piece.groupby("size")["error"].agg(["mean", "std", "count"])
        axis.plot(curve.index, curve["mean"], marker="o", markersize=5,
                  color=colours[count], linewidth=2.0, zorder=3)
        if curve["count"].max() > 1:
            axis.errorbar(curve.index, curve["mean"], yerr=curve["std"].fillna(0),
                          fmt="none", ecolor=colours[count], elinewidth=1.2,
                          capsize=3, zorder=2)
        ends.append((curve["mean"].iloc[-1], curve.index[-1], count))

    # Descolisão dos rótulos da ponta: as curvas convergem à direita, e sem isto
    # os quatro últimos números se sobrepõem exatamente onde a figura precisa
    # mostrar que eles empataram.
    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    minimum = max(span * 0.055, 0.12)
    placed = []
    for value, x, count in sorted(ends):
        target = value
        while placed and target - placed[-1] < minimum:
            target = placed[-1] + minimum
        placed.append(target)
        axis.annotate(f"{count}", (x, value), xytext=(11, 0),
                      textcoords="offset points", va="center", fontsize=10,
                      color=colours[count], fontweight="bold",
                      annotation_clip=False)
        if abs(target - value) > 1e-9:
            axis.annotate(f"{count}", (x, target), xytext=(11, 0),
                          textcoords="offset points", va="center", fontsize=10,
                          color=colours[count], fontweight="bold",
                          annotation_clip=False)

    sizes = sorted(data["size"].unique())
    axis.set_xscale("log")
    axis.set_xticks(sizes)
    # Sem isto o matplotlib desenha os ticks menores da escala log por cima dos
    # nossos (2x10^3, 3x10^3, ...) e o eixo fica ilegível.
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_xticklabels([f"{round(s / 1000, 1):g}k" if s >= 1000 else str(s)
                          for s in sizes])
    axis.set_xlabel("Training images")
    axis.set_ylabel("Balanced error (%)")
    axis.margins(x=0.09)
    axis.grid(alpha=0.28, linewidth=0.6, color=GRID)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.text(1.0, 1.01, "channels", transform=axis.transAxes, ha="right",
              fontsize=9.5, color=MUTED)

    inset = axis.inset_axes([0.09, 0.10, 0.34, 0.32])
    exponents = [exponent(data[data.n_groups == c]) for c in counts]
    inset.plot(counts, exponents, marker="o", markersize=4.5,
               color=colours[counts[-1]], linewidth=1.8)
    inset.set_title("error exponent", fontsize=9, color=MUTED, pad=4)
    inset.set_xlabel("channels", fontsize=8.5, color=MUTED)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.25, linewidth=0.5, color=GRID)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)

    footer = (f"WikiArt, 27 styles · LDA probe, peak over PCA components · "
              f"{folds} fold(s) · training subsampled, test fold never reduced\n"
              "channels added in greedy order (each step: the group with the "
              "largest marginal), fixed after the first pass")
    figure.text(0.5, 0.005, footer, ha="center", fontsize=8.4, color=MUTED)
    if with_title:
        provisional = " — PROVISIONAL" if folds < 5 else ""
        figure.suptitle(f"{TITLES.get(arm, arm)}{provisional}\n"
                        "more channels pull the model out of saturation",
                        fontsize=12.5, color=TEXT)
        figure.tight_layout(rect=(0, 0.075, 1, 0.93))
    else:
        figure.tight_layout(rect=(0, 0.075, 1, 1))
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"salvo: {output} ({folds} fold(s))", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "outputs/unsupervised_channel_grid/fold_results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    table = pd.read_csv(args.results)
    for arm in table.arm.unique():
        # Só desenha um braço quando ele tem a curva inteira; senão a inclinação
        # do encaixe seria de um pedaço de curva e o expoente, ficção.
        piece = table[table.arm == arm]
        if piece.groupby("n_groups")["size"].nunique().min() < 3:
            print(f"pulando {arm}: poucos tamanhos ainda", flush=True)
            continue
        draw(piece, arm, args.output_dir / f"channel_grid_{arm}.png", True)
        draw(piece, arm, args.output_dir / f"channel_grid_{arm}_paper.png", False)


if __name__ == "__main__":
    main()
