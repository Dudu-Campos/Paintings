"""Pareto custo x acurácia da grade de theta do Hilbert.

O que a figura tem de deixar claro, e que a tabela sozinha esconde: os dois
eixos de theta compram coisas muito diferentes por unidade de custo. Resolução
custa caro e não compra nada (16x os pixels rendem 0,72pp, e o custo multiplica
por 30); número de bandas é quase de graça e compra 3,56pp. A fronteira de
Pareto separa isso visualmente -- ela sobe pelo eixo de bandas, não pelo de
resolução.

`theta*` de produção é marcado à parte porque a leitura dele é o ponto da
seção de theta-robustez: ele **não** é o ótimo (12º de 19) e nem sequer está na
fronteira -- é dominado por configurações mais BARATAS e melhores. Isso é
evidência contra tuning, que é o que a seção precisa, e ao mesmo tempo diz onde
haveria ganho se alguém quisesse persegui-lo.

**Ressalva de medição:** a primeira configuração extraída pagou o aquecimento do
cache do NFS (41 s contra ~18 s das equivalentes) e o seu custo está inflado.
Ela é dominada de qualquer forma, então não afeta a fronteira, mas o ponto é
marcado como não-confiável em vez de silenciosamente corrigido.

`--loglog` desenha o mesmo dado com os dois eixos em log, onde a inclinação de
cada caminho vira a **elasticidade** `d log(BAcc) / d log(custo)` -- quanto de
acurácia se compra por unidade relativa de orçamento. É a leitura que separa os
dois eixos de theta em número, e não só visualmente:

    só resolução (bandas fixas)   +0,007 a +0,031
    só bandas (resolução fixa)    +0,287 a +1,400

Uma ordem de grandeza e meia de diferença. A escala log no eixo y comprime a
faixa de BAcc (17,2 a 21,5), então a versão linear continua sendo a melhor para
ler a fronteira; a log-log é para ler a troca.
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
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
BAND_COLOUR = {4: "#c9ddf5", 8: "#7fb0e5", 16: "#2a78d6", 32: "#12406f"}
COLD_CACHE = (32, 4)


def frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Maximiza BAcc, minimiza custo."""
    ordered = frame.sort_values(["extract_seconds", "lda_bacc"], ascending=[True, False])
    keep, best = [], -np.inf
    for row in ordered.itertuples():
        if row.lda_bacc > best:
            keep.append(row.Index)
            best = row.lda_bacc
    return frame.loc[keep].sort_values("extract_seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/theta_hilbert.csv")
    parser.add_argument("--metric", default="lda_bacc",
                        help="coluna de desempenho; `mlp_bacc` usa a grade do MLP")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--loglog", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("results/theta_hilbert_pareto_loglog.png" if args.loglog
                              else "results/theta_hilbert_pareto.png")

    data = pd.read_csv(args.input)
    if args.metric not in data.columns:
        # A grade do MLP vive num CSV separado; o custo só existe no da varredura.
        other = pd.read_csv(ROOT / "results/theta_hilbert_mlp.csv")
        data = data.merge(other[["n_size", "n_bands", args.metric]],
                          on=["n_size", "n_bands"], how="inner")
    # Renomear para o nome interno; a métrica não usada tem de SAIR antes,
    # senão sobram duas colunas `lda_bacc` e o sort quebra.
    data = data.drop(columns=[c for c in ("lda_bacc", "mlp_bacc")
                              if c != args.metric and c in data.columns])
    data = data.rename(columns={args.metric: "lda_bacc"})
    probe = "MLP" if args.metric.startswith("mlp") else "LDA"
    data["cold"] = [(s, b) == COLD_CACHE for s, b in zip(data.n_size, data.n_bands)]
    front = frontier(data[~data.cold])

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)

    axis.step(front.extract_seconds, front.lda_bacc, where="post",
              color="#9a9a9a", linewidth=1.4, linestyle="-", zorder=2)

    for bands, group in data.groupby("n_bands"):
        axis.scatter(group.extract_seconds, group.lda_bacc, s=95,
                     color=BAND_COLOUR[bands], edgecolor=SURFACE, linewidth=1.3,
                     zorder=4, label=f"{bands} bands")
    for row in data.itertuples():
        axis.annotate(f"{row.n_size}", (row.extract_seconds, row.lda_bacc),
                      xytext=(0, -14), textcoords="offset points", fontsize=9,
                      color=MUTED, ha="center")

    star = data[data.is_star].iloc[0]
    # Fixar o rank no texto quebra assim que a métrica ou a grade mudam.
    ranked = data.sort_values("lda_bacc", ascending=False).reset_index(drop=True)
    star_rank = int(ranked.index[ranked.is_star][0]) + 1
    dominated = ((data.extract_seconds < star.extract_seconds)
                 & (data.lda_bacc > star.lda_bacc)).sum()
    axis.scatter([star.extract_seconds], [star.lda_bacc], s=340, facecolor="none",
                 edgecolor="#d13c3c", linewidth=2.4, zorder=6)
    axis.annotate(f"θ* in production ({probe})\n(128 px, 8 bands)\n"
                  f"{star_rank} of {len(data)}, dominated by {dominated}",
                  (star.extract_seconds, star.lda_bacc), xytext=(16, -46),
                  textcoords="offset points", fontsize=11, color="#d13c3c",
                  fontweight="bold", ha="left")

    cold = data[data.cold]
    if len(cold):
        axis.annotate("cold NFS cache:\ncost inflated",
                      (cold.iloc[0].extract_seconds, cold.iloc[0].lda_bacc),
                      xytext=(10, 6), textcoords="offset points", fontsize=10,
                      color="#9a9a9a")

    axis.set_xscale("log")
    if args.loglog:
        axis.set_yscale("log")
        ticks = [t for t in (18, 19, 20, 21, 22, 24, 26, 28, 30, 32)
                 if data.lda_bacc.min() * 0.98 <= t <= data.lda_bacc.max() * 1.02]
        axis.set_yticks(ticks)
        axis.set_yticklabels([str(t) for t in ticks])
        axis.minorticks_off()
        # Caminhos de theta: um eixo por vez, para que a inclinação de cada
        # linha SEJA a elasticidade daquele eixo.
        clean = data[~data.cold]
        for bands, group in clean.groupby("n_bands"):
            group = group.sort_values("extract_seconds")
            axis.plot(group.extract_seconds, group.lda_bacc, color=BAND_COLOUR[bands],
                      linewidth=1.6, alpha=0.55, zorder=3)
        for size, group in clean.groupby("n_size"):
            group = group.sort_values("extract_seconds")
            axis.plot(group.extract_seconds, group.lda_bacc, color="#7a7a7a",
                      linewidth=1.3, alpha=0.6, linestyle="--", zorder=3)
        resolution = np.median([np.polyfit(np.log(g.extract_seconds), np.log(g.lda_bacc), 1)[0]
                                for _, g in clean.groupby("n_bands") if len(g) > 2])
        granularity = np.median([np.polyfit(np.log(g.extract_seconds), np.log(g.lda_bacc), 1)[0]
                                 for _, g in clean.groupby("n_size") if len(g) > 2])
        axis.text(0.035, 0.30,
                  "elasticity  d log(BAcc) / d log(cost)\n"
                  f"solid, resolution axis:   {resolution:+.3f}\n"
                  f"dashed, granularity axis: {granularity:+.3f}",
                  transform=axis.transAxes, fontsize=11.5, color=INK, va="top",
                  family="monospace",
                  bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.5"))
    axis.set_xlabel("extraction cost for the full corpus (seconds, 24 workers)",
                    fontsize=14.5, color=MUTED)
    axis.set_ylabel(f"balanced accuracy, Hilbert alone, {probe} (%)",
                    fontsize=14.5, color=MUTED)
    axis.set_title("Resolution costs a lot and buys nothing; granularity is nearly free",
                   fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)
    legend = axis.legend(loc="lower right", fontsize=11.5, frameon=True,
                         facecolor="white", edgecolor=GRID, title="point label = input side (px)")
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_fontsize(10.5)

    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    print("\nfronteira de Pareto:")
    print(front[["n_size", "n_bands", "dimensions", "extract_seconds", "lda_bacc"]]
          .to_string(index=False))
    better = data[(data.extract_seconds < star.extract_seconds)
                  & (data.lda_bacc > star.lda_bacc)]
    print(f"\nconfigurações que DOMINAM theta* (mais baratas E melhores): {len(better)}")
    if len(better):
        print(better[["n_size", "n_bands", "extract_seconds", "lda_bacc"]].to_string(index=False))


if __name__ == "__main__":
    main()
