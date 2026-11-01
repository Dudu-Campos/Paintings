"""Scatter de importancia x redundancia p95, SEM KDE -- o §5.9 em pontos.

Fonte: results/rich_importance_redundancy_kde_p95_pt.csv (1.642 features).
Substitui a leitura por contorno de `rich_importance_redundancy_kde_p95_pt.png`,
que era a saida do `attic/experiments/plot_importance_redundancy_kde_metric.py`.

Duas decisoes que o dado impoe, e as duas consertam a figura antiga:

1. O eixo x vai ate 0,60 e nao ate 1,0. A versao antiga esticava ate 1,0 para
   caber a linha de corte em 0,8, mas o maior p95_r2 do banco e **0,573** --
   43% do eixo era vazio. A linha saiu junto: ela marcava um corte que nenhuma
   feature alcanca.
2. Com 1.642 pontos e 87,8% deles num canto, um painel unico de quatro cores
   empasta. O modo `--facet` e o que sustenta a alegacao do §5.9 (*"nao e efeito
   de bloco: aparece dentro das quatro familias"*), porque mostra a tendencia
   quatro vezes com o resto da nuvem em cinza atras.

Cor: `experiments/view_names.py`, fonte unica do projeto (§7.18). O par pior
entre familias e dE 12,9 nos quatro cenarios de daltonismo -- conferido por
`audit_palette_cvd.py`, nao por inspecao.

Uso:
  python3 experiments/plot_importance_redundancy_scatter.py
  python3 experiments/plot_importance_redundancy_scatter.py --facet
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import FAMILY_SLOT, SLOTS

FAMILIES = ["Region graphs", "Image statistics", "Space-filling curve", "Spectral"]
INK, MUTED, GRID = "#1b1b19", "#6f6e6a", "#dedcd6"


def colour(family: str) -> str:
    key = family if family in FAMILY_SLOT else f"{family} & text"
    return SLOTS[FAMILY_SLOT[key]]


def style(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path,
                   default=ROOT / "results/rich_importance_redundancy_kde_p95_pt.csv")
    p.add_argument("--facet", action="store_true", help="2x2, uma familia por painel")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    d = d[d.family.isin(FAMILIES)].copy()
    d["y"] = d.importance_normalized_mean * 100.0
    xmax = float(np.ceil(d.p95_r2.max() * 20) / 20)
    ymax = float(np.ceil(d.y.max() * 200) / 200)   # passo de 0,005, nao 0,05

    rho_all = stats.spearmanr(d.p95_r2, d.y)[0]
    r_all = stats.pearsonr(d.p95_r2, d.y)[0]

    if args.facet:
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), sharex=True, sharey=True)
        for ax, fam in zip(axes.ravel(), FAMILIES):
            style(ax)
            g = d[d.family == fam]
            ax.scatter(d.p95_r2, d.y, s=7, c="#e8e6e0", linewidths=0, zorder=1)
            ax.scatter(g.p95_r2, g.y, s=9, c=colour(fam), alpha=0.65,
                       linewidths=0, zorder=2)
            b, a = np.polyfit(g.p95_r2, g.y, 1)
            xs = np.linspace(g.p95_r2.min(), g.p95_r2.max(), 2)
            ax.plot(xs, a + b * xs, color=colour(fam), linewidth=1.8, zorder=3)
            r = stats.pearsonr(g.p95_r2, g.y)[0]
            ax.set_title(f"{fam}   n = {len(g)}", fontsize=10.5, color=INK,
                         loc="left", pad=6)
            ax.text(0.97, 0.06, f"r = {r:+.3f}", transform=ax.transAxes,
                    ha="right", fontsize=10, color=MUTED)
            ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        fig.supxlabel("Feature redundancy  (p95 of $r^2$ against every other feature)",
                      fontsize=10.5, color=INK, y=0.035)
        fig.supylabel("Supervised importance  (share of the bank's total, %)", fontsize=10.5, color=INK, x=0.03)
        fig.suptitle("Redundant features are the ones the model uses",
                     fontsize=13, color=INK, x=0.5, y=0.985, fontweight="bold")
        out = args.output or ROOT / "results/importance_redundancy_scatter_facet.png"
        fig.tight_layout(rect=(0.035, 0.045, 1, 0.965))
    else:
        fig, ax = plt.subplots(figsize=(8.4, 6.0))
        style(ax)
        for fam in FAMILIES:
            g = d[d.family == fam]
            ax.scatter(g.p95_r2, g.y, s=10, c=colour(fam), alpha=0.5,
                       linewidths=0, label=f"{fam}  ({len(g)})", zorder=2)
        b, a = np.polyfit(d.p95_r2, d.y, 1)
        xs = np.linspace(0, d.p95_r2.max(), 2)
        ax.plot(xs, a + b * xs, color=INK, linewidth=1.6, linestyle="--",
                alpha=0.55, zorder=3)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.set_xlabel("Feature redundancy  (p95 of $r^2$ against every other feature)",
                      fontsize=10.5, color=INK)
        ax.set_ylabel("Supervised importance  (share of the bank's total, %)", fontsize=10.5, color=INK)
        ax.set_title("Redundant features are the ones the model uses",
                     fontsize=13, color=INK, loc="left", pad=10, fontweight="bold")
        ax.text(0.985, 0.05, f"n = {len(d)}      r = {r_all:+.3f}      $\\rho$ = {rho_all:+.3f}",
                transform=ax.transAxes, ha="right", fontsize=10, color=MUTED)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5,
                        handletextpad=0.4, borderaxespad=0.8, markerscale=2.6)
        for t in leg.get_texts():
            t.set_color(INK)
        out = args.output or ROOT / "results/importance_redundancy_scatter.png"
        fig.tight_layout()

    fig.savefig(out, dpi=200, facecolor="white")
    print(f"escrito: {out.relative_to(ROOT)}")
    print(f"  n={len(d)}  x in [0, {d.p95_r2.max():.3f}]  (eixo ate {xmax})")
    print(f"  global  r={r_all:+.3f}  rho={rho_all:+.3f}")


if __name__ == "__main__":
    main()
