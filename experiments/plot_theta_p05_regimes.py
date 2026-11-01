"""theta importa sozinho e deixa de importar na fusão -- os três regimes.

Cada painel é a distribuição de Δ = perf(theta) − perf(theta*) sobre a caixa
Theta_f interpolada, num regime de uso diferente:

  LDA isolado    a sonda linear sobre o bloco Hilbert sozinho
  MLP isolado    o mesmo bloco sozinho, sem teto de posto
  MLP produção   o bloco DENTRO do hand-8, substituindo só ele e mantendo as
                 outras sete modalidades idênticas

A leitura é a largura, não a posição: se a distribuição encolhe da esquerda
para a direita, a sensibilidade a theta é propriedade do bloco isolado e não do
método. O p05 marcado em cada painel é a perda de pior caso a 95%.

As distribuições são amostradas da superfície ajustada, não contadas na grade:
o quantil de uma grade é estatística da grade (5 níveis num eixo que não
importa contra 4 no que importa distorce nas duas direções, medido em ~0,5pp no
p05). O ajuste é validado por leave-one-out em `analyze_theta_robustness.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_theta_robustness import STAR, design, loo_rmse  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
REGIMES = [
    ("Hilbert alone, LDA probe", ROOT / "results/theta_hilbert.csv", "lda_bacc", "#c9ddf5"),
    ("Hilbert alone, MLP", ROOT / "results/theta_hilbert_mlp.csv", "mlp_bacc", "#5f9de0"),
    ("inside hand-8, production MLP", ROOT / "results/theta_hilbert_pipeline_grid.csv",
     "bacc", "#12406f"),
]


def surface_samples(frame: pd.DataFrame, metric: str, samples: int) -> tuple[np.ndarray, str]:
    size = frame.n_size.to_numpy(float)
    bands = frame.n_bands.to_numpy(float)
    star = frame[(frame.n_size == STAR[0]) & (frame.n_bands == STAR[1])][metric].iloc[0]
    delta = frame[metric].to_numpy(float) - star
    order = min(("additive", "interaction", "quadratic"),
                key=lambda o: loo_rmse(size, bands, delta, o))
    beta, *_ = np.linalg.lstsq(design(size, bands, order), delta, rcond=None)
    grid_s = np.linspace(np.log2(size.min()), np.log2(size.max()), samples)
    grid_b = np.linspace(np.log2(bands.min()), np.log2(bands.max()), samples)
    mesh_s, mesh_b = np.meshgrid(grid_s, grid_b)
    return design(2 ** mesh_s.ravel(), 2 ** mesh_b.ravel(), order) @ beta, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/theta_p05_regimes.png")
    args = parser.parse_args()

    panels = []
    for label, path, metric, colour in REGIMES:
        if not path.exists():
            print(f"faltando: {path}")
            continue
        frame = pd.read_csv(path)
        if metric not in frame.columns:
            continue
        if "n_size" not in frame.columns and "condition" in frame.columns:
            # O CSV do pipeline identifica theta pela string da condição
            # ("hilbert 128x8"); as linhas de controle não têm esse formato.
            parsed = frame.condition.str.extract(r"hilbert (\d+)x(\d+)$")
            frame = frame.assign(n_size=pd.to_numeric(parsed[0]),
                                 n_bands=pd.to_numeric(parsed[1])).dropna(
                                     subset=["n_size", "n_bands"])
            frame[["n_size", "n_bands"]] = frame[["n_size", "n_bands"]].astype(int)
        if "fold" in frame.columns:
            # A grade do pipeline vem por fold; a superfície é ajustada na média.
            frame = frame.groupby(["n_size", "n_bands"], as_index=False)[metric].mean()
        if not ((frame.n_size == STAR[0]) & (frame.n_bands == STAR[1])).any():
            print(f"{label}: theta* ausente, pulado")
            continue
        # Sem mínimo de pontos o ajuste é degenerado e o painel mente com
        # convicção: com 2 configurações o quadrático interpola exatamente e a
        # massa dentro de 0,5pp sai 100%.
        if len(frame) < args.min_points:
            print(f"{label}: só {len(frame)} configurações, insuficiente para a "
                  f"superfície (mínimo {args.min_points}) -- painel omitido")
            continue
        values, order = surface_samples(frame, metric, args.samples)
        panels.append((label, values, colour, order, len(frame)))
    if not panels:
        raise SystemExit("nenhum regime com dados")

    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, axis="x", zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=13)
    axis.axvline(0, color="#9a9a9a", linewidth=1.2, zorder=1)

    for index, (label, values, colour, order, points) in enumerate(panels):
        y = len(panels) - 1 - index
        parts = axis.violinplot([values], positions=[y], vert=False, widths=0.72,
                                showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.85)
            body.set_edgecolor(SURFACE)
        p05 = np.percentile(values, 5)
        axis.plot([p05], [y], marker="|", markersize=26, color="#d13c3c",
                  markeredgewidth=3, zorder=5)
        axis.annotate(f"p05 {p05:+.2f}", (p05, y), xytext=(0, -30),
                      textcoords="offset points", fontsize=11.5, color="#d13c3c",
                      fontweight="bold", ha="center")
        axis.text(axis.get_xlim()[0], y + 0.42, f"{label}   ({points} θ, {order})",
                  fontsize=12, color=INK, va="bottom")

    axis.set_yticks([])
    axis.set_xlabel("Δ balanced accuracy versus θ* (pp)", fontsize=14.5, color=MUTED)
    axis.set_title("θ matters alone and stops mattering in the fusion",
                   fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=14)
    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    for label, values, _, _, _ in panels:
        print(f"  {label:<34} p05 {np.percentile(values,5):+6.2f}   "
              f"|Δ|<=0,5 em {100*(np.abs(values)<=0.5).mean():5.1f}%")


if __name__ == "__main__":
    main()
