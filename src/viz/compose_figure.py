"""
compose_figure.py — Compõe figura final com 3 painéis:
  - Topo (largura total): strip de percolação cromática
  - Inferior esquerdo:    projeção 2D LDA
  - Inferior direito:     trade-off performance vs custo

Uso:
    python compose_figure.py --output ./plots_v4/figure_combined.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


STRIP  = './plots_v3/coarsening_evolution.png'
LDA2D  = './plots_v4/lda_eps_sweep_2d.png'
COST   = './plots_v4/eps_perf_vs_cost.png'


def load(path):
    return np.array(Image.open(path).convert('RGB'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./plots_v4/figure_combined.png')
    args = parser.parse_args()

    strip = load(STRIP)
    lda   = load(LDA2D)
    cost  = load(COST)

    # Altura do painel inferior = max das duas imagens, para manter proporção
    h_bot = max(lda.shape[0], cost.shape[0])
    h_top = strip.shape[0]

    # Proporção de altura: topo vs inferior
    top_frac = h_top / (h_top + h_bot)

    fig = plt.figure(figsize=(16, 16 * (h_top + h_bot) / strip.shape[1]))
    gs  = gridspec.GridSpec(
        2, 2,
        height_ratios=[top_frac, 1 - top_frac],
        hspace=0.04, wspace=0.03,
    )

    # Painel topo (span das 2 colunas)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(strip)
    ax_top.axis('off')

    # Painel inferior esquerdo — LDA 2D
    ax_lda = fig.add_subplot(gs[1, 0])
    ax_lda.imshow(lda)
    ax_lda.axis('off')
    ax_lda.set_title('(b)', loc='left', fontsize=13, fontweight='bold', pad=4)

    # Painel inferior direito — custo vs performance
    ax_cost = fig.add_subplot(gs[1, 1])
    ax_cost.imshow(cost)
    ax_cost.axis('off')
    ax_cost.set_title('(c)', loc='left', fontsize=13, fontweight='bold', pad=4)

    # Label do painel topo
    ax_top.set_title('(a)', loc='left', fontsize=13, fontweight='bold', pad=4)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Salvo: {out}')


if __name__ == '__main__':
    main()
