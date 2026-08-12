"""
plot_compare_wikiart.py — Descritores de Imagem vs Pipeline vs CNN, LDA e MLP.
Métrica: Balanced Accuracy (27 estilos, WikiArt ~80k).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
CSV_CMP = ROOT / 'paper' / 'results_wikiart_style.csv'
CSV_FS  = ROOT / 'paper' / 'forward_selection_3clf.csv'
OUT_PNG = ROOT / 'paper' / 'compare_wikiart_barplot.png'

# ── Carrega dados ─────────────────────────────────────────────────────────────
cmp  = pd.read_csv(CSV_CMP).set_index('method')
fs   = pd.read_csv(CSV_FS)
# Descritores = step 1; Pipeline completo = step 9 (mesmas 288 feat do CSV de comparação)
desc_row = fs[fs['group'] == 'Descriptors'].iloc[0]
pipe_row = fs[fs['step'] == fs['step'].max()].iloc[0]   # último step = pipeline completo

# Ordem das barras: simples → pipeline → CNN
methods = ['Descritores\n(130 feat)', 'Pipeline\n(288 feat)', 'CNN ResNet50\n(2048 feat)']

data = {
    'lda_mean': [
        desc_row['lda_mean'],
        pipe_row['lda_mean'],
        cmp.loc['CNN ResNet50 (2048 feat)', 'lda_bacc_mean'],
    ],
    'lda_std': [
        desc_row['lda_std'],
        pipe_row['lda_std'],
        cmp.loc['CNN ResNet50 (2048 feat)', 'lda_bacc_std'],
    ],
    'mlp_mean': [
        desc_row['mlp_mean'],
        pipe_row['mlp_mean'],
        cmp.loc['CNN ResNet50 (2048 feat)', 'mlp_bacc_mean'],
    ],
    'mlp_std': [
        desc_row['mlp_std'],
        pipe_row['mlp_std'],
        cmp.loc['CNN ResNet50 (2048 feat)', 'mlp_bacc_std'],
    ],
}

# Paleta: slot 3 aqua → Descritores, slot 1 azul → Pipeline, slot 2 laranja → CNN
COLORS  = ['#1baf7a', '#2a78d6', '#eb6834']
SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK_SEC = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'

chance = 1 / 27
x      = np.arange(len(methods))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False, facecolor=SURFACE)

for ax, (metric, title) in zip(axes, [
    ('lda', 'LDA (linear)'),
    ('mlp', 'MLP (512–512–256–128)'),
]):
    means = np.array(data[f'{metric}_mean'])
    stds  = np.array(data[f'{metric}_std'])

    ax.set_facecolor(SURFACE)

    ax.bar(x, means, color=COLORS, width=0.5, zorder=3,
           edgecolor=SURFACE, linewidth=1.5)

    ax.errorbar(x, means, yerr=stds, fmt='none',
                ecolor=INK_SEC, elinewidth=1.4, capsize=5, zorder=4)

    ax.axhline(chance, color=MUTED, linestyle='--', linewidth=1.0, zorder=2)
    ax.text(x[-1] + 0.38, chance + 0.008, 'Chance\n(1/27)',
            ha='right', va='bottom', fontsize=8.5, color=MUTED)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.014, f'{m:.3f}',
                ha='center', va='bottom', fontsize=12,
                fontweight='bold', color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11.5, color=INK)
    ax.tick_params(axis='y', labelsize=10.5, colors=MUTED)
    ax.set_ylabel('Acurácia Balanceada', fontsize=12, color=INK)
    ax.set_title(title, fontsize=13, pad=10, color=INK)
    ax.set_ylim(0, max(means) + max(stds) + 0.10)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)
    ax.set_xlim(-0.5, len(methods) - 0.5)

fig.suptitle(
    'Comparação de Métodos — Classificação de Estilo (27 classes, WikiArt ~80k)',
    fontsize=13, fontweight='bold', color=INK, y=1.01,
)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'Salvo: {OUT_PNG}')