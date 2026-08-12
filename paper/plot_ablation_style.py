"""
Gráfico de barras do ablation study por grupo de features — em português.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
df   = pd.read_csv(ROOT / 'paper/ablation_style_results.csv')

# ── Labels em português ───────────────────────────────────────────────────────
labels = [
    'RAG', 'kNN', 'Delaunay', 'MST',
    'Pipeline\nde Grafos',
    'Descritores\nde Imagem',
    'Hilbert', 'Cam.\nAleat.', 'Cam.\nHíbrida', 'Séries\nTemporais',
    'Pipeline\n(247)', 'Pipeline\n+TS',
    'Sentimento',
    'Pipeline\n+Sent.', 'Pipeline\n+TS+Sent.',
]

# ── Cores alinhadas com a paleta do diagrama do paper ─────────────────────────
# grafo   → rosa/pink
# imagem  → verde
# séries  → laranja
# tabular/sentimento → azul
# combinações → cinza neutro
# pipeline completo (melhor) → vermelho (destaque)
PINK   = '#d63384'
GREEN  = '#1baf7a'
ORANGE = '#eb6834'
BLUE   = '#2a78d6'
GREY   = '#6c757d'
RED    = '#e34948'

SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK_SEC = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'

colors = [
    PINK, PINK, PINK, PINK,    # grafos individuais
    PINK,                       # pipeline de grafos
    GREEN,                      # descritores de imagem
    ORANGE, ORANGE, ORANGE,     # séries temporais individuais
    ORANGE,                     # séries temporais combinadas
    GREY, GREY,                 # pipeline (247), pipeline+TS
    BLUE,                       # sentimento
    GREY,                       # pipeline+sent
    RED,                        # pipeline completo (melhor)
]

means  = df['bacc_mean'].values
stds   = df['bacc_std'].values
chance = 1 / 27

fig, ax = plt.subplots(figsize=(16, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.bar(range(len(means)), means, color=colors,
       edgecolor=SURFACE, linewidth=1.2, width=0.6, zorder=3)

ax.errorbar(range(len(means)), means, yerr=stds, fmt='none',
            ecolor=INK_SEC, elinewidth=1.2, capsize=3, zorder=4)

ax.axhline(chance, color=MUTED, linestyle='--', linewidth=1.0, zorder=2)
ax.text(len(means) - 0.5, chance + 0.005, f'Chance (1/27)',
        ha='right', va='bottom', fontsize=9, color=MUTED)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f'{m:.3f}',
            ha='center', va='bottom', fontsize=10.5, color=INK)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11, color=INK)
ax.tick_params(axis='y', labelsize=11, colors=MUTED)
ax.set_ylabel('Acurácia Balanceada', fontsize=13, color=INK)
ax.set_title('Ablation Study — Classificação de Estilo (27 classes, WikiArt ~80k)',
             fontsize=14, fontweight='bold', color=INK, pad=10)
ax.set_ylim(0, max(means) + stds[np.argmax(means)] + 0.06)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color(GRID)

legend_patches = [
    mpatches.Patch(color=PINK,   label='Grafo (RAG, kNN, Del., MST e combinações)'),
    mpatches.Patch(color=GREEN,  label='Imagem (descritores clássicos, 130 feat.)'),
    mpatches.Patch(color=ORANGE, label='Séries temporais (Hilbert, RW, Cam. Híbrida)'),
    mpatches.Patch(color=BLUE,   label='Tabular (sentimento ArtEmis)'),
    mpatches.Patch(color=GREY,   label='Combinações do pipeline'),
    mpatches.Patch(color=RED,    label='Pipeline completo (melhor)'),
]
ax.legend(handles=legend_patches, fontsize=10, loc='upper left',
          framealpha=0.9, frameon=True, edgecolor=GRID)

plt.tight_layout()
out = ROOT / 'paper/ablation_style_barplot.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'Salvo: {out}')
