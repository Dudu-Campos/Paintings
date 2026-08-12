"""
Top-N features por Mutual Information, coloridas por grupo.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).parent.parent
OUT_PNG = ROOT / 'paper' / 'feature_importance.png'
N_TOP   = 40

mi      = np.load(ROOT / 'paper/mi_scores.npy')
names   = np.load(ROOT / 'paper/mi_feat_names.npy', allow_pickle=True).astype(str)
groups  = np.load(ROOT / 'paper/mi_feat_groups.npy', allow_pickle=True).astype(str)

def desc_feature_names():
    names = []
    for ch in ['L','a','b']:
        names += [f'{ch}_hist_{i}' for i in range(30)]
    for ch in ['L','a','b']:
        names += [f'{ch}_mean', f'{ch}_std', f'{ch}_skew']
    names += [f'lbp_{i}' for i in range(18)]
    har_props = ['contrast','dissim','homogen','energy','corr','ASM']
    for p in har_props:
        names += [f'{p}_mean', f'{p}_std']
    names.append('corr_max')
    return names[:130]

DESC_NAMES = desc_feature_names()

GROUP_COLOR = {
    'RAG':         '#2E86AB',
    'kNN':         '#5BA4CF',
    'Delaunay':    '#1B4F72',
    'Hilbert':     '#DD6E42',
    'Descriptors': '#9B59B6',
    'Sentiment':   '#8FB339',
}

# label legível para exibição
def display_name(name, group):
    if group == 'Descriptors':
        idx = int(name.split('_')[-1])
        return DESC_NAMES[idx]
    if group == 'Hilbert':
        idx = name.split('_')[-1]
        return f'hilbert_{idx}'
    return name.replace('knn_', 'knn.').replace('del_', 'del.')

top_idx    = np.argsort(mi)[::-1][:N_TOP]
top_mi     = mi[top_idx]
top_names  = [display_name(names[i], groups[i]) for i in top_idx]
top_colors = [GROUP_COLOR[groups[i]] for i in top_idx]

legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLOR.items()]

half = N_TOP // 2
for part, (start, end, suffix) in enumerate([(0, half, 'a'), (half, N_TOP, 'b')], 1):
    idx_slice  = slice(start, end)
    m_part     = top_mi[idx_slice]
    n_part     = top_names[idx_slice]
    c_part     = top_colors[idx_slice]
    rank_start = start + 1
    rank_end   = end

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(m_part))
    ax.barh(y_pos, m_part[::-1], color=c_part[::-1], edgecolor='white',
            linewidth=0.4, height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(n_part[::-1], fontsize=10)
    ax.set_xlabel('Mutual Information', fontsize=13)
    ax.set_title(f'Top-{N_TOP} features (#{rank_start}–{rank_end})\n'
                 f'(WikiArt 80k, 27 estilos)', fontsize=14, pad=10)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_patches, fontsize=11, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    out = ROOT / f'paper/feature_importance_{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {out}')
