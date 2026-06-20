"""
plot_graph_correct_vs_wrong.py
Compara grafos RAG de pinturas classificadas certa vs erradamente pelo LDA.
Seleciona estilos com recall alto (discriminável) e recall=0 (indiscriminável).
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from skimage.color import lab2rgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

sys.path.insert(0, '.')
from lda_graph_features import (
    read_coarse2, GRAPH_FEATURE_COLS, STRUCT_FEATURE_COLS,
    SPATIAL_FEATURE_COLS, COLOR_FEATURE_COLS,
)

ALL_32 = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS + COLOR_FEATURE_COLS
OUT = Path('./plots_v3')

# ── Carregar features e treinar LDA ──────────────────────────────────────────
print('Carregando features...')
rag = pd.read_parquet('./plots_v3/full_rag_features.parquet')
knn = pd.read_parquet('./plots_v3/full_knn_features.parquet')
rag = rag[rag['style'] != 'unknown'].copy()
rag['stem'] = rag['stem'].astype(str)
knn['stem'] = knn['stem'].astype(str)
vc = rag['style'].value_counts()
rag = rag[rag['style'].isin(vc[vc >= 31].index)].reset_index(drop=True)
df = rag.merge(knn[['stem'] + ALL_32], on='stem', how='inner', suffixes=('', '_knn')).reset_index(drop=True)

le = LabelEncoder(); y = le.fit_transform(df['style'].to_numpy())
X = np.hstack([df[ALL_32].to_numpy(), df[[f'{c}_knn' for c in ALL_32]].to_numpy()]).astype(np.float32)
for j in range(X.shape[1]):
    m = np.isnan(X[:, j])
    if m.any(): X[m, j] = np.nanmedian(X[:, j])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, val = next(sss.split(X, y))
n_s = len(le.classes_)
lda = Pipeline([('sc', StandardScaler()),
                ('lda', LinearDiscriminantAnalysis(
                    n_components=min(n_s-1, X.shape[1]),
                    priors=np.ones(n_s)/n_s))])
lda.fit(X[tr], y[tr])
pred = lda.predict(X[val])

# ── Calcular recall por classe ────────────────────────────────────────────────
n_val    = np.bincount(y[val], minlength=n_s)
correct  = np.zeros(n_s)
for true, p in zip(y[val], pred):
    correct[true] += (p == true)
recall = np.where(n_val > 0, correct / np.maximum(n_val, 1), np.nan)

# df_val com predições
df_val = df.iloc[val].copy().reset_index(drop=True)
df_val['y_true'] = y[val]
df_val['y_pred'] = pred
df_val['correct'] = (df_val['y_true'] == df_val['y_pred'])
df_val['pred_style'] = le.inverse_transform(pred)

# ── Carregar lookup de coarse files ──────────────────────────────────────────
idx = pd.read_parquet('./coarsening_output_v3/index.parquet')
idx['stem'] = idx['stem'].astype(str)
coarse_map = dict(zip(idx['stem'], idx['coarse']))

# ── Estilos a exibir ──────────────────────────────────────────────────────────
HIGH_RECALL = ['Divisionism', 'Pointillism', 'Luminism']
LOW_RECALL  = ['Hard Edge Painting', 'Abstract Art', 'Dada']
SELECTED    = HIGH_RECALL + LOW_RECALL

# ── Função de plot de um grafo ────────────────────────────────────────────────
def plot_graph(ax, nodes, edges, title, title_color='black'):
    """Plota um grafo RAG: nós coloridos em Lab, arestas cinza, tamanho ∝ área."""
    n = len(nodes)
    if n == 0:
        ax.text(0.5, 0.5, 'vazio', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=6, color=title_color)
        ax.axis('off'); return

    # Coordenadas já normalizadas [0,1]
    xs, ys = nodes['X'], 1 - nodes['Y']  # inverter Y para visualização natural

    # Cor Lab → RGB (clamp para [0,1])
    L  = nodes['L'].clip(0, 100)
    a_ = nodes['a'].clip(-128, 127)
    b_ = nodes['b'].clip(-128, 127)
    lab_px = np.stack([L, a_, b_], axis=1).reshape(-1, 1, 3)
    rgb = lab2rgb(lab_px).reshape(-1, 3).clip(0, 1)

    # Arestas
    if len(edges) > 0:
        srcs = edges['src']; dsts = edges['dst']
        valid = (srcs < n) & (dsts < n)
        segs = np.stack([
            np.stack([xs[srcs[valid]], ys[srcs[valid]]], axis=1),
            np.stack([xs[dsts[valid]], ys[dsts[valid]]], axis=1),
        ], axis=1)
        # Subsample se muitas arestas
        if len(segs) > 3000:
            segs = segs[np.random.choice(len(segs), 3000, replace=False)]
        lc = LineCollection(segs, linewidths=0.3, colors='gray', alpha=0.25, zorder=1)
        ax.add_collection(lc)

    # Nós — tamanho proporcional a área
    sizes = (nodes['area_frac'] * 800).clip(1, 200)
    ax.scatter(xs, ys, s=sizes, c=rgb, zorder=2, linewidths=0)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=6, color=title_color, pad=2)

# ── Selecionar exemplos por estilo ────────────────────────────────────────────
N_EXAMPLES = 2  # corretos + errados por estilo
rng = np.random.default_rng(42)

rows_data = []  # lista de (style, recall, is_correct, stem, pred_style)
for style in SELECTED:
    rc = recall[le.transform([style])[0]] if style in le.classes_ else np.nan
    subset = df_val[df_val['style'] == style]
    correct_ex   = subset[subset['correct']].sample(min(N_EXAMPLES, len(subset[subset['correct']])),
                                                     random_state=42) if len(subset[subset['correct']]) > 0 else pd.DataFrame()
    incorrect_ex = subset[~subset['correct']].sample(min(N_EXAMPLES, len(subset[~subset['correct']])),
                                                      random_state=42) if len(subset[~subset['correct']]) > 0 else pd.DataFrame()
    for _, row in correct_ex.iterrows():
        rows_data.append((style, rc, True,  row['stem'], row['pred_style']))
    for _, row in incorrect_ex.iterrows():
        rows_data.append((style, rc, False, row['stem'], row['pred_style']))

# ── Filtrar: preferir imagens com n_nodes entre 50 e 800 ─────────────────────
def get_coarse(stem):
    if stem not in coarse_map: return None
    d = read_coarse2(coarse_map[stem])
    if d is None or len(d['nodes']) == 0: return None
    return d

# Organizar por estilo, certo/errado
from collections import defaultdict
by_style = defaultdict(lambda: {'correct': [], 'wrong': []})
for style, rc, is_correct, stem, pred in rows_data:
    key = 'correct' if is_correct else 'wrong'
    by_style[style][key].append((stem, pred, rc))

# ── Figura: grid estilos × (correto | errado) ─────────────────────────────────
N_COLS = N_EXAMPLES * 2 + 1  # separador no meio
n_styles  = len(SELECTED)
n_cols    = N_EXAMPLES * 2 + 1   # corretos | separador | errados
cell_w, cell_h = 2.5, 2.5
label_w   = 1.8

fig_w = label_w + n_cols * cell_w
fig_h = n_styles * cell_h + 1.0

fig = plt.figure(figsize=(fig_w, fig_h))

# Grid: coluna 0 = label, 1..N_EXAMPLES = corretos, N_EXAMPLES+1 = sep, N_EXAMPLES+2.. = errados
col_widths = [label_w] + [cell_w] * N_EXAMPLES + [0.3] + [cell_w] * N_EXAMPLES
gs = fig.add_gridspec(n_styles + 1, len(col_widths),
                       width_ratios=col_widths,
                       height_ratios=[0.35] + [1.0] * n_styles,
                       hspace=0.08, wspace=0.06)

# Cabeçalho
for c, txt, clr in [(1, '◀ correto', '#1a7a1a'),
                     (N_EXAMPLES + 2, 'errado ▶', '#b22222')]:
    ax_h = fig.add_subplot(gs[0, c])
    ax_h.text(0.5, 0.4, txt, ha='center', va='center',
              fontsize=9, fontweight='bold', color=clr, transform=ax_h.transAxes)
    ax_h.axis('off')

# Separador central (header row)
ax_sep_h = fig.add_subplot(gs[0, N_EXAMPLES + 1])
ax_sep_h.axis('off')

# Linha separadora horizontal entre alto e baixo recall
sep_row = len(HIGH_RECALL)  # após a 3ª linha

for row_i, style in enumerate(SELECTED):
    rc  = recall[le.transform([style])[0]] if style in le.classes_ else np.nan
    clr = '#1a7a1a' if rc > 0.3 else '#b22222'
    data = by_style[style]

    # Label do estilo
    ax_lbl = fig.add_subplot(gs[row_i + 1, 0])
    ax_lbl.text(0.95, 0.5, f'{style}\nrecall={rc:.2f}',
                ha='right', va='center', fontsize=8, color=clr,
                fontweight='bold', transform=ax_lbl.transAxes)
    ax_lbl.axis('off')

    # Separador vertical
    ax_sep = fig.add_subplot(gs[row_i + 1, N_EXAMPLES + 1])
    ax_sep.axvline(0.5, color='#cccccc', linewidth=1.5)
    ax_sep.axis('off')

    # Corretos e errados
    for group_i, (group_key, is_correct, group_clr, col_offset) in enumerate([
            ('correct', True,  '#1a7a1a', 1),
            ('wrong',   False, '#b22222', N_EXAMPLES + 2)]):
        items = data[group_key][:N_EXAMPLES]
        while len(items) < N_EXAMPLES:
            items.append((None, None, rc))
        for ex_i, (stem, pred, _) in enumerate(items):
            ax = fig.add_subplot(gs[row_i + 1, col_offset + ex_i])
            if stem is None:
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        fontsize=14, color='#aaaaaa', transform=ax.transAxes)
                ax.set_facecolor('#f8f8f8'); ax.axis('off')
                continue
            d = get_coarse(stem)
            title = '✓' if is_correct else f'→ {pred[:18]}'
            if d and len(d['nodes']) > 0:
                # Subsample nós se muitos (>800) para clareza visual
                nodes = d['nodes']
                if len(nodes) > 800:
                    idx_sub = np.random.choice(len(nodes), 800, replace=False)
                    nodes = nodes[idx_sub]
                    # Refiltrar arestas para esses nós
                    idx_set = set(idx_sub)
                    mask = np.isin(d['edges']['src'], idx_sub) & np.isin(d['edges']['dst'], idx_sub)
                    # Remapear índices
                    remap = {old: new for new, old in enumerate(idx_sub)}
                    edges = d['edges'][mask].copy()
                    edges['src'] = [remap[s] for s in edges['src']]
                    edges['dst'] = [remap[s] for s in edges['dst']]
                else:
                    edges = d['edges']
                plot_graph(ax, nodes, edges, title, title_color=group_clr)
            else:
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        fontsize=8, color='gray', transform=ax.transAxes)
                ax.set_title(title, fontsize=6, color=group_clr); ax.axis('off')

    # Linha separadora horizontal entre grupos
    if row_i == sep_row - 1:
        ax_line = fig.add_subplot(gs[row_i + 1, 0])
        ax_line.plot([0, fig_w], [-0.08, -0.08], color='#888888',
                     linewidth=1.0, transform=fig.transFigure,
                     clip_on=False, zorder=10)

fig.suptitle('RAG+kNN (64 feat, LDA) — grafos de pinturas classificadas certa vs erradamente',
             fontsize=10, y=1.002)

p = OUT / 'graph_correct_vs_wrong.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
print(f'Salvo: {p}')
plt.close()
