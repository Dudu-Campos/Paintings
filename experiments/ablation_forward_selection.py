"""
Forward selection greedy: adiciona um grupo de features por vez,
escolhendo sempre o que dá maior ganho incremental.

Gera:
  paper/forward_selection_results.csv
  paper/forward_selection_curve.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs/wikiart_features'
GRAPH_DIR = ROOT / 'outputs/wikiart_slic_features'

# ── definição dos grupos ──────────────────────────────────────────────────────

RAG_FEAT = [
    'n_nodes', 'n_edges', 'density', 'area_gini', 'area_entropy',
    'area_max_frac', 'area_mean', 'area_std', 'color_dist_mean', 'color_dist_std',
    'color_dist_max', 'spatial_dist_mean', 'spatial_dist_std', 'degree_mean',
    'degree_std', 'degree_max', 'hub_degree_ratio', 'n_leaf_frac', 'degree_gini',
    'area_second_max_frac', 'area_cv', 'spatial_density_cv',
    'spatial_density_max_block_frac',
    'L_mean_w', 'L_std_w', 'a_mean_w', 'a_std_w', 'b_mean_w', 'b_std_w',
    'chroma_mean_w', 'chroma_std_w', 'warm_frac',
]
KNN_FEAT = [
    'n_nodes', 'n_edges', 'density', 'area_gini', 'area_entropy',
    'area_max_frac', 'area_mean', 'area_std', 'color_dist_mean', 'color_dist_std',
    'color_dist_max', 'spatial_dist_mean', 'spatial_dist_std', 'degree_mean',
    'degree_std', 'degree_max', 'hub_degree_ratio', 'n_leaf_frac', 'degree_gini',
    'area_second_max_frac', 'area_cv',
]
NRC_CATS  = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
             'negative', 'positive', 'sadness', 'surprise', 'trust']
SENT_FEAT = (
    ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
     'tb_polarity', 'tb_subjectivity']
    + [f'nrc_{c}' for c in NRC_CATS]
)


def load_npy(feat_file, stem_file, prefix):
    feats = np.load(FEAT_DIR / feat_file)
    stems = np.load(FEAT_DIR / stem_file, allow_pickle=True).astype(str)
    cols  = [f'{prefix}_{i}' for i in range(feats.shape[1])]
    df    = pd.DataFrame(feats, columns=cols)
    df.insert(0, 'stem', stems)
    return df, cols


def run_lda(X, y):
    n_classes = len(np.unique(y))
    n_comps   = min(n_classes - 1, X.shape[1])
    priors    = np.ones(n_classes) / n_classes
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(n_components=n_comps, priors=priors)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    return scores.mean(), scores.std()


def main():
    # ── carregar todos os dados ───────────────────────────────────────────────
    graph = pd.read_parquet(GRAPH_DIR / 'graph_features.parquet')
    knn   = pd.read_parquet(GRAPH_DIR / 'knn_features.parquet')
    knn   = knn.rename(columns={c: f'knn_{c}' for c in KNN_FEAT})
    knn_cols = [f'knn_{c}' for c in KNN_FEAT]

    del_df = pd.read_parquet(GRAPH_DIR / 'delaunay_features.parquet')
    mst_df = pd.read_parquet(GRAPH_DIR / 'mst_features.parquet')
    # Exclude SLIC provenance/timing columns from the predictive matrix.
    del_feat = [c for c in RAG_FEAT if c in del_df.columns]
    mst_feat = [c for c in RAG_FEAT if c in mst_df.columns]
    del_df = del_df.rename(columns={c: f'del_{c}' for c in del_feat})
    mst_df = mst_df.rename(columns={c: f'mst_{c}' for c in mst_feat})
    del_cols = [f'del_{c}' for c in del_feat]
    mst_cols = [f'mst_{c}' for c in mst_feat]

    hil_df,  hil_cols  = load_npy('hilbert_features.npy',     'hilbert_stems.npy',     'hil')
    rw_df,   rw_cols   = load_npy('rw_features.npy',          'rw_stems.npy',          'rw')
    hw_df,   hw_cols   = load_npy('hybrid_walk_features.npy', 'hybrid_walk_stems.npy', 'hw')
    desc_df, desc_cols = load_npy('desc_full_features.npy',   'desc_full_stems.npy',   'desc')

    sent = pd.read_parquet(FEAT_DIR / 'sentiment_features.parquet')[['stem'] + SENT_FEAT]

    df = graph.merge(knn[['stem'] + knn_cols],   on='stem', how='inner')
    df = df.merge(del_df[['stem'] + del_cols],   on='stem', how='inner')
    df = df.merge(mst_df[['stem'] + mst_cols],   on='stem', how='inner')
    df = df.merge(hil_df,                         on='stem', how='inner')
    df = df.merge(rw_df,                          on='stem', how='inner')
    df = df.merge(hw_df,                          on='stem', how='inner')
    df = df.merge(desc_df,                        on='stem', how='inner')
    df = df.merge(sent,                           on='stem', how='inner')

    meta     = pd.read_csv(ROOT / 'data/artemis/wikiart_meta.csv')
    meta_idx = dict(zip(meta['stem'], meta['style']))
    df['label'] = df['stem'].map(meta_idx)
    df = df.dropna(subset=['label'])
    counts = df['label'].value_counts()
    df = df[df['label'].isin(counts[counts >= 5].index)]

    y = df['label'].to_numpy(dtype=str)
    print(f'Dataset: {len(df):,} imagens  classes: {len(np.unique(y))}')

    # ── grupos disponíveis ────────────────────────────────────────────────────
    groups = {
        'SLIC-RAG':    [c for c in RAG_FEAT   if c in df.columns],
        'kNN':         [c for c in knn_cols   if c in df.columns],
        'Delaunay':    [c for c in del_cols   if c in df.columns],
        'MST':         [c for c in mst_cols   if c in df.columns],
        'Hilbert':     [c for c in hil_cols   if c in df.columns],
        'RW':          [c for c in rw_cols    if c in df.columns],
        'Hybrid Walk': [c for c in hw_cols    if c in df.columns],
        'Descriptors': [c for c in desc_cols  if c in df.columns],
        'Sentiment':   [c for c in SENT_FEAT  if c in df.columns],
    }

    # ── forward selection ─────────────────────────────────────────────────────
    selected   = []
    remaining  = list(groups.keys())
    active_cols= []
    results    = []

    chance = 1 / len(np.unique(y))
    results.append({'step': 0, 'group': 'Chance', 'bacc_mean': chance, 'bacc_std': 0.0,
                    'selected': ''})
    print(f'\n  {"Step":<4} {"Added":<14} {"BAcc":>7}')

    for step in range(1, len(groups) + 1):
        best_name, best_mean, best_std, best_cols = None, -1, 0, []
        for name in remaining:
            cols = active_cols + groups[name]
            X    = df[cols].values.astype(np.float32)
            mean, std = run_lda(X, y)
            if mean > best_mean:
                best_name, best_mean, best_std, best_cols = name, mean, std, groups[name]

        active_cols += best_cols
        selected.append(best_name)
        remaining.remove(best_name)
        results.append({
            'step':       step,
            'group':      best_name,
            'bacc_mean':  best_mean,
            'bacc_std':   best_std,
            'selected':   ' + '.join(selected),
        })
        print(f'  {step:<4} +{best_name:<13} BAcc={best_mean:.4f} ± {best_std:.4f}  '
              f'[{" + ".join(selected)}]')

    # ── salvar CSV ────────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    res_df.to_csv(ROOT / 'paper/forward_selection_results.csv', index=False)

    # ── plot ──────────────────────────────────────────────────────────────────
    x     = res_df['step'].values
    means = res_df['bacc_mean'].values
    stds  = res_df['bacc_std'].values
    labels= ['Chance'] + [r['group'] for r in results[1:]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, means, 'o-', color='#2E86AB', linewidth=2, markersize=7, zorder=4)
    ax.fill_between(x, means - stds, means + stds, alpha=0.15, color='#2E86AB', zorder=3)

    for xi, (m, s, lbl) in enumerate(zip(means, stds, labels)):
        offset = 0.010 if xi % 2 == 0 else -0.018
        ax.annotate(f'{m:.3f}', (xi, m), xytext=(0, 10 + offset*100),
                    textcoords='offset points', ha='center', fontsize=11,
                    color='#1a1a1a')

    ax.set_xticks(x)
    ax.set_xticklabels(
        ['Chance'] + [r['group'] for r in results[1:]],
        fontsize=11
    )
    ax.set_ylabel('Balanced Accuracy', fontsize=13)
    ax.set_xlabel('Feature Group Added (Greedy Forward Selection)', fontsize=12)
    ax.set_title('Cumulative Feature Contribution — Style Classification (27 classes, WikiArt 80k)',
                 fontsize=13, pad=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # setas de ganho entre steps
    for i in range(1, len(means)):
        gain = means[i] - means[i-1]
        if gain > 0.005:
            ax.annotate('', xy=(i, means[i] - 0.005),
                        xytext=(i, means[i-1] + 0.005),
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
            ax.text(i + 0.08, (means[i] + means[i-1]) / 2,
                    f'+{gain:.3f}', fontsize=10, color='#555', va='center')

    plt.tight_layout()
    out = ROOT / 'paper/forward_selection_curve.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSalvo: {out}')


if __name__ == '__main__':
    main()
