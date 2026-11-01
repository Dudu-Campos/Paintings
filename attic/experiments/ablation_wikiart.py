"""
ablation_wikiart.py — Ablation study LDA no WikiArt completo.

Combina grafos sobre regiões SLIC (RAG + kNN + Delaunay + MST) com séries
temporais e sentimento.
Avaliação: LDA com 5-fold CV estratificado, balanced accuracy.
Duas tarefas: estilo (27 classes) e artista (1116 classes).

Uso:
    python paper/ablation_wikiart.py
    python paper/ablation_wikiart.py --task artist
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs/wikiart_features'
GRAPH_DIR = ROOT / 'outputs/wikiart_slic_features'

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


def load_npy_as_df(feat_file: str, stem_file: str, prefix: str) -> pd.DataFrame:
    feats = np.load(FEAT_DIR / feat_file)
    stems = np.load(FEAT_DIR / stem_file, allow_pickle=True).astype(str)
    cols  = [f'{prefix}_{i}' for i in range(feats.shape[1])]
    df    = pd.DataFrame(feats, columns=cols)
    df.insert(0, 'stem', stems)
    return df, cols


def run_lda(name: str, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n_classes = len(np.unique(y))
    n_comps   = min(n_classes - 1, X.shape[1])
    priors    = np.ones(n_classes) / n_classes
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(n_components=n_comps, priors=priors)),
    ])
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    print(f'  {name:<55} BAcc={scores.mean():.4f} ± {scores.std():.4f}')
    return scores.mean(), scores.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='style', choices=['style', 'artist'])
    args = parser.parse_args()

    # ── carregar parquets ──────────────────────────────────────────
    graph = pd.read_parquet(GRAPH_DIR / 'graph_features.parquet')
    knn   = pd.read_parquet(GRAPH_DIR / 'knn_features.parquet')
    knn   = knn.rename(columns={c: f'knn_{c}' for c in KNN_FEAT})
    knn_cols = [f'knn_{c}' for c in KNN_FEAT]

    df = graph.merge(knn[['stem'] + knn_cols], on='stem', how='inner')

    # Delaunay e MST — mesmas colunas do RAG, prefixadas
    del_df = pd.read_parquet(GRAPH_DIR / 'delaunay_features.parquet')
    mst_df = pd.read_parquet(GRAPH_DIR / 'mst_features.parquet')
    # Select only model features.  The SLIC parquet also carries provenance
    # and timing columns, which must never become predictors.
    del_feat_cols = [c for c in RAG_FEAT if c in del_df.columns]
    mst_feat_cols = [c for c in RAG_FEAT if c in mst_df.columns]
    del_df = del_df.rename(columns={c: f'del_{c}' for c in del_feat_cols})
    mst_df = mst_df.rename(columns={c: f'mst_{c}' for c in mst_feat_cols})
    del_cols = [f'del_{c}' for c in del_feat_cols]
    mst_cols = [f'mst_{c}' for c in mst_feat_cols]

    df = df.merge(del_df[['stem'] + del_cols], on='stem', how='inner')
    df = df.merge(mst_df[['stem'] + mst_cols], on='stem', how='inner')

    sent = pd.read_parquet(FEAT_DIR / 'sentiment_features.parquet')[['stem'] + SENT_FEAT]
    df   = df.merge(sent, on='stem', how='inner')

    # ── carregar timeseries e descritores (.npy) ──────────────────
    hil_df,  hil_cols  = load_npy_as_df('hilbert_features.npy',     'hilbert_stems.npy',     'hil')
    rw_df,   rw_cols   = load_npy_as_df('rw_features.npy',          'rw_stems.npy',          'rw')
    hw_df,   hw_cols   = load_npy_as_df('hybrid_walk_features.npy', 'hybrid_walk_stems.npy', 'hw')
    desc_df, desc_cols = load_npy_as_df('desc_full_features.npy',   'desc_full_stems.npy',   'desc')

    df = df.merge(hil_df,  on='stem', how='inner')
    df = df.merge(rw_df,   on='stem', how='inner')
    df = df.merge(hw_df,   on='stem', how='inner')
    df = df.merge(desc_df, on='stem', how='inner')

    # ── label ──────────────────────────────────────────────────────
    meta     = pd.read_csv(ROOT / 'data/artemis/wikiart_meta.csv')
    meta_idx = dict(zip(meta['stem'], meta[args.task]))
    df['label'] = df['stem'].map(meta_idx)
    df = df.dropna(subset=['label'])

    counts = df['label'].value_counts()
    df = df[df['label'].isin(counts[counts >= 5].index)]

    y = df['label'].to_numpy(dtype=str)
    print(f'Dataset: {len(df):,} imagens  classes ({args.task}): {len(np.unique(y))}')

    rag_c   = [c for c in RAG_FEAT   if c in df.columns]
    knn_c   = [c for c in knn_cols   if c in df.columns]
    sent_c  = [c for c in SENT_FEAT  if c in df.columns]
    ts_c    = hil_cols + rw_cols + hw_cols
    del_c   = [c for c in del_cols   if c in df.columns]
    mst_c   = [c for c in mst_cols   if c in df.columns]
    desc_c  = [c for c in desc_cols  if c in df.columns]

    graph_c = rag_c + knn_c + del_c + mst_c          # 4 grafos (117)
    pipe_c  = graph_c + desc_c                        # pipeline completo sem TS/Sent

    print(f'\n=== Ablation LDA — {args.task.upper()} ===')
    results = {}

    results['SLIC-RAG (32)']                   = run_lda('SLIC-RAG (32 feat)',                     df[rag_c].values,                              y)
    results['kNN (21)']                        = run_lda('kNN (21 feat)',                          df[knn_c].values,                              y)
    results['Delaunay (32)']                   = run_lda('Delaunay (32 feat)',                     df[del_c].values,                              y)
    results['MST (32)']                        = run_lda('MST (32 feat)',                          df[mst_c].values,                              y)
    results['Pipeline SLIC grafos (117)']      = run_lda('Pipeline SLIC — RAG+kNN+Del+MST',         df[graph_c].values,                            y)
    results['Descritores (130)']               = run_lda('Descritores (130 feat)',                  df[desc_c].values,                             y)
    results['Hilbert (57)']                    = run_lda('Hilbert (57 feat)',                       df[hil_cols].values,                           y)
    results['RW (57)']                         = run_lda('RW (57 feat)',                            df[rw_cols].values,                            y)
    results['Hybrid Walk (57)']                = run_lda('Hybrid Walk (57 feat)',                   df[hw_cols].values,                            y)
    results['Timeseries (171)']                = run_lda('Timeseries — Hil+RW+HW (171 feat)',       df[ts_c].values,                               y)
    results['Pipeline (247)']                  = run_lda('Pipeline — Grafos+Desc (247 feat)',        df[pipe_c].values,                             y)
    results['Pipeline + TS (418)']             = run_lda('Pipeline + TS (418 feat)',                df[pipe_c + ts_c].values,                      y)
    results['Sentiment (16)']                  = run_lda('Sentiment (16 feat)',                     df[sent_c].values,                             y)
    results['Pipeline + Sent (263)']           = run_lda('Pipeline + Sent (263 feat)',              df[pipe_c + sent_c].values,                    y)
    results['Pipeline + TS + Sent (434)']      = run_lda('Pipeline + TS + Sent (434 feat)',         df[pipe_c + ts_c + sent_c].values,             y)

    print(f'\n{"="*70}')
    print(f'{"Configuração":<55} {"BAcc":>7}  ±')
    print(f'{"-"*70}')
    for name, (mean, std) in results.items():
        print(f'{name:<55} {mean:.4f}  {std:.4f}')
    print(f'{"="*70}')

    rows = [{'config': n, 'bacc_mean': m, 'bacc_std': s} for n, (m, s) in results.items()]
    out  = ROOT / f'paper/ablation_{args.task}_results.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Salvo: {out}')


if __name__ == '__main__':
    main()
