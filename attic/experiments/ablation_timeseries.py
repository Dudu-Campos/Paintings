"""
ablation_timeseries.py — Ablation das representações temporais e HRW.

Avalia a contribuição incremental de cada representação dentro do pipeline,
usando LDA (5-fold CV, BAcc) — rápido o suficiente para iteração.

Pipeline base (X_pipe): RAG+kNN+Del+MST+Desc  (sem séries temporais)

Configs:
  Hilbert only        (57 feat)
  HRW only            (20 feat) — trap_content multi-escala (L=4,5,6,7)
  Pipeline            (258 feat)
  Pipeline + Hilbert  (315 feat)  ← resultado final atual
  Pipeline + HRW      (278 feat)
  Pipeline + Hil+HRW  (335 feat)
  [RW e Hybrid Walk mantidos para referência — já rodados]

Saída: paper/ablation_timeseries.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs' / 'features'
META_CSV = ROOT / 'meta' / 'meta.csv'
OUT_CSV  = Path(__file__).parent / 'ablation_timeseries.csv'

META_COLS = {'stem', 'artist', 'style', 'genre', 'filename', 'n_nodes_orig'}

from compare import run_lda, run_mlp, impute, log


def load():
    rag  = pd.read_parquet(FEAT_DIR / 'full_rag_features.parquet')
    knn  = pd.read_parquet(FEAT_DIR / 'full_knn_features.parquet')
    delx = pd.read_parquet(FEAT_DIR / 'delaunay_features.parquet')
    mst  = pd.read_parquet(FEAT_DIR / 'mst_features.parquet')
    feat_cols = [c for c in rag.columns if c not in META_COLS]

    desc_f = np.load(FEAT_DIR / 'desc_full_features.npy')
    desc_s = np.load(FEAT_DIR / 'desc_full_stems.npy', allow_pickle=True).astype(str)
    hil_f  = np.load(FEAT_DIR / 'hilbert_features.npy')
    hil_s  = np.load(FEAT_DIR / 'hilbert_stems.npy',      allow_pickle=True).astype(str)
    rw_f   = np.load(FEAT_DIR / 'rw_features.npy')
    rw_s   = np.load(FEAT_DIR / 'rw_stems.npy',           allow_pickle=True).astype(str)
    hw_f   = np.load(FEAT_DIR / 'hybrid_walk_features.npy')
    hw_s   = np.load(FEAT_DIR / 'hybrid_walk_stems.npy',  allow_pickle=True).astype(str)
    hrw_f  = np.load(FEAT_DIR / 'hrw_features.npy')
    hrw_s  = np.load(FEAT_DIR / 'hrw_stems.npy',          allow_pickle=True).astype(str)

    common = (set(rag['stem'].astype(str)) & set(knn['stem'].astype(str)) &
              set(delx['stem'].astype(str)) & set(mst['stem'].astype(str)) &
              set(desc_s) & set(hil_s) & set(rw_s) & set(hw_s) & set(hrw_s))

    meta = pd.read_csv(META_CSV)
    meta['stem'] = meta['filename'].astype(str).str.split('.').str[0]
    style_map = meta.set_index('stem')['style'].to_dict()

    stems  = np.array(sorted(common))
    styles = np.array([style_map.get(s, 'unknown') for s in stems])
    mask   = styles != 'unknown'
    stems  = stems[mask]; styles = styles[mask]
    cnt    = pd.Series(styles).value_counts()
    mask2  = np.array([s in set(cnt[cnt >= 5].index) for s in styles])
    stems  = stems[mask2]; styles = styles[mask2]
    log(f'Dataset: {len(stems):,} imagens, {len(set(styles))} estilos')

    idx = {
        'desc': {s: i for i, s in enumerate(desc_s)},
        'hil':  {s: i for i, s in enumerate(hil_s)},
        'rw':   {s: i for i, s in enumerate(rw_s)},
        'hw':   {s: i for i, s in enumerate(hw_s)},
        'hrw':  {s: i for i, s in enumerate(hrw_s)},
    }

    X_rag  = rag.set_index('stem').loc[stems, feat_cols].values.astype(float)
    X_knn  = knn.set_index('stem').loc[stems, feat_cols].values.astype(float)
    X_del  = delx.set_index('stem').loc[stems, feat_cols].values.astype(float)
    X_mst  = mst.set_index('stem').loc[stems, feat_cols].values.astype(float)
    X_desc = desc_f[[idx['desc'][s] for s in stems]]
    X_hil  = hil_f[[idx['hil'][s]  for s in stems]]
    X_rw   = rw_f[[idx['rw'][s]    for s in stems]]
    X_hw   = hw_f[[idx['hw'][s]    for s in stems]]
    X_hrw  = hrw_f[[idx['hrw'][s]  for s in stems]]
    y      = LabelEncoder().fit_transform(styles)

    X_pipe = np.hstack([X_rag, X_knn, X_del, X_mst, X_desc])

    return X_pipe, X_hil, X_rw, X_hw, X_hrw, y


def main():
    X_pipe, X_hil, X_rw, X_hw, X_hrw, y = load()

    configs = [
        ('Hilbert only',             X_hil),
        ('HRW only',                 X_hrw),
        ('Random Walk only',         X_rw),
        ('Hybrid Walk only',         X_hw),
        ('Pipeline (no timeseries)', X_pipe),
        ('Pipeline + Hilbert',       np.hstack([X_pipe, X_hil])),
        ('Pipeline + HRW',           np.hstack([X_pipe, X_hrw])),
        ('Pipeline + Hil + HRW',     np.hstack([X_pipe, X_hil, X_hrw])),
        ('Pipeline + RW',            np.hstack([X_pipe, X_rw])),
        ('Pipeline + Hybrid Walk',   np.hstack([X_pipe, X_hw])),
    ]

    # LDA — todos os configs
    lda_res = {}
    for label, X in configs:
        log(f'LDA {label} ({X.shape[1]} feat)...')
        t0 = time.time()
        m, s = run_lda(X, y)
        log(f'  BAcc={m:.4f}±{s:.4f}  ({time.time()-t0:.0f}s)')
        lda_res[label] = (m, s)

    # MLP — só configs relevantes para HRW (os demais já têm resultado)
    mlp_configs = [
        ('Pipeline + Hilbert',   np.hstack([X_pipe, X_hil])),
        ('Pipeline + HRW',       np.hstack([X_pipe, X_hrw])),
        ('Pipeline + Hil + HRW', np.hstack([X_pipe, X_hil, X_hrw])),
    ]
    log('MLP (GPU) para configs HRW...')
    mlp_res = run_mlp(mlp_configs, y)

    rows = []
    for label, X in configs:
        lda_m, lda_s = lda_res[label]
        mlp_m, mlp_s = mlp_res.get(label, (float('nan'), float('nan')))
        rows.append({'method': label, 'n_feat': X.shape[1],
                     'lda_mean': lda_m, 'lda_std': lda_s,
                     'mlp_mean': mlp_m, 'mlp_std': mlp_s})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print()
    print(df.to_string(index=False))
    log(f'Salvo: {OUT_CSV}')


if __name__ == '__main__':
    main()
