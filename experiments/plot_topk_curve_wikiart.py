"""
plot_topk_curve_wikiart.py — Curva BAcc × número de dimensões (K) no WikiArt.

Pipeline final (158 feat): SLIC-RAG + kNN + Delaunay + Hilbert + Sentiment
  Ranking por mutual information → top-K features → MLP

CNN ResNet50 (2048 feat):
  PCA → top-K componentes → MLP

Mesmo protocolo do compare_wikiart.py (5-fold, BS=4096, epochs=150, patience=15).

Saída:
  paper/results_topk_curve_wikiart.csv
  paper/topk_curve_wikiart.png
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs/wikiart_features'
GRAPH_DIR = ROOT / 'outputs/wikiart_slic_features'
META_CSV = ROOT / 'data/artemis/wikiart_meta.csv'
OUT_CSV  = Path(__file__).parent / 'results_topk_curve_wikiart.csv'
OUT_PNG  = Path(__file__).parent / 'topk_curve_wikiart.png'

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED     = 42
BS       = 4096
LR       = 3e-4
EPOCHS   = 150
PATIENCE = 15
HIDDENS  = (512, 512, 256, 128)
DROP     = 0.35
N_FOLDS  = 5

N_PIPE = 288   # dimensão total do pipeline final (com descritores)

K_PIPE = [N_PIPE, 200, 150, 100, 80, 60, 40, 20, 10, 5, 2]
K_CNN  = [2048, 1000, 500, 300, N_PIPE, 200, 100, 60, 30, 10, 5, 2]

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
NRC_CATS  = ['anger','anticipation','disgust','fear','joy',
             'negative','positive','sadness','surprise','trust']
SENT_FEAT = (
    ['vader_neg','vader_neu','vader_pos','vader_compound','tb_polarity','tb_subjectivity']
    + [f'nrc_{c}' for c in NRC_CATS]
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def impute(X):
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.any():
            X[m, j] = float(np.nanmedian(X[~m, j])) if (~m).any() else 0.0
    return X


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        layers, prev = [], n_in
        for h in HIDDENS:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(DROP)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_fold(X_tr, y_tr, X_va, y_va, n_classes, scale=True):
    torch.manual_seed(SEED)
    counts  = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = torch.tensor((1.0/counts) / (1.0/counts).mean(),
                           dtype=torch.float32, device=DEVICE)
    if scale:
        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_va  = sc.transform(X_va)
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long,    device=DEVICE)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True,
                        pin_memory=False, num_workers=0)
    model  = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit   = nn.CrossEntropyLoss(weight=weights)
    best_bacc, best_preds, no_imp = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = model(Xva).argmax(1).cpu().numpy()
        bacc = balanced_accuracy_score(y_va, preds)
        if bacc > best_bacc:
            best_bacc, best_preds, no_imp = bacc, preds, 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    return best_bacc, accuracy_score(y_va, best_preds), f1_score(y_va, best_preds, average='macro', zero_division=0)


def eval_k(X, y, label, k, idx, scale=True):
    n_classes = len(np.unique(y))
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    baccs, accs, f1s = [], [], []
    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        bacc, acc, f1 = _train_fold(X[tr], y[tr], X[va], y[va], n_classes, scale=scale)
        baccs.append(bacc); accs.append(acc); f1s.append(f1)
        log(f'  [{idx}] {label} K={k} fold {fold}/{N_FOLDS}  BAcc={bacc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')
    return (np.mean(baccs), np.std(baccs),
            np.mean(accs),  np.std(accs),
            np.mean(f1s),   np.std(f1s))


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_features():
    graph = pd.read_parquet(GRAPH_DIR / 'graph_features.parquet')
    knn   = pd.read_parquet(GRAPH_DIR / 'knn_features.parquet')
    knn   = knn.rename(columns={c: f'knn_{c}' for c in KNN_FEAT})
    knn_cols = [f'knn_{c}' for c in KNN_FEAT]

    del_df   = pd.read_parquet(GRAPH_DIR / 'delaunay_features.parquet')
    # Exclude SLIC provenance/timing columns from the predictive matrix.
    del_feat = [c for c in RAG_FEAT if c in del_df.columns]
    del_df   = del_df.rename(columns={c: f'del_{c}' for c in del_feat})
    del_cols = [f'del_{c}' for c in del_feat]

    hil_f  = np.load(FEAT_DIR / 'hilbert_features.npy')
    hil_s  = np.load(FEAT_DIR / 'hilbert_stems.npy',  allow_pickle=True).astype(str)
    desc_f = np.load(FEAT_DIR / 'desc_full_features.npy')
    desc_s = np.load(FEAT_DIR / 'desc_full_stems.npy', allow_pickle=True).astype(str)
    cnn_f  = np.load(FEAT_DIR / 'cnn_full_features.npy')
    cnn_s  = np.load(FEAT_DIR / 'cnn_full_stems.npy',  allow_pickle=True).astype(str)
    sent   = pd.read_parquet(FEAT_DIR / 'sentiment_features.parquet')[['stem'] + SENT_FEAT]

    df = graph.merge(knn[['stem'] + knn_cols],   on='stem', how='inner')
    df = df.merge(del_df[['stem'] + del_cols],   on='stem', how='inner')
    df = df.merge(sent,                           on='stem', how='inner')

    meta     = pd.read_csv(META_CSV)
    meta_idx = dict(zip(meta['stem'], meta['style']))
    df['style'] = df['stem'].map(meta_idx)
    df = df.dropna(subset=['style'])
    counts = df['style'].value_counts()
    df = df[df['style'].isin(counts[counts >= 5].index)]

    hil_set  = set(hil_s); desc_set = set(desc_s); cnn_set = set(cnn_s)
    df = df[df['stem'].isin(hil_set) & df['stem'].isin(desc_set) & df['stem'].isin(cnn_set)]
    stems = df['stem'].values

    hil_map  = {s: i for i, s in enumerate(hil_s)}
    desc_map = {s: i for i, s in enumerate(desc_s)}
    cnn_map  = {s: i for i, s in enumerate(cnn_s)}

    rag_c  = [c for c in RAG_FEAT  if c in df.columns]
    knn_c  = [c for c in knn_cols  if c in df.columns]
    del_c  = [c for c in del_cols  if c in df.columns]
    sent_c = [c for c in SENT_FEAT if c in df.columns]

    X_pipe = np.hstack([
        df[rag_c].values,
        df[knn_c].values,
        df[del_c].values,
        hil_f[[hil_map[s] for s in stems]],
        desc_f[[desc_map[s] for s in stems]],
        df[sent_c].values,
    ]).astype(np.float32)

    X_cnn = cnn_f[[cnn_map[s] for s in stems]].astype(np.float32)
    y     = LabelEncoder().fit_transform(df['style'].values)

    log(f'Dataset: {len(stems):,} imagens  estilos: {len(np.unique(y))}')
    log(f'Pipeline: {X_pipe.shape[1]} feat   CNN: {X_cnn.shape[1]} feat')
    return X_pipe, X_cnn, y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log(f'device={DEVICE}')
    X_pipe, X_cnn, y = load_features()
    X_pipe = impute(X_pipe)
    X_cnn  = impute(X_cnn)

    # Pipeline: MI ranking
    log('Calculando mutual information — Pipeline...')
    t0 = time.time()
    mi_pipe    = mutual_info_classif(X_pipe, y, random_state=SEED, n_jobs=32)
    order_pipe = np.argsort(mi_pipe)[::-1]
    log(f'  MI pipeline em {time.time()-t0:.1f}s  (top feature idx={order_pipe[0]})')

    # CNN: MI ranking (sem PCA — comparação simétrica)
    log('Calculando mutual information — CNN...')
    t0 = time.time()
    mi_cnn    = mutual_info_classif(X_cnn, y, random_state=SEED, n_jobs=32)
    order_cnn = np.argsort(mi_cnn)[::-1]
    log(f'  MI CNN em {time.time()-t0:.1f}s  (top feature idx={order_cnn[0]})')

    from itertools import zip_longest
    total = len(K_PIPE) + len(K_CNN)
    rows  = []
    idx   = 0

    log(f'\nAvaliação intercalada Pipeline / CNN ({N_FOLDS}-fold):')
    for k_pipe, k_cnn in zip_longest(K_PIPE, K_CNN):
        if k_pipe is not None:
            idx += 1
            cols = order_pipe[:k_pipe]
            bm, bs, am, as_, fm, fs = eval_k(X_pipe[:, cols], y, 'Pipeline', k_pipe, f'{idx}/{total}', scale=True)
            log(f'  Pipeline K={k_pipe}: BAcc={bm:.4f}±{bs:.4f}  Acc={am:.4f}  F1={fm:.4f}')
            rows.append({'source': 'Pipeline SLIC (MI top-K)', 'K': k_pipe,
                         'bacc_mean': bm, 'bacc_std': bs,
                         'acc_mean': am,  'acc_std': as_,
                         'f1_mean': fm,   'f1_std': fs})

        if k_cnn is not None:
            idx += 1
            cols_cnn = order_cnn[:k_cnn]
            bm, bs, am, as_, fm, fs = eval_k(X_cnn[:, cols_cnn], y, 'CNN', k_cnn, f'{idx}/{total}', scale=True)
            log(f'  CNN     K={k_cnn}: BAcc={bm:.4f}±{bs:.4f}  Acc={am:.4f}  F1={fm:.4f}')
            rows.append({'source': 'CNN ResNet50 (MI top-K)', 'K': k_cnn,
                         'bacc_mean': bm, 'bacc_std': bs,
                         'acc_mean': am,  'acc_std': as_,
                         'f1_mean': fm,   'f1_std': fs})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, float_format='%.4f')
    log(f'Salvo: {OUT_CSV}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    metrics = [
        ('bacc_mean', 'bacc_std', 'Balanced Accuracy'),
        ('acc_mean',  'acc_std',  'Accuracy'),
        ('f1_mean',   'f1_std',   'Macro F1'),
    ]
    colors = {
        'Pipeline SLIC (MI top-K)': '#2E86AB',
        'CNN ResNet50 (MI top-K)':        '#E74C3C',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('Informational efficiency — SLIC Pipeline vs CNN ResNet50 (WikiArt 80k, 27 styles)',
                 fontsize=12, fontweight='bold', y=1.02)

    for ax, (col_m, col_s, ylabel) in zip(axes, metrics):
        for src, grp in df_out.groupby('source'):
            grp = grp.sort_values('K')
            c   = colors[src]
            ax.plot(grp['K'], grp[col_m], '-o', color=c, lw=2, ms=5, label=src)
            ax.fill_between(grp['K'],
                            grp[col_m] - grp[col_s],
                            grp[col_m] + grp[col_s],
                            alpha=0.15, color=c)
        ax.axvline(N_PIPE, color='#2E86AB', lw=1, ls='--', alpha=0.5)
        ax.text(N_PIPE * 1.05, ax.get_ylim()[0], f'K={N_PIPE}', fontsize=7.5,
                color='#2E86AB', va='bottom')
        ax.axhline(1/27, color='gray', lw=1, ls=':', alpha=0.6, label='Chance')
        ax.set_xlabel('K (dimensions)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=9, framealpha=0.9, loc='upper left')

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f'Salvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
