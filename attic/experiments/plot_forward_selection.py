"""
plot_forward_selection.py — Curva de forward selection com LDA, XGBoost e MLP.

Estratégia:
  - Ordem de seleção já fixada em forward_selection_results.csv (por LDA).
  - Este script avalia XGBoost e MLP nas mesmas combinações cumulativas.
  - LDA lido do CSV existente; XGBoost e MLP computados aqui.

Saída:
  paper/forward_selection_3clf.csv   — resultados completos
  paper/forward_selection_3clf.png   — figura com 3 curvas

Uso:
    python -u paper/plot_forward_selection.py
    python -u paper/plot_forward_selection.py --skip_compute  # só plota o CSV
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs' / 'wikiart_features'
GRAPH_DIR = ROOT / 'outputs' / 'wikiart_slic_features'
META_CSV = ROOT / 'data' / 'artemis' / 'wikiart_meta.csv'
LDA_CSV  = Path(__file__).parent / 'forward_selection_results.csv'
OUT_CSV  = Path(__file__).parent / 'forward_selection_3clf.csv'
OUT_PNG  = Path(__file__).parent / 'forward_selection_3clf.png'

# ── MLP ──────────────────────────────────────────────────────────────────────
HIDDENS  = (512, 512, 256, 128)
DROP     = 0.35
LR       = 3e-4
BS       = 512
EPOCHS   = 150
PATIENCE = 15
SEED     = 42
N_FOLDS  = 5
N_WORKERS = 4        # folds em paralelo; ajustar ao número de cores disponíveis
THREADS_PER_WORKER = 4

# ── XGBoost ──────────────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.5, reg_lambda=2.0,
    objective='multi:softmax', eval_metric='mlogloss',
    n_jobs=THREADS_PER_WORKER, random_state=SEED,
)

# ── Mapeamento grupo → features ───────────────────────────────────────────────
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


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.any():
            X[m, j] = float(np.nanmedian(X[~m, j])) if (~m).any() else 0.0
    return X


# ── Carregamento de dados ─────────────────────────────────────────────────────

def load_data():
    graph = pd.read_parquet(GRAPH_DIR / 'graph_features.parquet')
    knn   = pd.read_parquet(GRAPH_DIR / 'knn_features.parquet')
    knn   = knn.rename(columns={c: f'knn_{c}' for c in KNN_FEAT if c in knn.columns})
    knn_cols = [f'knn_{c}' for c in KNN_FEAT if f'knn_{c}' in knn.columns]

    del_df = pd.read_parquet(GRAPH_DIR / 'delaunay_features.parquet')
    mst_df = pd.read_parquet(GRAPH_DIR / 'mst_features.parquet')
    # Exclude SLIC provenance/timing columns from the predictive matrix.
    del_feat = [c for c in RAG_FEAT if c in del_df.columns]
    mst_feat = [c for c in RAG_FEAT if c in mst_df.columns]
    del_df = del_df.rename(columns={c: f'del_{c}' for c in del_feat})
    mst_df = mst_df.rename(columns={c: f'mst_{c}' for c in mst_feat})
    del_cols = [f'del_{c}' for c in del_feat]
    mst_cols = [f'mst_{c}' for c in mst_feat]

    def load_npy(feat_f, stem_f, prefix):
        feats = np.load(FEAT_DIR / feat_f)
        stems = np.load(FEAT_DIR / stem_f, allow_pickle=True).astype(str)
        cols  = [f'{prefix}_{i}' for i in range(feats.shape[1])]
        return pd.DataFrame(feats, columns=cols).assign(stem=stems), cols

    hil_df, hil_cols   = load_npy('hilbert_features.npy',     'hilbert_stems.npy',     'hil')
    rw_df,  rw_cols    = load_npy('rw_features.npy',          'rw_stems.npy',          'rw')
    hw_df,  hw_cols    = load_npy('hybrid_walk_features.npy', 'hybrid_walk_stems.npy', 'hw')
    desc_df, desc_cols = load_npy('desc_full_features.npy',   'desc_full_stems.npy',   'desc')

    sent = pd.read_parquet(FEAT_DIR / 'sentiment_features.parquet')[['stem'] + SENT_FEAT]

    df = (graph
          .merge(knn[['stem'] + knn_cols],  on='stem', how='inner')
          .merge(del_df[['stem'] + del_cols], on='stem', how='inner')
          .merge(mst_df[['stem'] + mst_cols], on='stem', how='inner')
          .merge(hil_df,                       on='stem', how='inner')
          .merge(rw_df,                         on='stem', how='inner')
          .merge(hw_df,                         on='stem', how='inner')
          .merge(desc_df,                       on='stem', how='inner')
          .merge(sent,                          on='stem', how='inner'))

    meta      = pd.read_csv(META_CSV)
    meta_idx  = dict(zip(meta['stem'].astype(str), meta['style']))
    df['label'] = df['stem'].map(meta_idx)
    df = df.dropna(subset=['label'])
    cnt = df['label'].value_counts()
    df  = df[df['label'].isin(cnt[cnt >= 5].index)].reset_index(drop=True)
    log(f'Dataset: {len(df):,} imagens  {df["label"].nunique()} estilos')

    groups = {
        'SLIC-RAG':    [c for c in RAG_FEAT  if c in df.columns],
        'kNN':         [c for c in knn_cols  if c in df.columns],
        'Delaunay':    [c for c in del_cols  if c in df.columns],
        'MST':         [c for c in mst_cols  if c in df.columns],
        'Hilbert':     [c for c in hil_cols  if c in df.columns],
        'RW':          [c for c in rw_cols   if c in df.columns],
        'Hybrid Walk': [c for c in hw_cols   if c in df.columns],
        'Descriptors': [c for c in desc_cols if c in df.columns],
        'Sentiment':   [c for c in SENT_FEAT if c in df.columns],
    }
    return df, groups


# ── XGBoost ──────────────────────────────────────────────────────────────────

def run_xgb(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X   = impute(X)
    ns  = len(np.unique(y))
    kf  = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    baccs = []
    for tr, va in kf.split(X, y):
        sw  = compute_sample_weight('balanced', y[tr])
        clf = XGBClassifier(**XGB_PARAMS, num_class=ns)
        clf.fit(X[tr], y[tr], sample_weight=sw, verbose=False)
        baccs.append(balanced_accuracy_score(y[va], clf.predict(X[va])))
    return float(np.mean(baccs)), float(np.std(baccs))


# ── MLP ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        layers, prev = [], n_in
        for h in HIDDENS:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(DROP)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


def _mlp_fold_worker(args):
    X_tr, y_tr, X_va, y_va, n_classes = args
    torch.set_num_threads(THREADS_PER_WORKER)
    torch.manual_seed(SEED)

    counts  = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = torch.tensor((1.0 / counts) / (1.0 / counts).mean(), dtype=torch.float32)

    sc    = StandardScaler()
    Xtr   = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
    Xva   = torch.tensor(sc.transform(X_va),     dtype=torch.float32)
    ytr   = torch.tensor(y_tr, dtype=torch.long)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True)
    model  = MLP(X_tr.shape[1], n_classes)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit   = nn.CrossEntropyLoss(weight=weights)

    best_bacc, no_imp = 0.0, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = model(Xva).argmax(1).numpy()
        bacc = balanced_accuracy_score(y_va, preds)
        if bacc > best_bacc:
            best_bacc, no_imp = bacc, 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    return best_bacc


def run_mlp(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X   = impute(X)
    ns  = len(np.unique(y))
    kf  = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    baccs = []
    for fold_idx, (tr, va) in enumerate(kf.split(X, y)):
        bacc = _mlp_fold_worker((X[tr], y[tr], X[va], y[va], ns))
        log(f'    fold {fold_idx+1}/{N_FOLDS}: {bacc:.4f}')
        baccs.append(bacc)
    return float(np.mean(baccs)), float(np.std(baccs))


# ── Plot ─────────────────────────────────────────────────────────────────────

# Categorical palette (slots 1-3, validados all-pairs light e dark)
COLORS = {
    'LDA':      '#2a78d6',  # slot 1 — azul
    'XGBoost':  '#eb6834',  # slot 2 — laranja
    'MLP':      '#1baf7a',  # slot 3 — aqua
}
SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK_SEC = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'


def make_plot(df_all: pd.DataFrame):
    df_all = df_all.sort_values('step').reset_index(drop=True)
    data   = df_all[df_all['step'] > 0].reset_index(drop=True)
    xs     = np.arange(1, len(data) + 1)
    grp_labels = data['group'].values

    # Mapeamento de nomes de grupos para português
    PT = {
        'Descriptors':  'Descritores',
        'Delaunay':     'Delaunay',
        'Hilbert':      'Hilbert',
        'Sentiment':    'Sentimento',
        'SLIC-RAG':     'SLIC-RAG',
        'kNN':          'kNN',
        'MST':          'MST',
        'RW':           'Caminhada Aleatória',
        'Hybrid Walk':  'Caminhada Híbrida',
    }

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for clf, col_m, col_s, label_pt in [
        ('LDA',     'lda_mean', 'lda_std', 'LDA'),
        ('XGBoost', 'xgb_mean', 'xgb_std', 'XGBoost'),
        ('MLP',     'mlp_mean', 'mlp_std', 'MLP'),
    ]:
        ys = data[col_m].values
        ss = data[col_s].values
        c  = COLORS[clf]
        ax.plot(xs, ys, '-o', color=c, lw=2, ms=6, zorder=4,
                label=label_pt, solid_capstyle='round')
        ax.fill_between(xs, ys - ss, ys + ss, alpha=0.12, color=c, zorder=2)
        ax.annotate(f'{ys[-1]:.3f}', (xs[-1], ys[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8.5, color=c, va='center', fontweight='bold')

    tick_labels = [f'+{PT.get(g, g)}' for g in grp_labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9.5, color=INK)
    ax.set_xlim(0.5, xs[-1] + 0.7)

    ax.set_ylabel('Acurácia Balanceada', fontsize=11, color=INK)
    ax.set_xlabel('Grupo de features adicionado', fontsize=10, color=INK_SEC)
    ax.set_title('Seleção progressiva de grupos de features', fontsize=13,
                 fontweight='bold', color=INK, pad=10)

    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)
    ax.tick_params(colors=MUTED, which='both')

    ax.legend(framealpha=0.9, fontsize=10, loc='upper left',
              frameon=True, edgecolor=GRID)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    log(f'Figura salva: {OUT_PNG}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_compute', action='store_true',
                        help='Pula o cálculo e só plota o CSV existente')
    args = parser.parse_args()

    lda_df = pd.read_csv(LDA_CSV)

    if args.skip_compute:
        df_all = pd.read_csv(OUT_CSV)
    else:
        df, groups = load_data()
        y_raw = df['label'].to_numpy(str)
        le    = LabelEncoder()
        y     = le.fit_transform(y_raw)

        # Ordem de seleção fixada pelo LDA
        selection_order = lda_df.loc[lda_df['step'] > 0, 'group'].tolist()

        rows = [{'step': 0, 'group': 'Chance',
                 'lda_mean': lda_df.loc[0, 'bacc_mean'],
                 'lda_std':  lda_df.loc[0, 'bacc_std'],
                 'xgb_mean': np.nan, 'xgb_std': np.nan,
                 'mlp_mean': np.nan, 'mlp_std': np.nan}]

        active_cols = []
        for step, grp in enumerate(selection_order, 1):
            active_cols = active_cols + groups[grp]
            X = impute(df[active_cols].values.astype(np.float32))

            lda_row = lda_df[lda_df['step'] == step].iloc[0]

            log(f'Step {step}/{len(selection_order)}: {lda_row["selected"]}  ({X.shape[1]} feat)')

            log(f'  XGBoost...')
            t0 = time.time()
            xm, xs = run_xgb(X, y)
            log(f'  XGBoost BAcc={xm:.4f}±{xs:.4f}  ({time.time()-t0:.0f}s)')

            log(f'  MLP...')
            t0 = time.time()
            mm, ms = run_mlp(X, y)
            log(f'  MLP BAcc={mm:.4f}±{ms:.4f}  ({time.time()-t0:.0f}s)')

            rows.append({
                'step':     step,
                'group':    grp,
                'selected': lda_row['selected'],
                'lda_mean': lda_row['bacc_mean'],
                'lda_std':  lda_row['bacc_std'],
                'xgb_mean': xm, 'xgb_std': xs,
                'mlp_mean': mm, 'mlp_std': ms,
            })
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False, float_format='%.4f')

        df_all = pd.DataFrame(rows)
        log(f'Resultados salvos: {OUT_CSV}')

    make_plot(df_all)


if __name__ == '__main__':
    main()
