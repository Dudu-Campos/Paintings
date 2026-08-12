"""
plot_topk_curve.py — Curva acurácia × número de dimensões (K).

Pipeline: top-K features por mutual information → MLP
  MI ranking é supervisionado mas preserva as features originais sem
  distorção — MI K=315 = raw 315 features (reordenadas), bate com ablation.

CNN: top-K componentes PCA → MLP
  PCA necessário para CNN: 2048 features com muito ruído nos últimos
  componentes; PCA K~200 é o pico real da CNN.

MLP idêntico para os dois (5-fold, epochs=150, patience=15) —
  mesmo protocolo do compare.py para endpoints consistentes.

Saída:
  paper/results_topk_curve.csv
  outputs/figures/topk_curve.png
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
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs' / 'features'
META_CSV = ROOT / 'meta' / 'meta.csv'
OUT_CSV  = Path(__file__).parent / 'results_topk_curve.csv'
OUT_PNG  = ROOT / 'outputs' / 'figures' / 'topk_curve.png'

META_COLS = {'stem', 'artist', 'style', 'genre', 'filename', 'n_nodes_orig'}
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED    = 42
BS      = 512
LR      = 3e-4
EPOCHS   = 150
PATIENCE = 15
HIDDENS  = (512, 512, 256, 128)
DROP     = 0.35
N_FOLDS  = 5   # igual ao compare.py — endpoint deve bater com ablation

K_PIPE = [315, 250, 200, 150, 100, 50, 20, 10, 5]
K_CNN  = [2048, 1000, 500, 315, 200, 100, 50, 20, 10, 5]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def impute(X):
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.any():
            X[m, j] = float(np.nanmedian(X[~m, j])) if (~m).any() else 0.0
    return X


# ── MLP ──────────────────────────────────────────────────────────────────────

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
        Xtr_s = sc.fit_transform(X_tr)
        Xva_s = sc.transform(X_va)
    else:
        Xtr_s, Xva_s = X_tr, X_va
    Xtr = torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(Xva_s, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr,  dtype=torch.long,    device=DEVICE)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True)
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
    acc = accuracy_score(y_va, best_preds)
    f1  = f1_score(y_va, best_preds, average='macro', zero_division=0)
    return best_bacc, acc, f1


def eval_mlp(X, y, label, k_val, fold_i, n_folds, scale=True):
    n_classes = len(np.unique(y))
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    baccs, accs, f1s = [], [], []
    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        bacc, acc, f1 = _train_fold(X[tr], y[tr], X[va], y[va], n_classes, scale=scale)
        baccs.append(bacc); accs.append(acc); f1s.append(f1)
        log(f'  [{fold_i}/{n_folds}] {label} K={k_val} fold {fold}/{N_FOLDS}'
            f'  BAcc={bacc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')
    return (float(np.mean(baccs)), float(np.std(baccs)),
            float(np.mean(accs)),  float(np.std(accs)),
            float(np.mean(f1s)),   float(np.std(f1s)))


# ── Carregamento de dados ─────────────────────────────────────────────────────

def load_features():
    rag  = pd.read_parquet(FEAT_DIR / 'full_rag_features.parquet')
    knn  = pd.read_parquet(FEAT_DIR / 'full_knn_features.parquet')
    delx = pd.read_parquet(FEAT_DIR / 'delaunay_features.parquet')
    mst  = pd.read_parquet(FEAT_DIR / 'mst_features.parquet')
    feat_cols = [c for c in rag.columns if c not in META_COLS]

    desc_f = np.load(FEAT_DIR / 'desc_full_features.npy')
    desc_s = np.load(FEAT_DIR / 'desc_full_stems.npy', allow_pickle=True).astype(str)
    hil_f  = np.load(FEAT_DIR / 'hilbert_features.npy')
    hil_s  = np.load(FEAT_DIR / 'hilbert_stems.npy',   allow_pickle=True).astype(str)
    cnn_f  = np.load(FEAT_DIR / 'cnn_full_features.npy')
    cnn_s  = np.load(FEAT_DIR / 'cnn_full_stems.npy',  allow_pickle=True).astype(str)

    common = (set(rag['stem'].astype(str)) & set(knn['stem'].astype(str)) &
              set(delx['stem'].astype(str)) & set(mst['stem'].astype(str)) &
              set(desc_s) & set(hil_s) & set(cnn_s))

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

    ix = lambda s_arr: {s: i for i, s in enumerate(s_arr)}
    desc_m = ix(desc_s); hil_m = ix(hil_s); cnn_m = ix(cnn_s)

    X_pipe = np.hstack([
        rag.set_index('stem').loc[stems, feat_cols].values.astype(float),
        knn.set_index('stem').loc[stems, feat_cols].values.astype(float),
        delx.set_index('stem').loc[stems, feat_cols].values.astype(float),
        mst.set_index('stem').loc[stems, feat_cols].values.astype(float),
        desc_f[[desc_m[s] for s in stems]],
        hil_f[[hil_m[s]  for s in stems]],
    ])
    X_cnn = cnn_f[[cnn_m[s] for s in stems]]
    y     = LabelEncoder().fit_transform(styles)
    return X_pipe, X_cnn, y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log(f'device={DEVICE}')
    X_pipe, X_cnn, y = load_features()
    X_pipe = impute(X_pipe)
    X_cnn  = impute(X_cnn)

    # --- Pipeline: MI ranking ---
    log('Calculando mutual information para ranking do pipeline...')
    t0 = time.time()
    mi = mutual_info_classif(X_pipe, y, random_state=SEED, n_jobs=16)
    order_pipe = np.argsort(mi)[::-1]
    log(f'  MI calculado em {time.time()-t0:.1f}s')

    # --- CNN: PCA + global StandardScaler ---
    # StandardScaler aplicado UMA VEZ aqui fora (não dentro do fold) para
    # evitar amplificar as últimas componentes PCA que têm variância ~0.
    log(f'Fitting PCA no CNN (2048 → {max(K_CNN)})...')
    t0 = time.time()
    pca_cnn = PCA(n_components=max(K_CNN), random_state=SEED)
    X_cnn_pca = pca_cnn.fit_transform(StandardScaler().fit_transform(X_cnn))
    X_cnn_pca = StandardScaler().fit_transform(X_cnn_pca).astype(np.float32)
    ev_cnn = pca_cnn.explained_variance_ratio_.cumsum()[-1] * 100
    log(f'  PCA CNN em {time.time()-t0:.1f}s  var@{max(K_CNN)}: {ev_cnn:.1f}%')

    total_evals = len(K_PIPE) + len(K_CNN)
    rows = []
    done = 0

    # --- Pipeline MI curve ---
    log(f'\nPipeline top-K MI → MLP ({N_FOLDS}-fold):')
    for k in K_PIPE:
        done += 1
        cols = order_pipe[:k]
        bm, bs, am, as_, fm, fs = eval_mlp(X_pipe[:, cols], y, 'Pipeline', k, done, total_evals)
        log(f'  Pipeline K={k}: BAcc={bm:.4f}±{bs:.4f}  Acc={am:.4f}  F1={fm:.4f}')
        rows.append({'source': 'Pipeline cromático (MI top-K)', 'K': k,
                     'bacc_mean': bm, 'bacc_std': bs,
                     'acc_mean':  am, 'acc_std':  as_,
                     'f1_mean':   fm, 'f1_std':   fs})

    # --- CNN PCA curve ---
    log(f'\nCNN PCA K → MLP ({N_FOLDS}-fold):')
    for k in K_CNN:
        done += 1
        bm, bs, am, as_, fm, fs = eval_mlp(X_cnn_pca[:, :k], y, 'CNN', k, done, total_evals, scale=False)
        log(f'  CNN K={k}: BAcc={bm:.4f}±{bs:.4f}  Acc={am:.4f}  F1={fm:.4f}')
        rows.append({'source': 'CNN ResNet50 (PCA)', 'K': k,
                     'bacc_mean': bm, 'bacc_std': bs,
                     'acc_mean':  am, 'acc_std':  as_,
                     'f1_mean':   fm, 'f1_std':   fs})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    log(f'Salvo: {OUT_CSV}')

    # --- Plot: 3 subplots ---
    metrics = [
        ('bacc_mean', 'bacc_std', 'Balanced Accuracy'),
        ('acc_mean',  'acc_std',  'Accuracy'),
        ('f1_mean',   'f1_std',   'Macro F1'),
    ]
    colors = {'Pipeline cromático (MI top-K)': '#2980b9', 'CNN ResNet50 (PCA)': '#e74c3c'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('Eficiência informacional por dimensão — Pipeline cromático vs CNN ResNet50',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, (col_m, col_s, ylabel) in zip(axes, metrics):
        for src, grp in df.groupby('source'):
            grp = grp.sort_values('K')
            c = colors[src]
            ax.plot(grp['K'], grp[col_m], '-o', color=c, lw=2, ms=5,
                    label=src if ax is axes[0] else '_')
            ax.fill_between(grp['K'],
                            grp[col_m] - grp[col_s],
                            grp[col_m] + grp[col_s],
                            alpha=0.15, color=c)
        ax.axvline(315, color='#2980b9', lw=1, ls='--', alpha=0.4)
        ax.text(315 * 1.06, ax.get_ylim()[0], 'K=315', fontsize=7,
                color='#2980b9', va='bottom')
        ax.set_xlabel('K (dimensões)', fontsize=11)
        ax.set_ylabel(f'{ylabel} (3-fold MLP)', fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(fontsize=10, framealpha=0.9, loc='upper left')
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f'Salvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
