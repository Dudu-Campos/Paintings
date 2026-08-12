"""
compare.py — Comparação pipeline cromático vs CNN ResNet50.

Métodos avaliados:
  - RAG+kNN+Del+MST+Desc+Hilbert  (315 feat)
  - CNN ResNet50                   (2048 feat)

Protocolos:
  - LDA (priors uniformes, 5-fold CV estratificado, random_state=42)
  - MLP (512,512,256,128) + BN + GELU + Dropout(0.35),
        AdamW lr=3e-4, CosineAnnealing, early stop patience=15, epochs=150
        CrossEntropyLoss com pesos inversos de frequência de classe

Dataset: interseção de todos os feature sets (~96k imagens, 103 estilos)
Métrica: balanced accuracy (BAcc)

Saída: paper/results.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs' / 'features'
META_CSV = ROOT / 'meta' / 'meta.csv'
OUT_CSV  = Path(__file__).parent / 'results.csv'

# ── MLP hyperparameters ──────────────────────────────────────────────────────
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDENS  = (512, 512, 256, 128)
DROP     = 0.35
LR       = 3e-4
BS       = 512
EPOCHS   = 150
PATIENCE = 15
SEED     = 42
META_COLS = {'stem', 'artist', 'style', 'genre', 'filename', 'n_nodes_orig'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.any():
            X[m, j] = float(np.nanmedian(X[~m, j])) if (~m).any() else 0.0
    return X


# ── LDA ────────────────────────────────────────────────────────────────────

def run_lda(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X  = impute(X)
    ns = len(np.unique(y))
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('sc',  StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(
            n_components=min(ns - 1, X.shape[1]),
            priors=np.ones(ns) / ns)),
    ])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    sc = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=1)
    return float(sc.mean()), float(sc.std())


# ── MLP ────────────────────────────────────────────────────────────────────

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

    def forward(self, x):
        return self.net(x)


def _train_fold(X_tr, y_tr, X_va, y_va, n_classes):
    torch.manual_seed(SEED)
    counts  = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = (1.0 / counts) / (1.0 / counts).mean()
    w_t     = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xva_s = sc.transform(X_va)

    Xtr = torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(Xva_s, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr,  dtype=torch.long,    device=DEVICE)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True)
    model  = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit   = nn.CrossEntropyLoss(weight=w_t)

    best_bacc, best_ep, no_imp = 0.0, 0, 0
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
            best_bacc, best_ep, no_imp = bacc, ep, 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    return best_bacc, best_ep


def run_mlp(configs: list[tuple[str, np.ndarray]], y: np.ndarray) -> dict:
    ns = len(np.unique(y))
    kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results: dict[str, list] = {label: [] for label, _ in configs}
    t0 = time.time()
    total = len(configs) * 5
    done  = 0
    for label, X in configs:
        X = impute(X)
        for fold, (tr, va) in enumerate(kf.split(X, y), 1):
            bacc, ep = _train_fold(X[tr], y[tr], X[va], y[va], ns)
            results[label].append(bacc)
            done += 1
            log(f'  [{done}/{total}] {label} fold {fold}  BAcc={bacc:.4f}  ep={ep}  ({time.time()-t0:.0f}s)')
    return {label: (float(np.mean(v)), float(np.std(v))) for label, v in results.items()}


# ── Carregamento de dados ───────────────────────────────────────────────────

def load_features():
    rag  = pd.read_parquet(FEAT_DIR / 'full_rag_features.parquet')
    knn  = pd.read_parquet(FEAT_DIR / 'full_knn_features.parquet')
    delx = pd.read_parquet(FEAT_DIR / 'delaunay_features.parquet')
    mst  = pd.read_parquet(FEAT_DIR / 'mst_features.parquet')

    feat_cols = [c for c in rag.columns if c not in META_COLS]

    desc_f = np.load(FEAT_DIR / 'desc_full_features.npy')
    desc_s = np.load(FEAT_DIR / 'desc_full_stems.npy', allow_pickle=True).astype(str)
    hil_f  = np.load(FEAT_DIR / 'hilbert_features.npy')
    hil_s  = np.load(FEAT_DIR / 'hilbert_stems.npy', allow_pickle=True).astype(str)
    cnn_f  = np.load(FEAT_DIR / 'cnn_full_features.npy')
    cnn_s  = np.load(FEAT_DIR / 'cnn_full_stems.npy', allow_pickle=True).astype(str)

    common = (set(rag['stem'].astype(str)) & set(knn['stem'].astype(str)) &
              set(delx['stem'].astype(str)) & set(mst['stem'].astype(str)) &
              set(desc_s) & set(hil_s) & set(cnn_s))

    meta      = pd.read_csv(META_CSV)
    meta['stem'] = meta['filename'].astype(str).str.split('.').str[0]
    style_map = meta.set_index('stem')['style'].to_dict()

    stems  = sorted(common)
    styles = np.array([style_map.get(s, 'unknown') for s in stems])
    mask   = styles != 'unknown'
    stems  = np.array(stems)[mask]; styles = styles[mask]
    cnt    = pd.Series(styles).value_counts()
    mask2  = np.array([s in set(cnt[cnt >= 5].index) for s in styles])
    stems  = stems[mask2]; styles = styles[mask2]

    log(f'Dataset: {len(stems):,} imagens, {len(set(styles))} estilos')

    rag_ord = rag.set_index('stem')
    knn_ord = knn.set_index('stem')
    del_ord = delx.set_index('stem')
    mst_ord = mst.set_index('stem')
    desc_map = {s: i for i, s in enumerate(desc_s)}
    hil_map  = {s: i for i, s in enumerate(hil_s)}
    cnn_map  = {s: i for i, s in enumerate(cnn_s)}

    X_rag  = rag_ord.loc[stems, feat_cols].values.astype(float)
    X_knn  = knn_ord.loc[stems, feat_cols].values.astype(float)
    X_del  = del_ord.loc[stems, feat_cols].values.astype(float)
    X_mst  = mst_ord.loc[stems, feat_cols].values.astype(float)
    X_desc = desc_f[[desc_map[s] for s in stems]]
    X_hil  = hil_f[[hil_map[s] for s in stems]]
    X_cnn  = cnn_f[[cnn_map[s] for s in stems]]
    X_pipe = np.hstack([X_rag, X_knn, X_del, X_mst, X_desc, X_hil])   # 315 feat
    y      = LabelEncoder().fit_transform(styles)

    return X_pipe, X_cnn, y


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log(f'device={DEVICE}')
    X_pipe, X_cnn, y = load_features()

    configs = [
        ('CNN ResNet50',   X_cnn),
        ('Pipeline + CNN', np.hstack([X_pipe, X_cnn])),
    ]

    # LDA (rápido, sequencial)
    lda_results = {}
    for label, X in configs:
        log(f'LDA {label} ({X.shape[1]} feat)...')
        t0 = time.time()
        lda_results[label] = run_lda(X, y)
        log(f'  BAcc={lda_results[label][0]:.4f}±{lda_results[label][1]:.4f}  ({time.time()-t0:.0f}s)')

    # MLP — sequencial na GPU
    log(f'MLP sequencial (device={DEVICE})...')
    mlp_results = run_mlp(configs, y)

    rows = []
    for label, X in configs:
        lda_m, lda_s = lda_results[label]
        mlp_m, mlp_s = mlp_results[label]
        log(f'{label}: LDA={lda_m:.4f}±{lda_s:.4f}  MLP={mlp_m:.4f}±{mlp_s:.4f}')
        rows.append({'method': label, 'n_feat': X.shape[1],
                     'lda_mean': lda_m, 'lda_std': lda_s,
                     'mlp_mean': mlp_m, 'mlp_std': mlp_s})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    log(f'Salvo: {OUT_CSV}')
    print()
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
