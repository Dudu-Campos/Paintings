"""
MLP probe nas features estruturais — quanto do gap vs CNN é modelo vs features?

Mesmo dataset/split do LDA (96,949 imgs, 103 estilos, BAcc uniforme).
5-fold CV estratificado, mesmas features (357) + com binary (375).
CrossEntropyLoss com pesos inversos de frequência de classe.
"""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import sys
sys.path.insert(0, '/exp/duducampos/Paintings')
from lda_graph_features import GRAPH_FEATURE_COLS, STRUCT_FEATURE_COLS, SPATIAL_FEATURE_COLS, COLOR_FEATURE_COLS

# ── config ─────────────────────────────────────────────────────────────────
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS  = 150
PATIENCE= 15
BS      = 512
LR      = 3e-4
DROP    = 0.35
HIDDENS = (512, 512, 256, 128)
SEED    = 42
B       = '/exp/duducampos/Paintings'

EPS     = [2,4,6,8,10,12,15,18,22,27,33,40]
ALL_32  = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS + COLOR_FEATURE_COLS
nc_cols = [f'n_comp_e{e}' for e in EPS]
gf_cols = [f'giant_frac_e{e}' for e in EPS]
gi_cols = [f'gini_e{e}' for e in EPS]
# ───────────────────────────────────────────────────────────────────────────


def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def impute(X):
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:,j]) | np.isinf(X[:,j])
        if m.any():
            X[m,j] = float(np.nanmedian(X[~m,j])) if (~m).any() else 0.0
    return X


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_fold(X_tr, y_tr, X_va, y_va, n_classes):
    torch.manual_seed(SEED)

    # class weights: inversão de frequência
    counts  = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = (1.0 / counts)
    weights = weights / weights.mean()
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    sc   = StandardScaler()
    Xtr  = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32, device=DEVICE)
    Xva  = torch.tensor(sc.transform(X_va),     dtype=torch.float32, device=DEVICE)
    ytr  = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    yva  = torch.tensor(y_va, dtype=torch.long, device=DEVICE)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True)

    model  = MLP(X_tr.shape[1], n_classes, HIDDENS, DROP).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit   = nn.CrossEntropyLoss(weight=w_tensor)

    best_bacc, best_ep, no_imp = 0.0, 0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xva).argmax(1).cpu().numpy()
        bacc = balanced_accuracy_score(y_va, preds)
        if bacc > best_bacc:
            best_bacc, best_ep, no_imp = bacc, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break

    return best_bacc, best_ep


def cv5(X, y, label):
    le   = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_cl  = len(le.classes_)
    kf    = StratifiedKFold(5, shuffle=True, random_state=SEED)
    baccs = []
    for fold, (tr, va) in enumerate(kf.split(X, y_enc), 1):
        t0 = time.time()
        b, ep = train_fold(X[tr], y_enc[tr], X[va], y_enc[va], n_cl)
        baccs.append(b)
        log(f'  fold {fold}/5  BAcc={b:.4f}  stop_ep={ep}  ({time.time()-t0:.0f}s)')
    m, s = np.mean(baccs), np.std(baccs)
    log(f'{label:<45} {X.shape[1]:>5} feat  BAcc={m:.4f}+/-{s:.4f}')
    return m


def main():
    log(f'device={DEVICE}')

    # ── carregar features ──────────────────────────────────────────────────
    rag  = pd.read_parquet(f'{B}/plots_v3/full_rag_features.parquet')
    knn  = pd.read_parquet(f'{B}/plots_v3/full_knn_features.parquet')
    del_ = pd.read_parquet(f'{B}/plots_v3/delaunay_features.parquet')
    mst  = pd.read_parquet(f'{B}/plots_v3/mst_features.parquet')
    idx  = pd.read_parquet(f'{B}/coarsening_output_v3/index.parquet')

    rag = rag[rag['style'] != 'unknown'].copy()
    for d in [rag, knn, del_, mst, idx]: d['stem'] = d['stem'].astype(str)
    vc  = rag['style'].value_counts()
    rag = rag[rag['style'].isin(vc[vc >= 31].index)].reset_index(drop=True)

    df = (rag
          .merge(knn[['stem']+ALL_32],  on='stem', how='inner', suffixes=('','_knn'))
          .merge(del_[['stem']+ALL_32], on='stem', how='inner', suffixes=('','_del'))
          .merge(mst[['stem']+ALL_32],  on='stem', how='inner', suffixes=('','_mst'))
          .merge(idx[['stem']+nc_cols+gf_cols+gi_cols], on='stem', how='inner'))

    for name, npy, snpy, col in [
        ('desc', 'desc_full_features.npy', 'desc_full_stems.npy', 'desc_row'),
        ('hil',  'hilbert_features.npy',   'hilbert_stems.npy',   'hil_row'),
        ('bin',  'binary_features.npy',    'binary_stems.npy',    'bin_row'),
    ]:
        stems = np.load(f'{B}/plots_v3/{snpy}', allow_pickle=True).astype(str)
        df = df.merge(pd.DataFrame({'stem': stems, col: np.arange(len(stems))}),
                      on='stem', how='inner')
    df = df.reset_index(drop=True)
    log(f'{len(df):,} imagens, {df["style"].nunique()} estilos')

    NC = df[nc_cols].to_numpy(float); GF = df[gf_cols].to_numpy(float)
    GI = df[gi_cols].to_numpy(float)
    eps_arr = np.array(EPS, dtype=float)
    base     = NC[:,0].clip(1,None); nc_norm = NC/base[:,None]
    half_idx = np.argmax(nc_norm<=0.5,axis=1); half_idx[nc_norm[:,-1]>0.5]=len(EPS)-1
    p90_idx  = np.argmax(nc_norm<=0.10,axis=1); p90_idx[nc_norm[:,-1]>0.10]=len(EPS)-1
    auc_nc   = np.trapezoid(nc_norm, eps_arr, axis=1)
    slope_init = (nc_norm[:,0]-nc_norm[:,1])/(eps_arr[1]-eps_arr[0])
    peak_dgf   = (np.diff(GF,axis=1)/np.diff(eps_arr)).max(axis=1)
    X_perc = impute(np.column_stack([np.log1p(base), nc_norm, GF, GI,
                                       eps_arr[half_idx], eps_arr[p90_idx],
                                       auc_nc, slope_init, peak_dgf]).astype(np.float32))

    y      = df['style'].to_numpy()
    X_rag  = impute(df[ALL_32].to_numpy())
    X_knn  = impute(df[[f'{c}_knn' for c in ALL_32]].to_numpy())
    X_del  = impute(df[[f'{c}_del' for c in ALL_32]].to_numpy())
    X_mst  = impute(df[[f'{c}_mst' for c in ALL_32]].to_numpy())
    X_desc = impute(np.load(f'{B}/plots_v3/desc_full_features.npy')[df['desc_row'].to_numpy()])
    X_hil  = impute(np.load(f'{B}/plots_v3/hilbert_features.npy')[df['hil_row'].to_numpy()])
    X_bin  = impute(np.load(f'{B}/plots_v3/binary_features.npy')[df['bin_row'].to_numpy()])

    X_best = np.hstack([X_rag, X_knn, X_del, X_mst, X_perc, X_desc, X_hil])        # 357
    X_all  = np.hstack([X_best, X_bin])                                               # 375

    log('=== MLP 5-fold CV ===')
    cv5(X_best, y, 'MLP 357 feat (=LDA input)')
    cv5(X_all,  y, 'MLP 375 feat (+binary/LZ)')
    log('Pronto.')


if __name__ == '__main__':
    main()
