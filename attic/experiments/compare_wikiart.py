"""
compare_wikiart.py — Pipeline SLIC multigrafo vs CNN ResNet50 no WikiArt 80k.

Pipeline final: SLIC-RAG + kNN + Delaunay + Hilbert + Descriptors + Sentiment (288 feat)
  — determinado por forward selection greedy (RW, HW, MST não contribuem)

CNN baseline: ResNet50 (2048 feat, reduzido via PCA para n_pipe para comparação justa)

Protocolos:
  - LDA (priors uniformes, 5-fold CV, random_state=42)
  - MLP (512,512,256,128) + BN + GELU + Dropout(0.35)

Uso:
  python compare_wikiart.py --task style    # 27 classes
  python compare_wikiart.py --task artist   # ~1116 classes
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / 'outputs/wikiart_features'
GRAPH_DIR = ROOT / 'outputs/wikiart_slic_features'
META_CSV = ROOT / 'data/artemis/wikiart_meta.csv'

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDENS  = (512, 512, 256, 128)
DROP     = 0.35
LR       = 3e-4
BS       = 4096
EPOCHS   = 150
PATIENCE = 15
SEED     = 42

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


def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]) | np.isinf(X[:, j])
        if m.any():
            X[m, j] = float(np.nanmedian(X[~m, j])) if (~m).any() else 0.0
    return X


# ── LDA ───────────────────────────────────────────────────────────────────────

def run_lda(X: np.ndarray, y: np.ndarray, reduce_to: int | None = None) -> dict[str, tuple[float, float]]:
    X  = impute(X)
    ns = len(np.unique(y))
    from sklearn.pipeline import Pipeline
    steps = []
    if reduce_to is not None:
        steps.append(('pca', PCA(n_components=reduce_to, random_state=SEED)))
    steps.append(('lda', LinearDiscriminantAnalysis(
        n_components=min(ns - 1, reduce_to or X.shape[1]),
        priors=np.ones(ns) / ns)))
    pipe = Pipeline(steps)
    cv  = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = cross_validate(pipe, X, y, cv=cv, n_jobs=-1, scoring={
        'bacc': 'balanced_accuracy',
        'acc':  'accuracy',
        'f1':   'f1_macro',
    })
    return {
        'bacc': (float(res['test_bacc'].mean()), float(res['test_bacc'].std())),
        'acc':  (float(res['test_acc'].mean()),  float(res['test_acc'].std())),
        'f1':   (float(res['test_f1'].mean()),   float(res['test_f1'].std())),
    }


# ── MLP ───────────────────────────────────────────────────────────────────────

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


def _train_fold(X_tr, y_tr, X_va, y_va, n_classes, reduce_to=None):
    if reduce_to is not None:
        pca   = PCA(n_components=reduce_to, random_state=SEED)
        X_tr  = pca.fit_transform(X_tr)
        X_va  = pca.transform(X_va)
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

    # dataset inteiro já na GPU: sem transferência por batch
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True,
                        pin_memory=False, num_workers=0)
    model  = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit   = nn.CrossEntropyLoss(weight=w_t)

    best_bacc, best_preds, best_ep, no_imp = 0.0, None, 0, 0
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
            best_bacc, best_preds, best_ep, no_imp = bacc, preds, ep, 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    acc = accuracy_score(y_va, best_preds)
    f1  = f1_score(y_va, best_preds, average='macro')
    return best_bacc, acc, f1, best_ep


def run_mlp(configs: list[tuple[str, np.ndarray, int | None]], y: np.ndarray) -> dict:
    ns = len(np.unique(y))
    kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results: dict[str, dict] = {label: {'bacc':[],'acc':[],'f1':[]} for label, _, __ in configs}
    t0    = time.time()
    total = len(configs) * 5
    done  = 0
    for label, X, reduce_to in configs:
        X = impute(X)
        for fold, (tr, va) in enumerate(kf.split(X, y), 1):
            bacc, acc, f1, ep = _train_fold(X[tr], y[tr], X[va], y[va], ns, reduce_to)
            results[label]['bacc'].append(bacc)
            results[label]['acc'].append(acc)
            results[label]['f1'].append(f1)
            done += 1
            log(f'  [{done}/{total}] {label} fold {fold}  '
                f'BAcc={bacc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  ep={ep}  ({time.time()-t0:.0f}s)')
    return {
        label: {k: (float(np.mean(v)), float(np.std(v))) for k, v in m.items()}
        for label, m in results.items()
    }


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_features(task: str = 'style'):
    graph = pd.read_parquet(GRAPH_DIR / 'graph_features.parquet')
    knn   = pd.read_parquet(GRAPH_DIR / 'knn_features.parquet')
    knn   = knn.rename(columns={c: f'knn_{c}' for c in KNN_FEAT})
    knn_cols = [f'knn_{c}' for c in KNN_FEAT]

    del_df = pd.read_parquet(GRAPH_DIR / 'delaunay_features.parquet')
    # Exclude SLIC provenance/timing columns from the predictive matrix.
    del_feat = [c for c in RAG_FEAT if c in del_df.columns]
    del_df = del_df.rename(columns={c: f'del_{c}' for c in del_feat})
    del_cols = [f'del_{c}' for c in del_feat]

    hil_f  = np.load(FEAT_DIR / 'hilbert_features.npy')
    hil_s  = np.load(FEAT_DIR / 'hilbert_stems.npy',  allow_pickle=True).astype(str)
    desc_f = np.load(FEAT_DIR / 'desc_full_features.npy')
    desc_s = np.load(FEAT_DIR / 'desc_full_stems.npy', allow_pickle=True).astype(str)
    cnn_f  = np.load(FEAT_DIR / 'cnn_full_features.npy')
    cnn_s  = np.load(FEAT_DIR / 'cnn_full_stems.npy',  allow_pickle=True).astype(str)
    clip_f = np.load(FEAT_DIR / 'clip_features.npy')
    clip_s = np.load(FEAT_DIR / 'clip_stems.npy',      allow_pickle=True).astype(str)
    sent   = pd.read_parquet(FEAT_DIR / 'sentiment_features.parquet')[['stem'] + SENT_FEAT]

    df = graph.merge(knn[['stem'] + knn_cols],   on='stem', how='inner')
    df = df.merge(del_df[['stem'] + del_cols],   on='stem', how='inner')
    df = df.merge(sent,                           on='stem', how='inner')

    meta     = pd.read_csv(META_CSV)
    meta_idx = dict(zip(meta['stem'], meta[task]))
    df['label'] = df['stem'].map(meta_idx)
    df = df.dropna(subset=['label'])
    counts = df['label'].value_counts()
    df = df[df['label'].isin(counts[counts >= 5].index)]

    # interseção com Hilbert, Descritores e CNN
    hil_set  = set(hil_s)
    desc_set = set(desc_s)
    cnn_set  = set(cnn_s)
    clip_set = set(clip_s)
    df = df[df['stem'].isin(hil_set) & df['stem'].isin(desc_set)
            & df['stem'].isin(cnn_set) & df['stem'].isin(clip_set)]
    stems = df['stem'].values

    hil_map  = {s: i for i, s in enumerate(hil_s)}
    desc_map = {s: i for i, s in enumerate(desc_s)}
    cnn_map  = {s: i for i, s in enumerate(cnn_s)}
    clip_map = {s: i for i, s in enumerate(clip_s)}

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

    X_cnn  = cnn_f[[cnn_map[s]  for s in stems]].astype(np.float32)
    X_clip = clip_f[[clip_map[s] for s in stems]].astype(np.float32)
    y     = LabelEncoder().fit_transform(df['label'].values)

    n_classes = len(np.unique(y))
    log(f'Dataset: {len(stems):,} imagens  {task}: {n_classes}  chance=1/{n_classes}={1/n_classes:.4f}')
    log(f'Pipeline: {X_pipe.shape[1]} feat   CNN: {X_cnn.shape[1]} feat   CLIP: {X_clip.shape[1]} feat')
    return X_pipe, X_cnn, X_clip, y


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['style', 'artist'], default='style')
    args = ap.parse_args()

    log(f'device={DEVICE}  task={args.task}')
    X_pipe, X_cnn, X_clip, y = load_features(args.task)

    out_csv = ROOT / 'results' / f'compare_wikiart_{args.task}.csv'
    configs = [
        ('Pipeline SLIC (RAG+kNN+Del+Hil+Desc+Sent)', X_pipe, None),
        ('CNN ResNet50 (2048 feat)',              X_cnn,  None),
        ('CLIP ViT-L/14 (768 feat)',              X_clip, None),
    ]

    log('── LDA ──')
    lda_results = {}
    for label, X, reduce_to in configs:
        log(f'  {label} ({X.shape[1]} feat → {reduce_to or X.shape[1]})...')
        t0 = time.time()
        lda_results[label] = run_lda(X, y, reduce_to)
        r = lda_results[label]
        log(f'  BAcc={r["bacc"][0]:.4f}  Acc={r["acc"][0]:.4f}  F1={r["f1"][0]:.4f}  ({time.time()-t0:.0f}s)')

    log('── MLP ──')
    mlp_results = run_mlp(configs, y)

    rows = []
    hdr = f'{"Method":<40} {"LDA BAcc":>10} {"LDA Acc":>9} {"LDA F1":>8}  {"MLP BAcc":>10} {"MLP Acc":>9} {"MLP F1":>8}'
    print(f'\n{"="*len(hdr)}\n{hdr}\n{"-"*len(hdr)}')
    for label, X, reduce_to in configs:
        lr = lda_results[label]
        mr = mlp_results[label]
        print(f'{label:<40} '
              f'{lr["bacc"][0]:.4f}±{lr["bacc"][1]:.4f}  '
              f'{lr["acc"][0]:.4f}±{lr["acc"][1]:.4f}  '
              f'{lr["f1"][0]:.4f}±{lr["f1"][1]:.4f}  '
              f'{mr["bacc"][0]:.4f}±{mr["bacc"][1]:.4f}  '
              f'{mr["acc"][0]:.4f}±{mr["acc"][1]:.4f}  '
              f'{mr["f1"][0]:.4f}±{mr["f1"][1]:.4f}')
        rows.append({'method': label, 'n_feat': X.shape[1], 'n_feat_used': reduce_to or X.shape[1],
                     'lda_bacc_mean': lr['bacc'][0], 'lda_bacc_std': lr['bacc'][1],
                     'lda_acc_mean':  lr['acc'][0],  'lda_acc_std':  lr['acc'][1],
                     'lda_f1_mean':   lr['f1'][0],   'lda_f1_std':   lr['f1'][1],
                     'mlp_bacc_mean': mr['bacc'][0], 'mlp_bacc_std': mr['bacc'][1],
                     'mlp_acc_mean':  mr['acc'][0],  'mlp_acc_std':  mr['acc'][1],
                     'mlp_f1_mean':   mr['f1'][0],   'mlp_f1_std':   mr['f1'][1]})
    print('='*len(hdr))

    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format='%.4f')
    log(f'Salvo: {out_csv}')


if __name__ == '__main__':
    main()
