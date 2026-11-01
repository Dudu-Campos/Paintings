"""
lstm_hilbert.py — LSTM baseline na curva de Hilbert cromática.

Pipeline por imagem:
  1. Resize 32×32 (bilinear)
  2. RGB → Lab
  3. Ordena pixels pela curva de Hilbert → sequência (1024, 3)
  4. Normalização z-score pelo training fold

Modelo: BiLSTM (hidden=128, layers=2) → mean pooling sobre steps → FC(256) → n_classes
Protocolo: 5-fold CV estratificado (mesmos SEED/splits do compare.py)
Métrica: balanced accuracy

Baseline responde: a sequência completa (com estrutura temporal) supera
as 57 features de sub-banda extraídas da mesma curva?

Saída: paper/results_lstm_hilbert.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage import color as skcolor
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

ROOT     = Path(__file__).parent.parent.parent
RAW_DIR  = ROOT / 'raw' / 'images'
META_CSV = ROOT / 'meta' / 'meta.csv'
FEAT_DIR = ROOT / 'outputs' / 'features'
OUT_CSV  = ROOT / 'paper' / 'results_lstm_hilbert.csv'

N_SIZE   = 32          # 32×32 → sequência de 1024 steps
SEQ_LEN  = N_SIZE ** 2
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED     = 42
BS       = 512
LR       = 3e-4
EPOCHS   = 100
PATIENCE = 15


# ── Hilbert ─────────────────────────────────────────────────────────────────

def _hilbert_order(n):
    """Retorna (rows, cols) dos pixels na ordem da curva de Hilbert para grid n×n."""
    N = n * n
    d = np.arange(N, dtype=np.int64)
    x = np.zeros(N, dtype=np.int32)
    y = np.zeros(N, dtype=np.int32)
    s = 1
    while s < n:
        rx = ((d >> 1) & 1).astype(np.int32)
        ry = ((d & 1) ^ rx).astype(np.int32)
        m0 = ry == 0
        fl = m0 & (rx == 1)
        x[fl] = s - 1 - x[fl]
        y[fl] = s - 1 - y[fl]
        x[m0], y[m0] = y[m0].copy(), x[m0].copy()
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return y, x   # rows, cols


_ROWS, _COLS = _hilbert_order(N_SIZE)


def image_to_hilbert_seq(img_path):
    """Carrega imagem, retorna sequência Lab na ordem Hilbert (SEQ_LEN, 3)."""
    img  = Image.open(img_path).convert('RGB')
    arr  = np.array(img.resize((N_SIZE, N_SIZE), Image.BILINEAR), dtype=np.float32) / 255.
    lab  = skcolor.rgb2lab(arr).astype(np.float32)
    return lab[_ROWS, _COLS]   # (1024, 3)


# ── Pré-extração das sequências ──────────────────────────────────────────────

def _extract_one(args):
    idx, path = args
    try:
        return idx, image_to_hilbert_seq(path)
    except Exception:
        return idx, None


def extract_all_sequences(paths, n_workers=32):
    """Extrai sequências Hilbert para todas as imagens em paralelo."""
    from multiprocessing import Pool
    N   = len(paths)
    out = np.zeros((N, SEQ_LEN, 3), dtype=np.float32)

    tasks = list(enumerate(paths))
    t0    = time.time()
    errs  = 0
    with Pool(n_workers) as pool:
        for i, (idx, seq) in enumerate(pool.imap_unordered(_extract_one, tasks), 1):
            if seq is not None:
                out[idx] = seq
            else:
                errs += 1
            if i % 10000 == 0 or i == N:
                print(f'  {i:,}/{N:,}  errs={errs}  {(time.time()-t0)/60:.1f}m',
                      flush=True)
    return out


# ── Dataset ──────────────────────────────────────────────────────────────────

class HilbertDataset(Dataset):
    def __init__(self, seqs, labels, mean=None, std=None):
        self.seqs   = seqs    # (N, SEQ_LEN, 3) float32 numpy
        self.labels = labels
        self.mean   = mean    # (3,)
        self.std    = std     # (3,)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx].copy()
        if self.mean is not None:
            seq = (seq - self.mean) / (self.std + 1e-6)
        return torch.tensor(seq, dtype=torch.float32), int(self.labels[idx])


# ── Modelo ───────────────────────────────────────────────────────────────────

class HilbertLSTM(nn.Module):
    def __init__(self, n_classes, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=drop if layers > 1 else 0.)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)       # (B, T, hidden*2)
        pooled = out.mean(dim=1)    # mean pooling over steps
        return self.head(pooled)


# ── Treino de um fold ────────────────────────────────────────────────────────

def train_fold(seqs_tr, y_tr, seqs_va, y_va, n_classes):
    mean = seqs_tr.mean(axis=(0, 1)).astype(np.float32)
    std  = seqs_tr.std(axis=(0, 1)).astype(np.float32)

    ds_tr = HilbertDataset(seqs_tr, y_tr, mean, std)
    ds_va = HilbertDataset(seqs_va, y_va, mean, std)
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True,
                       num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BS, shuffle=False,
                       num_workers=4, pin_memory=True)

    counts  = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = torch.tensor((1. / counts) / (1. / counts).mean(),
                           dtype=torch.float32, device=DEVICE)

    torch.manual_seed(SEED)
    model  = HilbertLSTM(n_classes).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit   = nn.CrossEntropyLoss(weight=weights)

    best_bacc, best_ep, no_imp = 0., 0, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                preds_all.append(model(xb.to(DEVICE)).argmax(1).cpu())
                true_all.append(yb)
        preds = torch.cat(preds_all).numpy()
        true  = torch.cat(true_all).numpy()
        bacc  = balanced_accuracy_score(true, preds)

        if bacc > best_bacc:
            best_bacc, best_ep, no_imp = bacc, ep, 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break

    print(f'    ep={best_ep}  BAcc={best_bacc:.4f}', flush=True)
    return best_bacc


# ── Main ─────────────────────────────────────────────────────────────────────

def load_stems_and_labels():
    """Mesma filtragem do compare.py: ≥5 imagens por estilo."""
    meta = pd.read_csv(META_CSV)
    meta['stem'] = meta['filename'].astype(str).str.split('.').str[0]

    # Usa mesmos stems do pipeline (garantia de interseção)
    hil_s = np.load(FEAT_DIR / 'hilbert_stems.npy', allow_pickle=True).astype(str)
    cnn_s = np.load(FEAT_DIR / 'cnn_full_stems.npy', allow_pickle=True).astype(str)
    common = set(hil_s) & set(cnn_s)

    style_map = meta.set_index('stem')['style'].to_dict()
    stems  = sorted(common)
    styles = np.array([style_map.get(s, 'unknown') for s in stems])

    mask  = styles != 'unknown'
    stems = np.array(stems)[mask]; styles = styles[mask]
    cnt   = pd.Series(styles).value_counts()
    mask2 = np.array([s in set(cnt[cnt >= 5].index) for s in styles])
    stems = stems[mask2]; styles = styles[mask2]

    paths = np.array([str(RAW_DIR / f'{s}.jpg') for s in stems])
    y     = LabelEncoder().fit_transform(styles)
    print(f'Dataset: {len(stems):,} imagens, {len(set(styles))} estilos', flush=True)
    return paths, y


def main():
    print(f'device={DEVICE}', flush=True)
    paths, y = load_stems_and_labels()
    n_classes = len(np.unique(y))

    kf     = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = []
    t0     = time.time()

    print('\nPré-extraindo sequências Hilbert (32×32)...', flush=True)
    seqs = extract_all_sequences(paths, n_workers=32)
    print(f'Sequências shape: {seqs.shape}  ({seqs.nbytes/1e9:.2f} GB)', flush=True)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(paths, y), 1):
        print(f'\n[fold {fold}/5]  tr={len(tr_idx):,}  va={len(va_idx):,}', flush=True)
        bacc = train_fold(seqs[tr_idx], y[tr_idx],
                          seqs[va_idx], y[va_idx], n_classes)
        scores.append(bacc)
        elapsed = time.time() - t0
        print(f'  fold {fold} done  {elapsed/60:.1f} min elapsed', flush=True)

    m, s = float(np.mean(scores)), float(np.std(scores))
    print(f'\nBiLSTM Hilbert: BAcc = {m:.4f} ± {s:.4f}', flush=True)

    df = pd.DataFrame([{
        'method': 'BiLSTM on Hilbert curve (32×32)',
        'n_feat': f'seq {SEQ_LEN}×3',
        'lda_mean': '—', 'lda_std': '—',
        'mlp_mean': m, 'mlp_std': s,
    }])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f'Salvo: {OUT_CSV}')


if __name__ == '__main__':
    main()
