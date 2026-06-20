"""
extract_cnn_full.py — Extrai features ResNet50 para todas as imagens do dataset.
Usa DataLoader com múltiplos workers para paralelizar carregamento de imagens.
Salva:
  plots_v3/cnn_full_features.npy   — (N, 2048) float32
  plots_v3/cnn_full_stems.npy      — (N,) stems correspondentes

Uso:
    nohup python extract_cnn_full.py > logs/cnn_extract.log 2>&1 &
"""

import sys, time
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, '.')
from baseline_cnn import _build_resnet50, letterbox_resize

IMAGE_DIR   = './raw/images'
RAG_PARQUET = './plots_v3/full_rag_features.parquet'
OUT_FEATS   = './plots_v3/cnn_full_features.npy'
OUT_STEMS   = './plots_v3/cnn_full_stems.npy'
DEVICE      = 'cuda'
BATCH_SIZE  = 512
NUM_WORKERS = 16

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def letterbox_resize_fast(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Letterbox com BILINEAR + pré-downscale para imagens grandes."""
    w, h = img.size
    if max(w, h) > 1024:
        # pré-downscale rápido com BOX antes do resize fino
        scale = 1024 / max(w, h)
        img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.BOX)
        w, h = img.size
    scale = target_size / max(w, h)
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    canvas.paste(img, ((target_size - nw) // 2, (target_size - nh) // 2))
    return canvas


class PaintingDataset(Dataset):
    def __init__(self, paths, stems, transform):
        self.paths   = paths
        self.stems   = stems
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            img = letterbox_resize_fast(img, 224)
            return self.transform(img), self.stems[idx], True
        except Exception:
            dummy = torch.zeros(3, 224, 224)
            return dummy, self.stems[idx], False


if Path(OUT_FEATS).exists() and Path(OUT_STEMS).exists():
    log('Já existe — saindo.')
    sys.exit(0)

log('Carregando stems...')
rag = pd.read_parquet(RAG_PARQUET)[['stem']]
rag['stem'] = rag['stem'].astype(str)
stems = rag['stem'].tolist()

image_dir = Path(IMAGE_DIR)
paths = [str(image_dir / f'{s}.jpg') for s in stems]
missing = sum(1 for p in paths if not Path(p).exists())
if missing:
    log(f'AVISO: {missing} imagens não encontradas')

log('Carregando modelo ResNet50...')
model, preprocess, feat_dim = _build_resnet50()
model = model.to(DEVICE)
model.eval()

dataset = PaintingDataset(paths, stems, preprocess)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=True, prefetch_factor=2, persistent_workers=True)

log(f'Extraindo CNN para {len(paths):,} imagens  '
    f'device={DEVICE}  batch={BATCH_SIZE}  workers={NUM_WORKERS}')

all_feats  = []
all_stems  = []
n_ok = 0; n_fail = 0
t0 = time.time()

with torch.no_grad():
    for i, (imgs, batch_stems, valid_flags) in enumerate(loader):
        valid = valid_flags.numpy().astype(bool)
        if valid.any():
            out = model(imgs[valid].to(DEVICE)).cpu().numpy()
            all_feats.append(out)
            all_stems.extend([s for s, v in zip(batch_stems, valid) if v])
            n_ok += int(valid.sum())
        n_fail += int((~valid).sum())

        done = (i + 1) * BATCH_SIZE
        if done % 5000 < BATCH_SIZE or i == len(loader) - 1:
            elapsed = time.time() - t0
            log(f'  [{min(done, len(paths)):>6}/{len(paths):,}]  '
                f'{elapsed:.1f}s  {n_ok/elapsed:.1f} img/s  '
                f'ok={n_ok:,} fail={n_fail}')

X = np.concatenate(all_feats, axis=0).astype(np.float32)
valid_stems = np.array(all_stems)

log(f'Extraídos: {n_ok:,}/{len(paths):,}  shape={X.shape}')
Path(OUT_FEATS).parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_FEATS, X)
np.save(OUT_STEMS, valid_stems)
log(f'Salvo: {OUT_FEATS}  {OUT_STEMS}')
