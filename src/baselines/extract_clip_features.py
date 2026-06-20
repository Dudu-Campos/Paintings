"""
Extrai embeddings CLIP ViT-L/14 (openai) para todas as imagens do dataset.
Saída: outputs/features/clip_features.npy  (N, 768)
       outputs/features/clip_stems.npy     (N,)
"""
import time
import numpy as np
import pandas as pd
import torch
import open_clip
from pathlib import Path
from PIL import Image

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'images'
META_CSV  = ROOT / 'meta' / 'meta.csv'
OUT_FEATS = ROOT / 'outputs' / 'features' / 'clip_features.npy'
OUT_STEMS = ROOT / 'outputs' / 'features' / 'clip_stems.npy'

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH     = 256
MODEL_NAME = 'ViT-L-14'
PRETRAINED = 'openai'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    log(f'device={DEVICE}  model={MODEL_NAME}/{PRETRAINED}')

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    model.eval()

    meta = pd.read_csv(META_CSV)
    stems_meta = set(Path(f).stem for f in meta['filename'])
    paths = sorted(p for p in IMAGE_DIR.glob('*.jpg') if p.stem in stems_meta)
    log(f'{len(paths):,} imagens')

    all_feats, all_stems = [], []
    t0 = time.time()
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i:i + BATCH]
        imgs = []
        valid_stems = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert('RGB')))
                valid_stems.append(p.stem)
            except Exception:
                pass
        if not imgs:
            continue
        tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad(), torch.autocast(DEVICE):
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normaliza
        all_feats.append(feats.cpu().float().numpy())
        all_stems.extend(valid_stems)
        if (i // BATCH + 1) % 20 == 0 or i + BATCH >= len(paths):
            done = i + len(batch_paths)
            elapsed = time.time() - t0
            eta = elapsed / done * (len(paths) - done)
            log(f'  {done:,}/{len(paths):,}  {elapsed/60:.1f} min  ETA {eta/60:.0f} min')

    X = np.vstack(all_feats)
    np.save(OUT_FEATS, X)
    np.save(OUT_STEMS, np.array(all_stems))
    log(f'Salvo: {OUT_FEATS}  shape={X.shape}  ({(time.time()-t0)/60:.1f} min)')


if __name__ == '__main__':
    main()
