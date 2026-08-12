"""
Extrai features do ViT-S pré-treinado (ImageNet, SEM fine-tune de afeto).

Usado como baseline para motivar o fine-tuning: mostra que o ViT-S genérico
não discrimina estilo pictórico tão bem quanto o AffectViT fine-tunado.

Saída: outputs/features/vits_pretrained_features.npy  (N, 384)
       outputs/features/vits_pretrained_stems.npy     (N,)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'images'
META_CSV  = ROOT / 'meta' / 'meta.csv'
OUT_FEATS = ROOT / 'outputs' / 'features' / 'vits_pretrained_features.npy'
OUT_STEMS = ROOT / 'outputs' / 'features' / 'vits_pretrained_stems.npy'
DEVICE    = 'cuda'
BATCH     = 256
FEAT_DIM  = 384

INFER_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class ImageDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf    = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.tf(img), self.paths[idx].stem
        except Exception:
            return torch.zeros(3, 224, 224), ''


def main():
    log('Carregando ViT-S pré-treinado (ImageNet, sem fine-tune)...')
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()
    log(f'  Parâmetros: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    meta       = pd.read_csv(META_CSV)
    stems_meta = {Path(f).stem for f in meta['filename']}
    paths      = sorted(p for p in IMAGE_DIR.glob('*.jpg') if p.stem in stems_meta)
    log(f'{len(paths):,} pinturas WikiArt')

    loader = DataLoader(ImageDataset(paths, INFER_TF),
                        batch_size=BATCH, num_workers=8,
                        pin_memory=True, shuffle=False)

    all_feats, all_stems = [], []
    t0 = time.time()

    with torch.no_grad():
        for i, (imgs, stems) in enumerate(loader):
            mask  = [s != '' for s in stems]
            if not any(mask):
                continue
            imgs  = imgs[[j for j, m in enumerate(mask) if m]].to(DEVICE)
            feats = model(imgs).cpu().numpy()
            all_feats.append(feats)
            all_stems.extend(s for s, m in zip(stems, mask) if m)

            if (i + 1) % 50 == 0:
                done    = len(all_stems)
                elapsed = time.time() - t0
                eta     = elapsed / max(done, 1) * (len(paths) - done)
                log(f'  {done:,}/{len(paths):,}  {elapsed/60:.1f} min  ETA {eta/60:.0f} min')

    feats = np.vstack(all_feats).astype(np.float32)
    stems = np.array(all_stems)
    OUT_FEATS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FEATS, feats)
    np.save(OUT_STEMS, stems)
    log(f'Salvo: {OUT_FEATS}  shape={feats.shape}')


if __name__ == '__main__':
    main()
