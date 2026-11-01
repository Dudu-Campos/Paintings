"""
Passo 3: Extrai features de afeto das pinturas WikiArt usando AffectViT fine-tunado.

Inferência direta: imagem → ViT-S + proj → 768-dim (sem geração de texto).
~96k imagens em < 1 hora na GPU.

Saída: outputs/features/affect_features.npy  (N, 768)
       outputs/features/affect_stems.npy     (N,)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'images'
META_CSV  = ROOT / 'meta' / 'meta.csv'
MODEL_PT  = ROOT / 'outputs' / 'affect_model' / 'affect_vits.pt'
OUT_FEATS = ROOT / 'outputs' / 'features' / 'affect_features.npy'
OUT_STEMS = ROOT / 'outputs' / 'features' / 'affect_stems.npy'
DEVICE    = 'cuda'
BATCH     = 256
FEAT_DIM  = 384
SBERT_DIM = 768

INFER_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class AffectViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(FEAT_DIM, 512),
            nn.GELU(),
            nn.Linear(512, SBERT_DIM),
        )

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)


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
    log('Carregando AffectViT...')
    model = AffectViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    model.eval()
    log(f'Modelo carregado: {MODEL_PT}')

    meta       = pd.read_csv(META_CSV)
    stems_meta = {Path(f).stem for f in meta['filename']}
    paths      = sorted(p for p in IMAGE_DIR.glob('*.jpg') if p.stem in stems_meta)
    log(f'{len(paths):,} pinturas WikiArt')

    dataset = ImageDataset(paths, INFER_TF)
    loader  = DataLoader(dataset, batch_size=BATCH, num_workers=8,
                         pin_memory=True, shuffle=False)

    all_feats = []
    all_stems = []
    t0 = time.time()

    with torch.no_grad():
        for i, (imgs, stems) in enumerate(loader):
            valid = [s for s in stems if s != '']
            if not valid:
                continue
            mask  = [s != '' for s in stems]
            imgs  = imgs[mask].to(DEVICE)
            feats = model(imgs).cpu().numpy()
            all_feats.append(feats)
            all_stems.extend([s for s, m in zip(stems, mask) if m])

            if (i + 1) % 50 == 0:
                done    = len(all_stems)
                elapsed = time.time() - t0
                eta     = elapsed / max(done, 1) * (len(paths) - done)
                log(f'  {done:,}/{len(paths):,}  {elapsed/60:.1f} min  ETA {eta/60:.0f} min')

    feats = np.vstack(all_feats).astype(np.float32)
    stems = np.array(all_stems)
    np.save(OUT_FEATS, feats)
    np.save(OUT_STEMS, stems)
    log(f'Salvo: {OUT_FEATS}  shape={feats.shape}')


if __name__ == '__main__':
    main()
