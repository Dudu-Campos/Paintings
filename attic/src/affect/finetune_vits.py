"""
Passo 2: Fine-tune ViT-S por destilação de sentence-BERT.

Pipeline:
  captions → sentence-BERT (frozen) → embeddings 768-dim (targets)
  imagens  → ViT-S + projection head → embeddings 768-dim
  loss: 1 − cosine_similarity(pred, target)

Resultado: modelo leve (22M params) que mapeia imagem → espaço sentimental
sem geração de texto. Inferência nas 96k pinturas WikiArt em < 1 hora.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT      = Path(__file__).parent.parent.parent
CAPTIONS  = ROOT / 'outputs' / 'affect_captions' / 'painting_captions.jsonl'
IMG_DIRS  = [ROOT / 'raw' / 'artgan', ROOT / 'raw' / 'best_artworks']
OUT_DIR   = ROOT / 'outputs' / 'affect_model'
SBERT_ID  = 'sentence-transformers/all-mpnet-base-v2'
DEVICE    = 'cuda'

LR        = 1e-4
EPOCHS    = 20
BATCH     = 64
BACKBONE  = 'vit_small_patch16_224'
FEAT_DIM  = 384   # ViT-S output
SBERT_DIM = 768

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class AffectViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(FEAT_DIM, 512),
            nn.GELU(),
            nn.Linear(512, SBERT_DIM),
        )

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)


class PaintingDataset(Dataset):
    def __init__(self, stems, targets, stem2path, tf):
        self.stems    = stems
        self.targets  = targets
        self.stem2path = stem2path
        self.tf       = tf

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        img = Image.open(self.stem2path[self.stems[idx]]).convert('RGB')
        return self.tf(img), torch.tensor(self.targets[idx], dtype=torch.float32)


def main():
    # Índice stem → path
    stem2path = {}
    for d in IMG_DIRS:
        for p in Path(d).rglob('*'):
            if p.suffix.lower() in EXTS:
                stem2path[p.stem] = p

    # Captions
    records = []
    with open(CAPTIONS) as f:
        for line in f:
            r = json.loads(line)
            if r['stem'] in stem2path:
                records.append(r)
    log(f'{len(records):,} pares imagem-caption')

    # Encode captions com sentence-BERT (frozen, uma vez)
    log(f'Encodando captions com {SBERT_ID}...')
    sbert = SentenceTransformer(SBERT_ID, device=DEVICE)
    captions = [r['caption'] for r in records]
    targets  = sbert.encode(captions, batch_size=256,
                             normalize_embeddings=True, show_progress_bar=True)
    del sbert
    torch.cuda.empty_cache()
    log(f'Targets: {targets.shape}')

    stems = [r['stem'] for r in records]
    dataset = PaintingDataset(stems, targets, stem2path, TRAIN_TF)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                         num_workers=8, pin_memory=True)

    model = AffectViT().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))

    log(f'Treinando {BACKBONE} + proj  ({EPOCHS} épocas, {len(dataset):,} exemplos)...')
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, tgts in loader:
            imgs = imgs.to(DEVICE)
            tgts = tgts.to(DEVICE)
            pred = model(imgs)
            loss = (1 - F.cosine_similarity(pred, tgts)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        log(f'  Época {ep:02d}/{EPOCHS}  loss={avg:.4f}  ({(time.time()-t0)/60:.1f} min)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / 'affect_vits.pt')
    log(f'Modelo salvo: {OUT_DIR}/affect_vits.pt')


if __name__ == '__main__':
    main()
