"""
Extrai features do encoder ViT-Base do caption_model (pré-fine-tune de afeto).

Usa o encoder de outputs/caption_model/model.safetensors (ViT-Base + GPT-2
treinado para descrever fotos COCO). Extrai o token [CLS] do encoder (768-dim)
como representação visual baseline — antes do fine-tune de afeto.

Saída: outputs/features/caption_encoder_features.npy  (N, 768)
       outputs/features/caption_encoder_stems.npy     (N,)
"""
import time
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor

Image.MAX_IMAGE_PIXELS = None  # imagens grandes no WikiArt
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

ROOT        = Path(__file__).parent.parent.parent
IMAGE_DIR   = ROOT / 'raw' / 'images'
META_CSV    = ROOT / 'meta' / 'meta.csv'
WEIGHTS_PT  = ROOT / 'outputs' / 'caption_model' / 'model.safetensors'
ENCODER_ID  = 'google/vit-base-patch16-224-in21k'
DECODER_ID  = 'gpt2'
OUT_FEATS   = ROOT / 'outputs' / 'features' / 'caption_encoder_features.npy'
OUT_STEMS   = ROOT / 'outputs' / 'features' / 'caption_encoder_stems.npy'
DEVICE      = 'cuda'
BATCH       = 128


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return img, self.paths[idx].stem
        except Exception:
            return None, ''


def collate(batch):
    imgs   = [b[0] for b in batch if b[0] is not None]
    stems  = [b[1] for b in batch if b[0] is not None]
    return imgs, stems


def main():
    log('Carregando encoder do caption_model (ViT + GPT-2 fine-tunado)...')
    ved       = VisionEncoderDecoderModel.from_pretrained(str(WEIGHTS_PT.parent))
    encoder   = ved.encoder.to(DEVICE).eval()
    processor = ViTImageProcessor.from_pretrained(str(WEIGHTS_PT.parent))
    feat_dim  = encoder.config.hidden_size
    log(f'  Encoder: {feat_dim}-dim, '
        f'{sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params')
    del ved

    meta       = pd.read_csv(META_CSV)
    stems_meta = {Path(f).stem for f in meta['filename']}
    paths      = sorted(p for p in IMAGE_DIR.glob('*.jpg') if p.stem in stems_meta)
    log(f'{len(paths):,} pinturas WikiArt')

    loader = DataLoader(ImageDataset(paths), batch_size=BATCH,
                        num_workers=8, pin_memory=False,
                        shuffle=False, collate_fn=collate)

    all_feats, all_stems = [], []
    t0 = time.time()

    with torch.no_grad():
        for i, (imgs, stems) in enumerate(loader):
            if not imgs:
                continue
            pixel_values = processor(images=imgs, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(DEVICE)
            out   = encoder(pixel_values=pixel_values)
            feats = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
            all_feats.append(feats)
            all_stems.extend(stems)

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
