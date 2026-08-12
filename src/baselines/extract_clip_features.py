"""
Extrai embeddings CLIP ViT-L/14 (openai) para todas as imagens WikiArt.
Saída: outputs/wikiart_features/clip_features.npy  (N, 768)
       outputs/wikiart_features/clip_stems.npy     (N,)
"""
import time
import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'wikiart'
OUT_FEATS = ROOT / 'outputs' / 'wikiart_features' / 'clip_features.npy'
OUT_STEMS = ROOT / 'outputs' / 'wikiart_features' / 'clip_stems.npy'

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH       = 256
NUM_WORKERS = 8
MODEL_NAME  = 'ViT-L-14'
PRETRAINED  = 'openai'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class WikiArtDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths     = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            return self.preprocess(Image.open(p).convert('RGB')), p.stem
        except Exception:
            return torch.zeros(3, 224, 224), ''


def main():
    log(f'device={DEVICE}  model={MODEL_NAME}/{PRETRAINED}')

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    model.eval()

    paths = sorted(IMAGE_DIR.rglob('*.jpg'))
    log(f'{len(paths):,} imagens em {IMAGE_DIR}')

    loader = DataLoader(WikiArtDataset(paths, preprocess),
                        batch_size=BATCH, num_workers=NUM_WORKERS,
                        pin_memory=True, shuffle=False)

    all_feats, all_stems = [], []
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, (imgs, stems) in enumerate(loader):
            valid = [j for j, s in enumerate(stems) if s != '']
            if not valid:
                continue
            feats = model.encode_image(imgs[valid].to(DEVICE))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())
            all_stems.extend(stems[j] for j in valid)
            if (i + 1) % 50 == 0:
                done    = len(all_stems)
                elapsed = time.time() - t0
                eta     = elapsed / max(done, 1) * (len(paths) - done)
                log(f'  {done:,}/{len(paths):,}  {elapsed/60:.1f}min  ETA {eta/60:.0f}min')

    X = np.vstack(all_feats).astype(np.float32)
    np.save(OUT_FEATS, X)
    np.save(OUT_STEMS, np.array(all_stems))
    log(f'Salvo: {OUT_FEATS}  shape={X.shape}  ({(time.time()-t0)/60:.1f} min)')


if __name__ == '__main__':
    main()
