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
from baseline_cnn import MODEL_REGISTRY, letterbox_resize

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


def resize_center_crop(img: Image.Image, target_size: int = 224) -> Image.Image:
    """`Resize(256, bicubic) + CenterCrop(224)`, a avaliacao canonica do ImageNet.

    O Letterbox preserva proporcao e preenche o resto com cinza -- distribuicao
    que os pesos pre-treinados nunca viram, ja que eles foram treinados com
    `RandomResizedCrop` e avaliados assim. O §8 mediu o par valendo 4,91pp nos
    MESMOS pesos, e os backbones fine-tuned da grade ja usam esta transformacao
    (`--augment strong`), entao usar Letterbox so no congelado handicapa
    justamente a baseline adversaria (§7.17).
    """
    short = int(round(target_size * 256 / 224))
    width, height = img.size
    scale = short / min(width, height)
    img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                     Image.BICUBIC)
    width, height = img.size
    left, top = (width - target_size) // 2, (height - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


class PaintingDataset(Dataset):
    def __init__(self, paths, stems, transform, preprocess="letterbox"):
        self.paths   = paths
        self.stems   = stems
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            img = (letterbox_resize_fast(img, 224) if self.preprocess == 'letterbox'
                   else resize_center_crop(img, 224))
            return self.transform(img), self.stems[idx], True
        except Exception:
            dummy = torch.zeros(3, 224, 224)
            return dummy, self.stems[idx], False


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--image_dir', required=True)
ap.add_argument('--meta',      required=True)
ap.add_argument('--output',    required=True)
ap.add_argument('--init', choices=['imagenet', 'random'], default='imagenet',
                help='random = ancora de transfer zero (ver _build_resnet50)')
ap.add_argument('--model', default='resnet50', choices=sorted(MODEL_REGISTRY),
                help='arquitetura do extrator; a saída é sempre o vetor pós-GAP')
ap.add_argument('--preprocess', choices=['letterbox', 'resize_crop'], default='letterbox',
                help='letterbox = histórico do projeto (barras cinzas); '
                     'resize_crop = Resize(256)+CenterCrop(224), a avaliação '
                     'canônica do ImageNet e a que os fine-tuned usam (§7.17)')
ap.add_argument('--seed', type=int, default=42)
_args = ap.parse_args()

OUT_FEATS = str(Path(_args.output) / 'cnn_full_features.npy')
OUT_STEMS = str(Path(_args.output) / 'cnn_full_stems.npy')
Path(_args.output).mkdir(parents=True, exist_ok=True)

if Path(OUT_FEATS).exists() and Path(OUT_STEMS).exists():
    log('Já existe — saindo.')
    sys.exit(0)

log('Carregando stems...')
meta = pd.read_csv(_args.meta, dtype={'stem': str})
_images = [p for p in Path(_args.image_dir).rglob('*')
           if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')]

# Indexar por (estilo, arquivo) quando o metadado permite, e usar a coluna
# `stem` como está. Indexar só pelo nome do arquivo colapsa imagens homônimas
# em estilos diferentes -- no OilPainting são 258 pares de quadros DISTINTOS
# que dividem nome (`a-soldier.jpg` em Impressionism e em Realism), e a versão
# antiga perdia 378 imagens e ainda gravava um stem que não casava com o das
# demais modalidades. O ramo antigo fica para os datasets já extraídos, cujos
# arquivos precisam continuar reproduzíveis byte a byte.
if {'stem', 'style', 'filename'} <= set(meta.columns):
    # Casar por (estilo, arquivo) exige que as imagens estejam em pastas por
    # estilo. O Multitask tem diretório PLANO (`images/53219.jpg`) e ainda
    # assim traz a coluna `style` no metadado -- ali `p.parent.name` é sempre
    # "images" e o casamento por par devolve zero. O fallback por nome de
    # arquivo cobre esse caso, e só é usado quando o nome é inequívoco no
    # dataset inteiro; onde há homônimos (OilPainting: 258 pares de quadros
    # distintos com o mesmo nome) o par continua sendo a única via, e as
    # imagens ambíguas que ele não resolver são reportadas como ausentes.
    by_pair = {(p.parent.name, p.name): str(p) for p in _images}
    _by_name, _ambiguous = {}, set()
    for _p in _images:
        if _p.name in _by_name:
            _ambiguous.add(_p.name)
        _by_name[_p.name] = str(_p)
    for _name in _ambiguous:
        _by_name.pop(_name, None)
    if _ambiguous:
        log(f'{len(_ambiguous):,} nomes de arquivo repetidos: '
            f'só resolvem por (estilo, arquivo)')

    def _resolve(style: str, filename: str):
        return by_pair.get((style, filename)) or _by_name.get(filename)

    pairs = [(str(r.stem), _resolve(str(r.style), str(r.filename)))
             for r in meta.itertuples(index=False)]
    missing = [s for s, path in pairs if path is None]
    if missing:
        log(f'AVISO: {len(missing):,} sem imagem, ex.: {missing[:3]}')
    stems = [s for s, path in pairs if path is not None]
    paths = [path for _s, path in pairs if path is not None]
else:
    stems_meta = set(Path(f).stem for f in meta['filename'])
    stem_to_path = {p.stem: str(p) for p in _images}
    stems = [s for s in stems_meta if s in stem_to_path]
    paths = [stem_to_path[s] for s in stems]
log(f'{len(stems):,} imagens encontradas')

log(f'Carregando modelo {_args.model} (init={_args.init})...')
model, preprocess, feat_dim = MODEL_REGISTRY[_args.model]['builder'](
    pretrained=_args.init == 'imagenet', seed=_args.seed)
model = model.to(DEVICE)
model.eval()

dataset = PaintingDataset(paths, stems, preprocess, _args.preprocess)
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
