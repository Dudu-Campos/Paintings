"""
Extrai features do Hierarchical Random Walk (hrw_features.py) para todo o dataset.

Por imagem: 5 features × 4 níveis (L=4,5,6,7) = 20 features
  trap_cnt@L{k}, mean_d_cnt@L{k}, rl_mean@L{k}, rl_median@L{k}, frac_len1@L{k}

Saída:
  outputs/features/hrw_features.npy   (N, 20) float32
  outputs/features/hrw_stems.npy      (N,)    str
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent / 'viz'))   # hrw.py
sys.path.insert(0, str(Path(__file__).parent))                   # hrw_diag.py, hrw_features.py

import hrw_features
from hrw_features import feature_vector, _floor

META_CSV  = ROOT / 'meta' / 'meta.csv'
IMAGE_DIR = ROOT / 'raw' / 'images'
OUT_DIR   = ROOT / 'outputs' / 'features'

LEVELS     = (4, 5, 6, 7)
TAU        = 10.0
SEED       = 7
N_WORKERS  = 32
Image.MAX_IMAGE_PIXELS = None   # silence decompression bomb warning


def _load_lab_by_level(path: str, levels=LEVELS):
    img = Image.open(path).convert('RGB')
    w, h = img.size; sq = min(w, h)
    img_sq = img.crop(((w-sq)//2, (h-sq)//2, (w-sq)//2+sq, (h-sq)//2+sq))
    out = {}
    for L in levels:
        side = 2 ** L
        arr  = np.array(img_sq.resize((side, side), Image.BILINEAR),
                        dtype=np.float32) / 255.
        out[L] = skcolor.rgb2lab(arr).astype(np.float32)
    return out


def _worker(args):
    idx, stem, path = args
    try:
        lab_by_level   = _load_lab_by_level(path)
        _, vals, _     = feature_vector(lab_by_level, LEVELS, TAU, seed=SEED)
        return idx, stem, vals.astype(np.float32)
    except Exception as e:
        return idx, stem, None


def main():
    meta = pd.read_csv(META_CSV)
    meta['stem'] = meta['filename'].astype(str).str.split('.').str[0]
    stems = meta['stem'].unique().tolist()

    # filter to existing images
    stems = [s for s in stems if (IMAGE_DIR / f'{s}.jpg').exists()]
    print(f'{len(stems):,} imagens encontradas', flush=True)

    # pre-warm floor cache in main process — forked workers inherit it
    print('Pre-aquecendo geometric_floor...', flush=True)
    for L in LEVELS:
        t0 = time.time()
        _floor(2**L, L)
        print(f'  L={L}: {time.time()-t0:.1f}s', flush=True)

    tasks  = [(i, s, str(IMAGE_DIR / f'{s}.jpg')) for i, s in enumerate(stems)]
    N      = len(tasks)
    n_feat = len(LEVELS) * 5   # 20

    out_f  = np.full((N, n_feat), np.nan, dtype=np.float32)
    out_s  = np.array(stems, dtype=object)
    errs   = 0
    t0     = time.time()

    print(f'\nExtraindo {n_feat} features × {N:,} imagens com {N_WORKERS} workers...', flush=True)
    with Pool(N_WORKERS) as pool:
        for done, (idx, stem, vals) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            if vals is not None:
                out_f[idx] = vals
            else:
                errs += 1
            if done % 5000 == 0 or done == N:
                elapsed = time.time() - t0
                eta     = elapsed / done * (N - done)
                print(f'  {done:,}/{N:,}  errs={errs}  '
                      f'{elapsed/60:.1f}m elapsed  ETA {eta/60:.1f}m', flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / 'hrw_features.npy', out_f)
    np.save(OUT_DIR / 'hrw_stems.npy',    out_s)
    print(f'\nSalvo: hrw_features.npy {out_f.shape}  errs={errs}')
    print(f'Tempo total: {(time.time()-t0)/60:.1f}m')


if __name__ == '__main__':
    main()
