"""
extract_hilbert_features.py — Features de série temporal via curva de Hilbert.

Pipeline por imagem:
  1. Resize 128×128 (bilinear)
  2. RGB → Lab
  3. Ordena pixels pela curva de Hilbert → sinal 1D de 16384 pts por canal
  4. 8 sub-bandas: média e std de cada (16 feat por canal)
  5. Média e std globais do sinal (2 feat por canal)
  6. Correlação entre pares de canais Lab (3 feat)
  → 57 features por imagem

Saída:
  {output}/hilbert_features.npy   (N, 57) float32
  {output}/hilbert_stems.npy      (N,) stems

Uso:
    python src/baselines/extract_hilbert_features.py \
        --image_dir raw/images \
        --meta data/artemis/wikiart_meta.csv \
        --output outputs/features \
        --workers 32
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from slic_graph_features import REGIONS, crop_to_region

N_SIZE = 128
N_BANDS = 8

_hilbert_rows = None
_hilbert_cols = None


def _build_hilbert_indices(n):
    """Índices (row, col) em ordem de Hilbert para grade n×n (n potência de 2)."""
    N = n * n
    d = np.arange(N, dtype=np.int64)
    x = np.zeros(N, dtype=np.int32)
    y = np.zeros(N, dtype=np.int32)
    s = 1
    while s < n:
        rx = ((d >> 1) & 1).astype(np.int32)
        ry = ((d & 1) ^ rx).astype(np.int32)
        mask0 = ry == 0
        flip = mask0 & (rx == 1)
        x[flip] = s - 1 - x[flip]
        y[flip] = s - 1 - y[flip]
        x[mask0], y[mask0] = y[mask0].copy(), x[mask0].copy()
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return y, x  # rows, cols


def _init_worker(rows, cols):
    global _hilbert_rows, _hilbert_cols
    _hilbert_rows = rows
    _hilbert_cols = cols


def _extract_one(args):
    stem, img_path, region = args
    global _hilbert_rows, _hilbert_cols
    try:
        with Image.open(img_path) as opened:
            img = crop_to_region(opened.convert('RGB'), region)
        img = img.resize((N_SIZE, N_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        lab = color.rgb2lab(arr).astype(np.float32)  # (128, 128, 3)

        signal = lab[_hilbert_rows, _hilbert_cols, :]  # (16384, 3)

        band_size = (N_SIZE * N_SIZE) // N_BANDS
        feats = []
        for c in range(3):
            ch = signal[:, c]
            for b in range(N_BANDS):
                feats.append(ch[b * band_size:(b + 1) * band_size].mean())
            for b in range(N_BANDS):
                feats.append(ch[b * band_size:(b + 1) * band_size].std())
            feats.append(ch.mean())
            feats.append(ch.std())

        for i, j in [(0, 1), (0, 2), (1, 2)]:
            feats.append(float(np.corrcoef(signal[:, i], signal[:, j])[0, 1]))

        return stem, np.array(feats, dtype=np.float32)
    except Exception:
        return stem, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--meta',      help='Metadata CSV with filename (omit with --manifest)')
    parser.add_argument('--manifest',  help='Derived-image manifest with sample_id/image_relative_path')
    parser.add_argument('--output',    required=True)
    parser.add_argument('--workers',   type=int, default=32)
    parser.add_argument('--region', choices=REGIONS, default='global')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if bool(args.meta) == bool(args.manifest):
        parser.error('provide exactly one of --meta or --manifest')
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        required = {'sample_id', 'image_relative_path'}
        missing = required - set(manifest.columns)
        if missing:
            parser.error(f'manifest missing columns: {sorted(missing)}')
        tasks = [
            (str(sample_id), str(image_dir / relative_path), args.region)
            for sample_id, relative_path in manifest[['sample_id', 'image_relative_path']].itertuples(index=False)
        ]
        if any(not Path(path).is_file() for _, path, _ in tasks):
            parser.error('manifest references missing images')
    else:
        meta = pd.read_csv(args.meta)
        # build stem → path index (suporta subdirs como raw/wikiart/<Style>/)
        stem_to_path = {}
        for p in image_dir.rglob('*'):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                stem_to_path[p.stem] = str(p)
        meta_stems = set(Path(str(r)).stem for r in meta['filename'])
        tasks = [(s, stem_to_path[s], args.region) for s in meta_stems if s in stem_to_path]

    rows, cols = _build_hilbert_indices(N_SIZE)

    t0 = time.time()
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    log(f"{len(tasks)} imagens ({args.region}), {args.workers} workers")

    stems_out, feats_out = [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(rows, cols)) as pool:
        for i, (stem, feat) in enumerate(
            pool.imap_unordered(_extract_one, tasks, chunksize=64)
        ):
            if feat is not None:
                stems_out.append(stem)
                feats_out.append(feat)
            if (i + 1) % 5000 == 0:
                log(f"{i+1}/{len(tasks)}  ({(time.time()-t0)/60:.1f} min)")

    feats_arr = np.array(feats_out, dtype=np.float32)
    stems_arr = np.array(stems_out)

    np.save(out / 'hilbert_features.npy', feats_arr)
    np.save(out / 'hilbert_stems.npy', stems_arr)
    log(f"Salvo: {out}/hilbert_features.npy  shape={feats_arr.shape}")
    log(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
