"""
extract_random_walk_features.py — Features de série temporal via random walk cromático.

Pipeline por imagem:
  1. Resize 128×128 (bilinear)
  2. RGB → Lab
  3. Pré-computa probabilidades de transição: p(i→j) ∝ 1/(1 + ΔE_Lab(i,j))
     para os 4 vizinhos espaciais de cada pixel
  4. Random walk de N² passos com seed fixo
  5. Mesmas 57 features do Hilbert (8 sub-bandas × mean/std × 3 canais +
     mean/std globais × 3 + correlações par-a-par)

Propriedade-chave: o walk fica "preso" em regiões cromaticamente homogêneas
(baixa variância nas sub-bandas correspondentes) e cruza bordas cromáticas
com menor frequência — captura estrutura local diferente do Hilbert (espacial puro)
e do TSP (cromático sem restrição espacial).

Saída:
  {output}/rw_features.npy   (N, 57) float32
  {output}/rw_stems.npy      (N,) stems
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

N_SIZE      = 128
N_WALKS     = 128   # restarts — garante cobertura global da imagem
WALK_LEN    = 128   # passos por walk — captura vizinhança local antes de restartar
N_STEPS     = N_WALKS * WALK_LEN   # 16384 total — comparável ao Hilbert
N_BANDS     = 8
SEED        = 42


def _precompute_transitions(lab_flat, H, W):
    """Pré-computa matriz de transição vetorizada."""
    N  = H * W
    idx = np.arange(N, dtype=np.int32)
    r   = idx // W
    c   = idx % W

    up    = np.where(r > 0,   (r - 1) * W + c, -1).astype(np.int32)
    down  = np.where(r < H-1, (r + 1) * W + c, -1).astype(np.int32)
    left  = np.where(c > 0,    r * W + (c - 1), -1).astype(np.int32)
    right = np.where(c < W-1,  r * W + (c + 1), -1).astype(np.int32)
    nb    = np.stack([up, down, left, right], axis=1)   # (N, 4)

    probs = np.zeros((N, 4), dtype=np.float32)
    for d in range(4):
        valid = nb[:, d] >= 0
        nidx  = nb[valid, d]
        dist  = np.sqrt(np.sum((lab_flat[valid] - lab_flat[nidx]) ** 2, axis=1))
        probs[valid, d] = 1.0 / (1.0 + dist)

    row_sum = probs.sum(axis=1, keepdims=True)
    probs  /= np.maximum(row_sum, 1e-10)
    return nb, probs


def _run_walk(lab_flat, nb, probs, n_walks, walk_len, seed):
    """
    N_WALKS walks truncados de WALK_LEN passos, cada um com restart aleatório.
    Garante cobertura global: sem restart, o walk fica preso em regiões homogêneas.
    Retorna sinal concatenado (n_walks * walk_len, 3).
    """
    rng    = np.random.default_rng(seed)
    N      = len(lab_flat)
    signal = np.empty((n_walks * walk_len, 3), dtype=np.float32)

    for w in range(n_walks):
        curr = int(rng.integers(0, N))   # restart aleatório a cada walk
        base = w * walk_len
        for t in range(walk_len):
            signal[base + t] = lab_flat[curr]
            nb_curr = nb[curr]
            p_curr  = probs[curr]
            valid   = nb_curr >= 0
            vp = p_curr[valid]; vp = vp / vp.sum()
            vn = nb_curr[valid]
            curr = int(vn[rng.choice(len(vn), p=vp)])

    return signal


def _extract_features(signal):
    """Mesmas 57 features do Hilbert: sub-bandas + globais + correlações."""
    T        = signal.shape[0]
    band_len = T // N_BANDS
    feats    = []

    # Sub-bandas: mean + std por canal
    for b in range(N_BANDS):
        seg = signal[b * band_len:(b + 1) * band_len]
        feats.extend(seg.mean(axis=0).tolist())
        feats.extend(seg.std(axis=0).tolist())

    # Globais: mean + std por canal
    feats.extend(signal.mean(axis=0).tolist())
    feats.extend(signal.std(axis=0).tolist())

    # Correlações par-a-par: (L,a), (L,b), (a,b)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        c = float(np.corrcoef(signal[:, i], signal[:, j])[0, 1])
        feats.append(0.0 if np.isnan(c) else c)

    return np.array(feats, dtype=np.float32)   # 57


def _init_worker():
    pass


def _extract_one(args):
    stem, img_path = args
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((N_SIZE, N_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        lab = color.rgb2lab(arr).astype(np.float32)      # (128, 128, 3)

        lab_flat = lab.reshape(-1, 3)
        nb, probs = _precompute_transitions(lab_flat, N_SIZE, N_SIZE)
        signal    = _run_walk(lab_flat, nb, probs, N_WALKS, WALK_LEN, SEED)
        feats     = _extract_features(signal)
        return stem, feats
    except Exception as e:
        return stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--meta',      required=True)
    ap.add_argument('--output',    required=True)
    ap.add_argument('--workers',   type=int, default=16)
    args = ap.parse_args()

    meta  = pd.read_csv(args.meta)
    idir  = Path(args.image_dir)
    odir  = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    stems_meta = set(Path(f).stem for f in meta['filename'])
    stem_to_path = {p.stem: str(p) for p in idir.rglob('*')
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')}
    tasks = [(s, stem_to_path[s]) for s in stems_meta if s in stem_to_path]
    print(f'{len(tasks):,} imagens', flush=True)

    t0     = time.time()
    feats  = []
    stems  = []
    errors = 0

    with Pool(args.workers) as pool:
        for i, (stem, f) in enumerate(pool.imap_unordered(_extract_one, tasks), 1):
            if f is not None:
                feats.append(f)
                stems.append(stem)
            else:
                errors += 1
            if i % 2000 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta     = elapsed / i * (len(tasks) - i)
                print(f'  {i:,}/{len(tasks):,}  ok={len(stems):,} err={errors}'
                      f'  {elapsed/60:.1f} min  ETA {eta/60:.0f} min', flush=True)

    F = np.vstack(feats).astype(np.float32)
    S = np.array(stems)
    np.save(odir / 'rw_features.npy', F)
    np.save(odir / 'rw_stems.npy', S)
    print(f'Salvo: {odir}/rw_features.npy  shape={F.shape}')


if __name__ == '__main__':
    main()
