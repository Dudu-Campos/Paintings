"""
extract_hybrid_walk_features.py — Random walk híbrido (espacial + cromático).

Diferença do walk local (4-vizinhos): janela 5×5 (24 candidatos por passo).
Probabilidade de transição combina decaimento espacial E cromático:

    p(i→j) ∝ exp(-ΔE_Lab / σ_c) × exp(-d_pixel² / (2·σ_s²))

σ_c adaptativo por imagem (mediana ΔE entre todos os 4-vizinhos imediatos).
σ_s = 1.5 pixels (decaimento suave dentro da janela 5×5).

Mesma estrutura de walk: 128 restarts × 128 passos, 57 features (8 sub-bandas).
Saída:
  outputs/features/hybrid_walk_features.npy  (N, 57)
  outputs/features/hybrid_walk_stems.npy     (N,)
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
N_WALKS     = 128
WALK_LEN    = 128
N_STEPS     = N_WALKS * WALK_LEN
N_BANDS     = 8
SIGMA_S     = 1.5
SEED        = 42

# 5×5 window offsets, excluindo centro
_OFFSETS = np.array(
    [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if not (dr == 0 and dc == 0)],
    dtype=np.int32)   # (24, 2)

# Pesos espaciais pré-computados (constantes para todos os pixels)
_D_SPATIAL = np.sqrt((_OFFSETS[:, 0]**2 + _OFFSETS[:, 1]**2).astype(np.float32))
_W_SPATIAL = np.exp(-_D_SPATIAL**2 / (2 * SIGMA_S**2)).astype(np.float32)  # (24,)


def _precompute_transitions(lab_flat, H, W):
    N  = H * W
    K  = len(_OFFSETS)
    idx = np.arange(N, dtype=np.int32)
    r   = idx // W
    c   = idx % W

    nb    = np.full((N, K), -1, dtype=np.int32)
    probs = np.zeros((N, K), dtype=np.float32)

    # σ_c adaptativo: mediana ΔE entre vizinhos imediatos (4-conectados)
    up_v  = (r > 0)
    dE_up = np.sqrt(np.sum((lab_flat[idx[up_v]] -
                             lab_flat[(r[up_v]-1)*W + c[up_v]])**2, axis=1))
    sigma_c = float(np.median(dE_up)) if dE_up.size > 0 else 10.0
    sigma_c = max(sigma_c, 1.0)   # evita divisão por zero em imagens uniformes

    for k, (dr, dc) in enumerate(_OFFSETS):
        nr = r + dr
        nc = c + dc
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        nidx  = (nr * W + nc).astype(np.int32)

        vi = np.where(valid)[0]
        nb[vi, k] = nidx[vi]

        dE = np.sqrt(np.sum((lab_flat[vi] - lab_flat[nidx[vi]])**2, axis=1))
        probs[vi, k] = _W_SPATIAL[k] * np.exp(-dE / sigma_c)

    row_sum = probs.sum(axis=1, keepdims=True)
    probs  /= np.maximum(row_sum, 1e-10)
    return nb, probs


def _run_walk(lab_flat, nb, probs, seed):
    rng    = np.random.default_rng(seed)
    N      = len(lab_flat)
    signal = np.empty((N_STEPS, 3), dtype=np.float32)

    for w in range(N_WALKS):
        curr = int(rng.integers(0, N))
        base = w * WALK_LEN
        for t in range(WALK_LEN):
            signal[base + t] = lab_flat[curr]
            nb_curr = nb[curr]
            p_curr  = probs[curr]
            valid   = nb_curr >= 0
            vp = p_curr[valid]; vp = vp / vp.sum()
            vn = nb_curr[valid]
            curr = int(vn[rng.choice(len(vn), p=vp)])

    return signal


def _extract_features(signal):
    band_len = N_STEPS // N_BANDS
    feats = []
    for b in range(N_BANDS):
        seg = signal[b * band_len:(b + 1) * band_len]
        feats.extend(seg.mean(axis=0).tolist())
        feats.extend(seg.std(axis=0).tolist())
    feats.extend(signal.mean(axis=0).tolist())
    feats.extend(signal.std(axis=0).tolist())
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        c = float(np.corrcoef(signal[:, i], signal[:, j])[0, 1])
        feats.append(0.0 if np.isnan(c) else c)
    return np.array(feats, dtype=np.float32)   # 57


def _extract_one(args):
    stem, img_path = args
    try:
        img      = Image.open(img_path).convert('RGB')
        img      = img.resize((N_SIZE, N_SIZE), Image.BILINEAR)
        arr      = np.array(img, dtype=np.float32) / 255.0
        lab      = color.rgb2lab(arr).astype(np.float32)
        lab_flat = lab.reshape(-1, 3)
        nb, probs = _precompute_transitions(lab_flat, N_SIZE, N_SIZE)
        signal    = _run_walk(lab_flat, nb, probs, SEED)
        return stem, _extract_features(signal)
    except Exception:
        return stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--meta',      required=True)
    ap.add_argument('--output',    required=True)
    ap.add_argument('--workers',   type=int, default=16)
    args = ap.parse_args()

    meta       = pd.read_csv(args.meta)
    idir       = Path(args.image_dir)
    odir       = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)
    stems_meta = set(Path(f).stem for f in meta['filename'])
    stem_to_path = {p.stem: str(p) for p in idir.rglob('*')
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')}
    tasks = [(s, stem_to_path[s]) for s in stems_meta if s in stem_to_path]
    print(f'{len(tasks):,} imagens', flush=True)

    t0 = time.time(); feats = []; stems = []; errors = 0
    with Pool(args.workers) as pool:
        for i, (stem, f) in enumerate(pool.imap_unordered(_extract_one, tasks), 1):
            if f is not None:
                feats.append(f); stems.append(stem)
            else:
                errors += 1
            if i % 5000 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta     = elapsed / i * (len(tasks) - i)
                print(f'  {i:,}/{len(tasks):,}  ok={len(stems):,} err={errors}'
                      f'  {elapsed/60:.1f}m  ETA {eta/60:.0f}m', flush=True)

    F = np.vstack(feats).astype(np.float32)
    S = np.array(stems)
    np.save(odir / 'hybrid_walk_features.npy', F)
    np.save(odir / 'hybrid_walk_stems.npy', S)
    print(f'Salvo: {odir}/hybrid_walk_features.npy  shape={F.shape}')


if __name__ == '__main__':
    main()
