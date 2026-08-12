"""
Extrai features espectrais do Laplaciano normalizado do RAG para cada imagem.

Features (4 por imagem):
  fiedler   — λ₂ via método da potência em (2I-L) com deflação do espaço nulo
  eig_max   — λ_max via método da potência em L
  spec_gap  — fiedler / eig_max  (gap normalizado)
  spec_ratio— eig_max / fiedler  (dispersão espectral)

Método da potência: O(k×E), sem ARPACK, sem fatoração LU.

Saída:
  outputs/wikiart_features/spectral_features.parquet
"""
import struct
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

ROOT      = Path(__file__).parent.parent.parent
COARSE_DIR= ROOT / 'outputs' / 'wikiart_coarsening' / 'coarse'
OUT_PATH  = ROOT / 'outputs' / 'wikiart_features' / 'spectral_features.parquet'
N_WORKERS = 64
K_EIGS    = 11      # pede k+1 para descartar λ₁=0
MAX_NODES = 50_000  # pula grafos degenerados

dt_percol = np.dtype([
    ('epsilon','<f4'),('n_components','<u4'),('giant_size','<u4'),
    ('giant_frac','<f4'),('entropy','<f4'),('mean_size','<f4'),
    ('gini','<f4'),('n_singletons_frac','<f4'),
])
dt_node = np.dtype([
    ('L','<f4'),('a','<f4'),('b','<f4'),
    ('X','<f4'),('Y','<f4'),('area_frac','<f4'),('area_px','<u4'),
])
dt_edge = np.dtype([
    ('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4'),
])


def read_coarse(path):
    try:
        with open(path, 'rb') as f:
            mode         = struct.unpack('<I', f.read(4))[0]
            n_thresholds = struct.unpack('<I', f.read(4))[0]
            _            = struct.unpack('<f', f.read(4))[0] if mode == 1 else None
            f.read(n_thresholds * dt_percol.itemsize)
            if mode != 1:
                return None, None
            nn    = struct.unpack('<I', f.read(4))[0]
            nodes = np.frombuffer(f.read(nn * dt_node.itemsize), dtype=dt_node).copy()
            ne    = struct.unpack('<I', f.read(4))[0]
            edges = np.frombuffer(f.read(ne * dt_edge.itemsize), dtype=dt_edge).copy()
        return nodes, edges
    except Exception:
        return None, None


N_ITER = 60   # iterações do método da potência
RNG    = np.random.default_rng(42)


def _power_max(L, n_iter):
    """λ_max via método da potência em L."""
    v = RNG.standard_normal(L.shape[0]).astype(np.float64)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        v = L @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-14:
            return 0.0
        v /= nrm
    return float(v @ (L @ v))


def _power_fiedler(L, n_iter):
    """λ₂ (Fiedler) via método da potência em M=(2I-L) com deflação de 1/√n."""
    n    = L.shape[0]
    ones = np.ones(n, dtype=np.float64) / np.sqrt(n)
    v    = RNG.standard_normal(n).astype(np.float64)
    v   -= ones * (ones @ v)          # deflate null space
    v   /= np.linalg.norm(v)
    for _ in range(n_iter):
        w  = 2.0 * v - L @ v          # M @ v = (2I - L) @ v
        w -= ones * (ones @ w)
        nrm = np.linalg.norm(w)
        if nrm < 1e-14:
            break
        v = w / nrm
    return float(v @ (L @ v))         # Rayleigh quotient em L → λ₂


def spectral_features(path):
    stem = Path(path).stem
    nodes, edges = read_coarse(path)
    if nodes is None or len(nodes) < 4 or len(nodes) > MAX_NODES or len(edges) == 0:
        return stem, None

    n   = len(nodes)
    src = edges['src'].astype(np.int32)
    dst = edges['dst'].astype(np.int32)
    data = np.ones(len(src) * 2, dtype=np.float32)
    A    = csr_matrix((data, (np.concatenate([src, dst]),
                               np.concatenate([dst, src]))), shape=(n, n))
    L    = laplacian(A, normed=True).astype(np.float64)

    eig_max = _power_max(L, N_ITER)
    fiedler = _power_fiedler(L, N_ITER)
    fiedler = max(fiedler, 0.0)

    if eig_max < 1e-8:
        return stem, None

    return stem, {
        'fiedler':   fiedler,
        'eig_max':   eig_max,
        'spec_gap':  fiedler / eig_max,
        'spec_ratio': eig_max / max(fiedler, 1e-8),
    }


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    paths = [str(p) for p in sorted(COARSE_DIR.rglob('*.coarse'))]
    log(f'{len(paths):,} grafos encontrados')

    rows, skipped = [], 0
    t0   = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for done, (stem, feat) in enumerate(
                pool.imap_unordered(spectral_features, paths, chunksize=32), 1):
            if feat is None:
                skipped += 1
            else:
                rows.append({'stem': stem, **feat})
            if done % 5000 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / done * (len(paths) - done)
                log(f'  {done:,}/{len(paths):,}  ok={len(rows):,}  skip={skipped}  '
                    f'{elapsed/60:.1f}min  ETA {eta/60:.0f}min')

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    log(f'Salvo: {OUT_PATH}  shape={df.shape}  ({(time.time()-t0)/60:.1f} min)')
    log(f'Skipped: {skipped} ({100*skipped/len(paths):.1f}%)')


if __name__ == '__main__':
    main()
