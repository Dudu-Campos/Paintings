"""Extrai features espectrais do Laplaciano normalizado do RAG do SLIC.

Le os grafos ja exportados por `extract_slic_rag_graphs.py`
(outputs/slic_gnn_full/graphs/*.pt) -- sem reler imagem, CPU pura.

Substitui a versao legada (removida junto com o pipeline de coarsening em
outputs/wikiart_coarsening/, que so cobria 69.762/79.207 imagens e nunca
entrou no pipeline): mesma ideia -- Laplaciano normalizado do RAG -- mas o
espectro completo via `eigvalsh` em vez de so fiedler/eig_max via metodo da
potencia, e um bloco bem mais rico.

Features por imagem (~41):
  quantile_05..quantile_95 (19, passo 5%)  -- quantis do espectro sem lambda_1=0
  mean, std, skew, kurtosis (4)            -- momentos do espectro
  heat_t{0.05,0.1,0.25,0.5,1,2,4,8,16,32} (10) -- tr(exp(-tL)), heat kernel
  spectral_entropy (1)                     -- entropia de Shannon do espectro normalizado
  fiedler, eig_max, spec_gap, spec_ratio (4) -- compatibilidade com a versao legada
  n_nodes, n_edges, n_components (3)       -- estrutural

Saida: outputs/wikiart_features/spectral_features_v2.parquet
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigvalsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "outputs/slic_gnn_full/graphs"
OUT_PATH = ROOT / "outputs/wikiart_features/spectral_features_v2.parquet"

QUANTILE_LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)
HEAT_SCALES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
EPS_ZERO = 1e-8


def spectral_features(pt_path: str) -> tuple[str, dict | None]:
    stem = Path(pt_path).stem
    try:
        graph = torch.load(pt_path, weights_only=False)
        n_nodes = int(graph["n_nodes"])
        edge_index = graph["edge_index"]
        if n_nodes < 4 or edge_index.shape[1] == 0:
            return stem, None

        adjacency = csr_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float64), (edge_index[0], edge_index[1])),
            shape=(n_nodes, n_nodes),
        )
        laplacian_matrix = laplacian(adjacency, normed=True).toarray().astype(np.float64)
        eigenvalues = np.clip(eigvalsh(laplacian_matrix), 0.0, None)
        eigenvalues.sort()

        n_components = int(np.sum(eigenvalues < EPS_ZERO))
        nonzero = eigenvalues[n_components:]
        if len(nonzero) < 2:
            return stem, None

        quantiles = np.quantile(nonzero, QUANTILE_LEVELS)
        heat = {f"heat_t{t}": float(np.sum(np.exp(-t * eigenvalues))) for t in HEAT_SCALES}

        total = eigenvalues.sum()
        probs = eigenvalues[eigenvalues > 0] / total if total > 0 else np.array([])
        entropy = float(-np.sum(probs * np.log(probs))) if len(probs) else 0.0

        fiedler = float(nonzero[0])
        eig_max = float(eigenvalues[-1])
        result = {
            **{f"quantile_{int(round(q * 100)):02d}": float(v)
               for q, v in zip(QUANTILE_LEVELS, quantiles)},
            "mean": float(nonzero.mean()), "std": float(nonzero.std()),
            "skew": float(skew(nonzero)), "kurtosis": float(kurtosis(nonzero)),
            **heat,
            "spectral_entropy": entropy,
            "fiedler": fiedler, "eig_max": eig_max,
            "spec_gap": fiedler / eig_max if eig_max > 1e-8 else 0.0,
            "spec_ratio": eig_max / max(fiedler, 1e-8),
            "n_nodes": n_nodes, "n_edges": int(edge_index.shape[1] // 2),
            "n_components": n_components,
        }
        return stem, result
    except Exception:
        return stem, None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    paths = [str(p) for p in sorted(args.graph_dir.glob("*.pt"))]
    log(f"{len(paths):,} grafos encontrados em {args.graph_dir}")

    rows, skipped = [], 0
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Pool(processes=args.workers) as pool:
        for done, (stem, feat) in enumerate(
            pool.imap_unordered(spectral_features, paths, chunksize=32), 1
        ):
            if feat is None:
                skipped += 1
            else:
                rows.append({"stem": stem, **feat})
            if done % 5_000 == 0 or done == len(paths):
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (len(paths) - done)
                log(
                    f"{done:,}/{len(paths):,}  ok={len(rows):,}  skip={skipped}  "
                    f"{elapsed / 60:.1f}min  ETA {eta / 60:.0f}min"
                )
            if done % 20_000 == 0:
                pd.DataFrame(rows).to_parquet(args.output, index=False)

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output, index=False)
    log(f"Salvo: {args.output}  shape={frame.shape}  ({(time.perf_counter() - started) / 60:.1f} min)")
    log(f"Skipped: {skipped} ({100 * skipped / len(paths):.1f}%)")


if __name__ == "__main__":
    main()
