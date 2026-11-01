"""Cotovelo do limiar de r^2 usado no grafo de redundancia (achado: era
parametro arbitrario, --threshold=0.30 sem justificativa estatistica).
Varre o limiar e mede duas curvas:
  - grau medio (numero medio de conexoes por feature)
  - fracao de nos no maior componente conectado (percolacao)
A fracao do componente gigante tem transicao de fase mais abrupta que o grau
medio, dando um cotovelo mais defensavel -- mesma matematica do achado 26
(persistencia via Kruskal sobre o RAG), so que a "aresta" aqui e r^2 entre
features em vez de distancia cromatica entre superpixels.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_rich_redundancy_layout import load_matrix  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/redundancy_graph_elbow.csv")
    args = parser.parse_args()

    values, family, feature_names = load_matrix(args)
    log(f"{values.shape[0]:,} images x {values.shape[1]} rich visual features")
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
    std = values.std(axis=0)
    keep = std > 1e-9
    values, feature_names = values[:, keep], feature_names[keep]
    n = values.shape[1]

    standard = (values - values.mean(axis=0)) / values.std(axis=0)
    log("computing pairwise r^2 matrix...")
    correlation = (standard.T @ standard) / len(standard)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2
    np.fill_diagonal(r2, 0.0)
    left, right = np.triu_indices(n, 1)
    pair_r2 = r2[left, right]
    log(f"{n} features, {len(pair_r2):,} pares")

    thresholds = np.concatenate([
        np.linspace(0.0, 0.1, 21),
        np.linspace(0.1, 0.6, 51)[1:],
        np.linspace(0.6, 1.0, 21)[1:],
    ])
    rows = []
    for tau in thresholds:
        mask = pair_r2 >= tau
        mean_degree = 2 * mask.sum() / n
        adjacency = csr_matrix((np.ones(mask.sum()), (left[mask], right[mask])), shape=(n, n))
        n_components, labels = connected_components(adjacency, directed=False)
        giant_size = np.bincount(labels).max()
        giant_fraction = giant_size / n
        rows.append({"threshold": tau, "mean_degree": mean_degree,
                    "n_components": n_components, "giant_fraction": giant_fraction})

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    # cotovelo do grau medio: segunda derivada (curvatura maxima) em log-log
    nonzero = result[result.mean_degree > 0]
    log_t = np.log(nonzero.threshold.clip(lower=1e-4))
    log_d = np.log(nonzero.mean_degree.clip(lower=1e-4))
    curvature = np.gradient(np.gradient(log_d, log_t), log_t)
    elbow_degree = nonzero.threshold.iloc[np.abs(curvature).argmax()]

    # cotovelo da percolacao: onde a derivada da fracao do componente gigante e maxima (queda mais abrupta)
    slope = np.gradient(result.giant_fraction, result.threshold)
    elbow_percolation = result.threshold.iloc[np.abs(slope).argmax()]

    print()
    print(result.to_string(index=False))
    print()
    print(f"cotovelo (curvatura do grau medio em log-log): threshold ~= {elbow_degree:.3f}")
    print(f"cotovelo (percolacao, maior derivada da fracao do componente gigante): threshold ~= {elbow_percolation:.3f}")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
