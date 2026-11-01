"""Mesmo grafo de redundancia (r^2 >= threshold), mas colorido por cluster
espectral NAO-SUPERVISIONADO (heuristica eigengap + SpectralClustering sobre
a afinidade r^2, mesmo metodo do achado 28 em analyze_unsupervised_feature_groups.py)
em vez de por familia/procedencia. Responde: "os clusters ficam evidentes sem
usar informacao de rotulo?" -- aqui nem a cor usa rotulo, so topologia do r^2.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_rich_redundancy_layout import load_matrix  # noqa: E402

SEED = 42
TAB20 = plt.get_cmap("tab20")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def choose_k(affinity: np.ndarray, low: int, high: int) -> tuple[int, np.ndarray]:
    degree = affinity.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(len(affinity)) - affinity * inverse[:, None] * inverse[None, :]
    values = np.linalg.eigvalsh(laplacian)[: high + 2]
    gaps = np.diff(values)
    window = range(low - 1, min(high, len(gaps)))
    best = max(window, key=lambda i: gaps[i])
    return best + 1, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=14)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/rich_feature_redundancy_r2_by_cluster/layout.png")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    values, family, feature_names = load_matrix(args)
    log(f"{values.shape[0]:,} images x {values.shape[1]} rich visual features")
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
    std = values.std(axis=0)
    keep = std > 1e-9
    values, family, feature_names = values[:, keep], family[keep], feature_names[keep]
    n = values.shape[1]

    standard = (values - values.mean(axis=0)) / values.std(axis=0)
    log("computing pairwise r^2 matrix...")
    correlation = (standard.T @ standard) / len(standard)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2

    affinity = r2.copy()
    np.fill_diagonal(affinity, 1.0)
    k, spectrum = choose_k(affinity, args.min_k, args.max_k)
    log(f"espectro do Laplaciano: " + " ".join(f"{v:.3f}" for v in spectrum))
    log(f"k escolhido (eigengap) = {k}")
    labels = SpectralClustering(n_clusters=k, affinity="precomputed",
                                assign_labels="kmeans", random_state=SEED,
                                n_init=20).fit_predict(affinity)
    for cluster_id in range(k):
        members = feature_names[labels == cluster_id]
        log(f"  cluster {cluster_id}: {len(members)} features, ex: {list(members[:4])}")

    np.fill_diagonal(r2, 0.0)
    left, right = np.triu_indices(n, 1)
    pair_r2 = r2[left, right]
    strong = pair_r2 >= args.threshold
    log(f"{strong.sum():,} arestas em r^2 >= {args.threshold}")

    edges = [(int(a), int(b), float(w)) for a, b, w in
             zip(left[strong], right[strong], pair_r2[strong])]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    isolated = n - graph.number_of_nodes()
    layout = nx.spring_layout(
        graph, weight="weight", seed=SEED, iterations=300,
        k=2.2 / np.sqrt(max(graph.number_of_nodes(), 1)),
    )
    nodes = np.array(sorted(graph.nodes))
    points = np.array([layout[node] for node in nodes])
    index = {node: position for position, node in enumerate(nodes)}
    node_clusters = labels[nodes]

    figure, axis = plt.subplots(figsize=(9.4, 8.4))
    for a, b, weight in edges:
        pair = [index[a], index[b]]
        axis.plot(points[pair, 0], points[pair, 1], zorder=1, color="#8a8a8a",
                  linewidth=0.30, alpha=0.04 + 0.32 * (weight - args.threshold) / (1 - args.threshold + 1e-9))
    for cluster_id in range(k):
        mask = node_clusters == cluster_id
        if not mask.any():
            continue
        axis.scatter(points[mask, 0], points[mask, 1], s=22, zorder=3,
                     color=TAB20(cluster_id % 20), edgecolor="white", linewidth=0.45,
                     label=f"cluster {cluster_id} ({int(mask.sum())})")
    axis.legend(frameon=False, fontsize=8.5, loc="upper left",
               bbox_to_anchor=(0.0, 1.14), ncol=3, handletextpad=0.4, borderpad=0.3)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.text(0.5, -0.02,
              "cor = cluster espectral nao-supervisionado (so topologia do r^2, sem rotulo/familia)\n"
              f"{isolated} features sem parceiro acima de r^2 = {args.threshold:g}, nao desenhadas",
              transform=axis.transAxes, ha="center", fontsize=8.8, color="#555555")
    figure.tight_layout()
    figure.savefig(args.output, dpi=220, facecolor="white")
    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
