"""Estatísticas de triângulo dos RAGs — a ordem que o bloco de 71 features não tem.

Auditoria do bloco de grafo atual: **nenhuma feature de triângulo, aglomeração
ou transitividade**. O que existe é tudo de primeira ordem — arestas,
densidade, distâncias de cor/espaço, grau, razão de hub. Triângulo é a
estatística de segunda ordem mais básica de um grafo e está ausente.

Onde vale procurar, e onde não:

    RAG (SLIC/Felz)  planar, então clique ≤ 4 (Kuratowski) -- mas a contagem
                     de triângulos varia de 0 a 3n−8, é quantidade real
    kNN (k=10)       NÃO planar; cliques maiores são possíveis e informativas
    Delaunay         é uma triangulação: ~2n−h−2 triângulos por Euler, ou seja
                     determinado por n. Vazio.
    MST              árvore, zero triângulos. Trivialmente vazio.

Por isso só RAG entra aqui (é o que está exportado em `.pt`). O kNN exigiria
reconstruir o grafo a partir dos nós, o que é outro script.

Calibração de expectativa, dada a noite de 15/ago: **três modalidades
ortogonais e informativas sozinhas somaram zero**. "É quantidade nova e não
medida" não prevê contribuição -- foi exatamente esse raciocínio que o achado
de redundância derrubou. Só o marginal decide.

Features por imagem (13): contagem e densidade de triângulos, coeficiente de
aglomeração global e local (média/desvio/máx/gini), fração de nós em algum
triângulo, e as mesmas quantidades ponderadas por similaridade de cor das
arestas -- porque num RAG o triângulo "fechado por cor" é evento diferente do
triângulo meramente topológico.

Saída: outputs/triangle_features_{slic,felz}/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COLUMNS = ["n_nodes", "n_edges", "n_triangles", "triangle_density",
           "transitivity", "clustering_mean", "clustering_std",
           "clustering_max", "clustering_gini", "frac_in_triangle",
           "tri_color_mean", "tri_color_std", "tri_color_ratio"]


def gini(values: np.ndarray) -> float:
    if len(values) == 0 or values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, len(ordered) + 1)
    return float((2 * index - len(ordered) - 1).dot(ordered)
                 / (len(ordered) * ordered.sum()))


def triangle_stats(edges: np.ndarray, n_nodes: int,
                   colors: np.ndarray | None) -> np.ndarray:
    """Triângulos por multiplicação esparsa: A² ∘ A conta caminhos fechados."""
    from scipy.sparse import coo_matrix

    if len(edges) == 0 or n_nodes < 3:
        return np.full(len(COLUMNS), np.nan, dtype=np.float32)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    adjacency = coo_matrix((data, (rows, cols)),
                           shape=(n_nodes, n_nodes)).tocsr()
    adjacency.data[:] = 1.0
    # Triângulos por nó = diag(A³)/2; obtido sem materializar A³.
    squared = adjacency @ adjacency
    per_node = np.asarray(adjacency.multiply(squared).sum(axis=1)).ravel() / 2.0
    total = per_node.sum() / 3.0

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    pairs = degree * (degree - 1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(pairs > 0, per_node / pairs, 0.0)
    transitivity = (3.0 * total / pairs.sum()) if pairs.sum() > 0 else 0.0

    # Triângulo "fechado por cor": num RAG, fechar um triângulo entre regiões
    # cromaticamente parecidas é evento diferente de fechar entre regiões
    # contrastantes, e a topologia sozinha não distingue os dois.
    if colors is not None and len(colors) == n_nodes:
        difference = np.linalg.norm(colors[edges[:, 0]] - colors[edges[:, 1]], axis=1)
        weight = np.exp(-difference / (difference.mean() + 1e-9))
        weighted = coo_matrix((np.concatenate([weight, weight]), (rows, cols)),
                              shape=(n_nodes, n_nodes)).tocsr()
        closed = np.asarray(adjacency.multiply(weighted @ weighted).sum(axis=1)).ravel() / 2.0
        colour_mean = float(closed.sum() / 3.0)
        colour_std = float(closed.std())
        colour_ratio = colour_mean / total if total > 0 else 0.0
    else:
        colour_mean = colour_std = colour_ratio = np.nan

    return np.array([
        n_nodes, len(edges), total,
        total / n_nodes if n_nodes else 0.0,
        transitivity, local.mean(), local.std(), local.max(), gini(local),
        float((per_node > 0).mean()),
        colour_mean, colour_std, colour_ratio], dtype=np.float32)


def process(path: Path) -> tuple[str, np.ndarray | None]:
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
        edges = np.asarray(graph["edge_index"]).T if "edge_index" in graph else None
        if edges is None or edges.ndim != 2:
            return path.stem, None
        if edges.shape[0] == 2 and edges.shape[1] != 2:
            edges = edges.T
        nodes = np.asarray(graph.get("x", graph.get("node_features")))
        n_nodes = len(nodes) if nodes is not None else int(edges.max()) + 1
        colors = nodes[:, :3].astype(np.float64) if nodes is not None and nodes.shape[1] >= 3 else None
        return path.stem, triangle_stats(edges.astype(np.int64), n_nodes, colors)
    except Exception:
        return path.stem, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.graph_dir.glob("*.pt"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] {len(paths):,} grafos", flush=True)

    from multiprocessing import Pool
    stems, values, started = [], [], time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (stem, feature) in enumerate(
                pool.imap_unordered(process, paths, chunksize=64), start=1):
            if feature is not None:
                stems.append(stem)
                values.append(feature)
            if index % 10000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}]   {index:,}/{len(paths):,}  "
                      f"{index / (time.perf_counter() - started):.0f} graf/s  "
                      f"ok={len(values):,}", flush=True)

    frame = pd.DataFrame(np.stack(values), columns=COLUMNS)
    frame.insert(0, "stem", stems)
    frame.to_parquet(args.output / "triangle_features.parquet", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] salvo: {frame.shape} em {args.output}", flush=True)


if __name__ == "__main__":
    main()
