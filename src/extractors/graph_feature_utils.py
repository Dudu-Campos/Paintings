"""Segmentation-agnostic graph construction and aggregate features."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


dt_node = np.dtype(
    [
        ("L", "<f4"), ("a", "<f4"), ("b", "<f4"),
        ("X", "<f4"), ("Y", "<f4"),
        ("area_frac", "<f4"), ("area_px", "<u4"),
    ]
)
dt_edge = np.dtype(
    [
        ("src", "<u4"), ("dst", "<u4"),
        ("color_dist", "<f4"), ("spatial_dist", "<f4"),
    ]
)


GRAPH_FEATURE_COLS = [
    "n_nodes", "n_edges", "density",
    "area_gini", "area_entropy", "area_max_frac", "area_mean", "area_std",
    "color_dist_mean", "color_dist_std", "color_dist_max",
    "spatial_dist_mean", "spatial_dist_std", "degree_mean", "degree_std",
]
STRUCT_FEATURE_COLS = [
    "degree_max", "hub_degree_ratio", "n_leaf_frac", "degree_gini",
    "area_second_max_frac", "area_cv",
]
SPATIAL_FEATURE_COLS = ["spatial_density_cv", "spatial_density_max_block_frac"]
COLOR_FEATURE_COLS = [
    "L_mean_w", "L_std_w", "a_mean_w", "a_std_w", "b_mean_w", "b_std_w",
    "chroma_mean_w", "chroma_std_w", "warm_frac",
]


def gini_coefficient(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float(
        (len(values) + 1 - 2 * np.sum(cumulative) / cumulative[-1])
        / len(values)
    )


def shannon_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def extract_graph_features(nodes: np.ndarray, edges: np.ndarray) -> dict:
    n_nodes = len(nodes)
    n_edges = len(edges)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    area = nodes["area_frac"]
    color_dist = edges["color_dist"] if n_edges else np.array([0.0])
    spatial_dist = edges["spatial_dist"] if n_edges else np.array([0.0])

    degree = np.zeros(n_nodes)
    if n_edges:
        np.add.at(degree, edges["src"], 1)
        np.add.at(degree, edges["dst"], 1)

    area_mean = float(area.mean()) if n_nodes else 0.0
    area_std = float(area.std()) if n_nodes else 0.0
    degree_mean = float(degree.mean()) if n_nodes else 0.0
    degree_max = float(degree.max()) if n_nodes else 0.0
    sorted_area = np.sort(area)[::-1]
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": n_edges / max_edges,
        "area_gini": gini_coefficient(area),
        "area_entropy": shannon_entropy(area),
        "area_max_frac": float(area.max()) if n_nodes else 0.0,
        "area_mean": area_mean,
        "area_std": area_std,
        "color_dist_mean": float(color_dist.mean()),
        "color_dist_std": float(color_dist.std()),
        "color_dist_max": float(color_dist.max()),
        "spatial_dist_mean": float(spatial_dist.mean()),
        "spatial_dist_std": float(spatial_dist.std()),
        "degree_mean": degree_mean,
        "degree_std": float(degree.std()) if n_nodes else 0.0,
        "degree_max": degree_max,
        "hub_degree_ratio": degree_max / degree_mean if degree_mean else 0.0,
        "n_leaf_frac": float((degree == 1).sum() / n_nodes) if n_nodes else 0.0,
        "degree_gini": gini_coefficient(degree),
        "area_second_max_frac": float(sorted_area[1]) if n_nodes > 1 else 0.0,
        "area_cv": area_std / area_mean if area_mean else 0.0,
    }


def spatial_density_features(nodes: np.ndarray, grid_size: int = 3) -> dict:
    n_nodes = len(nodes)
    if not n_nodes:
        return {name: 0.0 for name in SPATIAL_FEATURE_COLS}
    x_bin = (np.clip(nodes["X"], 0, 0.999999) * grid_size).astype(int)
    y_bin = (np.clip(nodes["Y"], 0, 0.999999) * grid_size).astype(int)
    counts = np.bincount(
        y_bin * grid_size + x_bin, minlength=grid_size * grid_size
    ).astype(float)
    mean = counts.mean()
    return {
        "spatial_density_cv": float(counts.std() / mean) if mean else 0.0,
        "spatial_density_max_block_frac": float(counts.max() / n_nodes),
    }


def color_features(nodes: np.ndarray) -> dict:
    if not len(nodes):
        return {name: 0.0 for name in COLOR_FEATURE_COLS}
    weights = nodes["area_frac"].astype(np.float64)
    if weights.sum() <= 0:
        return {name: 0.0 for name in COLOR_FEATURE_COLS}
    weights /= weights.sum()
    lightness = nodes["L"].astype(np.float64)
    a_axis = nodes["a"].astype(np.float64)
    b_axis = nodes["b"].astype(np.float64)
    chroma = np.sqrt(a_axis**2 + b_axis**2)

    def mean(values: np.ndarray) -> float:
        return float(np.sum(weights * values))

    def std(values: np.ndarray) -> float:
        center = np.sum(weights * values)
        return float(np.sqrt(np.sum(weights * (values - center) ** 2)))

    return {
        "L_mean_w": mean(lightness), "L_std_w": std(lightness),
        "a_mean_w": mean(a_axis), "a_std_w": std(a_axis),
        "b_mean_w": mean(b_axis), "b_std_w": std(b_axis),
        "chroma_mean_w": mean(chroma), "chroma_std_w": std(chroma),
        "warm_frac": float(np.sum(weights[b_axis > 0])),
    }


def all_graph_features(nodes: np.ndarray, edges: np.ndarray) -> dict:
    features = extract_graph_features(nodes, edges)
    features.update(spatial_density_features(nodes))
    features.update(color_features(nodes))
    return features


def make_edges(
    pairs: list[tuple[int, int]],
    color_dist: np.ndarray,
    spatial_dist: np.ndarray,
) -> np.ndarray:
    edges = np.zeros(len(pairs), dtype=dt_edge)
    if pairs:
        edges["src"] = [pair[0] for pair in pairs]
        edges["dst"] = [pair[1] for pair in pairs]
        edges["color_dist"] = color_dist
        edges["spatial_dist"] = spatial_dist
    return edges


def build_knn_edges(nodes: np.ndarray, k: int = 10) -> np.ndarray:
    n_nodes = len(nodes)
    if n_nodes < 2:
        return np.zeros(0, dtype=dt_edge)
    k_effective = min(k, n_nodes - 1)
    color = np.column_stack([nodes["L"], nodes["a"], nodes["b"]])
    position = np.column_stack([nodes["X"], nodes["Y"]])
    color_scale = color.std(axis=0)
    position_scale = position.std(axis=0)
    color_scale[color_scale == 0] = 1.0
    position_scale[position_scale == 0] = 1.0
    attributes = np.column_stack([color / color_scale, position / position_scale])
    neighbours = NearestNeighbors(n_neighbors=k_effective + 1).fit(attributes)
    indices = neighbours.kneighbors(attributes, return_distance=False)
    pairs = sorted(
        {
            (min(index, int(neighbour)), max(index, int(neighbour)))
            for index in range(n_nodes)
            for neighbour in indices[index, 1:]
        }
    )
    src = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
    dst = np.fromiter((pair[1] for pair in pairs), dtype=np.int64)
    color_dist = np.linalg.norm(color[src] - color[dst], axis=1).astype(np.float32)
    spatial_dist = np.linalg.norm(position[src] - position[dst], axis=1).astype(np.float32)
    return make_edges(pairs, color_dist, spatial_dist)


def build_delaunay_edges(nodes: np.ndarray) -> np.ndarray:
    if len(nodes) < 3:
        return np.zeros(0, dtype=dt_edge)
    position = np.column_stack([nodes["X"], nodes["Y"]])
    color = np.column_stack([nodes["L"], nodes["a"], nodes["b"]])
    try:
        triangulation = Delaunay(position)
    except Exception:
        return np.zeros(0, dtype=dt_edge)
    pairs = sorted(
        {
            tuple(sorted((int(simplex[index]), int(simplex[(index + 1) % 3]))))
            for simplex in triangulation.simplices
            for index in range(3)
        }
    )
    src = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
    dst = np.fromiter((pair[1] for pair in pairs), dtype=np.int64)
    return make_edges(
        pairs,
        np.linalg.norm(color[src] - color[dst], axis=1).astype(np.float32),
        np.linalg.norm(position[src] - position[dst], axis=1).astype(np.float32),
    )


def build_mst_edges(nodes: np.ndarray, rag_edges: np.ndarray) -> np.ndarray:
    n_nodes = len(nodes)
    if n_nodes < 2 or not len(rag_edges):
        return np.zeros(0, dtype=dt_edge)
    src = rag_edges["src"].astype(np.int32)
    dst = rag_edges["dst"].astype(np.int32)
    weights = rag_edges["color_dist"].astype(np.float32)
    adjacency = csr_matrix((weights, (src, dst)), shape=(n_nodes, n_nodes))
    tree = minimum_spanning_tree(adjacency + adjacency.T).tocoo()
    pairs = list(zip(tree.row.tolist(), tree.col.tolist()))
    if not pairs:
        return np.zeros(0, dtype=dt_edge)
    position = np.column_stack([nodes["X"], nodes["Y"]])
    tree_src = tree.row.astype(np.int64)
    tree_dst = tree.col.astype(np.int64)
    return make_edges(
        pairs,
        tree.data.astype(np.float32),
        np.linalg.norm(position[tree_src] - position[tree_dst], axis=1).astype(np.float32),
    )
