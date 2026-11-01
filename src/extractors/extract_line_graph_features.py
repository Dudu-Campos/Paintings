#!/usr/bin/env python3
"""Extract interpretable boundary-junction features from line graphs of SLIC RAGs.

The nodes of L(G) are undirected RAG edges (region boundaries).  Two line-graph
nodes are adjacent when the corresponding boundaries meet at one SLIC region.
Besides basic L(G) topology, this extractor summarizes attributed wedges:
contrast continuity, centroid-junction angle, outer-region colour distance and
luminance reversals.  It consumes exported graphs and never rereads images.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((values.size + 1 - 2 * cumulative.sum() / cumulative[-1]) / values.size)


def corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def stats(values: np.ndarray, prefix: str, quantiles: tuple[int, ...] = (10, 50, 90)) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
            **{f"{prefix}_q{q}": 0.0 for q in quantiles},
        }
    result = {f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std())}
    result.update({f"{prefix}_q{q}": float(np.percentile(values, q)) for q in quantiles})
    return result


def spatial_grid(values: np.ndarray, grid_size: int = 3) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    bins = np.clip(values, 0, 0.999999)
    x = (bins[:, 0] * grid_size).astype(int)
    y = (bins[:, 1] * grid_size).astype(int)
    counts = np.bincount(y * grid_size + x, minlength=grid_size * grid_size).astype(float)
    return float(counts.std() / counts.mean()), float(counts.max() / len(values))


def extract_one(task: tuple[str, str, str]) -> tuple[dict | None, str | None]:
    stem, style, path_value = task
    try:
        graph = torch.load(path_value, map_location="cpu", weights_only=False)
        nodes = np.asarray(graph["x"], dtype=np.float64)
        n_nodes = len(nodes)
        n_edges = int(graph["n_edges"])
        # Exporter stores undirected edges once, then their reverse copies.
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)[:, :n_edges]
        edge_attr = np.asarray(graph["edge_attr"], dtype=np.float64)[:n_edges]
        source, target = edge_index
        if n_edges and (np.any(source >= target) or len(np.unique(source * n_nodes + target)) != n_edges):
            raise ValueError("first edge block is not a unique src<dst undirected RAG")

        degree = np.bincount(np.concatenate([source, target]), minlength=n_nodes)
        line_degree = degree[source] + degree[target] - 2 if n_edges else np.zeros(0)
        wedge_load = degree * np.maximum(degree - 1, 0) / 2
        n_line_edges = int(wedge_load.sum())
        possible = n_edges * (n_edges - 1) / 2

        color = edge_attr[:, 0] if n_edges else np.zeros(0)
        spatial = edge_attr[:, 1] if n_edges else np.zeros(0)
        position = nodes[:, 3:5]
        midpoint = (position[source] + position[target]) / 2 if n_edges else np.empty((0, 2))
        edge_vector = position[target] - position[source] if n_edges else np.empty((0, 2))
        orientation = np.arctan2(edge_vector[:, 1], edge_vector[:, 0]) if n_edges else np.zeros(0)
        orientation_coherence = (
            float(np.hypot(np.cos(2 * orientation).mean(), np.sin(2 * orientation).mean()))
            if n_edges else 0.0
        )
        area = np.exp(nodes[:, 5])
        area_ratio = (
            np.abs(np.log(np.clip(area[source], 1e-12, None)) - np.log(np.clip(area[target], 1e-12, None)))
            if n_edges else np.zeros(0)
        )

        first_edges: list[np.ndarray] = []
        second_edges: list[np.ndarray] = []
        shared_nodes: list[np.ndarray] = []
        outer_first: list[np.ndarray] = []
        outer_second: list[np.ndarray] = []
        incident = [[] for _ in range(n_nodes)]
        for edge_id, (left, right) in enumerate(zip(source, target, strict=True)):
            incident[int(left)].append(edge_id)
            incident[int(right)].append(edge_id)
        for shared, edge_ids_list in enumerate(incident):
            if len(edge_ids_list) < 2:
                continue
            edge_ids = np.asarray(edge_ids_list, dtype=np.int64)
            left_index, right_index = np.triu_indices(len(edge_ids), 1)
            first = edge_ids[left_index]
            second = edge_ids[right_index]
            first_edges.append(first)
            second_edges.append(second)
            shared_nodes.append(np.full(len(first), shared, dtype=np.int64))
            outer_first.append(np.where(source[first] == shared, target[first], source[first]))
            outer_second.append(np.where(source[second] == shared, target[second], source[second]))

        if first_edges:
            edge_a = np.concatenate(first_edges)
            edge_b = np.concatenate(second_edges)
            shared = np.concatenate(shared_nodes)
            outer_a = np.concatenate(outer_first)
            outer_b = np.concatenate(outer_second)
        else:
            edge_a = edge_b = shared = outer_a = outer_b = np.zeros(0, dtype=np.int64)
        if len(edge_a) != n_line_edges:
            raise ValueError(f"line-edge count mismatch: {len(edge_a)} != {n_line_edges}")

        vector_a = position[outer_a] - position[shared]
        vector_b = position[outer_b] - position[shared]
        norm = np.linalg.norm(vector_a, axis=1) * np.linalg.norm(vector_b, axis=1)
        cosine = np.divide(
            np.sum(vector_a * vector_b, axis=1), norm,
            out=np.ones(len(norm), dtype=np.float64), where=norm > 1e-12,
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        lab = nodes[:, :3]
        outer_lab_distance = np.linalg.norm(lab[outer_a] - lab[outer_b], axis=1)
        outer_spatial_distance = np.linalg.norm(position[outer_a] - position[outer_b], axis=1)
        delta_l_a = lab[outer_a, 0] - lab[shared, 0]
        delta_l_b = lab[outer_b, 0] - lab[shared, 0]
        strong_threshold = float(np.percentile(color, 75)) if n_edges else 0.0
        strong_a = color[edge_a] >= strong_threshold
        strong_b = color[edge_b] >= strong_threshold

        grid_cv, grid_max = spatial_grid(midpoint)
        result: dict[str, float | str] = {
            "stem": stem, "style": style,
            "lg_n_nodes": float(n_edges),
            "lg_n_edges": float(n_line_edges),
            "lg_density": float(n_line_edges / possible) if possible else 0.0,
            "lg_degree_mean": float(line_degree.mean()) if n_edges else 0.0,
            "lg_degree_std": float(line_degree.std()) if n_edges else 0.0,
            "lg_degree_max": float(line_degree.max()) if n_edges else 0.0,
            "lg_degree_q90": float(np.percentile(line_degree, 90)) if n_edges else 0.0,
            "lg_degree_gini": gini(line_degree),
            "lg_wedge_load_gini": gini(wedge_load),
            "lg_wedge_load_max_frac": float(wedge_load.max() / n_line_edges) if n_line_edges else 0.0,
            "lg_degree_assortativity": corr(line_degree[edge_a], line_degree[edge_b]),
            "lg_color_gini": gini(color),
            "lg_spatial_gini": gini(spatial),
            "lg_color_spatial_corr": corr(color, spatial),
            "lg_orientation_coherence": orientation_coherence,
            "lg_midpoint_grid_cv": grid_cv,
            "lg_midpoint_grid_max_frac": grid_max,
            "lg_contrast_assortativity": corr(color[edge_a], color[edge_b]),
            "lg_spatial_assortativity": corr(spatial[edge_a], spatial[edge_b]),
            "lg_acute_junction_frac": float(np.mean(angle < 60)) if len(angle) else 0.0,
            "lg_straight_junction_frac": float(np.mean(angle > 150)) if len(angle) else 0.0,
            "lg_luminance_reversal_frac": float(np.mean(delta_l_a * delta_l_b < 0)) if len(angle) else 0.0,
            "lg_both_strong_frac": float(np.mean(strong_a & strong_b)) if len(angle) else 0.0,
            "lg_one_strong_frac": float(np.mean(strong_a ^ strong_b)) if len(angle) else 0.0,
        }
        result.update(stats(color, "lg_color"))
        result.update(stats(spatial, "lg_spatial"))
        result.update(stats(area_ratio, "lg_endpoint_log_area_ratio", (50, 90)))
        result.update(stats(np.abs(color[edge_a] - color[edge_b]), "lg_contrast_delta", (50, 90)))
        result.update(stats(np.abs(spatial[edge_a] - spatial[edge_b]), "lg_spatial_delta", (50, 90)))
        result.update(stats(angle, "lg_junction_angle"))
        result.update(stats(outer_lab_distance, "lg_outer_lab_distance", (50, 90)))
        result.update(stats(outer_spatial_distance, "lg_outer_spatial_distance", (50, 90)))
        result.update(stats(area[shared], "lg_shared_area", (50, 90)))
        if not all(np.isfinite(value) for key, value in result.items() if key not in {"stem", "style"}):
            raise ValueError("non-finite feature")
        return result, None
    except Exception as exc:
        return None, f"{style}/{stem}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/super_resolution_3k_sample.csv"))
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/line_graph_3k_features.parquet"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    completed = pd.DataFrame()
    if args.output.exists() and not args.overwrite:
        completed = pd.read_parquet(args.output)
        if completed["stem"].duplicated().any():
            raise ValueError(f"duplicate stems in existing checkpoint: {args.output}")
        print(f"resume={len(completed)} rows from {args.output}", flush=True)
    completed_stems = set(completed["stem"].astype(str)) if not completed.empty else set()
    tasks = [
        (str(row.stem), str(row.style), str(args.graph_dir / f"{row.stem}.pt"))
        for row in sample.itertuples(index=False)
        if str(row.stem) not in completed_stems
    ]
    missing = [path for _, _, path in tasks if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} exported RAGs are missing; first={missing[0]}")
    rows: list[dict] = completed.to_dict("records") if not completed.empty else []
    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            if row is None:
                failures.append(error or "unknown failure")
            else:
                rows.append(row)
            if done % 250 == 0 or done == len(tasks):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(
                    f"{done}/{len(tasks)} pending complete; total={len(rows)}/{len(sample)}; "
                    f"failures={len(failures)}; {rate:.2f} graphs/s",
                    flush=True,
                )
            if done % args.checkpoint_every == 0:
                checkpoint = pd.DataFrame(rows).sort_values(["style", "stem"]).reset_index(drop=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.to_parquet(temporary, index=False)
                temporary.replace(args.output)
    if failures:
        failure_path = args.output.with_suffix(".failures.txt")
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} failures written to {failure_path}")
    frame = pd.DataFrame(rows).sort_values(["style", "stem"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    print(f"saved={args.output} shape={frame.shape} elapsed={time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
