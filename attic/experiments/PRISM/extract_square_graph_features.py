#!/usr/bin/env python3
"""Extract interpretable features from the strict distance-2 shell of SLIC RAGs.

For a RAG G, the square graph G^2 joins vertices whose shortest-path distance is
at most two.  The original edges are already described by SLIC-71, so this file
focuses on the genuinely new edges E(G^2) \ E(G): pairs of nonadjacent regions
that share one or more neighbours.  It consumes cached graph exports and never
rereads or segments an image.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from extract_line_graph_features import corr, gini, spatial_grid, stats


def undirected_assortativity(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Pearson assortativity without depending on an arbitrary edge direction."""
    if len(left) < 2:
        return 0.0
    return corr(
        np.concatenate([values[left], values[right]]),
        np.concatenate([values[right], values[left]]),
    )


def extract_one(task: tuple[str, str, str]) -> tuple[dict | None, str | None]:
    stem, style, path_value = task
    try:
        graph = torch.load(path_value, map_location="cpu", weights_only=False)
        nodes = np.asarray(graph["x"], dtype=np.float64)
        n_nodes = len(nodes)
        n_edges = int(graph["n_edges"])
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)[:, :n_edges]
        edge_attr = np.asarray(graph["edge_attr"], dtype=np.float64)[:n_edges]
        source, target = edge_index
        if n_edges and (
            np.any(source >= target)
            or len(np.unique(source * n_nodes + target)) != n_edges
        ):
            raise ValueError("first edge block is not a unique src<dst undirected RAG")

        degree = np.bincount(np.concatenate([source, target]), minlength=n_nodes)
        neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
        incident_edges: list[list[int]] = [[] for _ in range(n_nodes)]
        for edge_id, (left, right) in enumerate(zip(source, target, strict=True)):
            neighbours[int(left)].append(int(right))
            neighbours[int(right)].append(int(left))
            incident_edges[int(left)].append(edge_id)
            incident_edges[int(right)].append(edge_id)

        wedge_left: list[np.ndarray] = []
        wedge_right: list[np.ndarray] = []
        wedge_center: list[np.ndarray] = []
        wedge_edge_a: list[np.ndarray] = []
        wedge_edge_b: list[np.ndarray] = []
        for center, neighbour_list in enumerate(neighbours):
            if len(neighbour_list) < 2:
                continue
            neighbour_array = np.asarray(neighbour_list, dtype=np.int64)
            edge_array = np.asarray(incident_edges[center], dtype=np.int64)
            first, second = np.triu_indices(len(neighbour_array), 1)
            left = neighbour_array[first]
            right = neighbour_array[second]
            wedge_left.append(np.minimum(left, right))
            wedge_right.append(np.maximum(left, right))
            wedge_center.append(np.full(len(first), center, dtype=np.int64))
            wedge_edge_a.append(edge_array[first])
            wedge_edge_b.append(edge_array[second])

        if wedge_left:
            path_left = np.concatenate(wedge_left)
            path_right = np.concatenate(wedge_right)
            path_center = np.concatenate(wedge_center)
            path_edge_a = np.concatenate(wedge_edge_a)
            path_edge_b = np.concatenate(wedge_edge_b)
        else:
            path_left = path_right = path_center = np.zeros(0, dtype=np.int64)
            path_edge_a = path_edge_b = np.zeros(0, dtype=np.int64)

        original_keys = source * n_nodes + target
        path_keys = path_left * n_nodes + path_right
        is_closed = np.isin(path_keys, original_keys, assume_unique=False)
        open_mask = ~is_closed
        open_keys = path_keys[open_mask]
        open_left = path_left[open_mask]
        open_right = path_right[open_mask]
        open_center = path_center[open_mask]
        open_edge_a = path_edge_a[open_mask]
        open_edge_b = path_edge_b[open_mask]

        if len(open_keys):
            unique_keys, first_occurrence, inverse, common_count = np.unique(
                open_keys, return_index=True, return_inverse=True, return_counts=True
            )
            second_left = unique_keys // n_nodes
            second_right = unique_keys % n_nodes
        else:
            unique_keys = first_occurrence = inverse = common_count = np.zeros(0, dtype=np.int64)
            second_left = second_right = np.zeros(0, dtype=np.int64)
        n_second_edges = len(unique_keys)

        second_degree = np.bincount(
            np.concatenate([second_left, second_right]), minlength=n_nodes
        ) if n_second_edges else np.zeros(n_nodes, dtype=np.int64)
        square_degree = degree + second_degree
        degree_expansion = second_degree / np.maximum(degree, 1)
        square_edges = n_edges + n_second_edges
        possible = n_nodes * (n_nodes - 1) / 2

        lab = nodes[:, :3]
        position = nodes[:, 3:5]
        log_area = nodes[:, 5]
        chroma = np.linalg.norm(lab[:, 1:3], axis=1)
        second_lab = np.linalg.norm(lab[second_left] - lab[second_right], axis=1)
        second_spatial = np.linalg.norm(
            position[second_left] - position[second_right], axis=1
        )
        second_log_area = np.abs(log_area[second_left] - log_area[second_right])
        neighbour_union = degree[second_left] + degree[second_right] - common_count
        neighbour_jaccard = np.divide(
            common_count, neighbour_union,
            out=np.zeros(n_second_edges, dtype=np.float64), where=neighbour_union > 0,
        )

        midpoint = (
            (position[second_left] + position[second_right]) / 2
            if n_second_edges else np.empty((0, 2))
        )
        vector = (
            position[second_right] - position[second_left]
            if n_second_edges else np.empty((0, 2))
        )
        orientation = np.arctan2(vector[:, 1], vector[:, 0]) if n_second_edges else np.zeros(0)
        orientation_coherence = (
            float(np.hypot(np.cos(2 * orientation).mean(), np.sin(2 * orientation).mean()))
            if n_second_edges else 0.0
        )
        grid_cv, grid_max = spatial_grid(midpoint)

        # Each open wedge is one explicit two-hop path.  Ratios near one mean
        # that its endpoints continue nearly straight through the middle region.
        path_lab_chord = np.linalg.norm(lab[open_left] - lab[open_right], axis=1)
        path_spatial_chord = np.linalg.norm(
            position[open_left] - position[open_right], axis=1
        )
        path_color_sum = edge_attr[open_edge_a, 0] + edge_attr[open_edge_b, 0]
        path_spatial_sum = edge_attr[open_edge_a, 1] + edge_attr[open_edge_b, 1]
        color_chord_ratio = np.divide(
            path_lab_chord, path_color_sum,
            out=np.zeros(len(open_left), dtype=np.float64), where=path_color_sum > 1e-12,
        )
        spatial_chord_ratio = np.divide(
            path_spatial_chord, path_spatial_sum,
            out=np.zeros(len(open_left), dtype=np.float64), where=path_spatial_sum > 1e-12,
        )
        delta_l_a = lab[open_left, 0] - lab[open_center, 0]
        delta_l_b = lab[open_right, 0] - lab[open_center, 0]

        result: dict[str, float | str] = {
            "stem": stem,
            "style": style,
            "sq_added_edges": float(n_second_edges),
            "sq_total_edges": float(square_edges),
            "sq_edge_expansion_ratio": float(square_edges / n_edges) if n_edges else 0.0,
            "sq_density": float(square_edges / possible) if possible else 0.0,
            "sq_open_wedges": float(len(open_keys)),
            "sq_closed_wedge_frac": float(is_closed.mean()) if len(is_closed) else 0.0,
            "sq_duplicate_path_frac": (
                float(1 - n_second_edges / len(open_keys)) if len(open_keys) else 0.0
            ),
            "sq_multi_common_frac": float(np.mean(common_count > 1)) if n_second_edges else 0.0,
            "sq_added_degree_mean": float(second_degree.mean()),
            "sq_added_degree_std": float(second_degree.std()),
            "sq_added_degree_max": float(second_degree.max()) if n_nodes else 0.0,
            "sq_added_degree_q90": float(np.percentile(second_degree, 90)) if n_nodes else 0.0,
            "sq_added_degree_gini": gini(second_degree),
            "sq_degree_mean": float(square_degree.mean()),
            "sq_degree_std": float(square_degree.std()),
            "sq_degree_max": float(square_degree.max()) if n_nodes else 0.0,
            "sq_degree_q90": float(np.percentile(square_degree, 90)) if n_nodes else 0.0,
            "sq_degree_gini": gini(square_degree),
            "sq_degree_expansion_mean": float(degree_expansion.mean()),
            "sq_degree_expansion_std": float(degree_expansion.std()),
            "sq_degree_expansion_q90": float(np.percentile(degree_expansion, 90)),
            "sq_common_count_gini": gini(common_count),
            "sq_second_lab_gini": gini(second_lab),
            "sq_second_spatial_gini": gini(second_spatial),
            "sq_second_lab_spatial_corr": corr(second_lab, second_spatial),
            "sq_orientation_coherence": orientation_coherence,
            "sq_midpoint_grid_cv": grid_cv,
            "sq_midpoint_grid_max_frac": grid_max,
            "sq_luminance_assortativity": undirected_assortativity(
                lab[:, 0], second_left, second_right
            ),
            "sq_chroma_assortativity": undirected_assortativity(
                chroma, second_left, second_right
            ),
            "sq_log_area_assortativity": undirected_assortativity(
                log_area, second_left, second_right
            ),
            "sq_degree_assortativity": undirected_assortativity(
                degree.astype(float), second_left, second_right
            ),
            "sq_luminance_reversal_frac": (
                float(np.mean(delta_l_a * delta_l_b < 0)) if len(open_left) else 0.0
            ),
        }
        result.update(stats(common_count, "sq_common_count", (50, 90)))
        result.update(stats(neighbour_jaccard, "sq_neighbor_jaccard"))
        result.update(stats(second_lab, "sq_second_lab"))
        result.update(stats(second_spatial, "sq_second_spatial"))
        result.update(stats(second_log_area, "sq_second_log_area", (50, 90)))
        result.update(stats(color_chord_ratio, "sq_color_chord_ratio", (50, 90)))
        result.update(stats(spatial_chord_ratio, "sq_spatial_chord_ratio", (50, 90)))
        if not all(
            np.isfinite(value) for key, value in result.items()
            if key not in {"stem", "style"}
        ):
            raise ValueError("non-finite feature")
        return result, None
    except Exception as exc:
        return None, f"{style}/{stem}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/super_resolution_3k_sample.csv"))
    parser.add_argument("--graph-dir", type=Path, default=Path("outputs/slic_gnn_full/graphs"))
    parser.add_argument("--output", type=Path, default=Path("paper/square_graph_3k_features.parquet"))
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
