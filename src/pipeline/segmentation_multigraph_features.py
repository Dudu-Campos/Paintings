"""Generate the 71-feature multigraph representation from several segmenters.

Supported methods are SLICO, Felzenszwalb, Quickshift, and marker-controlled
watershed.  Every method feeds the same region-node builder and the same RAG,
kNN, Delaunay, and MST graph builders used by the production SLIC pipeline.

This module is intended for segmentation ablations.  Parameters are global,
fixed before classification, and recorded in every output row.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed

from graph_feature_utils import (
    build_delaunay_edges,
    build_knn_edges,
    build_mst_edges,
    extract_graph_features,
)
from slic_graph_features import (
    build_slic_edges,
    build_slic_nodes,
    graph_feature_dict,
    log,
    resize_and_convert,
    resolve_wikiart_records,
    stratified_sample,
)
from slic_multigraph_features import (
    OUTPUT_NAMES,
    read_existing,
    save_combined_features,
    save_rows,
)


METHODS = ("slic", "slico", "felzenszwalb", "quickshift", "watershed")


def segment_labels(
    rgb: np.ndarray,
    lab: np.ndarray,
    method: str,
    parameters: dict,
) -> np.ndarray:
    """Return a 2-D integer label map for one configured method."""
    target_segments = min(
        int(parameters["segments"]), int(rgb.shape[0] * rgb.shape[1])
    )
    if method in {"slic", "slico"}:
        return slic(
            rgb,
            n_segments=target_segments,
            compactness=float(parameters["compactness"]),
            sigma=float(parameters["sigma"]),
            convert2lab=True,
            enforce_connectivity=True,
            slic_zero=method == "slico",
            start_label=0,
            channel_axis=-1,
        )
    if method == "felzenszwalb":
        return felzenszwalb(
            rgb,
            scale=float(parameters["felz_scale"]),
            sigma=float(parameters["felz_sigma"]),
            min_size=int(parameters["felz_min_size"]),
            channel_axis=-1,
        )
    if method == "quickshift":
        return quickshift(
            rgb,
            ratio=float(parameters["quick_ratio"]),
            kernel_size=float(parameters["quick_kernel_size"]),
            max_dist=float(parameters["quick_max_dist"]),
            sigma=float(parameters["quick_sigma"]),
            convert2lab=True,
            rng=int(parameters["seed"]),
            channel_axis=-1,
        )
    if method == "watershed":
        gradient = sobel(lab[..., 0])
        return watershed(
            gradient,
            markers=target_segments,
            compactness=float(parameters["watershed_compactness"]),
            watershed_line=False,
        )
    raise ValueError(f"Unknown segmentation method: {method}")


def segment_region_graph(
    path: str,
    method: str,
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment an image and return generic region nodes and adjacency edges."""
    rgb, lab = resize_and_convert(path, int(parameters["max_side"]))
    labels = segment_labels(rgb, lab, method, parameters)
    nodes, labels = build_slic_nodes(labels, lab)
    edges = build_slic_edges(labels, lab, nodes)
    return nodes, edges


def extract_one(
    task: tuple[str, str, str, str, dict],
) -> tuple[dict[str, dict] | None, str | None]:
    stem, style, path, method, parameters = task
    started = time.perf_counter()
    try:
        nodes, rag_edges = segment_region_graph(path, method, parameters)
        knn_edges = build_knn_edges(nodes, k=int(parameters["k"]))
        delaunay_edges = build_delaunay_edges(nodes)
        mst_edges = build_mst_edges(nodes, rag_edges)
        metadata = {
            "stem": stem,
            "style": style,
            "segmentation": method,
            "segmentation_target_segments": int(parameters["segments"]),
            "segmentation_actual_segments": int(len(nodes)),
            "segmentation_max_side": int(parameters["max_side"]),
            "segmentation_parameters": json.dumps(
                parameters, sort_keys=True, separators=(",", ":")
            ),
        }
        rows = {
            "rag": {**graph_feature_dict(nodes, rag_edges), **metadata},
            "knn": {
                **extract_graph_features(nodes, knn_edges),
                **metadata,
            },
            "delaunay": {
                **graph_feature_dict(nodes, delaunay_edges),
                **metadata,
            },
            "mst": {**graph_feature_dict(nodes, mst_edges), **metadata},
        }
        elapsed = float(time.perf_counter() - started)
        for row in rows.values():
            row["elapsed_seconds_all_graphs"] = elapsed
        return rows, None
    except Exception as exc:
        return None, (
            f"{style}/{stem}: {type(exc).__name__}: {exc}"
        )


def parameter_dict(args: argparse.Namespace) -> dict:
    return {
        "segments": args.segments,
        "max_side": args.max_side,
        "k": args.k,
        "seed": args.seed,
        "compactness": args.compactness,
        "sigma": args.sigma,
        "felz_scale": args.felz_scale,
        "felz_sigma": args.felz_sigma,
        "felz_min_size": args.felz_min_size,
        "quick_ratio": args.quick_ratio,
        "quick_kernel_size": args.quick_kernel_size,
        "quick_max_dist": args.quick_max_dist,
        "quick_sigma": args.quick_sigma,
        "watershed_compactness": args.watershed_compactness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--segments", type=int, default=2048)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample-per-style", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--felz-scale", type=float, default=100.0)
    parser.add_argument("--felz-sigma", type=float, default=0.8)
    parser.add_argument("--felz-min-size", type=int, default=20)
    parser.add_argument("--quick-ratio", type=float, default=1.0)
    parser.add_argument("--quick-kernel-size", type=float, default=5.0)
    parser.add_argument("--quick-max-dist", type=float, default=10.0)
    parser.add_argument("--quick-sigma", type=float, default=0.0)
    parser.add_argument("--watershed-compactness", type=float, default=0.001)
    args = parser.parse_args()

    if args.segments < 2:
        parser.error("--segments must be at least 2")
    if args.max_side < 16:
        parser.error("--max-side must be at least 16")
    if args.k < 1:
        parser.error("--k must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    records = resolve_wikiart_records(args.meta, args.image_dir)
    records = records.drop_duplicates("stem").copy()
    records = stratified_sample(
        records, args.sample_per_style, args.seed
    )
    parameters = parameter_dict(args)
    rows = read_existing(args.output_dir, args.overwrite)
    completed = {str(row["stem"]) for row in rows["rag"]}
    records = records[
        ~records["stem"].astype(str).isin(completed)
    ].copy()
    tasks = [
        (
            str(row.stem),
            str(row.style),
            str(row.path),
            args.method,
            parameters,
        )
        for row in records.itertuples(index=False)
    ]
    tasks.sort(key=lambda task: Path(task[2]).stat().st_size, reverse=True)
    log(
        f"{args.method} multigraph: {len(tasks):,} tasks, "
        f"target={args.segments}, max_side={args.max_side}, "
        f"k={args.k}, workers={args.workers}"
    )
    log(
        "Parameters: "
        + json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    )

    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            result, error = future.result()
            if result is None:
                failures.append(error or "unknown error")
            else:
                for name in OUTPUT_NAMES:
                    rows[name].append(result[name])
            if done % args.checkpoint_every == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                log(
                    f"{done:,}/{len(tasks):,} done, "
                    f"ok={len(rows['rag']):,}, "
                    f"fail={len(failures):,}, {rate:.2f} images/s"
                )
                save_rows(rows, args.output_dir)

    save_rows(rows, args.output_dir)
    combined_path = save_combined_features(args.output_dir)
    log(f"Saved 71-feature multigraph table to {combined_path}")
    if failures:
        failure_path = args.output_dir / "failures.txt"
        failure_path.write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
        log(
            f"Failures written to {failure_path}: "
            f"{len(failures):,}"
        )
    log(
        f"Saved {len(rows['rag']):,} complete multigraph rows "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
