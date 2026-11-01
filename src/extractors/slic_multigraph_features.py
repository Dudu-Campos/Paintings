"""Generate the complete graph-feature family from one SLIC segmentation.

For each image this script computes SLIC once, then derives:

* the spatial region-adjacency graph (32 features),
* a k-nearest-neighbour graph in Lab+XY space (21 features),
* a Delaunay graph over region centroids (32 features), and
* a colour-weighted minimum spanning tree of the SLIC RAG (32 features).

Its segment budget is fixed and explicit.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from graph_feature_utils import (
    build_delaunay_edges,
    build_knn_edges,
    build_mst_edges,
    extract_graph_features,
)
from slic_graph_features import (
    REGIONS,
    graph_feature_dict,
    log,
    resolve_wikiart_records,
    segment_slic_graph,
    stratified_sample,
)


OUTPUT_NAMES = {
    "rag": "graph_features.parquet",
    "knn": "knn_features.parquet",
    "delaunay": "delaunay_features.parquet",
    "mst": "mst_features.parquet",
}

# Only these features change when the edge set changes.  Node statistics are
# stored once in the unprefixed SLIC-RAG columns of the combined table.
EDGE_FEATURE_COLS = [
    "n_edges", "density", "color_dist_mean", "color_dist_std",
    "color_dist_max", "spatial_dist_mean", "spatial_dist_std",
    "degree_mean", "degree_std", "degree_max", "hub_degree_ratio",
    "n_leaf_frac", "degree_gini",
]


def add_metadata(features: dict, stem: str, style: str, config: dict) -> dict:
    features.update({"stem": stem, "style": style, **config})
    return features


def extract_one(task: tuple) -> tuple[dict[str, dict] | None, str | None]:
    stem, style, path, segments, compactness, sigma, max_side, k, region = task
    started = time.perf_counter()
    try:
        nodes, rag_edges = segment_slic_graph(
            path,
            requested_segments=segments,
            compactness=compactness,
            sigma=sigma,
            max_side=max_side,
            enforce_connectivity=True,
            region=region,
        )
        knn_edges = build_knn_edges(nodes, k=k)
        delaunay_edges = build_delaunay_edges(nodes)
        mst_edges = build_mst_edges(nodes, rag_edges)

        config = {
            "segmentation": "slic",
            "slic_requested_segments": int(segments),
            "slic_actual_segments": int(len(nodes)),
            "slic_compactness": float(compactness),
            "slic_sigma": float(sigma),
            "slic_max_side": int(max_side),
        }
        rows = {
            "rag": add_metadata(
                graph_feature_dict(nodes, rag_edges), stem, style, config
            ),
            # The historical kNN branch uses the 21 basic+structural features.
            "knn": add_metadata(
                extract_graph_features(nodes, knn_edges), stem, style, config
            ),
            "delaunay": add_metadata(
                graph_feature_dict(nodes, delaunay_edges), stem, style, config
            ),
            "mst": add_metadata(
                graph_feature_dict(nodes, mst_edges), stem, style, config
            ),
        }
        elapsed = float(time.perf_counter() - started)
        for row in rows.values():
            row["elapsed_seconds_all_graphs"] = elapsed
        return rows, None
    except Exception as exc:
        return None, f"{style}/{stem}: {type(exc).__name__}: {exc}"


def read_existing(output_dir: Path, overwrite: bool) -> dict[str, list[dict]]:
    rows = {name: [] for name in OUTPUT_NAMES}
    if overwrite:
        return rows
    paths = {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}
    if not all(path.exists() for path in paths.values()):
        return rows
    completed_sets = []
    frames = {}
    for name, path in paths.items():
        frame = pd.read_parquet(path)
        frames[name] = frame
        completed_sets.append(set(frame["stem"].astype(str)))
    completed = set.intersection(*completed_sets)
    for name, frame in frames.items():
        rows[name] = frame[frame["stem"].astype(str).isin(completed)].to_dict("records")
    log(f"Resume: {len(completed):,} complete images across all graph families")
    return rows


def save_rows(rows: dict[str, list[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in OUTPUT_NAMES.items():
        frame = pd.DataFrame(rows[name])
        if not frame.empty:
            frame = frame.sort_values(["style", "stem"]).reset_index(drop=True)
        path = output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)


def save_combined_features(output_dir: Path) -> Path:
    """Store 32 shared/RAG features plus 13 edge features per other graph."""
    frames = {
        name: pd.read_parquet(output_dir / filename)
        for name, filename in OUTPUT_NAMES.items()
    }
    expected_stems = set(frames["rag"]["stem"].astype(str))
    for name, frame in frames.items():
        if frame["stem"].duplicated().any():
            raise ValueError(f"Duplicate stems in {OUTPUT_NAMES[name]}")
        if set(frame["stem"].astype(str)) != expected_stems:
            raise ValueError(f"Stem mismatch in {OUTPUT_NAMES[name]}")

    combined = frames["rag"].copy()
    for name, prefix in (("knn", "knn"), ("delaunay", "del"), ("mst", "mst")):
        selected = frames[name][["stem"] + EDGE_FEATURE_COLS].rename(
            columns={column: f"{prefix}_{column}" for column in EDGE_FEATURE_COLS}
        )
        combined = combined.merge(selected, on="stem", how="inner", validate="one_to_one")

    path = output_dir / "combined_features.parquet"
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def resolve_records_from_manifest(manifest_path: Path, image_dir: Path) -> pd.DataFrame:
    """Load a derived-image manifest while preserving variant ``sample_id`` keys."""
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "style", "image_relative_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    records = manifest[["sample_id", "style", "image_relative_path"]].rename(
        columns={"sample_id": "stem", "image_relative_path": "relative_path"}
    )
    records["stem"] = records["stem"].astype(str)
    records["style"] = records["style"].astype(str)
    records["path"] = records["relative_path"].map(lambda path: str(image_dir / path))
    missing_images = records[~records["path"].map(lambda path: Path(path).is_file())]
    if not missing_images.empty:
        raise FileNotFoundError(
            f"Manifest references {len(missing_images):,} missing images "
            f"(first: {missing_images.iloc[0]['path']})"
        )
    return records[["stem", "style", "path"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path,
                        help="Canonical WikiArt metadata CSV (omit with --manifest).")
    parser.add_argument("--manifest", type=Path,
                        help="Derived-image manifest with sample_id/style/image_relative_path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=2048)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--region", choices=REGIONS, default="global",
        help="Fixed spatial view to extract; default is the entire image.",
    )
    parser.add_argument("--sample-per-style", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.segments < 1:
        parser.error("--segments must be positive")
    if args.k < 1:
        parser.error("--k must be positive")
    if bool(args.meta) == bool(args.manifest):
        parser.error("provide exactly one of --meta or --manifest")

    records = (
        resolve_records_from_manifest(args.manifest, args.image_dir)
        if args.manifest else resolve_wikiart_records(args.meta, args.image_dir)
    )
    # Downstream feature tables use stem as their join key.  Conflicting
    # duplicate metadata stems cannot coexist under that schema.
    records = records.drop_duplicates("stem").copy()
    records = stratified_sample(records, args.sample_per_style, args.seed)

    rows = read_existing(args.output_dir, args.overwrite)
    completed = set(row["stem"] for row in rows["rag"])
    records = records[~records["stem"].isin(completed)].copy()
    tasks = [
        (
            row.stem,
            row.style,
            row.path,
            args.segments,
            args.compactness,
            args.sigma,
            args.max_side,
            args.k,
            args.region,
        )
        for row in records.itertuples(index=False)
    ]
    tasks.sort(key=lambda task: Path(task[2]).stat().st_size, reverse=True)
    log(
        f"SLIC multigraph: {len(tasks):,} tasks, segments={args.segments}, "
        f"compactness={args.compactness:g}, k={args.k}, workers={args.workers}"
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
                    f"ok={len(rows['rag']):,}, fail={len(failures):,}, "
                    f"{rate:.2f} images/s"
                )
                save_rows(rows, args.output_dir)

    save_rows(rows, args.output_dir)
    combined_path = save_combined_features(args.output_dir)
    log(f"Saved 71-feature multigraph table to {combined_path}")
    if failures:
        failure_path = args.output_dir / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        log(f"Failures written to {failure_path}: {len(failures):,}")
    log(f"Saved {len(rows['rag']):,} complete multigraph rows to {args.output_dir}")


if __name__ == "__main__":
    main()
