"""Extract a SLIC graph baseline with the standard 32 graph features.

The segment budget is explicit and fixed for all images in a run.

Example (stratified pilot, 20 images per style)::

    python src/pipeline/slic_graph_features.py \
        --image-dir wikiart \
        --meta data/artemis/wikiart_meta.csv \
        --output outputs/slic_pilot/slic_c10_features.parquet \
        --compactness 10 --fixed-segments 2048 --sample-per-style 20 --workers 8
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from skimage.segmentation import slic

from graph_feature_utils import (
    COLOR_FEATURE_COLS,
    GRAPH_FEATURE_COLS,
    SPATIAL_FEATURE_COLS,
    STRUCT_FEATURE_COLS,
    all_graph_features,
    dt_edge,
    dt_node,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FEATURE_COLS = (
    GRAPH_FEATURE_COLS
    + STRUCT_FEATURE_COLS
    + SPATIAL_FEATURE_COLS
    + COLOR_FEATURE_COLS
)
assert len(FEATURE_COLS) == 32

Image.MAX_IMAGE_PIXELS = None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _title_part(filename: str) -> str:
    """Return the title portion of WikiArt's ``artist_title.jpg`` name."""
    return filename.partition("_")[2] or filename


def resolve_wikiart_records(meta_path: Path, image_dir: Path) -> pd.DataFrame:
    """Resolve metadata rows to images, including the known ZIP/CSV encoding drift."""
    meta = pd.read_csv(meta_path)
    required = {"stem", "filename", "style"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    by_exact: dict[tuple[str, str], Path] = {}
    by_style_title: dict[tuple[str, str], list[Path]] = {}
    for path in image_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        exact_key = (path.parent.name, path.name)
        if exact_key in by_exact:
            raise ValueError(f"Duplicate style/filename: {exact_key}")
        by_exact[exact_key] = path
        by_style_title.setdefault(
            (path.parent.name, _title_part(path.name)), []
        ).append(path)

    rows: list[dict] = []
    unresolved: list[str] = []
    exact_matches = fallback_matches = 0
    for row in meta[["stem", "filename", "style"]].dropna().itertuples(index=False):
        stem = str(row.stem)
        filename = str(row.filename)
        style = str(row.style)
        path = by_exact.get((style, filename))
        if path is not None:
            exact_matches += 1
        else:
            candidates = by_style_title.get((style, _title_part(filename)), [])
            if not candidates:
                unresolved.append(f"{style}/{filename}")
                continue
            if len(candidates) > 1:
                artist = filename.partition("_")[0]
                scored = sorted(
                    (
                        SequenceMatcher(
                            None, artist, candidate.name.partition("_")[0]
                        ).ratio(),
                        candidate,
                    )
                    for candidate in candidates
                )
                scored.sort(key=lambda item: item[0], reverse=True)
                if len(scored) > 1 and scored[0][0] == scored[1][0]:
                    unresolved.append(f"ambiguous {style}/{filename}")
                    continue
                path = scored[0][1]
            else:
                path = candidates[0]
            fallback_matches += 1
        rows.append({"stem": stem, "style": style, "path": str(path)})

    if unresolved:
        preview = "; ".join(unresolved[:5])
        raise ValueError(
            f"Could not resolve {len(unresolved):,} metadata images (first: {preview})"
        )
    log(
        f"Image matching: {exact_matches:,} exact, {fallback_matches:,} "
        "recovered from filename encoding"
    )
    return pd.DataFrame(rows)


REGIONS = ("global", "top_left", "top_right", "bottom_left", "bottom_right")


def crop_to_region(image: Image.Image, region: str) -> Image.Image:
    """Return one fixed spatial view without introducing interpolation artefacts."""
    if region == "global":
        return image
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; expected one of {REGIONS}")
    width, height = image.size
    midpoint_x, midpoint_y = width // 2, height // 2
    boxes = {
        "top_left": (0, 0, midpoint_x, midpoint_y),
        "top_right": (midpoint_x, 0, width, midpoint_y),
        "bottom_left": (0, midpoint_y, midpoint_x, height),
        "bottom_right": (midpoint_x, midpoint_y, width, height),
    }
    return image.crop(boxes[region])


def resize_and_convert(
    path: str, max_side: int, region: str = "global"
) -> tuple[np.ndarray, np.ndarray]:
    """Return a resized RGB/Lab global image or fixed quadrant view."""
    with Image.open(path) as image:
        image = crop_to_region(image.convert("RGB"), region)
        width, height = image.size
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    return rgb, rgb2lab(rgb).astype(np.float32)


def build_slic_nodes(labels: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build region records and return contiguous labels."""
    _, inverse = np.unique(labels, return_inverse=True)
    labels = inverse.reshape(labels.shape).astype(np.int32, copy=False)
    flat = labels.ravel()
    n_nodes = int(flat.max()) + 1
    counts = np.bincount(flat, minlength=n_nodes).astype(np.float64)

    nodes = np.zeros(n_nodes, dtype=dt_node)
    for channel, name in enumerate(("L", "a", "b")):
        sums = np.bincount(flat, weights=lab[..., channel].ravel(), minlength=n_nodes)
        nodes[name] = (sums / counts).astype(np.float32)

    height, width = labels.shape
    x_coords = np.tile(np.arange(width, dtype=np.float64), height)
    y_coords = np.repeat(np.arange(height, dtype=np.float64), width)
    sum_x = np.bincount(flat, weights=x_coords, minlength=n_nodes)
    sum_y = np.bincount(flat, weights=y_coords, minlength=n_nodes)
    nodes["X"] = (sum_x / counts / max(width - 1, 1)).astype(np.float32)
    nodes["Y"] = (sum_y / counts / max(height - 1, 1)).astype(np.float32)
    nodes["area_frac"] = (counts / labels.size).astype(np.float32)
    nodes["area_px"] = counts.astype(np.uint32)
    return nodes, labels


def build_slic_edges(labels: np.ndarray, lab: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Build a 4-neighbour RAG with color and spatial boundary weights."""
    n_nodes = len(nodes)
    keys: list[np.ndarray] = []
    distances: list[np.ndarray] = []

    def collect(
        labels_a: np.ndarray,
        labels_b: np.ndarray,
        lab_a: np.ndarray,
        lab_b: np.ndarray,
    ) -> None:
        mask = labels_a != labels_b
        if not np.any(mask):
            return
        a = labels_a[mask].astype(np.int64)
        b = labels_b[mask].astype(np.int64)
        src = np.minimum(a, b)
        dst = np.maximum(a, b)
        keys.append(src * n_nodes + dst)
        delta = lab_a[mask] - lab_b[mask]
        distances.append(np.sqrt(np.sum(delta * delta, axis=1)))

    collect(labels[:, :-1], labels[:, 1:], lab[:, :-1], lab[:, 1:])
    collect(labels[:-1, :], labels[1:, :], lab[:-1, :], lab[1:, :])
    if not keys:
        return np.zeros(0, dtype=dt_edge)

    pair_keys = np.concatenate(keys)
    color_dist = np.concatenate(distances).astype(np.float32, copy=False)
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]
    color_dist = color_dist[order]
    unique_keys, first = np.unique(pair_keys, return_index=True)
    min_color_dist = np.minimum.reduceat(color_dist, first)

    src = (unique_keys // n_nodes).astype(np.uint32)
    dst = (unique_keys % n_nodes).astype(np.uint32)
    dx = nodes["X"][src] - nodes["X"][dst]
    dy = nodes["Y"][src] - nodes["Y"][dst]

    edges = np.zeros(len(unique_keys), dtype=dt_edge)
    edges["src"] = src
    edges["dst"] = dst
    edges["color_dist"] = min_color_dist
    edges["spatial_dist"] = np.sqrt(dx * dx + dy * dy)
    return edges


def segment_slic_graph(
    path: str,
    requested_segments: int,
    compactness: float,
    sigma: float,
    max_side: int,
    enforce_connectivity: bool,
    region: str = "global",
) -> tuple[np.ndarray, np.ndarray]:
    """Segment one image and return region records and RAG edges."""
    rgb, lab = resize_and_convert(path, max_side, region)
    labels = slic(
        rgb,
        n_segments=requested_segments,
        compactness=compactness,
        sigma=sigma,
        convert2lab=True,
        enforce_connectivity=enforce_connectivity,
        start_label=0,
        channel_axis=-1,
    )
    nodes, labels = build_slic_nodes(labels, lab)
    edges = build_slic_edges(labels, lab, nodes)
    return nodes, edges


def graph_feature_dict(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """Return the standard 32 aggregate graph features."""
    return all_graph_features(nodes, edges)


def extract_one(task: tuple) -> tuple[dict | None, str | None]:
    (
        stem,
        style,
        path,
        reference_n_nodes,
        requested_segments,
        compactness,
        sigma,
        max_side,
        enforce_connectivity,
    ) = task
    started = time.perf_counter()
    try:
        nodes, edges = segment_slic_graph(
            path,
            requested_segments,
            compactness,
            sigma,
            max_side,
            enforce_connectivity,
        )
        features = graph_feature_dict(nodes, edges)
        features.update(
            {
                "stem": stem,
                "style": style,
                "reference_n_nodes": int(reference_n_nodes),
                "slic_requested_segments": int(requested_segments),
                "slic_actual_segments": int(len(nodes)),
                "slic_compactness": float(compactness),
                "slic_sigma": float(sigma),
                "slic_max_side": int(max_side),
                "slic_enforce_connectivity": bool(enforce_connectivity),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        return features, None
    except Exception as exc:
        return None, f"{style}/{stem}: {type(exc).__name__}: {exc}"


def stratified_sample(records: pd.DataFrame, per_style: int, seed: int) -> pd.DataFrame:
    if per_style <= 0:
        return records
    sampled = []
    for _, group in records.groupby("style", sort=True):
        sampled.append(group.sample(n=min(per_style, len(group)), random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def save_rows(rows: list[dict], output: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["style", "stem"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--min-segments", type=int, default=1)
    parser.add_argument("--max-segments", type=int, default=100_000)
    parser.add_argument("--fixed-segments", type=int, default=0)
    parser.add_argument("--sample-per-style", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--no-enforce-connectivity", action="store_true",
        help="Disable SLIC's connectivity post-processing",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Discard rows already present in the output instead of resuming",
    )
    args = parser.parse_args()

    if args.min_segments < 1 or args.max_segments < args.min_segments:
        parser.error("invalid min/max segment limits")
    if args.fixed_segments < 0:
        parser.error("--fixed-segments must be non-negative")
    if args.fixed_segments == 0:
        parser.error("--fixed-segments is required")

    records = resolve_wikiart_records(args.meta, args.image_dir)
    records = records.drop_duplicates("stem").copy()
    records["reference_n_nodes"] = 0
    log(
        f"SLIC cohort: {len(records):,} images, "
        f"{records['style'].nunique()} styles"
    )
    records = stratified_sample(records, args.sample_per_style, args.seed)

    existing_rows: list[dict] = []
    completed: set[str] = set()
    if args.output.exists() and not args.overwrite:
        existing = pd.read_parquet(args.output)
        existing_rows = existing.to_dict("records")
        completed = set(existing["stem"].astype(str))
        log(f"Resume: {len(completed):,} existing rows")
    records = records[~records["stem"].isin(completed)].copy()

    tasks = []
    for row in records.itertuples(index=False):
        reference_n_nodes = int(row.reference_n_nodes)
        requested = args.fixed_segments or reference_n_nodes
        requested = int(np.clip(requested, args.min_segments, args.max_segments))
        tasks.append(
            (
                row.stem,
                row.style,
                row.path,
                reference_n_nodes,
                requested,
                args.compactness,
                args.sigma,
                args.max_side,
                not args.no_enforce_connectivity,
            )
        )

    # Largest images first improves process-pool load balancing.
    tasks.sort(key=lambda task: Path(task[2]).stat().st_size, reverse=True)
    log(
        f"SLIC tasks: {len(tasks):,}, workers={args.workers}, "
        f"compactness={args.compactness:g}, max_side={args.max_side}"
    )

    rows = existing_rows
    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            if row is not None:
                rows.append(row)
            else:
                failures.append(error or "unknown error")
            if done % args.checkpoint_every == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                log(
                    f"{done:,}/{len(tasks):,} done, ok={len(rows):,}, "
                    f"fail={len(failures):,}, {rate:.2f} images/s"
                )
                save_rows(rows, args.output)

    save_rows(rows, args.output)
    if failures:
        failure_path = args.output.with_suffix(".failures.txt")
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        log(f"Failures written to {failure_path}: {len(failures):,}")
    log(f"Saved {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
