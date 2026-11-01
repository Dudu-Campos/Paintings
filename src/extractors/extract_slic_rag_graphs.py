"""Persist raw SLIC region-adjacency graphs for graph-neural-network experiments.

The existing SLIC pipeline writes aggregate statistics only.  This exporter
keeps the same SLIC configuration but saves the graph itself: Lab, centroid
and area node attributes; RAG colour-distance and spatial-distance edge
attributes.  Each graph is an independent ``.pt`` file, so interrupted pilot
extractions can resume safely.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from slic_graph_features import resolve_wikiart_records, segment_slic_graph, stratified_sample


NODE_FIELDS = ("L", "a", "b", "X", "Y", "area_frac")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def extract_one(task: tuple[str, str, str, int, float, float, int]) -> tuple[dict | None, str | None]:
    stem, style, path, segments, compactness, sigma, max_side = task
    try:
        nodes, edges = segment_slic_graph(path, segments, compactness, sigma, max_side, True)
        node_values = np.column_stack([nodes[field] for field in NODE_FIELDS]).astype(np.float32, copy=False)
        # Area is positive and highly skewed; the log is a numerically useful
        # node attribute while retaining the original information.
        node_values[:, -1] = np.log(np.clip(node_values[:, -1], 1e-8, None))
        source = edges["src"].astype(np.int64, copy=False)
        target = edges["dst"].astype(np.int64, copy=False)
        edge_index = np.concatenate([
            np.stack([source, target]), np.stack([target, source]),
        ], axis=1)
        edge_attr = np.column_stack([edges["color_dist"], edges["spatial_dist"]]).astype(np.float32, copy=False)
        edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
        return {
            "stem": stem, "style": style,
            "x": node_values, "edge_index": edge_index, "edge_attr": edge_attr,
            "n_nodes": len(nodes), "n_edges": len(edges),
        }, None
    except Exception as exc:  # pragma: no cover - corrupt image handling
        return None, f"{style}/{stem}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--image-dir", type=Path, default=root / "wikiart")
    parser.add_argument("--meta", type=Path, default=root / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/slic_gnn_pilot/graphs")
    parser.add_argument("--sample-per-style", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segments", type=int, default=2048)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    records = stratified_sample(records, args.sample_per_style, args.seed).sort_values("stem").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in args.output_dir.glob("*.pt")}
    pending = records.loc[~records.stem.isin(existing)]
    tasks = [
        (row.stem, row.style, row.path, args.segments, args.compactness, args.sigma, args.max_side)
        for row in pending.itertuples(index=False)
    ]
    tasks.sort(key=lambda task: Path(task[2]).stat().st_size, reverse=True)
    log(
        f"SLIC RAG export: {len(records):,} requested, {len(existing):,} already saved, "
        f"{len(tasks):,} pending; segments={args.segments}, workers={args.workers}"
    )
    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            graph, error = future.result()
            if graph is None:
                failures.append(error or "unknown error")
            else:
                stem = str(graph.pop("stem"))
                style = str(graph.pop("style"))
                torch.save(graph, args.output_dir / f"{stem}.pt")
                # Keeping the label beside the graph makes the artefacts easy
                # to audit without mixing it into learned node attributes.
                (args.output_dir / f"{stem}.label").write_text(style + "\n", encoding="utf-8")
            if done % args.checkpoint_every == 0 or done == len(tasks):
                rate = done / max(time.perf_counter() - started, 1e-9)
                log(f"{done:,}/{len(tasks):,}; failures={len(failures):,}; {rate:.2f} graphs/s")
    # The training index only needs a graph identifier and its style.  Avoid
    # reloading every serialized graph here: on the full WikiArt export that
    # would reread roughly 25 GB merely to recover node/edge counts.
    rows = []
    for graph_path in sorted(args.output_dir.glob("*.pt")):
        label_path = graph_path.with_suffix(".label")
        if not label_path.exists():
            continue
        rows.append({
            "stem": graph_path.stem, "style": label_path.read_text(encoding="utf-8").strip(),
        })
    pd.DataFrame(rows).sort_values("stem").to_csv(args.output_dir.parent / "index.csv", index=False)
    if failures:
        (args.output_dir.parent / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    log(f"Saved {len(rows):,} graphs to {args.output_dir}")


if __name__ == "__main__":
    main()
