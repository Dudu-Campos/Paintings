"""Calibrate alternative segmenters against SLIC's actual region budget.

The calibration set contains one deterministic image per WikiArt style by
default.  No classifier is involved: configurations are ranked only by their
node-count proximity to the cached SLIC-2048 output and by runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src/pipeline"))

from segmentation_multigraph_features import segment_labels  # noqa: E402
from slic_graph_features import (  # noqa: E402
    log,
    resize_and_convert,
    resolve_wikiart_records,
    stratified_sample,
)


OUTPUT_DETAIL = ROOT / "paper/segmentation_calibration.csv"
OUTPUT_SUMMARY = ROOT / "paper/segmentation_calibration_summary.csv"


def base_parameters() -> dict:
    return {
        "segments": 2048,
        "max_side": 1024,
        "k": 10,
        "seed": 42,
        "compactness": 10.0,
        "sigma": 0.0,
        "felz_scale": 100.0,
        "felz_sigma": 0.8,
        "felz_min_size": 20,
        "quick_ratio": 1.0,
        "quick_kernel_size": 5.0,
        "quick_max_dist": 10.0,
        "quick_sigma": 0.0,
        "watershed_compactness": 0.001,
    }


def configurations() -> list[tuple[str, str, dict]]:
    base = base_parameters()
    configs: list[tuple[str, str, dict]] = [
        ("slic", "slic_c10", dict(base)),
        ("slico", "slico", dict(base)),
        ("slico", "slico_n1800", {
            **base, "segments": 1800,
        }),
        ("watershed", "watershed_c0", {
            **base, "watershed_compactness": 0.0,
        }),
        ("watershed", "watershed_c001", {
            **base, "watershed_compactness": 0.001,
        }),
        ("watershed", "watershed_n1800_c0", {
            **base,
            "segments": 1800,
            "watershed_compactness": 0.0,
        }),
    ]
    for scale in (50.0, 100.0, 200.0):
        for min_size in (20, 50, 100):
            configs.append((
                "felzenszwalb",
                f"felz_s{scale:g}_m{min_size}",
                {
                    **base,
                    "felz_scale": scale,
                    "felz_min_size": min_size,
                },
            ))
    for kernel_size in (3.0, 5.0):
        for max_dist in (6.0, 10.0, 20.0):
            configs.append((
                "quickshift",
                f"quick_k{kernel_size:g}_d{max_dist:g}",
                {
                    **base,
                    "quick_kernel_size": kernel_size,
                    "quick_max_dist": max_dist,
                },
            ))
    for max_dist in (12.0, 14.0, 16.0):
        configs.append((
            "quickshift",
            f"quick_k3_d{max_dist:g}",
            {
                **base,
                "quick_kernel_size": 3.0,
                "quick_max_dist": max_dist,
            },
        ))
    return configs


def calibrate_one(task: tuple) -> dict:
    stem, style, path, method, config_name, parameters, target_nodes = task
    started = time.perf_counter()
    try:
        rgb, lab = resize_and_convert(path, int(parameters["max_side"]))
        labels = segment_labels(rgb, lab, method, parameters)
        actual_nodes = int(np.unique(labels).size)
        elapsed = time.perf_counter() - started
        return {
            "stem": stem,
            "style": style,
            "method": method,
            "config": config_name,
            "parameters": json.dumps(
                parameters, sort_keys=True, separators=(",", ":")
            ),
            "target_slic_nodes": int(target_nodes),
            "actual_nodes": actual_nodes,
            "node_ratio_to_slic": actual_nodes / target_nodes,
            "abs_log_node_ratio": abs(
                np.log(max(actual_nodes, 1) / target_nodes)
            ),
            "elapsed_seconds": elapsed,
            "error": "",
        }
    except Exception as exc:
        return {
            "stem": stem,
            "style": style,
            "method": method,
            "config": config_name,
            "parameters": json.dumps(
                parameters, sort_keys=True, separators=(",", ":")
            ),
            "target_slic_nodes": int(target_nodes),
            "actual_nodes": np.nan,
            "node_ratio_to_slic": np.nan,
            "abs_log_node_ratio": np.nan,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, config), group in detail.groupby(
        ["method", "config"], sort=False
    ):
        valid = group[group["error"] == ""]
        rows.append({
            "method": method,
            "config": config,
            "n_images": len(valid),
            "n_failures": len(group) - len(valid),
            "nodes_mean": valid["actual_nodes"].mean(),
            "nodes_median": valid["actual_nodes"].median(),
            "nodes_p05": valid["actual_nodes"].quantile(0.05),
            "nodes_p95": valid["actual_nodes"].quantile(0.95),
            "node_ratio_median": valid["node_ratio_to_slic"].median(),
            "abs_log_node_ratio_median": valid[
                "abs_log_node_ratio"
            ].median(),
            "elapsed_mean": valid["elapsed_seconds"].mean(),
            "elapsed_median": valid["elapsed_seconds"].median(),
            "elapsed_p95": valid["elapsed_seconds"].quantile(0.95),
            "parameters": group["parameters"].iloc[0],
        })
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["method", "abs_log_node_ratio_median", "elapsed_median"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-style", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = resolve_wikiart_records(
        ROOT / "data/artemis/wikiart_meta.csv", ROOT / "wikiart"
    ).drop_duplicates("stem")
    records = stratified_sample(
        records, args.sample_per_style, args.seed
    )
    cached_slic = pd.read_parquet(
        ROOT / "outputs/wikiart_slic_features/graph_features.parquet",
        columns=["stem", "n_nodes"],
    ).drop_duplicates("stem")
    target_map = dict(
        zip(cached_slic["stem"].astype(str), cached_slic["n_nodes"].astype(int))
    )
    records = records[
        records["stem"].astype(str).isin(target_map)
    ].copy()
    configs = configurations()
    tasks = [
        (
            str(row.stem),
            str(row.style),
            str(row.path),
            method,
            name,
            parameters,
            target_map[str(row.stem)],
        )
        for row in records.itertuples(index=False)
        for method, name, parameters in configs
    ]
    log(
        f"Calibration: {len(records)} images x {len(configs)} configs "
        f"= {len(tasks)} tasks, workers={args.workers}"
    )

    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(calibrate_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % len(configs) == 0 or done == len(tasks):
                log(
                    f"Calibration {done}/{len(tasks)} "
                    f"({time.perf_counter() - started:.1f}s)"
                )

    detail = pd.DataFrame(rows).sort_values(
        ["method", "config", "style", "stem"]
    )
    detail.to_csv(OUTPUT_DETAIL, index=False)
    summary = summarize(detail)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    log("\n" + summary[[
        "method", "config", "nodes_median", "node_ratio_median",
        "elapsed_median", "n_failures",
    ]].to_string(index=False))
    log(f"Saved: {OUTPUT_DETAIL}")
    log(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
