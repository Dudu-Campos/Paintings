"""Neighborhood redundancy metrics for the rich feature pool.

``max_r2`` is too sensitive when the feature pool is large: one near-duplicate
is enough to mark a feature as redundant.  This script recomputes the rich
pairwise-r2 matrix and reports more local neighborhood summaries per feature:

* p90/p95/p99 r2 against all other features
* mean r2 of the top 5, top 10 and top 20 neighbors
* number of neighbors above r2 thresholds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from plot_rich_redundancy_layout import load_matrix  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[1] <= k:
        return np.nanmean(values, axis=1)
    partition = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
    return partition.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_feature_redundancy_r2/neighborhood_metrics.csv")
    args = parser.parse_args()

    values, family, feature_names = load_matrix(args)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
    std = values.std(axis=0)
    keep = std > 1e-9
    values, family, feature_names, std = values[:, keep], family[keep], feature_names[keep], std[keep]
    log(f"{values.shape[0]:,} images x {values.shape[1]} nonconstant rich features")

    standard = (values - values.mean(axis=0)) / std
    correlation = (standard.T @ standard) / len(standard)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2
    np.fill_diagonal(r2, np.nan)
    finite_r2 = np.where(np.isfinite(r2), r2, -np.inf)

    rows = {
        "feature": feature_names,
        "family": family,
        "max_r2": np.nanmax(r2, axis=1),
        "p90_r2": np.nanquantile(r2, 0.90, axis=1),
        "p95_r2": np.nanquantile(r2, 0.95, axis=1),
        "p99_r2": np.nanquantile(r2, 0.99, axis=1),
        "top5_mean_r2": topk_mean(finite_r2, 5),
        "top10_mean_r2": topk_mean(finite_r2, 10),
        "top20_mean_r2": topk_mean(finite_r2, 20),
        "neighbors_ge_0_3": np.nansum(r2 >= 0.3, axis=1),
        "neighbors_ge_0_5": np.nansum(r2 >= 0.5, axis=1),
        "neighbors_ge_0_8": np.nansum(r2 >= 0.8, axis=1),
        "neighbors_ge_0_9": np.nansum(r2 >= 0.9, axis=1),
    }
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.8f")
    summary = result.groupby("family", as_index=False).agg(
        features=("feature", "count"),
        max_r2_median=("max_r2", "median"),
        p95_r2_median=("p95_r2", "median"),
        top10_mean_r2_median=("top10_mean_r2", "median"),
        neighbors_ge_0_8_median=("neighbors_ge_0_8", "median"),
    )
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False, float_format="%.8f")
    log(f"saved: {args.output}")
    log(summary.to_string(index=False))


if __name__ == "__main__":
    main()
