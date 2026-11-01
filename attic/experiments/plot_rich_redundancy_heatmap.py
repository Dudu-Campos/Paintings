"""Pairwise r^2 heatmap for the rich visual feature library."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_rich_redundancy_layout import FAMILY_ORDER, load_matrix, log  # noqa: E402


COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
}
MUTED = "#555555"


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
        log(f"imputed {int(bad.sum()):,} non-finite values by column median")
    std = values.std(axis=0)
    keep = std > 1e-9
    if (~keep).any():
        log(f"dropping {int((~keep).sum())} constant features")
    values = values[:, keep]
    std = std[keep]
    standard = (values - values.mean(axis=0)) / std
    return standard.astype(np.float32), keep


def family_internal_order(r2: np.ndarray, families: np.ndarray) -> tuple[np.ndarray, list[int]]:
    order_parts: list[np.ndarray] = []
    boundaries: list[int] = []
    offset = 0
    for family in FAMILY_ORDER:
        indices = np.flatnonzero(families == family)
        if len(indices) <= 2:
            local = np.arange(len(indices))
        else:
            block = np.nan_to_num(r2[np.ix_(indices, indices)], nan=1.0)
            distance = 1.0 - np.sqrt(np.clip(block, 0.0, 1.0))
            np.fill_diagonal(distance, 0.0)
            local = leaves_list(linkage(squareform(distance, checks=False), method="average"))
        order_parts.append(indices[local])
        offset += len(indices)
        boundaries.append(offset)
    return np.concatenate(order_parts), boundaries


def plot_heatmap(
    r2: np.ndarray,
    families: np.ndarray,
    feature_names: np.ndarray,
    output: Path,
    title: bool,
) -> pd.DataFrame:
    order, boundaries = family_internal_order(r2, families)
    ordered = np.nan_to_num(r2[np.ix_(order, order)], nan=1.0)
    ordered_families = families[order]

    figure, axis = plt.subplots(figsize=(11.5, 10.2))
    image = axis.imshow(ordered, cmap="magma", vmin=0, vmax=1, interpolation="nearest", rasterized=True)
    starts = [0] + boundaries[:-1]
    centres = [(start + end - 1) / 2 for start, end in zip(starts, boundaries)]

    for edge in boundaries[:-1]:
        axis.axhline(edge - 0.5, color="white", linewidth=1.0, alpha=0.85)
        axis.axvline(edge - 0.5, color="white", linewidth=1.0, alpha=0.85)
    axis.set_xticks(centres)
    axis.set_yticks(centres)
    axis.set_xticklabels(FAMILY_ORDER, rotation=35, ha="right", fontsize=9)
    axis.set_yticklabels(FAMILY_ORDER, fontsize=9)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)

    if title:
        axis.set_title(
            "Rich-feature redundancy heatmap\n"
            "global descriptors, Hilbert and Fourier replaced by rich versions",
            fontsize=14,
            pad=14,
        )
    axis.set_xlabel("Features ordered within each family by r2 clustering", fontsize=10)
    axis.set_ylabel("Features ordered within each family by r2 clustering", fontsize=10)
    bar = figure.colorbar(image, ax=axis, fraction=0.042, pad=0.025)
    bar.set_label("pairwise r2", fontsize=10)
    bar.outline.set_visible(False)

    for centre, family in zip(centres, FAMILY_ORDER):
        axis.text(
            -0.045,
            centre / len(order),
            str(int(np.sum(ordered_families == family))),
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
    figure.text(
        0.01,
        0.012,
        f"WikiArt, no labels used; {len(order):,} non-constant features. "
        "White lines mark extraction families; ordering uses feature-feature redundancy only within each family.",
        fontsize=8.5,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.02, 0.035, 1, 1))
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)

    return pd.DataFrame({
        "heatmap_position": np.arange(len(order)),
        "feature_index": order,
        "feature": feature_names[order],
        "family": families[order],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rich_feature_redundancy_r2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values, families, feature_names = load_matrix(args)
    log(f"{values.shape[0]:,} images x {values.shape[1]} rich visual features")
    standard, keep = standardize(values)
    families = families[keep]
    feature_names = feature_names[keep]

    log("computing pairwise r^2 matrix...")
    correlation = (standard.T @ standard) / len(standard)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2
    np.fill_diagonal(r2, np.nan)

    order = plot_heatmap(
        r2,
        families,
        feature_names,
        args.output_dir / "heatmap.png",
        title=True,
    )
    plot_heatmap(
        r2,
        families,
        feature_names,
        args.output_dir / "heatmap_paper.png",
        title=False,
    )
    order.to_csv(args.output_dir / "heatmap_order.csv", index=False)
    log(f"saved: {args.output_dir / 'heatmap.png'}")
    log(f"saved: {args.output_dir / 'heatmap_paper.png'}")
    log(f"saved: {args.output_dir / 'heatmap_order.csv'}")


if __name__ == "__main__":
    main()
