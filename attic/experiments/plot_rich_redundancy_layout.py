"""Force-layout plot of pairwise r^2 for the new rich visual features.

This mirrors ``plot_redundancy_layout.py`` but replaces the compact global
Descriptors, Hilbert and Fourier blocks by the rich versions extracted after
the machine migration.  Multiscale quadrants are intentionally excluded: this
plot is about representation families, not image position.

Outputs live in ``results/rich_feature_redundancy_r2/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from registry import PARAMETER_PREFIXES  # noqa: E402


SEED = 42
THRESHOLDS = (0.30, 0.50, 0.80, 0.90, 0.95, 0.99)
METADATA = {"stem", "style", "label", "period", "filename", "sample_id", "elapsed_seconds_all_graphs"}
FAMILY_ORDER = ("Region graphs", "Image statistics", "Space-filling curve", "Spectral")
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
}
TEXT, MUTED = "#0b0b0b", "#52514e"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [
        column for column in table.columns
        if column not in METADATA
        and not column.startswith(PARAMETER_PREFIXES)
        and table[column].dtype.kind in "fiub"
    ]


def read_npy(values_path: Path, stems_path: Path, prefix: str) -> pd.DataFrame:
    values = np.load(values_path, mmap_mode="r")
    stems = np.load(stems_path, allow_pickle=True).astype(str)
    if values.shape[0] != len(stems):
        raise ValueError(f"Shape/stem mismatch for {values_path}")
    frame = pd.DataFrame(np.asarray(values, dtype=np.float32), columns=[
        f"{prefix}_{index:03d}" for index in range(values.shape[1])
    ])
    frame.insert(0, "stem", stems)
    if frame["stem"].duplicated().any():
        raise ValueError(f"Duplicate stems in {stems_path}")
    return frame


def read_parquet(path: Path, prefix: str) -> pd.DataFrame:
    table = pd.read_parquet(path)
    columns = feature_columns(table)
    frame = table[["stem", *columns]].rename(columns={
        column: f"{prefix}_{column}" for column in columns
    })
    frame["stem"] = frame["stem"].astype(str)
    if frame["stem"].duplicated().any():
        frame = frame.drop_duplicates("stem")
    return frame


def load_matrix(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = pd.read_csv(args.meta).dropna(subset=["style"]).drop_duplicates("stem", keep="last")
    frame = meta[["stem", "style"]].copy()
    frame["stem"] = frame["stem"].astype(str)

    pieces = [
        ("Region graphs", read_parquet(args.slic, "slic")),
        ("Region graphs", read_parquet(args.felzenszwalb, "felz")),
        ("Region graphs", read_parquet(args.line_graph, "line")),
        ("Region graphs", read_parquet(args.line_graph_felz, "line_felz")),
        (
            "Image statistics",
            read_npy(
                args.rich_dir / "descriptors" / "desc_full_features.npy",
                args.rich_dir / "descriptors" / "desc_full_stems.npy",
                "descriptor_rich",
            ),
        ),
        (
            "Space-filling curve",
            read_npy(
                args.rich_dir / "hilbert" / "hilbert_features.npy",
                args.rich_dir / "hilbert" / "hilbert_stems.npy",
                "hilbert_rich",
            ),
        ),
        ("Spectral", read_parquet(args.rich_fourier, "fourier_rich")),
    ]

    blocks, families, feature_names = [], [], []
    for family, piece in pieces:
        before = len(frame)
        frame = frame.merge(piece, on="stem", how="inner", validate="one_to_one")
        columns = [column for column in piece.columns if column != "stem"]
        blocks.append(columns)
        families.extend([family] * len(columns))
        feature_names.extend(columns)
        if len(frame) != before:
            log(f"{family}: cohort changed {before:,} -> {len(frame):,}")

    counts = frame["style"].value_counts()
    frame = frame[frame["style"].isin(counts[counts >= 5].index)].sort_values("stem").reset_index(drop=True)
    values = frame[feature_names].to_numpy(dtype=np.float32)
    return values, np.asarray(families, dtype=object), np.asarray(feature_names, dtype=object)


def write_group_summary(output_dir: Path, family: np.ndarray, r2: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, left in enumerate(FAMILY_ORDER):
        left_mask = family == left
        for right in FAMILY_ORDER[index:]:
            right_mask = family == right
            if left == right:
                sub = r2[np.ix_(left_mask, right_mask)]
                values = sub[np.triu_indices(sub.shape[0], k=1)]
            else:
                values = r2[np.ix_(left_mask, right_mask)].ravel()
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            rows.append({
                "group_a": left,
                "group_b": right,
                "n_pairs": int(len(values)),
                "mean_r2": float(values.mean()),
                "median_r2": float(np.median(values)),
                "p95_r2": float(np.quantile(values, 0.95)),
                "max_r2": float(values.max()),
                **{f"pairs_r2_ge_{str(t).replace('.', '_')}": int((values >= t).sum()) for t in THRESHOLDS},
            })
    summary = pd.DataFrame(rows).sort_values(["median_r2", "p95_r2"], ascending=False)
    summary.to_csv(output_dir / "group_summary.csv", index=False, float_format="%.6f")
    return summary


def plot_layout(
    output: Path,
    family: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    title: bool,
) -> None:
    edges = [(int(a), int(b), float(w)) for a, b, w in zip(rows, columns, weights)]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    isolated = len(family) - graph.number_of_nodes()
    layout = nx.spring_layout(
        graph,
        weight="weight",
        seed=SEED,
        iterations=300,
        k=2.2 / np.sqrt(max(graph.number_of_nodes(), 1)),
    )
    nodes = np.array(sorted(graph.nodes))
    points = np.array([layout[node] for node in nodes])
    index = {node: position for position, node in enumerate(nodes)}
    kinds = family[nodes]

    figure, axis = plt.subplots(figsize=(9.4, 8.4))
    for a, b, weight in edges:
        pair = [index[a], index[b]]
        axis.plot(
            points[pair, 0],
            points[pair, 1],
            zorder=1,
            color="#8a8a8a",
            linewidth=0.30,
            alpha=0.04 + 0.32 * (weight - threshold) / (1 - threshold + 1e-9),
        )
    for name in FAMILY_ORDER:
        mask = kinds == name
        if not mask.any():
            continue
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=22,
            zorder=3,
            color=COLOURS[name],
            edgecolor="white",
            linewidth=0.45,
            label=f"{name} - {int(mask.sum())} of {int((family == name).sum())}",
        )
    axis.legend(
        frameon=False,
        fontsize=9.5,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),
        ncol=2,
        handletextpad=0.4,
        borderpad=0.3,
    )
    axis.set_aspect("equal")
    axis.axis("off")
    axis.text(
        0.5,
        -0.02,
        "the layout is given only the r^2 edges - it never sees which family a feature belongs to\n"
        f"{isolated} features have no partner above r^2 = {threshold:g} and are not drawn",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.8,
        color=MUTED,
    )
    if title:
        figure.suptitle(
            "Rich-feature redundancy structure without labels\n"
            "global descriptors, Hilbert and Fourier replaced by their rich versions",
            fontsize=12.5,
            color=TEXT,
        )
        figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    else:
        figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rich_feature_redundancy_r2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values, family, feature_names = load_matrix(args)
    log(f"{values.shape[0]:,} images x {values.shape[1]} rich visual features")
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
    values, family, feature_names, std = values[:, keep], family[keep], feature_names[keep], std[keep]

    standard = (values - values.mean(axis=0)) / std
    log("computing pairwise r^2 matrix...")
    correlation = (standard.T @ standard) / len(standard)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2
    np.fill_diagonal(r2, np.nan)

    pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "feature": feature_names,
        "family": family,
    }).to_csv(args.output_dir / "feature_index.csv", index=False)
    summary = write_group_summary(args.output_dir, family, r2)

    left, right = np.triu_indices(len(feature_names), 1)
    pair_r2 = r2[left, right]
    strong = pair_r2 >= args.threshold
    high = pd.DataFrame({
        "feature_a": feature_names[left[strong]],
        "feature_b": feature_names[right[strong]],
        "family_a": family[left[strong]],
        "family_b": family[right[strong]],
        "r2": pair_r2[strong],
        "pearson_r": correlation[left[strong], right[strong]],
    }).sort_values("r2", ascending=False)
    high.to_csv(args.output_dir / "pairs_ge_threshold.csv", index=False, float_format="%.6f")
    high[high["r2"] >= 0.80].to_csv(args.output_dir / "high_pairs_ge_0_8.csv", index=False, float_format="%.6f")

    feature_rows = []
    for index, feature in enumerate(feature_names):
        other = r2[index].copy()
        other[index] = np.nan
        best = int(np.nanargmax(other))
        feature_rows.append({
            "feature": feature,
            "family": family[index],
            "max_r2": float(other[best]),
            "best_partner": feature_names[best],
            "best_partner_family": family[best],
            **{f"partners_r2_ge_{str(t).replace('.', '_')}": int(np.nansum(other >= t)) for t in THRESHOLDS},
        })
    feature_summary = pd.DataFrame(feature_rows).sort_values("max_r2", ascending=False)
    feature_summary.to_csv(args.output_dir / "feature_summary.csv", index=False, float_format="%.6f")

    rows, columns = left[strong], right[strong]
    weights = pair_r2[strong]
    log(f"{len(weights):,} edges at r^2 >= {args.threshold:g}")
    plot_layout(args.output_dir / "layout.png", family, rows, columns, weights, args.threshold, title=True)
    plot_layout(args.output_dir / "layout_paper.png", family, rows, columns, weights, args.threshold, title=False)

    counts = feature_summary.groupby("family").agg(
        features=("feature", "count"),
        drawn=("partners_r2_ge_0_3", lambda s: int((s > 0).sum())),
        isolated=("partners_r2_ge_0_3", lambda s: int((s == 0).sum())),
        median_max_r2=("max_r2", "median"),
    ).reset_index()
    counts.to_csv(args.output_dir / "family_counts.csv", index=False, float_format="%.6f")
    report = {
        "method": "pairwise Pearson r squared; no labels used",
        "features": int(len(feature_names)),
        "threshold": args.threshold,
        "edges": int(len(weights)),
        "isolated": int((feature_summary["partners_r2_ge_0_3"] == 0).sum()),
        "families": counts.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("\n=== family counts ===")
    log(counts.to_string(index=False))
    log("\n=== group r^2 top ===")
    log(summary.head(12)[["group_a", "group_b", "median_r2", "p95_r2", "pairs_r2_ge_0_9"]].round(4).to_string(index=False))
    log(f"\nsaved: {args.output_dir}")


if __name__ == "__main__":
    main()
