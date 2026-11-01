"""Force-layout plot of pairwise r^2 for rich visual features plus text scores."""

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
sys.path.insert(0, str(ROOT / "experiments"))

from plot_rich_redundancy_layout import (  # noqa: E402
    SEED,
    THRESHOLDS,
    read_npy,
    read_parquet,
)


FAMILY_ORDER = (
    "Region graphs",
    "Image statistics",
    "Space-filling curve",
    "Spectral",
    "Text (external)",
)
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
    "Text (external)": "#e94149",
}
TEXT, MUTED = "#0b0b0b", "#52514e"
BACKDROP = "#eef2f6"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_text_scores(stems_path: Path, decisions_path: Path, classes_path: Path) -> pd.DataFrame:
    stems = np.load(stems_path, allow_pickle=True).astype(str)
    decisions = np.load(decisions_path).astype(np.float32)
    classes = np.load(classes_path, allow_pickle=True).astype(str)
    if len(stems) != len(decisions):
        raise ValueError("Text stem/score length mismatch")
    columns = [f"text_score_{name}" for name in classes]
    frame = pd.DataFrame(decisions, columns=columns)
    frame.insert(0, "stem", stems)
    if frame["stem"].duplicated().any():
        raise ValueError("Duplicate stems in text score file")
    return frame


def load_matrix_with_text(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(args.meta).dropna(subset=["style"]).drop_duplicates("stem", keep="last")
    frame = frame[["stem", "style"]].copy()
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

    families, feature_names = [], []
    for name, piece in pieces:
        before = len(frame)
        frame = frame.merge(piece, on="stem", how="inner", validate="one_to_one")
        columns = [column for column in piece.columns if column != "stem"]
        families.extend([name] * len(columns))
        feature_names.extend(columns)
        if len(frame) != before:
            log(f"{name}: cohort changed {before:,} -> {len(frame):,}")

    counts = frame["style"].value_counts()
    frame = (
        frame[frame["style"].isin(counts[counts >= 5].index)]
        .sort_values("stem")
        .reset_index(drop=True)
    )
    values = frame[feature_names].to_numpy(dtype=np.float32)
    family = np.asarray(families, dtype=object)
    feature_names = np.asarray(feature_names, dtype=object)

    text = load_text_scores(args.text_stems, args.text_decisions, args.text_classes)
    text_columns = [column for column in text.columns if column != "stem"]
    merged = frame[["stem"]].merge(text, on="stem", how="inner", validate="one_to_one")
    if len(merged) != values.shape[0]:
        raise ValueError(
            f"Text cohort mismatch: visual matrix has {values.shape[0]:,} rows, "
            f"text merge has {len(merged):,}"
        )
    text_values = merged[text_columns].to_numpy(dtype=np.float32)
    values = np.concatenate([values, text_values], axis=1)
    family = np.concatenate([family, np.array(["Text (external)"] * len(text_columns), dtype=object)])
    feature_names = np.concatenate([feature_names, np.array(text_columns, dtype=object)])
    return values, family, feature_names

def write_group_summary_with_text(output_dir: Path, family: np.ndarray, r2: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, left in enumerate(FAMILY_ORDER):
        left_mask = family == left
        if not left_mask.any():
            continue
        for right in FAMILY_ORDER[index:]:
            right_mask = family == right
            if not right_mask.any():
                continue
            if left == right:
                sub = r2[np.ix_(left_mask, right_mask)]
                values = sub[np.triu_indices(sub.shape[0], k=1)]
            else:
                values = r2[np.ix_(left_mask, right_mask)].ravel()
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            rows.append(
                {
                    "group_a": left,
                    "group_b": right,
                    "n_pairs": int(len(values)),
                    "mean_r2": float(values.mean()),
                    "median_r2": float(np.median(values)),
                    "p95_r2": float(np.quantile(values, 0.95)),
                    "max_r2": float(values.max()),
                    **{f"pairs_r2_ge_{str(t).replace('.', '_')}": int((values >= t).sum()) for t in THRESHOLDS},
                }
            )
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
    backdrop: bool = False,
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

    figure, axis = plt.subplots(figsize=(9.8, 8.6))
    if backdrop and len(points):
        pad = 0.10
        x_min, y_min = points.min(axis=0) - pad
        x_max, y_max = points.max(axis=0) + pad
        density, x_edges, y_edges = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=90,
            range=((x_min, x_max), (y_min, y_max)),
        )
        density = np.sqrt(density.T)
        if density.max() > 0:
            density /= density.max()
        axis.contourf(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
            density,
            levels=[0.02, 0.08, 0.18, 1.0],
            colors=[BACKDROP, "#e5ebf1", "#dbe3eb"],
            alpha=0.24,
            zorder=0,
        )
        axis.imshow(
            density,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap="Greys",
            alpha=0.10,
            interpolation="bicubic",
            zorder=0,
        )
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
            s=24 if name == "Text (external)" else 22,
            zorder=4 if name == "Text (external)" else 3,
            color=COLOURS[name],
            edgecolor="white",
            linewidth=0.45,
            label=f"{name} - {int(mask.sum())} of {int((family == name).sum())}",
        )
    axis.legend(
        frameon=False,
        fontsize=9.3,
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
        f"{isolated} features have no partner above r^2 = {threshold:g} and are not drawn; "
        "text points are external TF-IDF decision-score features",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.7,
        color=MUTED,
    )
    if title:
        figure.suptitle(
            "Rich-feature redundancy structure with external text scores",
            fontsize=12.8,
            color=TEXT,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.93))
    else:
        figure.tight_layout(rect=(0, 0.035, 1, 1))
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
    parser.add_argument("--text-stems", type=Path, default=ROOT / "results/tfidf_caption_style_stems.npy")
    parser.add_argument("--text-decisions", type=Path, default=ROOT / "results/tfidf_caption_style_oof_decisions.npy")
    parser.add_argument("--text-classes", type=Path, default=ROOT / "results/tfidf_caption_style_classes.npy")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rich_feature_redundancy_r2_with_text")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values, family, feature_names = load_matrix_with_text(args)
    log(f"{values.shape[0]:,} images x {values.shape[1]:,} rich features including Text")
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

    pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names)),
            "feature": feature_names,
            "family": family,
        }
    ).to_csv(args.output_dir / "feature_index.csv", index=False)
    summary = write_group_summary_with_text(args.output_dir, family, r2)

    left, right = np.triu_indices(len(feature_names), 1)
    pair_r2 = r2[left, right]
    strong = pair_r2 >= args.threshold
    high = pd.DataFrame(
        {
            "feature_a": feature_names[left[strong]],
            "feature_b": feature_names[right[strong]],
            "family_a": family[left[strong]],
            "family_b": family[right[strong]],
            "r2": pair_r2[strong],
            "pearson_r": correlation[left[strong], right[strong]],
        }
    ).sort_values("r2", ascending=False)
    high.to_csv(args.output_dir / "pairs_ge_threshold.csv", index=False, float_format="%.6f")
    high[high["r2"] >= 0.80].to_csv(args.output_dir / "high_pairs_ge_0_8.csv", index=False, float_format="%.6f")

    feature_rows = []
    for index, feature in enumerate(feature_names):
        other = r2[index].copy()
        other[index] = np.nan
        best = int(np.nanargmax(other))
        feature_rows.append(
            {
                "feature": feature,
                "family": family[index],
                "max_r2": float(other[best]),
                "best_partner": feature_names[best],
                "best_partner_family": family[best],
                **{f"partners_r2_ge_{str(t).replace('.', '_')}": int(np.nansum(other >= t)) for t in THRESHOLDS},
            }
        )
    feature_summary = pd.DataFrame(feature_rows).sort_values("max_r2", ascending=False)
    feature_summary.to_csv(args.output_dir / "feature_summary.csv", index=False, float_format="%.6f")

    rows, columns = left[strong], right[strong]
    weights = pair_r2[strong]
    log(f"{len(weights):,} edges at r^2 >= {args.threshold:g}")
    plot_layout(args.output_dir / "layout.png", family, rows, columns, weights, args.threshold, title=True)
    plot_layout(args.output_dir / "layout_paper.png", family, rows, columns, weights, args.threshold, title=False)
    plot_layout(args.output_dir / "layout_backdrop.png", family, rows, columns, weights, args.threshold, title=True, backdrop=True)
    plot_layout(args.output_dir / "layout_backdrop_paper.png", family, rows, columns, weights, args.threshold, title=False, backdrop=True)

    counts = (
        feature_summary.groupby("family")
        .agg(
            features=("feature", "count"),
            drawn=("partners_r2_ge_0_3", lambda s: int((s > 0).sum())),
            isolated=("partners_r2_ge_0_3", lambda s: int((s == 0).sum())),
            median_max_r2=("max_r2", "median"),
        )
        .reset_index()
    )
    counts["family"] = pd.Categorical(counts["family"], FAMILY_ORDER, ordered=True)
    counts = counts.sort_values("family")
    counts.to_csv(args.output_dir / "family_counts.csv", index=False, float_format="%.6f")
    report = {
        "method": "pairwise Pearson r squared; layout uses no family labels; text features are external TF-IDF decision scores",
        "features": int(len(feature_names)),
        "threshold": args.threshold,
        "edges": int(len(weights)),
        "isolated": int((feature_summary["partners_r2_ge_0_3"] == 0).sum()),
        "families": counts.astype({"family": str}).to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("\n=== family counts ===")
    log(counts.astype({"family": str}).to_string(index=False))
    log("\n=== group r^2 top ===")
    log(summary.head(12)[["group_a", "group_b", "median_r2", "p95_r2", "pairs_r2_ge_0_9"]].round(4).to_string(index=False))
    log(f"\nsaved: {args.output_dir}")


if __name__ == "__main__":
    main()
