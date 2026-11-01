"""Pairwise-r2 layout for rich visual features plus the global similarity graph."""

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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_felzenszwalb_line_graph import load_felz_line  # noqa: E402
from evaluate_full_factorial_ablation_text import selected_blocks  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_prior import scale_blocks  # noqa: E402
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks  # noqa: E402
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich,
    rich_similarity_features,
)
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402
from plot_rich_redundancy_layout import THRESHOLDS, read_npy, read_parquet  # noqa: E402


FAMILY_ORDER = (
    "Region graphs",
    "Image statistics",
    "Space-filling curve",
    "Spectral",
    "Global similarity graph",
    "Text",
)
COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
    "Global similarity graph": "#cdae31",
    "Text": "#e94149",
}
MARKERS = {
    "Region graphs": "o",
    "Image statistics": "s",
    "Space-filling curve": "^",
    "Spectral": "D",
    "Global similarity graph": "P",
    "Text": "*",
}
TEXT, MUTED, BACKDROP = "#0b0b0b", "#52514e", "#eef2f6"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_rich_visual(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], np.ndarray]:
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
    for family, piece in pieces:
        before = len(frame)
        frame = frame.merge(piece, on="stem", how="inner", validate="one_to_one")
        columns = [column for column in piece.columns if column != "stem"]
        families.extend([family] * len(columns))
        feature_names.extend(columns)
        if len(frame) != before:
            log(f"{family}: cohort changed {before:,} -> {len(frame):,}")

    counts = frame["style"].value_counts()
    frame = (
        frame[frame["style"].isin(counts[counts >= N_FOLDS].index)]
        .sort_values("stem")
        .reset_index(drop=True)
    )
    return frame, feature_names, np.asarray(families, dtype=object)


def load_text_scores(stems_path: Path, decisions_path: Path, classes_path: Path) -> pd.DataFrame:
    stems = np.load(stems_path, allow_pickle=True).astype(str)
    decisions = np.load(decisions_path).astype(np.float32)
    classes = np.load(classes_path, allow_pickle=True).astype(str)
    if len(stems) != len(decisions):
        raise ValueError("Text stem/score length mismatch")
    frame = pd.DataFrame(decisions, columns=[f"text_score_{name}" for name in classes])
    frame.insert(0, "stem", stems)
    if frame["stem"].duplicated().any():
        raise ValueError("Duplicate stems in text score file")
    return frame


def compute_global_graph_features(args: argparse.Namespace) -> pd.DataFrame:
    cache = args.global_cache or (args.output_dir / "global_similarity_graph_extended_features.parquet")
    if cache.exists() and not args.recompute_global:
        log(f"loading cached global graph features: {cache}")
        return pd.read_parquet(cache)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, _ = family_quadrant_blocks("Descriptors", group_blocks)
    visual_core, visual_names = selected_blocks(BASELINE_MASK & ~64, group_blocks, fixed_extra)
    graph_space = visual_core + [felzline_block] + descriptor_quadrants
    log(
        f"global graph feature space: {sum(block.shape[1] for block in graph_space)}d; "
        f"blocks={visual_names + ['L(G_felz)', 'Descriptors quadrants']}"
    )

    graph_feature = None
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    for fold, (train, test) in enumerate(splits, start=1):
        scaled = scale_blocks(graph_space, train)
        started = time.perf_counter()
        train_graph, test_graph = rich_similarity_features(
            scaled[train],
            labels[train],
            scaled[test],
            n_classes,
            args.n_neighbors,
            args.alpha,
            args.max_iter,
            SEED + fold,
        )
        train_graph = prior_correct_rich(train_graph, labels[train], n_classes)
        test_graph = prior_correct_rich(test_graph, labels[train], n_classes)
        if graph_feature is None:
            graph_feature = np.zeros((len(frame), train_graph.shape[1]), dtype=np.float32)
        graph_feature[train] = train_graph
        graph_feature[test] = test_graph
        log(f"fold={fold}/{N_FOLDS} global graph {graph_feature.shape[1]}d in {time.perf_counter() - started:.1f}s")

    assert graph_feature is not None
    columns = [f"global_graph_{index:03d}" for index in range(graph_feature.shape[1])]
    output = pd.DataFrame(graph_feature, columns=columns)
    output.insert(0, "stem", frame["stem"].astype(str).to_numpy())
    cache.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(cache, index=False)
    log(f"saved cached global graph features: {cache}")
    return output


def write_group_summary(output_dir: Path, family: np.ndarray, r2: np.ndarray) -> pd.DataFrame:
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
    backdrop: bool,
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

    figure, axis = plt.subplots(figsize=(9.9, 8.7))
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
            s=86 if name == "Text" else (58 if name == "Global similarity graph" else 24),
            zorder=4 if name in {"Global similarity graph", "Text"} else 3,
            color=COLOURS[name],
            marker=MARKERS[name],
            edgecolor="white",
            linewidth=0.55,
            label=f"{name} - {int(mask.sum())} of {int((family == name).sum())}",
        )
    axis.legend(frameon=False, fontsize=9.2, loc="upper left", bbox_to_anchor=(0.0, 1.10), ncol=2)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.text(
        0.5,
        -0.02,
        "the layout is given only the r^2 edges - it never sees which family a feature belongs to\n"
        f"{isolated} features have no partner above r^2 = {threshold:g} and are not drawn",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.7,
        color=MUTED,
    )
    if title:
        figure.suptitle("Rich-feature redundancy structure with global similarity graph", fontsize=12.8, color=TEXT)
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
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-line-features", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--felz-features", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--recompute-global", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--text-stems", type=Path, default=ROOT / "results/tfidf_caption_style_stems.npy")
    parser.add_argument("--text-decisions", type=Path, default=ROOT / "results/tfidf_caption_style_oof_decisions.npy")
    parser.add_argument("--text-classes", type=Path, default=ROOT / "results/tfidf_caption_style_classes.npy")
    parser.add_argument("--global-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rich_feature_redundancy_r2_with_global_graph")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rich_frame, rich_features, rich_families = load_rich_visual(args)
    global_graph = compute_global_graph_features(args)
    global_columns = [column for column in global_graph.columns if column != "stem"]
    before = len(rich_frame)
    rich_frame = rich_frame.merge(global_graph, on="stem", how="inner", validate="one_to_one")
    if len(rich_frame) != before:
        log(f"global graph merge changed cohort {before:,} -> {len(rich_frame):,}")

    feature_names = np.asarray([*rich_features, *global_columns], dtype=object)
    family = np.asarray([*rich_families.tolist(), *["Global similarity graph"] * len(global_columns)], dtype=object)
    if args.include_text:
        text_scores = load_text_scores(args.text_stems, args.text_decisions, args.text_classes)
        text_columns = [column for column in text_scores.columns if column != "stem"]
        before = len(rich_frame)
        rich_frame = rich_frame.merge(text_scores, on="stem", how="inner", validate="one_to_one")
        if len(rich_frame) != before:
            log(f"text merge changed cohort {before:,} -> {len(rich_frame):,}")
        feature_names = np.asarray([*feature_names.tolist(), *text_columns], dtype=object)
        family = np.asarray([*family.tolist(), *["Text"] * len(text_columns)], dtype=object)

    values = rich_frame[feature_names].to_numpy(dtype=np.float32)
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
    suffix = "including Global graph and Text" if args.include_text else "including Global graph"
    log(f"{values.shape[0]:,} images x {values.shape[1]:,} features {suffix}")

    standard = (values - values.mean(axis=0)) / std
    log("computing pairwise r^2 matrix...")
    correlation = (standard.T @ standard) / len(standard)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2
    np.fill_diagonal(r2, np.nan)

    pd.DataFrame({"feature_index": np.arange(len(feature_names)), "feature": feature_names, "family": family}).to_csv(
        args.output_dir / "feature_index.csv", index=False
    )
    summary = write_group_summary(args.output_dir, family, r2)
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

    rows, columns, weights = left[strong], right[strong], pair_r2[strong]
    log(f"{len(weights):,} edges at r^2 >= {args.threshold:g}")
    plot_layout(args.output_dir / "layout.png", family, rows, columns, weights, args.threshold, title=True, backdrop=False)
    plot_layout(args.output_dir / "layout_paper.png", family, rows, columns, weights, args.threshold, title=False, backdrop=False)
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
        "method": "pairwise Pearson r squared; layout uses no family labels",
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
    log(summary.head(14)[["group_a", "group_b", "median_r2", "p95_r2", "pairs_r2_ge_0_9"]].round(4).to_string(index=False))
    log(f"\nsaved: {args.output_dir}")


if __name__ == "__main__":
    main()
