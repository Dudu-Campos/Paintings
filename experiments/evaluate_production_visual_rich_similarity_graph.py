"""Production visual-only model with a richer inductive similarity-graph branch.

Text is excluded from both the classifier and the graph construction.  The
graph branch is built on the production visual space (1127d): global visual
families, Line graph, Felzenszwalb, Fourier, L(G_felz), and native Descriptor
quadrants.  It includes the 27 prior-corrected propagated class scores plus
inductive neighbourhood statistics from the same k-NN query.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_felzenszwalb_line_graph import load_felz_line  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_full_factorial_ablation_text import selected_blocks  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_prior import (  # noqa: E402
    prior_correct,
    scale_blocks,
)
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks  # noqa: E402
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402


torch.set_num_threads(int(THREADS_PER_WORKER))
INNER_FOLDS = 5
OUTPUT_DIR = ROOT / "outputs/production_visual_rich_similarity_graph"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-8, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def query_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    n_classes: int,
    n_neighbors: int,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    model.fit(train_x, train_y)
    if len(model.classes_) != n_classes:
        raise ValueError(f"LabelSpreading saw {len(model.classes_)}/{n_classes} classes")

    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(train_x)
    distances, indices = neighbors.kneighbors(query_x)
    propagated = model.label_distributions_[indices].mean(axis=1).astype(np.float32)

    # A robust local scale from the train graph makes density features comparable
    # across folds without fitting anything on query/test rows.
    train_distances, _ = neighbors.kneighbors(train_x)
    scale = float(np.median(train_distances[:, 1:])) if n_neighbors > 1 else float(np.median(train_distances))
    scale = max(scale, 1e-6)
    similarity = np.exp(-distances / scale).astype(np.float32)

    sorted_scores = np.sort(propagated, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    distance_quantiles = np.quantile(distances, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    similarity_sum = similarity.sum(axis=1)
    effective_neighbors = similarity_sum**2 / np.square(similarity).sum(axis=1).clip(min=1e-8)

    stats = np.column_stack(
        [
            distances.mean(axis=1),
            distances.std(axis=1),
            distances.min(axis=1),
            distances.max(axis=1),
            distance_quantiles,
            similarity.mean(axis=1),
            similarity.std(axis=1),
            similarity.max(axis=1),
            similarity_sum,
            effective_neighbors,
            entropy(propagated),
            propagated.max(axis=1),
            margin,
        ]
    ).astype(np.float32)
    return np.hstack([propagated, stats]).astype(np.float32)


def rich_similarity_features(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    target_x: np.ndarray,
    n_classes: int,
    n_neighbors: int,
    alpha: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = None
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        values = query_features(
            fit_x[inner_train],
            fit_y[inner_train],
            fit_x[inner_val],
            n_classes,
            n_neighbors,
            alpha,
            max_iter,
        )
        if oof is None:
            oof = np.zeros((len(fit_x), values.shape[1]), dtype=np.float32)
        oof[inner_val] = values
    assert oof is not None
    target = query_features(fit_x, fit_y, target_x, n_classes, n_neighbors, alpha, max_iter)
    return oof, target


def prior_correct_rich(features: np.ndarray, train_y: np.ndarray, n_classes: int) -> np.ndarray:
    corrected = features.copy()
    corrected[:, :n_classes] = prior_correct(corrected[:, :n_classes], train_y, n_classes)
    return corrected


def write_summary(detail: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for classifier in sorted(detail["classifier"].unique()):
        base = detail[
            detail["classifier"].eq(classifier) & detail["condition"].eq("production visual-only")
        ].set_index("fold")["bacc"]
        for condition in sorted(
            c
            for c in detail.loc[detail["classifier"].eq(classifier), "condition"].unique()
            if c != "production visual-only"
        ):
            paired = detail[
                detail["classifier"].eq(classifier) & detail["condition"].eq(condition)
            ].set_index("fold")["bacc"]
            common = base.index.intersection(paired.index)
            delta = paired.loc[common] - base.loc[common]
            statistic, p_value = ttest_rel(paired.loc[common], base.loc[common]) if len(delta) > 1 else (np.nan, np.nan)
            rows.append(
                {
                    "classifier": classifier,
                    "condition": condition,
                    "folds": len(common),
                    "base_bacc_mean": 100 * float(base.loc[common].mean()),
                    "condition_bacc_mean": 100 * float(paired.loc[common].mean()),
                    "delta_bacc_pp": 100 * float(delta.mean()),
                    "delta_std_pp": 100 * float(delta.std()),
                    "t_statistic": float(statistic),
                    "p_paired": float(p_value),
                    "positive_folds": int((delta > 0).sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-line-features", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--felz-features", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "lda"], choices=["mlp", "lda"])
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    visual_core, visual_core_names = selected_blocks(BASELINE_MASK & ~64, group_blocks, fixed_extra)
    production_visual_blocks = visual_core + [felzline_block] + descriptor_quadrants
    production_visual_names = visual_core_names + [
        "L(G_felz)",
        "Descriptors top_left",
        "Descriptors top_right",
        "Descriptors bottom_left",
        "Descriptors bottom_right",
    ]

    log(
        f"rich similarity graph: {len(frame):,} images, {n_classes} classes; "
        f"production visual space={sum(b.shape[1] for b in production_visual_blocks)}d; "
        f"blocks={production_visual_names}"
    )

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels), dtype=np.uint8), labels))
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(row["condition"]), str(row["classifier"]), int(row["fold"])) for row in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        graph_scaled = scale_blocks(production_visual_blocks, train)
        started = time.perf_counter()
        train_graph, test_graph = rich_similarity_features(
            graph_scaled[train],
            labels[train],
            graph_scaled[test],
            n_classes,
            args.n_neighbors,
            args.alpha,
            args.max_iter,
            SEED + fold,
        )
        graph_feature = np.zeros((len(frame), train_graph.shape[1]), dtype=np.float32)
        graph_feature[train] = train_graph
        graph_feature[test] = test_graph
        graph_feature = prior_correct_rich(graph_feature, labels[train], n_classes)
        graph_elapsed = time.perf_counter() - started
        log(f"fold={fold}/{N_FOLDS} rich graph {graph_feature.shape[1]}d calculated in {graph_elapsed:.1f}s")

        conditions = {
            "production visual-only": production_visual_blocks,
            "production visual-only + rich similarity graph": production_visual_blocks + [graph_feature],
        }
        for condition, blocks in conditions.items():
            dimensions = sum(block.shape[1] for block in blocks)
            for classifier in args.classifiers:
                if (condition, classifier, fold) in done:
                    continue
                started = time.perf_counter()
                if classifier == "mlp":
                    metrics = evaluate_mlp(blocks, labels, train, test, fold, args.epochs, args.patience)
                else:
                    metrics = evaluate_lda(blocks, labels, train, test)
                rows.append(
                    {
                        "condition": condition,
                        "classifier": classifier,
                        "fold": fold,
                        "dimensions": dimensions,
                        "n_branches": len(blocks),
                        "elapsed_seconds": time.perf_counter() - started + (graph_elapsed if "rich similarity" in condition else 0.0),
                        **{key: metrics[key] for key in ("bacc", "accuracy", "macro_f1")},
                    }
                )
                done.add((condition, classifier, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(
                    f"fold={fold}/{N_FOLDS} {condition:<47} | {classifier:<3} | "
                    f"{dimensions:4d}d {len(blocks):2d}br | BAcc={100 * metrics['bacc']:.2f}%"
                )

    detail = pd.DataFrame(rows)
    summary = write_summary(detail, args.output_dir)
    log(f"saved: {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
