"""Inductive visual-only similarity graph feature for LDA and MLP.

The k-NN similarity space is the full visual 7-factor pipeline, explicitly
excluding Text.  Test samples are never inserted into the LabelSpreading graph:
the model is fit on train nodes only, then each query receives the average
label distribution of its nearest train neighbours.
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

from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_full_factorial_ablation_text import selected_blocks  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_prior import (  # noqa: E402
    VISUAL_FULL_MASK,
    prior_correct,
    scale_blocks,
)
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402

torch.set_num_threads(int(THREADS_PER_WORKER))

INNER_FOLDS = 5
OUTPUT_DIR = ROOT / "outputs/label_spreading_feature_visual_graph_inductive"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def inductive_spread(
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
    train_distributions = model.label_distributions_
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(train_x)
    _, indices = neighbors.kneighbors(query_x)
    return train_distributions[indices].mean(axis=1).astype(np.float32)


def inductive_node_feature_scores(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    target_x: np.ndarray,
    n_classes: int,
    n_neighbors: int,
    alpha: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(fit_x), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        oof[inner_val] = inductive_spread(
            fit_x[inner_train],
            fit_y[inner_train],
            fit_x[inner_val],
            n_classes,
            n_neighbors,
            alpha,
            max_iter,
        )
    target_scores = inductive_spread(fit_x, fit_y, target_x, n_classes, n_neighbors, alpha, max_iter)
    return oof, target_scores


def write_comparison(detail: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for classifier in sorted(detail["classifier"].unique()):
        baseline_name = f"1: visual-only {classifier.upper()}"
        baseline = detail[
            detail["classifier"].eq(classifier) & detail["condition"].eq(baseline_name)
        ].set_index("fold")["bacc"]
        for condition in sorted(
            c
            for c in detail.loc[detail["classifier"].eq(classifier), "condition"].unique()
            if c != baseline_name
        ):
            paired = detail[
                detail["classifier"].eq(classifier) & detail["condition"].eq(condition)
            ].set_index("fold")["bacc"]
            common_folds = baseline.index.intersection(paired.index)
            delta = paired.loc[common_folds] - baseline.loc[common_folds]
            if len(delta) > 1:
                statistic, p_value = ttest_rel(paired.loc[common_folds], baseline.loc[common_folds])
            else:
                statistic, p_value = float("nan"), float("nan")
            rows.append(
                {
                    "classifier": classifier,
                    "condition": condition,
                    "folds": len(common_folds),
                    "base_bacc_mean": 100 * float(baseline.loc[common_folds].mean()),
                    "graph_bacc_mean": 100 * float(paired.loc[common_folds].mean()),
                    "delta_bacc_pp": 100 * float(delta.mean()),
                    "delta_std_pp": 100 * float(delta.std()),
                    "t_statistic": float(statistic),
                    "p_paired": float(p_value),
                    "positive_folds": int((delta > 0).sum()),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False, float_format="%.6f")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"], choices=["lda", "mlp"])
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument(
        "--felz-features",
        type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    visual_blocks, visual_names = selected_blocks(VISUAL_FULL_MASK, group_blocks, fixed_extra_blocks)
    log(
        f"inductive visual-only similarity graph: {len(frame):,} images, {n_classes} classes, "
        f"{sum(block.shape[1] for block in visual_blocks)} visual dims, blocks={visual_names}"
    )

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(row["classifier"]), str(row["condition"]), int(row["fold"])) for row in rows}

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        pipeline_scaled = scale_blocks(visual_blocks, outer_train)
        started = time.perf_counter()
        oof_fit, target_scores = inductive_node_feature_scores(
            pipeline_scaled[outer_train],
            labels[outer_train],
            pipeline_scaled[outer_test],
            n_classes,
            args.n_neighbors,
            args.alpha,
            args.max_iter,
            SEED + fold,
        )
        elapsed_graph = time.perf_counter() - started
        node_feature = np.zeros((len(frame), n_classes), dtype=np.float32)
        node_feature[outer_train] = oof_fit
        node_feature[outer_test] = target_scores
        corrected_feature = prior_correct(node_feature, labels[outer_train], n_classes)
        log(f"fold={fold}/{N_FOLDS} inductive visual graph calculated in {elapsed_graph:.1f}s")

        condition_blocks = {
            "raw": visual_blocks + [node_feature],
            "prior-corrected": visual_blocks + [corrected_feature],
        }
        for classifier in args.classifiers:
            classifier_name = classifier.upper()
            baseline_condition = f"1: visual-only {classifier_name}"
            conditions = {
                baseline_condition: visual_blocks,
                f"2: visual-only {classifier_name} + inductive similarity graph": condition_blocks["raw"],
                f"3: visual-only {classifier_name} + inductive similarity graph (prior-corrected)": condition_blocks[
                    "prior-corrected"
                ],
            }
            for condition, blocks in conditions.items():
                if (classifier, condition, fold) in completed:
                    continue
                dimensions = sum(block.shape[1] for block in blocks)
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, outer_train, outer_test)
                else:
                    metrics = evaluate_mlp(blocks, labels, outer_train, outer_test, fold, args.epochs, args.patience)
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "fold": fold,
                        "classifier": classifier,
                        "condition": condition,
                        "dimensions": dimensions,
                        "elapsed_seconds": elapsed + (elapsed_graph if "similarity graph" in condition else 0.0),
                        **metrics,
                    }
                )
                completed.add((classifier, condition, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(
                    f"fold={fold}/{N_FOLDS} {classifier:<3} {condition:<76} "
                    f"dims={dimensions:4d} BAcc={metrics['bacc']:.4f}"
                )

    detail = pd.DataFrame(rows)
    comparison = write_comparison(detail, args.output_dir)
    log(f"saved: {args.output_dir}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
