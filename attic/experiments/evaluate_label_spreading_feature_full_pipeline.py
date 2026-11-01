"""Same propagated-label node-feature test as `evaluate_label_spreading_feature.py`
(graph built on Descriptors similarity), but concatenated onto the full
8-factor pipeline (mask=255, ~1603d) instead of Descriptors alone -- does the
node feature still help once everything else (SLIC, Felzenszwalb, Hilbert,
Line graph, Fourier, Text, Multiscale) is already present, or was its signal
already redundant with the rest of the pipeline?
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from evaluate_full_factorial_ablation import evaluate_lda
from evaluate_full_factorial_ablation_text import FULL_MASK, load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


INNER_FOLDS = 5
OUTPUT_DIR = ROOT / "outputs/label_spreading_feature_full_pipeline"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def spread(
    labeled_X: np.ndarray, labeled_y: np.ndarray, unlabeled_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int,
) -> np.ndarray:
    combined_X = np.vstack([labeled_X, unlabeled_X])
    combined_y = np.concatenate([labeled_y, -np.ones(len(unlabeled_X), dtype=int)])
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    model.fit(combined_X, combined_y)
    if len(model.classes_) != n_classes:
        raise ValueError(
            f"LabelSpreading only saw {len(model.classes_)}/{n_classes} classes among labeled rows"
        )
    return model.label_distributions_[len(labeled_X):].astype(np.float32)


def node_feature_scores(
    fit_X: np.ndarray, fit_y: np.ndarray, target_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(fit_X), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        oof[inner_val] = spread(
            fit_X[inner_train], fit_y[inner_train], fit_X[inner_val],
            n_classes, n_neighbors, alpha, max_iter,
        )
    target_scores = spread(fit_X, fit_y, target_X, n_classes, n_neighbors, alpha, max_iter)
    return oof, target_scores


def inductive_spread(
    train_X: np.ndarray, train_y: np.ndarray, query_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int,
) -> np.ndarray:
    """Out-of-sample extension: fit LabelSpreading on `train_X`/`train_y` alone
    (fully labeled), then score each `query_X` row from its k nearest TRAIN
    neighbors only -- query points never see each other."""
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    model.fit(train_X, train_y)
    if len(model.classes_) != n_classes:
        raise ValueError(
            f"LabelSpreading only saw {len(model.classes_)}/{n_classes} classes among labeled rows"
        )
    train_distributions = model.label_distributions_
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(train_X)
    _, indices = neighbors.kneighbors(query_X)
    return train_distributions[indices].mean(axis=1).astype(np.float32)


def inductive_node_feature_scores(
    fit_X: np.ndarray, fit_y: np.ndarray, target_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(fit_X), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        oof[inner_val] = inductive_spread(
            fit_X[inner_train], fit_y[inner_train], fit_X[inner_val],
            n_classes, n_neighbors, alpha, max_iter,
        )
    target_scores = inductive_spread(fit_X, fit_y, target_X, n_classes, n_neighbors, alpha, max_iter)
    return oof, target_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features"
    )
    parser.add_argument(
        "--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet"
    )
    parser.add_argument(
        "--felz-features", type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument(
        "--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet"
    )
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features
    )
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()
    }
    descriptor_columns = groups["Descriptors"]
    descriptors_raw = frame[descriptor_columns].to_numpy(dtype=np.float64)

    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    log(f"LDA completo + node features: {len(frame):,} imagens, {n_classes} estilos")

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(row["condition"]), int(row["fold"])) for row in rows}

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra_blocks = {**fixed_extra_blocks, "Text": text_block}
        pipeline_blocks, block_names = selected_blocks(FULL_MASK, group_blocks, extra_blocks)

        descriptor_scaler = StandardScaler().fit(descriptors_raw[outer_train])
        descriptors_scaled = descriptor_scaler.transform(descriptors_raw)

        started = time.perf_counter()
        oof_fit, target_scores = node_feature_scores(
            descriptors_scaled[outer_train], labels[outer_train], descriptors_scaled[outer_test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
        )
        elapsed_graph = time.perf_counter() - started
        node_feature = np.zeros((len(frame), n_classes), dtype=np.float32)
        node_feature[outer_train] = oof_fit
        node_feature[outer_test] = target_scores

        started = time.perf_counter()
        oof_fit_ind, target_scores_ind = inductive_node_feature_scores(
            descriptors_scaled[outer_train], labels[outer_train], descriptors_scaled[outer_test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
        )
        elapsed_graph_ind = time.perf_counter() - started
        node_feature_ind = np.zeros((len(frame), n_classes), dtype=np.float32)
        node_feature_ind[outer_train] = oof_fit_ind
        node_feature_ind[outer_test] = target_scores_ind

        conditions = {
            "1: LDA completo (8 fatores)": pipeline_blocks,
            "2: LDA completo + node feature (transdutivo)": pipeline_blocks + [node_feature],
            "3: LDA completo + node feature (indutivo)": pipeline_blocks + [node_feature_ind],
        }
        condition_elapsed_extra = {
            "1: LDA completo (8 fatores)": 0.0,
            "2: LDA completo + node feature (transdutivo)": elapsed_graph,
            "3: LDA completo + node feature (indutivo)": elapsed_graph_ind,
        }
        for condition, condition_blocks in conditions.items():
            if (condition, fold) in completed:
                continue
            dimensions = sum(block.shape[1] for block in condition_blocks)
            started = time.perf_counter()
            metrics = evaluate_lda(condition_blocks, labels, outer_train, outer_test)
            elapsed = time.perf_counter() - started
            rows.append({
                "fold": fold, "condition": condition, "dimensions": dimensions,
                "elapsed_seconds": elapsed + condition_elapsed_extra[condition],
                **metrics,
            })
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            log(
                f"fold={fold}/{N_FOLDS} {condition:<32} dims={dimensions:4d} "
                f"BAcc={metrics['bacc']:.4f} seconds={elapsed:.1f}"
            )

    detail = pd.DataFrame(rows)
    baseline = detail[detail.condition.eq("1: LDA completo (8 fatores)")].set_index("fold")["bacc"]
    comparison_rows = []
    for label, condition in (
        ("transdutivo", "2: LDA completo + node feature (transdutivo)"),
        ("indutivo", "3: LDA completo + node feature (indutivo)"),
    ):
        paired = detail[detail.condition.eq(condition)].set_index("fold")["bacc"]
        common_folds = baseline.index.intersection(paired.index)
        delta = paired.loc[common_folds] - baseline.loc[common_folds]
        if len(delta) > 1:
            _, p_value = ttest_rel(paired.loc[common_folds], baseline.loc[common_folds])
        else:
            p_value = float("nan")
        comparison_rows.append({
            "variant": label, "n_neighbors": args.n_neighbors, "alpha": args.alpha,
            "folds": len(common_folds),
            "delta_bacc_pp": 100 * float(delta.mean()), "delta_std_pp": 100 * float(delta.std()),
            "p_paired": float(p_value), "positive_folds": int((delta > 0).sum()),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False, float_format="%.6f")
    log(f"Saved to {args.output_dir}")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
