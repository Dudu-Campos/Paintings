"""Same node-feature idea, but the k-NN similarity graph is now built in the
full 8-factor pipeline's feature space (~1603d, per-block scaled) instead of
Descriptors alone (`evaluate_label_spreading_feature_full_pipeline.py`).
Tests whether a richer similarity basis for the graph itself carries signal
that a Descriptors-only graph didn't -- since the Descriptors-graph node
feature's benefit mostly evaporated once combined with everything else,
maybe a same-space graph (built on what LDA already sees) doesn't help
either, or maybe local/nonlinear neighborhood structure in the full space is
still informative beyond LDA's single global linear discriminant.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from evaluate_full_factorial_ablation import evaluate_lda
from evaluate_full_factorial_ablation_text import FULL_MASK, load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


INNER_FOLDS = 5
OUTPUT_DIR = ROOT / "outputs/label_spreading_feature_full_graph"


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


def scale_blocks(
    blocks: list[np.ndarray], fit_index: np.ndarray,
) -> np.ndarray:
    """Per-block StandardScaler (fit on fit_index only), then hstack -- same
    convention as `scaled_splits`, but returns one full-length matrix."""
    parts = []
    for block in blocks:
        scaler = StandardScaler().fit(block[fit_index])
        parts.append(scaler.transform(block).astype(np.float32))
    return np.hstack(parts).astype(np.float32, copy=False)


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
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
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
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    log(f"LDA completo + node feature (grafo no espaco completo): {len(frame):,} imagens, {n_classes} estilos")

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
        if args.only_fold is not None and fold != args.only_fold:
            continue
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra_blocks = {**fixed_extra_blocks, "Text": text_block}
        pipeline_blocks, block_names = selected_blocks(FULL_MASK, group_blocks, extra_blocks)

        pipeline_scaled = scale_blocks(pipeline_blocks, outer_train)

        started = time.perf_counter()
        oof_fit, target_scores = node_feature_scores(
            pipeline_scaled[outer_train], labels[outer_train], pipeline_scaled[outer_test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
        )
        elapsed_graph = time.perf_counter() - started
        node_feature = np.zeros((len(frame), n_classes), dtype=np.float32)
        node_feature[outer_train] = oof_fit
        node_feature[outer_test] = target_scores
        log(f"fold={fold}/{N_FOLDS} grafo (1603d) calculado em {elapsed_graph:.1f}s")

        conditions = {
            "1: LDA completo (8 fatores)": pipeline_blocks,
            "2: LDA completo + node feature (grafo completo)": pipeline_blocks + [node_feature],
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
                "elapsed_seconds": elapsed + (elapsed_graph if "grafo completo" in condition else 0.0),
                **metrics,
            })
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            log(
                f"fold={fold}/{N_FOLDS} {condition:<48} dims={dimensions:4d} "
                f"BAcc={metrics['bacc']:.4f} seconds={elapsed:.1f}"
            )

    detail = pd.DataFrame(rows)
    baseline = detail[detail.condition.eq("1: LDA completo (8 fatores)")].set_index("fold")["bacc"]
    paired = detail[detail.condition.eq("2: LDA completo + node feature (grafo completo)")].set_index("fold")["bacc"]
    common_folds = baseline.index.intersection(paired.index)
    delta = paired.loc[common_folds] - baseline.loc[common_folds]
    if len(delta) > 1:
        statistic, p_value = ttest_rel(paired.loc[common_folds], baseline.loc[common_folds])
    else:
        p_value = float("nan")
    comparison = pd.DataFrame([{
        "n_neighbors": args.n_neighbors, "alpha": args.alpha, "folds": len(common_folds),
        "delta_bacc_pp": 100 * float(delta.mean()), "delta_std_pp": 100 * float(delta.std()),
        "p_paired": float(p_value), "positive_folds": int((delta > 0).sum()),
    }])
    comparison.to_csv(args.output_dir / "comparison.csv", index=False, float_format="%.6f")
    log(f"Saved to {args.output_dir}")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
