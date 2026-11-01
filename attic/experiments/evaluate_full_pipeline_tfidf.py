"""Full 7-factor visual pipeline (mask=127) versus the same pipeline fused with
TF-IDF caption scores, on LDA, XGBoost and MLP.

Reuses the exact block-loading code path of `evaluate_full_factorial_ablation.py`
(`selected_blocks(FULL_MASK, ...)`), so the "Pipeline completo" condition is the
same 1,576-dimension representation as mask=127 there. The TF-IDF branch is the
nested OOF LinearSVC decision-score scheme from `final_tfidf_visual_mlp.py`.
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
import torch
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

torch.set_num_threads(int(THREADS_PER_WORKER))

from evaluate_full_factorial_ablation import FULL_MASK, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from evaluate_robustness_mlp_aug import train_one
from final_tfidf_visual_mlp import (
    N_FOLDS, ROOT, SEED, fit_text_scores, load_dataset,
)
from tfidf_caption_style import make_stop_words


CLASSIFIERS = ("lda", "xgboost", "mlp")
CONDITIONS = ("Pipeline completo", "Pipeline completo + TF-IDF")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def scale_and_stack(
    train_blocks: list[np.ndarray], validation_blocks: list[np.ndarray], test_blocks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts, validation_parts, test_parts = [], [], []
    for train_block, validation_block, test_block in zip(
        train_blocks, validation_blocks, test_blocks, strict=True
    ):
        scaler = StandardScaler()
        train_parts.append(scaler.fit_transform(train_block).astype(np.float32))
        validation_parts.append(scaler.transform(validation_block).astype(np.float32))
        test_parts.append(scaler.transform(test_block).astype(np.float32))
    return (
        np.hstack(train_parts).astype(np.float32, copy=False),
        np.hstack(validation_parts).astype(np.float32, copy=False),
        np.hstack(test_parts).astype(np.float32, copy=False),
    )


def evaluate_lda(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, n_classes: int,
) -> dict[str, float]:
    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
    model.fit(train_x, train_y)
    return prediction_metrics(test_y, model.predict(test_x))


def evaluate_xgb(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
    n_classes: int, seed: int,
) -> dict[str, float]:
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0, objective="multi:softmax",
        num_class=n_classes, eval_metric="mlogloss", tree_method="hist",
        n_jobs=16, random_state=seed,
    )
    model.fit(train_x, train_y, sample_weight=compute_sample_weight("balanced", train_y))
    return prediction_metrics(test_y, model.predict(test_x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/full_pipeline_tfidf")
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features
    )
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()
    }
    extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    blocks, block_names = selected_blocks(FULL_MASK, group_blocks, extra_blocks)
    dimensions = sum(block.shape[1] for block in blocks)

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    texts = frame["caption"].to_numpy(dtype=str)
    stop_words = make_stop_words(encoder.classes_)

    log(
        f"Full pipeline vs +TF-IDF: {len(frame):,} images, {n_classes} styles, "
        f"{dimensions} dims / {len(blocks)} branches ({' | '.join(block_names)})"
    )

    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y), dtype=np.uint8), y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() and not args.restart else []
    completed = {
        (str(row["classifier"]), str(row["condition"]), int(row["fold"])) for row in rows
    }

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold
        )
        target_texts = np.concatenate([texts[validation], texts[outer_test]])
        fit_text, target_text = fit_text_scores(
            texts[fit], y[fit], target_texts, stop_words, n_classes, SEED + fold,
        )
        validation_text = target_text[:len(validation)]
        test_text = target_text[len(validation):]

        visual_train = [block[fit] for block in blocks]
        visual_validation = [block[validation] for block in blocks]
        visual_test = [block[outer_test] for block in blocks]
        branch_sizes = [block.shape[1] for block in blocks]

        conditions = {
            "Pipeline completo": (visual_train, visual_validation, visual_test, branch_sizes),
            "Pipeline completo + TF-IDF": (
                visual_train + [fit_text], visual_validation + [validation_text],
                visual_test + [test_text], branch_sizes + [n_classes],
            ),
        }

        for condition, (train_blocks, validation_blocks, test_blocks, sizes) in conditions.items():
            train_x, validation_x, test_x = scale_and_stack(train_blocks, validation_blocks, test_blocks)
            for classifier in args.classifiers:
                key = (classifier, condition, fold)
                if key in completed:
                    continue
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(train_x, y[fit], test_x, y[outer_test], n_classes)
                elif classifier == "xgboost":
                    metrics = evaluate_xgb(train_x, y[fit], test_x, y[outer_test], n_classes, SEED + fold)
                elif classifier == "mlp":
                    metrics, _ = train_one(
                        train_x, y[fit], validation_x, y[validation], test_x, y[outer_test],
                        sizes, n_classes, SEED + fold, args.epochs, args.patience,
                    )
                else:  # pragma: no cover
                    raise ValueError(classifier)
                elapsed = time.perf_counter() - started
                rows.append({
                    "classifier": classifier, "condition": condition, "fold": fold,
                    "n_images": len(frame), "n_fit": len(fit), "n_validation": len(validation),
                    "n_test": len(outer_test), "dimensions": sum(sizes), "branches": len(sizes),
                    "elapsed_seconds": elapsed, **metrics,
                })
                completed.add(key)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(
                    f"{classifier:<8} {condition:<28} fold={fold}/{N_FOLDS} "
                    f"dims={sum(sizes):4d} BAcc={metrics['bacc']:.4f} seconds={elapsed:.1f}"
                )

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["classifier", "condition"], as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"), folds=("fold", "nunique"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")

    comparisons = []
    for classifier, classifier_rows in detail.groupby("classifier"):
        pivot = classifier_rows.pivot(index="fold", columns="condition", values="bacc")
        if set(CONDITIONS) - set(pivot.columns) or pivot.isna().any().any():
            continue
        baseline = pivot["Pipeline completo"].to_numpy()
        fused = pivot["Pipeline completo + TF-IDF"].to_numpy()
        statistic, p_value = ttest_rel(fused, baseline)
        comparisons.append({
            "classifier": classifier,
            "delta_bacc_pp": 100 * float(np.mean(fused - baseline)),
            "p_paired": float(p_value),
            "positive_folds": int(np.sum(fused > baseline)),
            **{f"delta_fold_{index + 1}_pp": 100 * float(value) for index, value in enumerate(fused - baseline)},
        })
    pd.DataFrame(comparisons).to_csv(args.output_dir / "comparisons.csv", index=False, float_format="%.6f")

    log(f"Saved results to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
