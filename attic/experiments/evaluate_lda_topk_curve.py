"""Top-k LDA saturation curve over the full 8-factor pipeline's individual
features (not blocks).

Ranks every individual feature by LDA structure coefficient (the
collinearity-robust ranking already used in
`analyze_lda_structure_importance.py`: correlation of the standardized
feature with each discriminant axis, aggregated by that axis's explained
variance), computed once per fold from the training split. Then re-fits LDA
at log-spaced checkpoints k = 1, 2, 4, ... up to the full feature count and
records balanced accuracy, to see how much of the gap between a single
representation alone (e.g. Descriptors ~30pp standalone) and the full
pipeline (~42-45pp) comes from a handful of features versus needing (nearly)
all of them.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_full_factorial_ablation_text import FULL_MASK, load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def flat_column_names(
    block_names: list[str],
    groups: dict[str, list[str]],
    line_columns: list[str],
    felz_columns: list[str],
    fourier_columns: list[str],
    text_columns: list[str],
) -> list[str]:
    lookup = {
        **groups,
        "Line graph": line_columns,
        "Felzenszwalb": felz_columns,
        "Fourier": fourier_columns,
        "Text": text_columns,
    }
    names = []
    for block in block_names:
        names.extend(lookup[block])
    return names


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def structure_importance(model: Pipeline, train_values: np.ndarray) -> np.ndarray:
    standardized = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(train_values)
    ).astype(np.float64)
    scores = model.named_steps["lda"].transform(standardized)
    scores -= scores.mean(axis=0)
    score_std = scores.std(axis=0)
    valid = score_std > 1e-12
    scores[:, valid] /= score_std[valid]
    scores[:, ~valid] = 0.0
    correlations = (standardized.T @ scores) / len(standardized)
    weights = model.named_steps["lda"].explained_variance_ratio_.astype(np.float64)
    weights = weights[: correlations.shape[1]]
    weights /= weights.sum()
    return np.sqrt(np.sum(correlations ** 2 * weights[None, :], axis=1))


def checkpoints(n_features: int) -> list[int]:
    values = sorted({min(n_features, 2 ** exponent) for exponent in range(0, 20)} | {n_features})
    return [value for value in values if value <= n_features]


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
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lda_topk_curve")
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
    text_columns = [f"text_score_{index}" for index in range(n_classes)]

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    log(f"LDA top-k curve: {len(frame):,} images, {n_classes} styles")

    curve_rows, ranking_rows = [], []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra_blocks = {**fixed_extra_blocks, "Text": text_block}
        blocks, block_names = selected_blocks(FULL_MASK, group_blocks, extra_blocks)
        columns = flat_column_names(
            block_names, groups, line_columns, felz_columns, fourier_columns, text_columns
        )
        full_train = np.hstack([block[outer_train] for block in blocks]).astype(np.float32)
        full_test = np.hstack([block[outer_test] for block in blocks]).astype(np.float32)
        n_features = full_train.shape[1]

        started = time.perf_counter()
        full_model = make_lda(n_classes, n_features)
        full_model.fit(full_train, labels[outer_train])
        importance = structure_importance(full_model, full_train)
        order = np.argsort(-importance)
        log(f"fold {fold}/{N_FOLDS}: ranked {n_features} features ({time.perf_counter() - started:.1f}s)")

        for rank, feature_index in enumerate(order):
            ranking_rows.append({
                "fold": fold, "rank": rank + 1, "feature": columns[feature_index],
                "importance": float(importance[feature_index]),
            })

        for k in checkpoints(n_features):
            selected = order[:k]
            started = time.perf_counter()
            model = make_lda(n_classes, k)
            model.fit(full_train[:, selected], labels[outer_train])
            prediction = model.predict(full_test[:, selected])
            bacc = float(balanced_accuracy_score(labels[outer_test], prediction))
            elapsed = time.perf_counter() - started
            curve_rows.append({"fold": fold, "k": k, "bacc": bacc, "elapsed_seconds": elapsed})
            log(f"fold {fold}/{N_FOLDS} k={k:<5d} BAcc={bacc:.4f} seconds={elapsed:.1f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(args.output_dir / "curve_fold_results.csv", index=False)
    summary = curve.groupby("k", as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"), folds=("fold", "nunique"),
    ).sort_values("k")
    summary.to_csv(args.output_dir / "curve_summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(ranking_rows).to_csv(args.output_dir / "feature_ranking_by_fold.csv", index=False)

    log(f"Saved to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
