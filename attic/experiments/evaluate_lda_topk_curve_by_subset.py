"""Eight top-k LDA saturation curves, one per nested subset of representations
(1 representation alone, the 2 best combined, ..., all 8).

Representations are added in exact-Shapley rank order (from
`outputs/full_factorial_ablation_text/`): Descriptors, Felzenszwalb, SLIC,
Text, Line graph, Hilbert, Fourier, Multiscale. For each of the 8 nested
subsets, features are ranked *within that subset only* (a fresh LDA fit on
just those columns, ranked by structure coefficient -- not a slice of the
global 8-factor ranking), then LDA is re-fit at log-spaced checkpoints
k = 1, 2, 4, ... up to the subset's full width. This shows whether a lone
representation's growth/saturation shape differs from combined ones, and
whether any subset is starved for features (curve still rising steeply at
its own max k, suggesting more features for that representation are worth
extracting) rather than genuinely saturating.
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

from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


# Cumulative masks in exact-Shapley rank order (best representation first).
# FACTORS bits: SLIC=1, Descriptors=2, Hilbert=4, Line graph=8,
# Felzenszwalb=16, Fourier=32, Text=64, Multiscale=128.
SUBSETS = [
    (1, 2, "1: Descriptors"),
    (2, 18, "2: + Felzenszwalb"),
    (3, 19, "3: + SLIC"),
    (4, 83, "4: + Text"),
    (5, 91, "5: + Line graph"),
    (6, 95, "6: + Hilbert"),
    (7, 127, "7: + Fourier"),
    (8, 255, "8: + Multiscale (tudo)"),
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lda_topk_curve_by_subset")
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

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    log(f"LDA top-k curves by subset: {len(frame):,} images, {n_classes} styles, {len(SUBSETS)} subsets")

    curve_rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra_blocks = {**fixed_extra_blocks, "Text": text_block}

        for n_representations, mask, subset_label in SUBSETS:
            blocks, block_names = selected_blocks(mask, group_blocks, extra_blocks)
            train_x = np.hstack([block[outer_train] for block in blocks]).astype(np.float32)
            test_x = np.hstack([block[outer_test] for block in blocks]).astype(np.float32)
            n_features = train_x.shape[1]

            started = time.perf_counter()
            full_model = make_lda(n_classes, n_features)
            full_model.fit(train_x, labels[outer_train])
            importance = structure_importance(full_model, train_x)
            order = np.argsort(-importance)
            rank_time = time.perf_counter() - started

            for k in checkpoints(n_features):
                selected = order[:k]
                started = time.perf_counter()
                model = make_lda(n_classes, k)
                model.fit(train_x[:, selected], labels[outer_train])
                prediction = model.predict(test_x[:, selected])
                bacc = float(balanced_accuracy_score(labels[outer_test], prediction))
                elapsed = time.perf_counter() - started
                curve_rows.append({
                    "fold": fold, "n_representations": n_representations,
                    "subset": subset_label, "subset_features": n_features,
                    "k": k, "bacc": bacc, "elapsed_seconds": elapsed,
                })
                log(
                    f"fold={fold}/{N_FOLDS} subset={subset_label:<24} "
                    f"({n_features:4d}d) k={k:<5d} BAcc={bacc:.4f} seconds={elapsed:.1f}"
                )
            log(f"fold={fold}/{N_FOLDS} subset={subset_label:<24} ranked in {rank_time:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(args.output_dir / "curve_fold_results.csv", index=False)
    summary = curve.groupby(["n_representations", "subset", "subset_features", "k"], as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"), folds=("fold", "nunique"),
    ).sort_values(["n_representations", "k"])
    summary.to_csv(args.output_dir / "curve_summary.csv", index=False, float_format="%.6f")

    log(f"Saved to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
