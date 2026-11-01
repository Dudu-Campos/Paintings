"""PCA-component LDA curves for subsets 1, 4 and 8, to compare against the
supervised top-k (structure coefficient) curves already in
`outputs/lda_topk_curve_by_subset/`.

For each subset, features are standardized and projected with PCA (fit on
the training split only), then LDA is re-fit using the first k principal
components at the same log-spaced checkpoints as the top-k analysis. PCA
orders components by variance explained -- unsupervised -- unlike the
structure-coefficient ranking, which is supervised (uses the discriminant
axes). Comparing the two curves' shape/exponent for the same subsets tests
whether the "some features matter much more than others" pattern is a
property of the label-aware ranking or already present in the raw variance
structure of the representations.
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


SUBSETS = [
    (1, 2, "1: Descriptors"),
    (3, 19, "3: + SLIC"),
    (6, 95, "6: + Hilbert"),
]
OUTPUT_DIR = ROOT / "outputs/lda_pca_curve_by_subset_136"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_lda(n_classes: int, n_features: int) -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(
        n_components=min(n_classes - 1, n_features),
        priors=np.ones(n_classes) / n_classes,
    )


def checkpoints(n_features: int) -> list[int]:
    values = sorted({min(n_features, 2 ** exponent) for exponent in range(0, 20)} | {n_features})
    return [value for value in values if value <= n_features]


def main() -> None:
    frame, groups = load_dataset(ROOT / "outputs/wikiart_quadrant_features")
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, ROOT / "results/line_graph_full_features.parquet",
        ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    frame, fourier_columns = load_fourier(frame, ROOT / "results/fourier_full_features.parquet")
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
    log(f"LDA PCA-component curves: {len(frame):,} images, {n_classes} styles")

    curve_rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        text_block = load_text_block(ROOT / "outputs/tfidf_text_cache", fold, frame, n_classes)
        extra_blocks = {**fixed_extra_blocks, "Text": text_block}

        for n_representations, mask, subset_label in SUBSETS:
            blocks, block_names = selected_blocks(mask, group_blocks, extra_blocks)
            train_x = np.hstack([block[outer_train] for block in blocks]).astype(np.float32)
            test_x = np.hstack([block[outer_test] for block in blocks]).astype(np.float32)
            n_features = train_x.shape[1]
            n_components = min(n_features, len(outer_train) - 1)

            started = time.perf_counter()
            prep = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=SEED)),
            ])
            train_pca = prep.fit_transform(train_x)
            test_pca = prep.transform(test_x)
            fit_time = time.perf_counter() - started
            log(
                f"fold={fold}/{N_FOLDS} subset={subset_label:<24} ({n_features:5d}d) "
                f"PCA -> {n_components} comps in {fit_time:.1f}s"
            )

            for k in checkpoints(n_components):
                started = time.perf_counter()
                model = make_lda(n_classes, k)
                model.fit(train_pca[:, :k], labels[outer_train])
                prediction = model.predict(test_pca[:, :k])
                bacc = float(balanced_accuracy_score(labels[outer_test], prediction))
                elapsed = time.perf_counter() - started
                curve_rows.append({
                    "fold": fold, "n_representations": n_representations,
                    "subset": subset_label, "subset_features": n_features,
                    "k": k, "bacc": bacc, "elapsed_seconds": elapsed,
                })
                log(
                    f"fold={fold}/{N_FOLDS} subset={subset_label:<24} "
                    f"k={k:<5d} BAcc={bacc:.4f} seconds={elapsed:.1f}"
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUTPUT_DIR / "curve_fold_results.csv", index=False)
    summary = curve.groupby(["n_representations", "subset", "subset_features", "k"], as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"), folds=("fold", "nunique"),
    ).sort_values(["n_representations", "k"])
    summary.to_csv(OUTPUT_DIR / "curve_summary.csv", index=False, float_format="%.6f")

    log(f"Saved to {OUTPUT_DIR}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
