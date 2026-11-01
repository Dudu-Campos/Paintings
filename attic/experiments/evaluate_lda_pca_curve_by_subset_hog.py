"""PCA-component LDA curves for subsets 1, 3 and 6, with HOG folded into
Descriptors (130 + 1,764 = 1,894-dim Descriptors) -- same protocol as
`evaluate_lda_pca_curve_by_subset.py`, testing whether PCA's smoother,
non-monotonic-avoiding behavior (seen for Multiscale in subset 8) also
rescues HOG, whose supervised top-k curve peaked around k=256 and then
declined for every subset it touched.
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
    (1, 2, "1: Descriptors + HOG"),
    (3, 19, "3: + SLIC"),
    (6, 95, "6: + Hilbert"),
]
HOG_DIR = ROOT / "outputs/haralick_hog_features"
OUTPUT_DIR = ROOT / "outputs/lda_pca_curve_by_subset_136_hog"


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

    hog_values = np.load(HOG_DIR / "features.npy")[:, 13:]
    hog_stems = np.load(HOG_DIR / "stems.npy", allow_pickle=True).astype(str)
    hog = pd.DataFrame(hog_values, columns=[f"hog_{index}" for index in range(hog_values.shape[1])])
    hog.insert(0, "stem", hog_stems)
    frame = frame.merge(hog, on="stem", how="inner", validate="one_to_one")
    frame = frame.sort_values("stem").reset_index(drop=True)
    hog_columns = [f"hog_{index}" for index in range(hog_values.shape[1])]

    group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()
    }
    group_blocks["Descriptors"] = np.hstack(
        [group_blocks["Descriptors"], frame[hog_columns].to_numpy(dtype=np.float32)]
    ).astype(np.float32)
    log(f"Descriptors block is now {group_blocks['Descriptors'].shape[1]}d (130 + {len(hog_columns)} HOG)")

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
    log(f"LDA PCA-component curves (Descriptors+HOG): {len(frame):,} images, {n_classes} styles")

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
