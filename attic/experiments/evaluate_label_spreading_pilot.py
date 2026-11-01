"""Pilot: label propagation on a k-NN similarity graph between whole
paintings, using Descriptors as the similarity basis -- a genuinely
non-instance-based (transductive) alternative to the per-instance
classifiers (LDA/MLP) used everywhere else in this project.

Per fold: outer_train labels are visible, outer_test labels are masked as
unknown (-1); `LabelSpreading` builds a k-NN graph over ALL sample images
(train+test features, never test labels) and propagates. This is standard
transductive semi-supervised practice, not label leakage -- no test label
ever enters the graph.

Compared against a plain LDA on the same Descriptors features, same folds,
same 12k pilot sample (`results/fourier_12k_canonical_sample.csv`), for a
direct instance-based vs. transductive comparison.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from fusion_curve_slic import ROOT, SEED, load_npy


N_FOLDS = 5
SAMPLE_PATH = ROOT / "results/fourier_12k_canonical_sample.csv"
OUTPUT_DIR = ROOT / "outputs/label_spreading_pilot"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    sample = pd.read_csv(SAMPLE_PATH)[["stem", "style"]].drop_duplicates("stem")
    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    frame = sample.merge(descriptors, on="stem", how="inner", validate="one_to_one")
    if len(frame) != len(sample):
        raise ValueError(f"lost images merging descriptors: {len(frame):,}/{len(sample):,}")
    frame = frame.sort_values("stem").reset_index(drop=True)

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["style"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    X = frame[descriptor_columns].to_numpy(dtype=np.float64)
    log(f"Piloto label spreading: {len(frame):,} imagens, {n_classes} estilos, {X.shape[1]} features (Descriptors)")

    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y), dtype=np.uint8), y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        # Fit the scaler on train only, then transform the full (train+test)
        # matrix used by the transductive graph -- avoids leaking test
        # statistics into the standardization itself.
        scaler = StandardScaler().fit(X[outer_train])
        X_scaled = scaler.transform(X)

        # --- LabelSpreading (transductive) ---
        y_masked = y.copy()
        y_masked[outer_test] = -1
        started = time.perf_counter()
        model = LabelSpreading(
            kernel="knn", n_neighbors=args.n_neighbors, alpha=args.alpha,
            max_iter=args.max_iter, n_jobs=int(THREADS_PER_WORKER),
        )
        model.fit(X_scaled, y_masked)
        prediction = model.transduction_[outer_test]
        elapsed_spreading = time.perf_counter() - started
        metrics_spreading = prediction_metrics(y[outer_test], prediction)
        rows.append({
            "fold": fold, "method": "label_spreading", "n_neighbors": args.n_neighbors,
            "alpha": args.alpha, "elapsed_seconds": elapsed_spreading, **metrics_spreading,
        })
        log(
            f"fold={fold}/{N_FOLDS} label_spreading BAcc={metrics_spreading['bacc']:.4f} "
            f"seconds={elapsed_spreading:.1f} n_iter={model.n_iter_}"
        )

        # --- LDA baseline (instance-based), same fold/features ---
        started = time.perf_counter()
        lda = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
        lda.fit(X_scaled[outer_train], y[outer_train])
        lda_prediction = lda.predict(X_scaled[outer_test])
        elapsed_lda = time.perf_counter() - started
        metrics_lda = prediction_metrics(y[outer_test], lda_prediction)
        rows.append({
            "fold": fold, "method": "lda", "n_neighbors": np.nan,
            "alpha": np.nan, "elapsed_seconds": elapsed_lda, **metrics_lda,
        })
        log(f"fold={fold}/{N_FOLDS} lda              BAcc={metrics_lda['bacc']:.4f} seconds={elapsed_lda:.1f}")

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary = detail.groupby("method", as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"Saved to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
