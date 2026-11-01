"""Nested, normalised k-NN baseline for the canonical 258 visual features."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset


ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (5, 15, 31, 63)
WEIGHTS = ("uniform", "distance")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def predict_from_neighbors(
    distances: np.ndarray, neighbor_indices: np.ndarray, train_labels: np.ndarray,
    n_classes: int, k: int, weight_mode: str,
) -> np.ndarray:
    """Vote from a single max-k neighbour query, avoiding repeated searches."""
    labels = train_labels[neighbor_indices[:, :k]]
    if weight_mode == "uniform":
        weights = np.ones_like(distances[:, :k], dtype=np.float64)
    elif weight_mode == "distance":
        weights = 1.0 / np.maximum(distances[:, :k], 1e-8)
    else:  # pragma: no cover
        raise ValueError(weight_mode)
    votes = np.zeros((len(labels), n_classes), dtype=np.float64)
    rows = np.repeat(np.arange(len(labels)), k)
    np.add.at(votes, (rows, labels.ravel()), weights.ravel())
    return votes.argmax(axis=1)


def neighbours(train_values: np.ndarray, query_values: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=max(K_VALUES), metric="minkowski", p=2, n_jobs=workers)
    model.fit(train_values)
    return model.kneighbors(query_values, return_distance=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/visual_knn")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    frame, groups = load_dataset()
    columns = groups["SLIC"] + groups["Descriptors"] + groups["Hilbert"]
    values = frame[columns].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, predictions = [], []
    log(f"Nested normalised k-NN: {len(frame):,} images, {n_classes} styles, {values.shape[1]} features")
    for fold, (outer_train, outer_test) in enumerate(splitter.split(values, y), start=1):
        fit_train, validation = train_test_split(
            outer_train, test_size=.10, stratify=y[outer_train], random_state=SEED + fold,
        )
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(values[fit_train])
        validation_x = scaler.transform(values[validation])
        started = time.perf_counter()
        validation_distances, validation_indices = neighbours(fit_x, validation_x, args.workers)
        candidates = []
        for k in K_VALUES:
            for weight_mode in WEIGHTS:
                prediction = predict_from_neighbors(
                    validation_distances, validation_indices, y[fit_train], n_classes, k, weight_mode,
                )
                candidates.append((balanced_accuracy_score(y[validation], prediction), k, weight_mode))
        validation_bacc, selected_k, selected_weights = max(candidates)

        # Refit preprocessing and neighbour index on the entire outer-train
        # partition only after choosing hyperparameters on its validation split.
        outer_scaler = StandardScaler()
        outer_x = outer_scaler.fit_transform(values[outer_train])
        test_x = outer_scaler.transform(values[outer_test])
        test_distances, test_indices = neighbours(outer_x, test_x, args.workers)
        prediction = predict_from_neighbors(
            test_distances, test_indices, y[outer_train], n_classes, selected_k, selected_weights,
        )
        metrics = {
            "bacc": balanced_accuracy_score(y[outer_test], prediction),
            "accuracy": accuracy_score(y[outer_test], prediction),
            "macro_f1": f1_score(y[outer_test], prediction, average="macro"),
        }
        elapsed = time.perf_counter() - started
        rows.append({
            "fold": fold, "n_images": len(frame), "n_train": len(outer_train), "n_test": len(outer_test),
            "n_features": values.shape[1], "selected_k": selected_k, "selected_weights": selected_weights,
            "validation_bacc": validation_bacc, "elapsed_seconds": elapsed, **metrics,
        })
        predictions.extend({
            "fold": fold, "stem": stem, "true_style": encoder.classes_[truth], "predicted_style": encoder.classes_[pred],
        } for stem, truth, pred in zip(frame.iloc[outer_test]["stem"], y[outer_test], prediction, strict=True))
        log(f"fold {fold}/{N_FOLDS}: BAcc={metrics['bacc']:.4f}; val={validation_bacc:.4f}; k={selected_k}, {selected_weights}; {elapsed / 60:.1f} min")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_dir / "fold_results.csv", index=False)
    detail = pd.DataFrame(rows)
    detail.agg({"bacc": ["mean", "std"], "accuracy": "mean", "macro_f1": "mean", "elapsed_seconds": "sum"}).to_csv(args.output_dir / "summary.csv")
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)
    log(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
