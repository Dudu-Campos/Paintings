"""Paired MLP evaluation of frozen VGG pooled feature controls."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_robustness_mlp_aug import train_one
from fusion_curve_slic import ROOT, SEED


FEATURES = {
    "VGG-19 conv_5 GAP + MLP": "conv5_gap_features.npy",
    "VGG-19 final GAP + MLP": "final_gap_features.npy",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "outputs/vgg_pool_features")
    parser.add_argument(
        "--cohort-predictions", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/vgg_pool_mlp")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    stems = np.load(args.features_dir / "stems.npy", allow_pickle=True).astype(str)
    predictions = pd.read_csv(args.cohort_predictions, dtype=str)
    cohort = predictions.loc[predictions.condition.eq("Visual GIST"), ["stem", "true_style"]]
    cohort = cohort.drop_duplicates("stem").set_index("stem")
    encoder = LabelEncoder().fit(cohort.true_style.to_numpy())
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for condition, filename in FEATURES.items():
        values = np.load(args.features_dir / filename, mmap_mode="r")
        features = pd.DataFrame(values, index=pd.Index(stems, name="stem"))
        if features.index.duplicated().any():
            raise ValueError("Duplicate VGG feature stems")
        frame = cohort.join(features, how="inner", validate="one_to_one").sort_index()
        if len(frame) != len(cohort):
            raise ValueError(f"Feature/cohort mismatch: {len(frame):,} versus {len(cohort):,}")
        labels = encoder.transform(frame.true_style.to_numpy())
        x = frame.drop(columns="true_style").to_numpy(dtype=np.float32)
        splits = list(splitter.split(np.zeros(len(labels)), labels))
        log(f"{condition}: {len(frame):,} images, {x.shape[1]} features")
        for fold, (outer_train, outer_test) in enumerate(splits, 1):
            fit, validation = train_test_split(
                outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
            )
            scaler = StandardScaler()
            train_x = scaler.fit_transform(x[fit]).astype(np.float32)
            validation_x = scaler.transform(x[validation]).astype(np.float32)
            test_x = scaler.transform(x[outer_test]).astype(np.float32)
            started = time.perf_counter()
            metrics, _ = train_one(
                train_x, labels[fit], validation_x, labels[validation], test_x, labels[outer_test],
                [x.shape[1]], len(encoder.classes_), SEED + fold, args.epochs, args.patience,
            )
            rows.append({"condition": condition, "fold": fold, "n_images": len(frame),
                         "elapsed_seconds": time.perf_counter() - started, **metrics})
            log(f"{condition} fold {fold}/5: BAcc={metrics['bacc']:.4f}, epoch={metrics['best_epoch']}")
    detail = pd.DataFrame(rows)
    summary = detail.groupby("condition", as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
