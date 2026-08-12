"""Evaluate the compact Gatys-style Gram representation with the project MLP."""

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


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "outputs/gatys_compact_gram_features")
    parser.add_argument(
        "--cohort-predictions", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gatys_compact_gram_mlp")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    values = np.load(args.features_dir / "features.npy", mmap_mode="r")
    stems = np.load(args.features_dir / "stems.npy", allow_pickle=True).astype(str)
    features = pd.DataFrame(values, index=pd.Index(stems, name="stem"))
    if features.index.duplicated().any():
        raise ValueError("Duplicate compact Gram stems")
    predictions = pd.read_csv(args.cohort_predictions, dtype=str)
    cohort = predictions.loc[predictions.condition.eq("Visual GIST"), ["stem", "true_style"]]
    cohort = cohort.drop_duplicates("stem").set_index("stem")
    frame = cohort.join(features, how="inner", validate="one_to_one").sort_index()
    if len(frame) != len(cohort):
        raise ValueError(f"Feature/cohort mismatch: {len(frame):,} versus {len(cohort):,}")
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame.true_style.to_numpy())
    x = frame.drop(columns="true_style").to_numpy(dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    log(f"Compact Gatys Gram MLP: {len(frame):,} images, {x.shape[1]:,} features")
    for fold, (outer_train, outer_test) in enumerate(splitter.split(x, y), 1):
        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold
        )
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[fit]).astype(np.float32)
        validation_x = scaler.transform(x[validation]).astype(np.float32)
        test_x = scaler.transform(x[outer_test]).astype(np.float32)
        started = time.perf_counter()
        metrics, _ = train_one(
            train_x, y[fit], validation_x, y[validation], test_x, y[outer_test],
            [x.shape[1]], len(encoder.classes_), SEED + fold, args.epochs, args.patience,
        )
        rows.append({"fold": fold, "n_images": len(frame), "elapsed_seconds": time.perf_counter() - started,
                     **metrics})
        log(f"fold {fold}/5: BAcc={metrics['bacc']:.4f}, val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}")
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "method": "Compact Gatys Gram + MLP", "n_images": len(frame),
        "bacc_mean": detail.bacc.mean(), "bacc_std": detail.bacc.std(),
        "accuracy_mean": detail.accuracy.mean(), "macro_f1_mean": detail.macro_f1.mean(),
        "elapsed_seconds": detail.elapsed_seconds.sum(),
    }])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
