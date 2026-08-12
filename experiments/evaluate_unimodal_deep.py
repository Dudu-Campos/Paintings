"""Measure the non-linear style signal in each existing visual transformation.

This is intentionally a *unimodal* probe: SLIC graph summaries, local
descriptors, and Hilbert features each receive the same deep MLP separately.
It answers which transformed modalities deserve a learned embedding encoder
before any contrastive or multimodal-fusion objective is introduced.

The split protocol is the canonical 5-fold stratified WikiArt protocol with a
validation slice inside each outer training fold.  ``--only-fold`` is useful
for a cheap pilot, but its number must not be reported as a final CV result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_robustness_mlp_aug import train_one
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/unimodal_deep_probe",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--only-fold", type=int, choices=range(1, N_FOLDS + 1),
        help="Run one outer fold only as a pilot; omit for final 5-fold CV.",
    )
    args = parser.parse_args()

    frame, groups = load_dataset()
    modalities = {
        "SLIC": groups["SLIC"],
        "Descritores locais": groups["Descriptors"],
        "Hilbert": groups["Hilbert"],
        "Visual completo (SLIC + descritores + Hilbert)": (
            groups["SLIC"] + groups["Descriptors"] + groups["Hilbert"]
        ),
    }
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict[str, object]] = []
    log(f"Deep unimodal probe: {len(frame):,} images, {len(encoder.classes_)} styles")
    if args.only_fold:
        log(f"Pilot only: outer fold {args.only_fold}/{N_FOLDS}; not a final CV estimate")

    for fold, (outer_train, outer_test) in enumerate(splitter.split(frame, y), start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold,
        )
        for modality, columns in modalities.items():
            values = frame[columns].to_numpy(dtype=np.float32)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(values[fit_train]).astype(np.float32)
            validation_x = scaler.transform(values[validation]).astype(np.float32)
            test_x = scaler.transform(values[outer_test]).astype(np.float32)
            started = time.perf_counter()
            metrics, _ = train_one(
                train_x, y[fit_train], validation_x, y[validation], test_x, y[outer_test],
                [len(columns)], len(encoder.classes_), SEED + fold,
                args.epochs, args.patience,
            )
            elapsed = time.perf_counter() - started
            rows.append({
                "fold": fold, "modality": modality, "feature_dimensions": len(columns),
                "n_images": len(frame), "n_fit_train": len(fit_train),
                "n_validation": len(validation), "n_test": len(outer_test),
                "elapsed_seconds": elapsed, **metrics,
            })
            log(
                f"fold {fold}/{N_FOLDS} {modality}: BAcc={metrics['bacc']:.4f}, "
                f"val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}"
            )

    detail = pd.DataFrame(rows)
    summary = detail.groupby("modality", as_index=False).agg(
        feature_dimensions=("feature_dimensions", "first"),
        folds_completed=("fold", "nunique"),
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"),
        best_epoch_mean=("best_epoch", "mean"), elapsed_seconds=("elapsed_seconds", "sum"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"Saved to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
