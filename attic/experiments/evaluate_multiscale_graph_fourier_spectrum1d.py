#!/usr/bin/env python3
"""Evaluate Hilbert 1D spectra on top of the completed Fourier visual model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_multiscale_graph_fourier import BASELINE as PRE_FOURIER_BASELINE
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from evaluate_robustness_mlp_aug import train_one
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


BASELINE = PRE_FOURIER_BASELINE + " + Fourier"
CONDITION = BASELINE + " + Hilbert 1D spectrum"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_spectrum(frame: pd.DataFrame, directory: Path) -> tuple[pd.DataFrame, list[str]]:
    values = np.load(directory / "features.npy")
    stems = np.load(directory / "stems.npy", allow_pickle=True).astype(str)
    if values.ndim != 2 or values.shape[1] != 108 or len(values) != len(stems):
        raise ValueError("Expected aligned 1D spectral features with shape (N, 108)")
    columns = [f"hilbert_spectrum1d_{index:03d}" for index in range(values.shape[1])]
    spectrum = pd.DataFrame(values, columns=columns)
    spectrum.insert(0, "stem", stems)
    if spectrum.stem.duplicated().any() or not np.isfinite(values).all():
        raise ValueError("Invalid Hilbert spectrum features")
    expected_stems = frame.stem.to_numpy(dtype=str)
    result = frame.merge(spectrum, on="stem", how="inner", validate="one_to_one")
    result = result.sort_values("stem").reset_index(drop=True)
    if len(result) != len(frame) or not np.array_equal(result.stem.to_numpy(str), expected_stems):
        raise ValueError(f"Spectrum merge changed canonical cohort: {len(result)}/{len(frame)}")
    return result, columns


def save(rows: list[dict], output_dir: Path) -> None:
    details = pd.DataFrame(rows).sort_values("fold")
    details.to_csv(output_dir / "fold_results.csv", index=False)
    baseline = pd.read_csv(output_dir.parent / "multiscale_graph_fourier" / "fold_results.csv")
    baseline = baseline.loc[baseline.condition.eq(BASELINE)].sort_values("fold")
    candidate = details.sort_values("fold")
    if len(baseline) == len(candidate) == N_FOLDS:
        delta = 100 * (candidate.bacc.to_numpy() - baseline.bacc.to_numpy())
        pd.DataFrame([{
            "baseline": BASELINE, "condition": CONDITION, "paired_folds": len(delta),
            "delta_bacc_pp": float(delta.mean()), "positive_folds": int((delta > 0).sum()),
            "paired_t": float(ttest_rel(candidate.bacc, baseline.bacc).statistic),
            "paired_p": float(ttest_rel(candidate.bacc, baseline.bacc).pvalue),
            **{f"delta_fold_{fold}_pp": float(value) for fold, value in enumerate(delta, 1)},
        }]).to_csv(output_dir / "comparison.csv", index=False)
    summary = pd.DataFrame([{
        "condition": CONDITION, "folds_completed": details.fold.nunique(),
        "n_images": int(details.n_images.iloc[0]), "dimensions": int(details.dimensions.iloc[0]),
        "branches": int(details.branches.iloc[0]), "bacc_mean": details.bacc.mean(),
        "bacc_std": details.bacc.std(), "accuracy_mean": details.accuracy.mean(),
        "macro_f1_mean": details.macro_f1.mean(),
        "validation_bacc_mean": details.validation_bacc.mean(), "best_epoch_mean": details.best_epoch.mean(),
        "elapsed_seconds": details.elapsed_seconds.sum(),
    }])
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-features", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--spectrum-dir", type=Path, default=ROOT / "outputs/hilbert_spectrum_1d_full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/multiscale_graph_fourier_spectrum1d")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, spectrum_columns = load_spectrum(frame, args.spectrum_dir)
    visual_columns = [column for columns in groups.values() for column in columns]
    blocks = [
        frame[visual_columns].to_numpy(dtype=np.float32), frame[line_columns].to_numpy(dtype=np.float32),
        frame[felz_columns].to_numpy(dtype=np.float32), frame[fourier_columns].to_numpy(dtype=np.float32),
        frame[spectrum_columns].to_numpy(dtype=np.float32),
    ]
    branch_sizes = [len(columns) for columns in groups.values()] + [len(line_columns), len(felz_columns), len(fourier_columns), len(spectrum_columns)]
    values = np.hstack(blocks)
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(str))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    completed = {int(row["fold"]) for row in rows}
    log(
        f"1D spectrum fusion: {len(frame):,} images, {len(np.unique(labels))} styles; "
        f"dimensions={values.shape[1]}, branches={len(branch_sizes)} (spectrum={len(spectrum_columns)})"
    )
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        if fold in completed:
            log(f"fold {fold}/{N_FOLDS}: already complete")
            continue
        fit_train, validation = train_test_split(outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(values[fit_train]).astype(np.float32)
        validation_x = scaler.transform(values[validation]).astype(np.float32)
        test_x = scaler.transform(values[outer_test]).astype(np.float32)
        started = time.perf_counter()
        metrics, _ = train_one(
            train_x, labels[fit_train], validation_x, labels[validation], test_x, labels[outer_test],
            branch_sizes, len(np.unique(labels)), SEED + fold, args.epochs, args.patience,
        )
        rows.append({
            "fold": fold, "condition": CONDITION, "n_images": len(frame),
            "n_fit_train": len(fit_train), "n_validation": len(validation), "n_test": len(outer_test),
            "dimensions": values.shape[1], "branches": len(branch_sizes),
            "elapsed_seconds": time.perf_counter() - started, **metrics,
        })
        save(rows, args.output_dir)
        log(f"fold {fold}/{N_FOLDS}: BAcc={metrics['bacc']:.4f}; val={metrics['validation_bacc']:.4f}")
    if rows:
        save(rows, args.output_dir)


if __name__ == "__main__":
    main()
