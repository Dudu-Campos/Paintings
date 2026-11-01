#!/usr/bin/env python3
"""Evaluate compact Fourier-spectrum features on the paired 12k cohort."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_segmentations_pilot import FEATURE_COLUMNS  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402
from extract_fourier_spectrum_features import FEATURE_NAMES  # noqa: E402
from model_evaluation import DEVICE, evaluate_mlp  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def fold_scores(row: pd.Series) -> np.ndarray:
    return np.asarray([row[f"fold_{fold}"] for fold in range(1, 6)], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "results/fourier_12k_sample.csv"
    )
    parser.add_argument(
        "--fourier-features", type=Path,
        default=ROOT / "results/fourier_12k_features.parquet",
    )
    parser.add_argument(
        "--line-features", type=Path,
        default=ROOT / "results/line_graph_full_features.parquet",
    )
    parser.add_argument(
        "--felz-features", type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/fourier_12k_mlp_results.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    visual, visual_columns = load_original(
        ROOT / "outputs/wikiart_slic_features", ROOT / "outputs/wikiart_features"
    )
    line_raw = pd.read_parquet(args.line_features)
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    line_columns = list(line_rename.values())
    felz_raw = pd.read_parquet(args.felz_features)
    felz_rename = {column: f"felz_{column}" for column in FEATURE_COLUMNS}
    felz = (
        felz_raw[["stem"] + FEATURE_COLUMNS]
        .drop_duplicates("stem")
        .rename(columns=felz_rename)
    )
    felz_columns = list(felz_rename.values())
    fourier = pd.read_parquet(args.fourier_features)
    if len(visual_columns) != 258 or len(line_columns) != 63 or len(felz_columns) != 71:
        raise ValueError("unexpected canonical feature dimensions")
    if tuple(column for column in fourier if column not in {"stem", "style"}) != FEATURE_NAMES:
        raise ValueError("unexpected Fourier feature schema")

    frame = sample.merge(visual, on="stem", validate="one_to_one")
    frame = frame.merge(line, on="stem", validate="one_to_one")
    frame = frame.merge(felz, on="stem", validate="one_to_one")
    frame = frame.merge(
        fourier.drop(columns="style"), on="stem", validate="one_to_one"
    )
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)
    base_columns = visual_columns + line_columns + felz_columns
    fourier_columns = list(FEATURE_NAMES)
    rng = np.random.default_rng(args.seed)
    shuffled_columns = [f"shuffled_{column}" for column in fourier_columns]
    shuffled = frame[fourier_columns].iloc[rng.permutation(len(frame))].reset_index(drop=True)
    shuffled.columns = shuffled_columns
    frame = pd.concat((frame, shuffled), axis=1)

    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    conditions = {
        "Visual 392": base_columns,
        "Fourier 152": fourier_columns,
        "Visual 392 + shuffled Fourier": base_columns + shuffled_columns,
        "Visual 392 + Fourier": base_columns + fourier_columns,
    }
    print(
        f"cohort={len(frame):,}; styles={len(np.unique(labels))}; "
        f"base={len(base_columns)}; Fourier={len(fourier_columns)}; device={DEVICE}",
        flush=True,
    )

    rows = []
    for condition, columns in conditions.items():
        started = time.perf_counter()
        scores = np.asarray(
            evaluate_mlp(frame[columns].to_numpy(dtype=np.float32), labels, splits),
            dtype=np.float64,
        )
        rows.append({
            "classifier": "MLP",
            "condition": condition,
            "n_images": len(frame),
            "n_styles": len(np.unique(labels)),
            "n_features": len(columns),
            "bacc_mean": float(scores.mean()),
            "bacc_std": float(scores.std()),
            "elapsed_seconds": time.perf_counter() - started,
            **{f"fold_{fold}": float(score) for fold, score in enumerate(scores, 1)},
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(
            f"{condition:<32} {scores.mean():.4f} +/- {scores.std():.4f} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    result = pd.DataFrame(rows).set_index("condition")
    base = fold_scores(result.loc["Visual 392"])
    placebo = fold_scores(result.loc["Visual 392 + shuffled Fourier"])
    real = fold_scores(result.loc["Visual 392 + Fourier"])
    comparisons = []
    for comparison, reference in (("vs baseline", base), ("vs shuffled", placebo)):
        delta = 100 * (real - reference)
        comparisons.append({
            "comparison": comparison,
            "delta_bacc_pp": float(delta.mean()),
            "p_paired": float(ttest_rel(real, reference).pvalue),
            "positive_folds": int(np.sum(delta > 0)),
            **{f"delta_fold_{fold}_pp": float(value) for fold, value in enumerate(delta, 1)},
        })
    comparison_path = args.output.with_name(args.output.stem + "_comparisons.csv")
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(comparison_path, index=False)
    print(comparison_frame.to_string(index=False), flush=True)
    print(f"saved={args.output} comparisons={comparison_path}", flush=True)


if __name__ == "__main__":
    main()
