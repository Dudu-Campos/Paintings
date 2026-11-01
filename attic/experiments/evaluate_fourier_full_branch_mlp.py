#!/usr/bin/env python3
"""Full-cohort BranchFusionMLP evaluation of compact Fourier features."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_segmentations_pilot import FEATURE_COLUMNS  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402
from extract_fourier_spectrum_features import FEATURE_NAMES  # noqa: E402
from slic_branch_mlp_fusion import (  # noqa: E402
    BranchFusionMLP,
    DEVICE,
    parameter_count,
    train_and_evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONDITION = "Visual 258 + line + Felzenszwalb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "results/line_graph_full_sample.csv"
    )
    parser.add_argument(
        "--fourier-features", type=Path,
        default=ROOT / "results/fourier_full_features.parquet",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "results/felzenszwalb_full_mlp_final_results.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/fourier_full_branch_mlp_results.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    visual, visual_columns = load_original(
        ROOT / "outputs/wikiart_slic_features", ROOT / "outputs/wikiart_features"
    )
    line_raw = pd.read_parquet(ROOT / "results/line_graph_full_features.parquet")
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    line_columns = list(line_rename.values())
    felz_raw = pd.read_parquet(
        ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet"
    )
    felz_rename = {column: f"felz_{column}" for column in FEATURE_COLUMNS}
    felz = (
        felz_raw[["stem"] + FEATURE_COLUMNS]
        .drop_duplicates("stem")
        .rename(columns=felz_rename)
    )
    felz_columns = list(felz_rename.values())
    fourier = pd.read_parquet(args.fourier_features).drop(columns="style")

    frame = sample.merge(visual, on="stem", validate="one_to_one")
    frame = frame.merge(line, on="stem", validate="one_to_one")
    frame = frame.merge(felz, on="stem", validate="one_to_one")
    frame = frame.merge(fourier, on="stem", validate="one_to_one")
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)
    base_columns = visual_columns + line_columns + felz_columns
    fourier_columns = list(FEATURE_NAMES)
    base_values = frame[base_columns].to_numpy(dtype=np.float32)
    fourier_values = frame[fourier_columns].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    shuffled_fourier = fourier_values[rng.permutation(len(frame))]

    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    n_classes = len(np.unique(labels))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    baseline = pd.read_csv(args.baseline).iloc[0]
    if baseline["condition"] != BASELINE_CONDITION or int(baseline["n_images"]) != len(frame):
        raise ValueError("cached baseline is not the paired 392-feature result")
    baseline_scores = np.asarray(
        [baseline[f"fold_{fold}"] for fold in range(1, 6)], dtype=np.float64
    )
    branch_sizes = [len(base_columns), len(fourier_columns)]
    conditions = {
        "Branch visual + shuffled Fourier": np.concatenate(
            (base_values, shuffled_fourier), axis=1
        ),
        "Branch visual + Fourier": np.concatenate((base_values, fourier_values), axis=1),
    }
    print(
        f"cohort={len(frame):,}; base={len(base_columns)}; Fourier={len(fourier_columns)}; "
        f"cached_baseline={baseline_scores.mean():.4f}; device={DEVICE}",
        flush=True,
    )

    rows = []
    fold_path = args.output.with_name(args.output.stem + "_folds.csv")
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=args.seed + fold,
        )
        for condition, values in conditions.items():
            scaler = StandardScaler()
            train_values = scaler.fit_transform(values[fit_train]).astype(np.float32)
            validation_values = scaler.transform(values[validation]).astype(np.float32)
            test_values = scaler.transform(values[outer_test]).astype(np.float32)
            score, best_epoch, elapsed = train_and_evaluate(
                lambda: BranchFusionMLP(branch_sizes, n_classes),
                train_values,
                labels[fit_train],
                validation_values,
                labels[validation],
                test_values,
                labels[outer_test],
                n_classes,
                args.seed + fold,
            )
            rows.append({
                "fold": fold,
                "condition": condition,
                "bacc": score,
                "best_epoch": best_epoch,
                "elapsed_seconds": elapsed,
            })
            pd.DataFrame(rows).to_csv(fold_path, index=False)
            print(
                f"fold={fold}/5 {condition:<34} bacc={score:.4f} epoch={best_epoch}",
                flush=True,
            )

    folds = pd.DataFrame(rows)
    summaries = [{
        "classifier": "MLP",
        "condition": BASELINE_CONDITION,
        "n_images": len(frame),
        "n_styles": n_classes,
        "n_features": len(base_columns),
        "parameters": np.nan,
        "bacc_mean": float(baseline_scores.mean()),
        "bacc_std": float(baseline_scores.std()),
        **{f"fold_{fold}": float(score) for fold, score in enumerate(baseline_scores, 1)},
    }]
    for condition, group in folds.groupby("condition", sort=False):
        scores = group.sort_values("fold")["bacc"].to_numpy()
        summaries.append({
            "classifier": "BranchFusionMLP",
            "condition": condition,
            "n_images": len(frame),
            "n_styles": n_classes,
            "n_features": len(base_columns) + len(fourier_columns),
            "parameters": parameter_count(BranchFusionMLP(branch_sizes, n_classes)),
            "bacc_mean": float(scores.mean()),
            "bacc_std": float(scores.std()),
            **{f"fold_{fold}": float(score) for fold, score in enumerate(scores, 1)},
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output, index=False)
    lookup = summary.set_index("condition")
    real = lookup.loc["Branch visual + Fourier", [f"fold_{fold}" for fold in range(1, 6)]].to_numpy(float)
    comparisons = []
    for name, reference in (
        ("vs cached flat baseline", baseline_scores),
        (
            "vs shuffled branch",
            lookup.loc[
                "Branch visual + shuffled Fourier",
                [f"fold_{fold}" for fold in range(1, 6)],
            ].to_numpy(float),
        ),
    ):
        delta = 100 * (real - reference)
        comparisons.append({
            "comparison": name,
            "delta_bacc_pp": float(delta.mean()),
            "p_paired": float(ttest_rel(real, reference).pvalue),
            "positive_folds": int(np.sum(delta > 0)),
            **{f"delta_fold_{fold}_pp": float(value) for fold, value in enumerate(delta, 1)},
        })
    comparison_path = args.output.with_name(args.output.stem + "_comparisons.csv")
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(comparison_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"saved={args.output} folds={fold_path} comparisons={comparison_path}", flush=True)


if __name__ == "__main__":
    main()
