#!/usr/bin/env python3
"""Full-cohort visual-only control for the Fourier branch architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_segmentations_pilot import FEATURE_COLUMNS  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402
from slic_branch_mlp_fusion import (  # noqa: E402
    BranchFusionMLP,
    DEVICE,
    parameter_count,
    train_and_evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/fourier_full_branch_visual_baseline.csv"


def main() -> None:
    sample = pd.read_csv(ROOT / "results/line_graph_full_sample.csv")[["stem", "style"]]
    sample = sample.drop_duplicates("stem")
    visual, visual_columns = load_original(
        ROOT / "outputs/wikiart_slic_features", ROOT / "outputs/wikiart_features"
    )
    line_raw = pd.read_parquet(ROOT / "results/line_graph_full_features.parquet")
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    felz_raw = pd.read_parquet(
        ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet"
    )
    felz_rename = {column: f"felz_{column}" for column in FEATURE_COLUMNS}
    felz = (
        felz_raw[["stem"] + FEATURE_COLUMNS]
        .drop_duplicates("stem")
        .rename(columns=felz_rename)
    )
    columns = visual_columns + list(line_rename.values()) + list(felz_rename.values())
    frame = sample.merge(visual, on="stem", validate="one_to_one")
    frame = frame.merge(line, on="stem", validate="one_to_one")
    frame = frame.merge(felz, on="stem", validate="one_to_one")
    frame = frame.sort_values("stem").reset_index(drop=True)
    if len(frame) != 79_207 or len(columns) != 392:
        raise ValueError(f"unexpected cohort/features: {len(frame)}/{len(columns)}")

    values = frame[columns].to_numpy(dtype=np.float32)
    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=42 + fold,
        )
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[fit_train]).astype(np.float32)
        validation_values = scaler.transform(values[validation]).astype(np.float32)
        test_values = scaler.transform(values[outer_test]).astype(np.float32)
        score, best_epoch, elapsed = train_and_evaluate(
            lambda: BranchFusionMLP([len(columns)], len(np.unique(labels))),
            train_values,
            labels[fit_train],
            validation_values,
            labels[validation],
            test_values,
            labels[outer_test],
            len(np.unique(labels)),
            42 + fold,
        )
        rows.append({
            "fold": fold,
            "bacc": score,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed,
        })
        pd.DataFrame(rows).to_csv(OUTPUT.with_name(OUTPUT.stem + "_folds.csv"), index=False)
        print(f"fold={fold}/5 bacc={score:.4f} epoch={best_epoch}", flush=True)
    scores = np.asarray([row["bacc"] for row in rows])
    summary = pd.DataFrame([{
        "classifier": "BranchFusionMLP",
        "condition": "Branch visual 392",
        "n_images": len(frame),
        "n_styles": len(np.unique(labels)),
        "n_features": len(columns),
        "parameters": parameter_count(BranchFusionMLP([len(columns)], len(np.unique(labels)))),
        "bacc_mean": float(scores.mean()),
        "bacc_std": float(scores.std()),
        **{f"fold_{fold}": float(score) for fold, score in enumerate(scores, 1)},
    }])
    summary.to_csv(OUTPUT, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"saved={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
