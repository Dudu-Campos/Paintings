#!/usr/bin/env python3
"""Paired classifier evaluation of Real-ESRGAN with a bicubic placebo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_big_lama_visual_pilot import (  # noqa: E402
    SLIC_METADATA,
    lda_scores,
    load_npy,
    load_representation,
)
from compare_fusion_classifiers import evaluate_mlp, evaluate_xgb  # noqa: E402


def load_original(slic_dir: Path, feature_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    graph = pd.read_parquet(slic_dir / "combined_features.parquet")
    graph_columns = [column for column in graph.columns if column not in SLIC_METADATA]
    rename = {column: f"original_slic_{column}" for column in graph_columns}
    frame = graph[["stem"] + graph_columns].drop_duplicates("stem").rename(columns=rename)
    columns = list(rename.values())
    descriptors, descriptor_columns = load_npy(
        feature_dir, "desc_full_features.npy", "desc_full_stems.npy", "original_descriptor"
    )
    hilbert, hilbert_columns = load_npy(
        feature_dir, "hilbert_features.npy", "hilbert_stems.npy", "original_hilbert"
    )
    for addition in (descriptors, hilbert):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
    columns.extend(descriptor_columns + hilbert_columns)
    return frame, columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("outputs/super_resolution_pilot/sample.csv"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/super_resolution_pilot/features"))
    parser.add_argument("--original-slic", type=Path, default=Path("outputs/wikiart_slic_features"))
    parser.add_argument("--original-features", type=Path, default=Path("outputs/wikiart_features"))
    parser.add_argument("--output", type=Path, default=Path("paper/super_resolution_pilot_results.csv"))
    parser.add_argument(
        "--classifiers", nargs="+", choices=("LDA", "XGBoost", "MLP"),
        default=("LDA", "XGBoost", "MLP"),
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    original = load_original(args.original_slic, args.original_features)
    representations = {"Original": original}
    for label, folder in (
        ("Normalized 1024", "normalized_1024"),
        ("Bicubic x2", "bicubic_x2"),
        ("Real-ESRGAN x2", "realesrgan_x2"),
    ):
        representations[label] = load_representation(args.feature_root / folder, folder)

    frame = sample.copy()
    columns_by_condition: dict[str, list[str]] = {}
    for name, (addition, columns) in representations.items():
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
        columns_by_condition[name] = columns
    columns_by_condition["Original + Bicubic x2"] = (
        columns_by_condition["Original"] + columns_by_condition["Bicubic x2"]
    )
    columns_by_condition["Original + Real-ESRGAN x2"] = (
        columns_by_condition["Original"] + columns_by_condition["Real-ESRGAN x2"]
    )
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["style"].astype(str))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels))
    evaluators = {
        "LDA": lambda values: lda_scores(values.astype(np.float64), labels, splits),
        "XGBoost": lambda values: evaluate_xgb(values.astype(np.float32), labels, splits),
        "MLP": lambda values: evaluate_mlp(values.astype(np.float32), labels, splits),
    }

    rows = []
    for classifier in args.classifiers:
        for condition, columns in columns_by_condition.items():
            started = time.perf_counter()
            scores = np.asarray(evaluators[classifier](frame[columns].to_numpy(dtype=np.float32)))
            rows.append({
                "classifier": classifier,
                "condition": condition,
                "n_images": len(frame),
                "n_styles": len(encoder.classes_),
                "n_features": len(columns),
                "bacc_mean": float(scores.mean()),
                "bacc_std": float(scores.std()),
                "elapsed_seconds": time.perf_counter() - started,
                **{f"fold_{index + 1}": float(score) for index, score in enumerate(scores)},
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"{classifier:<8} {condition:<28} {scores.mean():.4f} +/- {scores.std():.4f}", flush=True)

    result = pd.DataFrame(rows)
    lookup = result.set_index(["classifier", "condition"])["bacc_mean"]
    result["delta_vs_original_pp"] = [
        100 * (row.bacc_mean - lookup.loc[(row.classifier, "Original")])
        for row in result.itertuples()
    ]
    result["delta_vs_bicubic_pp"] = [
        100 * (row.bacc_mean - lookup.loc[(row.classifier, "Bicubic x2")])
        if row.condition in {"Normalized 1024", "Bicubic x2", "Real-ESRGAN x2"}
        else np.nan
        for row in result.itertuples()
    ]
    result.to_csv(args.output, index=False)
    print(result[[
        "classifier", "condition", "n_features", "bacc_mean", "bacc_std",
        "delta_vs_original_pp", "delta_vs_bicubic_pp",
    ]].to_string(index=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
