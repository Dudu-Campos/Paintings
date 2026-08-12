#!/usr/bin/env python3
"""Locate the Real-ESRGAN pilot effect in SLIC, descriptor, or Hilbert features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fusion_classifiers import evaluate_mlp, evaluate_xgb  # noqa: E402
from evaluate_big_lama_visual_pilot import lda_scores, load_representation  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402


def branch(columns: list[str], name: str) -> list[str]:
    tokens = {"SLIC": "_slic_", "Descriptors": "_descriptor_", "Hilbert": "_hilbert_"}
    return [column for column in columns if tokens[name] in column]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/super_resolution_pilot"))
    parser.add_argument(
        "--output", type=Path, default=Path("paper/super_resolution_pilot_branch_results.csv")
    )
    args = parser.parse_args()
    root = args.root
    sample = pd.read_csv(root / "sample.csv")[["stem", "style"]]
    original, original_columns = load_original(
        Path("outputs/wikiart_slic_features"), Path("outputs/wikiart_features")
    )
    bicubic, bicubic_columns = load_representation(root / "features/bicubic_x2", "bicubic_x2")
    sr, sr_columns = load_representation(root / "features/realesrgan_x2", "realesrgan_x2")
    frame = sample.merge(original, on="stem", validate="one_to_one")
    frame = frame.merge(bicubic, on="stem", validate="one_to_one")
    frame = frame.merge(sr, on="stem", validate="one_to_one").sort_values("stem").reset_index(drop=True)
    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels))
    evaluators = {
        "LDA": lambda x: lda_scores(x.astype(np.float64), labels, splits),
        "XGBoost": lambda x: evaluate_xgb(x.astype(np.float32), labels, splits),
        "MLP": lambda x: evaluate_mlp(x.astype(np.float32), labels, splits),
    }
    rows = []
    for branch_name in ("SLIC", "Descriptors", "Hilbert"):
        o = branch(original_columns, branch_name)
        b = branch(bicubic_columns, branch_name)
        s = branch(sr_columns, branch_name)
        conditions = {
            "Bicubic x2": b,
            "Real-ESRGAN x2": s,
            "Original + Bicubic x2": o + b,
            "Original + Real-ESRGAN x2": o + s,
        }
        for classifier, evaluator in evaluators.items():
            for condition, columns in conditions.items():
                scores = np.asarray(evaluator(frame[columns].to_numpy(dtype=np.float32)))
                rows.append({
                    "branch": branch_name,
                    "classifier": classifier,
                    "condition": condition,
                    "n_features": len(columns),
                    "bacc_mean": scores.mean(),
                    "bacc_std": scores.std(),
                    **{f"fold_{i + 1}": score for i, score in enumerate(scores)},
                })
                print(
                    f"{branch_name:<12} {classifier:<8} {condition:<28} "
                    f"{scores.mean():.4f} +/- {scores.std():.4f}", flush=True,
                )
    result = pd.DataFrame(rows)
    lookup = result.set_index(["branch", "classifier", "condition"])["bacc_mean"]
    result["delta_vs_matched_bicubic_pp"] = [
        100 * (
            row.bacc_mean
            - lookup.loc[(
                row.branch,
                row.classifier,
                "Original + Bicubic x2" if row.condition.startswith("Original +") else "Bicubic x2",
            )]
        )
        for row in result.itertuples()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
