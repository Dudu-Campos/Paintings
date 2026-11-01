#!/usr/bin/env python3
"""Evaluate line-graph boundary features on the fixed 3k/27-style cohort."""

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
from model_evaluation import evaluate_mlp, evaluate_xgb  # noqa: E402
from evaluate_big_lama_visual_pilot import lda_scores  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402


def nonredundant_columns(
    frame: pd.DataFrame, candidates: list[str], reference: list[str], threshold: float = 0.90,
) -> list[str]:
    left = frame[candidates].to_numpy(dtype=np.float64)
    right = frame[reference].to_numpy(dtype=np.float64)
    left = (left - left.mean(axis=0)) / np.where(left.std(axis=0) > 0, left.std(axis=0), 1)
    right = (right - right.mean(axis=0)) / np.where(right.std(axis=0) > 0, right.std(axis=0), 1)
    maximum_r2 = np.max((left.T @ right / len(frame)) ** 2, axis=1)
    return [column for column, value in zip(candidates, maximum_r2, strict=True) if value < threshold]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/super_resolution_3k_sample.csv"))
    parser.add_argument("--line-features", type=Path, default=Path("paper/line_graph_3k_features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("paper/line_graph_3k_results.csv"))
    parser.add_argument(
        "--classifiers", nargs="+", choices=("LDA", "XGBoost", "MLP"),
        default=("LDA", "XGBoost", "MLP"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    visual, visual_columns = load_original(
        Path("outputs/wikiart_slic_features"), Path("outputs/wikiart_features")
    )
    slic_columns = [column for column in visual_columns if "_slic_" in column]
    if len(slic_columns) != 71 or len(visual_columns) != 258:
        raise ValueError(f"unexpected canonical dimensions: SLIC={len(slic_columns)}, visual={len(visual_columns)}")
    line = pd.read_parquet(args.line_features)
    line_columns_raw = [column for column in line if column not in {"stem", "style"}]
    rename = {column: f"line_{column}" for column in line_columns_raw}
    line = line[["stem"] + line_columns_raw].rename(columns=rename)
    line_columns = list(rename.values())

    frame = sample.merge(visual, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(line, on="stem", how="inner", validate="one_to_one")
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(frame))
    shuffled_columns = []
    for column in line_columns:
        shuffled = f"shuffled_{column}"
        frame[shuffled] = frame[column].to_numpy()[permutation]
        shuffled_columns.append(shuffled)

    unique_for_slic = nonredundant_columns(frame, line_columns, slic_columns)
    unique_for_visual = nonredundant_columns(frame, line_columns, visual_columns)
    unique_slic_shuffled = [f"shuffled_{column}" for column in unique_for_slic]
    unique_visual_shuffled = [f"shuffled_{column}" for column in unique_for_visual]
    print(
        f"Nonredundant line features at r2<0.90: vs SLIC={len(unique_for_slic)}, "
        f"vs visual={len(unique_for_visual)}",
        flush=True,
    )

    conditions = {
        "Line graph 63": line_columns,
        "SLIC 71": slic_columns,
        "SLIC 71 + line graph": slic_columns + line_columns,
        "SLIC 71 + shuffled line": slic_columns + shuffled_columns,
        "SLIC 71 + unique line": slic_columns + unique_for_slic,
        "SLIC 71 + shuffled unique": slic_columns + unique_slic_shuffled,
        "Visual 258": visual_columns,
        "Visual 258 + line graph": visual_columns + line_columns,
        "Visual 258 + shuffled line": visual_columns + shuffled_columns,
        "Visual 258 + unique line": visual_columns + unique_for_visual,
        "Visual 258 + shuffled unique": visual_columns + unique_visual_shuffled,
    }
    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(np.zeros(len(labels)), labels)
    )
    evaluators = {
        "LDA": lambda values: lda_scores(values.astype(np.float64), labels, splits),
        "XGBoost": lambda values: evaluate_xgb(values.astype(np.float32), labels, splits),
        "MLP": lambda values: evaluate_mlp(values.astype(np.float32), labels, splits),
    }

    rows = []
    for classifier in args.classifiers:
        for condition, columns in conditions.items():
            started = time.perf_counter()
            values = frame[columns].to_numpy(dtype=np.float32)
            scores = np.asarray(evaluators[classifier](values))
            rows.append({
                "classifier": classifier,
                "condition": condition,
                "n_images": len(frame),
                "n_styles": len(np.unique(labels)),
                "n_features": len(columns),
                "bacc_mean": float(scores.mean()),
                "bacc_std": float(scores.std()),
                "elapsed_seconds": time.perf_counter() - started,
                **{f"fold_{index + 1}": float(score) for index, score in enumerate(scores)},
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"{classifier:<8} {condition:<29} {scores.mean():.4f} +/- {scores.std():.4f}", flush=True)

    result = pd.DataFrame(rows)
    lookup = result.set_index(["classifier", "condition"])
    comparisons = []
    for classifier in args.classifiers:
        for scope, baseline, real, placebo in (
            ("SLIC", "SLIC 71", "SLIC 71 + line graph", "SLIC 71 + shuffled line"),
            ("SLIC unique", "SLIC 71", "SLIC 71 + unique line", "SLIC 71 + shuffled unique"),
            ("Visual", "Visual 258", "Visual 258 + line graph", "Visual 258 + shuffled line"),
            ("Visual unique", "Visual 258", "Visual 258 + unique line", "Visual 258 + shuffled unique"),
        ):
            base = lookup.loc[(classifier, baseline)]
            line_row = lookup.loc[(classifier, real)]
            placebo_row = lookup.loc[(classifier, placebo)]
            base_scores = np.array([base[f"fold_{i}"] for i in range(1, 6)])
            line_scores = np.array([line_row[f"fold_{i}"] for i in range(1, 6)])
            placebo_scores = np.array([placebo_row[f"fold_{i}"] for i in range(1, 6)])
            comparisons.append({
                "classifier": classifier, "scope": scope,
                "delta_line_vs_baseline_pp": 100 * float((line_scores - base_scores).mean()),
                "p_line_vs_baseline": float(ttest_rel(line_scores, base_scores).pvalue),
                "delta_line_vs_placebo_pp": 100 * float((line_scores - placebo_scores).mean()),
                "p_line_vs_placebo": float(ttest_rel(line_scores, placebo_scores).pvalue),
                **{f"line_minus_baseline_fold_{i}_pp": 100 * float(line_scores[i - 1] - base_scores[i - 1])
                   for i in range(1, 6)},
            })
    comparison_path = args.output.with_name(args.output.stem + "_comparisons.csv")
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"saved={args.output} comparisons={comparison_path}")


if __name__ == "__main__":
    main()
