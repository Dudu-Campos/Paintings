#!/usr/bin/env python3
"""Evaluate strict distance-2 graph features on the fixed 3k cohort."""

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
from compare_fusion_classifiers import evaluate_mlp, evaluate_xgb  # noqa: E402
from evaluate_big_lama_visual_pilot import lda_scores  # noqa: E402
from evaluate_line_graph_3k import nonredundant_columns  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/super_resolution_3k_sample.csv"))
    parser.add_argument(
        "--square-features", type=Path,
        default=Path("paper/square_graph_3k_features.parquet"),
    )
    parser.add_argument(
        "--line-features", type=Path,
        default=Path("paper/line_graph_3k_features.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/square_graph_3k_results.csv"))
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
        raise ValueError(
            f"unexpected canonical dimensions: SLIC={len(slic_columns)}, visual={len(visual_columns)}"
        )

    square = pd.read_parquet(args.square_features)
    square_columns = [column for column in square if column not in {"stem", "style"}]
    line = pd.read_parquet(args.line_features)
    line_columns = [column for column in line if column not in {"stem", "style"}]
    frame = sample.merge(visual, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(
        line[["stem"] + line_columns], on="stem", how="inner", validate="one_to_one"
    )
    frame = frame.merge(
        square[["stem"] + square_columns], on="stem", how="inner", validate="one_to_one"
    )
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(frame))
    shuffled_columns = [f"shuffled_{column}" for column in square_columns]
    shuffled = frame[square_columns].iloc[permutation].reset_index(drop=True)
    shuffled.columns = shuffled_columns
    frame = pd.concat([frame, shuffled], axis=1)

    graph_branch_columns = slic_columns + line_columns
    visual_line_columns = visual_columns + line_columns
    unique_for_slic = nonredundant_columns(frame, square_columns, slic_columns)
    unique_for_graph_branch = nonredundant_columns(
        frame, square_columns, graph_branch_columns
    )
    unique_for_visual = nonredundant_columns(frame, square_columns, visual_columns)
    unique_for_visual_line = nonredundant_columns(
        frame, square_columns, visual_line_columns
    )
    shuffled_lookup = {column: f"shuffled_{column}" for column in square_columns}
    shuffled_unique_slic = [shuffled_lookup[column] for column in unique_for_slic]
    shuffled_unique_graph = [shuffled_lookup[column] for column in unique_for_graph_branch]
    shuffled_unique_visual = [shuffled_lookup[column] for column in unique_for_visual]
    shuffled_unique_visual_line = [
        shuffled_lookup[column] for column in unique_for_visual_line
    ]
    print(
        "Nonredundant square features at r2<0.90: "
        f"vs SLIC={len(unique_for_slic)}, "
        f"vs SLIC+line={len(unique_for_graph_branch)}, "
        f"vs visual={len(unique_for_visual)}, "
        f"vs visual+line={len(unique_for_visual_line)}",
        flush=True,
    )

    n_square = len(square_columns)
    conditions = {
        f"Square shell {n_square}": square_columns,
        "SLIC 71": slic_columns,
        "SLIC 71 + square": slic_columns + square_columns,
        "SLIC 71 + shuffled square": slic_columns + shuffled_columns,
        "SLIC 71 + unique square": slic_columns + unique_for_slic,
        "SLIC 71 + shuffled unique": slic_columns + shuffled_unique_slic,
        "SLIC 71 + line graph": graph_branch_columns,
        "SLIC 71 + line + square": graph_branch_columns + square_columns,
        "SLIC 71 + line + shuffled": graph_branch_columns + shuffled_columns,
        "SLIC 71 + line + unique": graph_branch_columns + unique_for_graph_branch,
        "SLIC 71 + line + shuffled unique": graph_branch_columns + shuffled_unique_graph,
        "Visual 258": visual_columns,
        "Visual 258 + square": visual_columns + square_columns,
        "Visual 258 + shuffled square": visual_columns + shuffled_columns,
        "Visual 258 + unique square": visual_columns + unique_for_visual,
        "Visual 258 + shuffled unique": visual_columns + shuffled_unique_visual,
        "Visual 258 + line graph": visual_line_columns,
        "Visual 258 + line + square": visual_line_columns + square_columns,
        "Visual 258 + line + shuffled": visual_line_columns + shuffled_columns,
        "Visual 258 + line + unique": visual_line_columns + unique_for_visual_line,
        "Visual 258 + line + shuffled unique": (
            visual_line_columns + shuffled_unique_visual_line
        ),
    }

    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            np.zeros(len(labels)), labels
        )
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
            scores = np.asarray(
                evaluators[classifier](frame[columns].to_numpy(dtype=np.float32))
            )
            rows.append({
                "classifier": classifier,
                "condition": condition,
                "n_images": len(frame),
                "n_styles": len(np.unique(labels)),
                "n_features": len(columns),
                "bacc_mean": float(scores.mean()),
                "bacc_std": float(scores.std()),
                "elapsed_seconds": time.perf_counter() - started,
                **{
                    f"fold_{index + 1}": float(score)
                    for index, score in enumerate(scores)
                },
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(
                f"{classifier:<8} {condition:<38} "
                f"{scores.mean():.4f} +/- {scores.std():.4f}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    lookup = result.set_index(["classifier", "condition"])
    comparisons = []
    comparison_specs = (
        ("SLIC", "SLIC 71", "SLIC 71 + square", "SLIC 71 + shuffled square"),
        ("SLIC unique", "SLIC 71", "SLIC 71 + unique square", "SLIC 71 + shuffled unique"),
        (
            "SLIC+line", "SLIC 71 + line graph", "SLIC 71 + line + square",
            "SLIC 71 + line + shuffled",
        ),
        (
            "SLIC+line unique", "SLIC 71 + line graph", "SLIC 71 + line + unique",
            "SLIC 71 + line + shuffled unique",
        ),
        ("Visual", "Visual 258", "Visual 258 + square", "Visual 258 + shuffled square"),
        (
            "Visual unique", "Visual 258", "Visual 258 + unique square",
            "Visual 258 + shuffled unique",
        ),
        (
            "Visual+line", "Visual 258 + line graph", "Visual 258 + line + square",
            "Visual 258 + line + shuffled",
        ),
        (
            "Visual+line unique", "Visual 258 + line graph", "Visual 258 + line + unique",
            "Visual 258 + line + shuffled unique",
        ),
    )
    for classifier in args.classifiers:
        for scope, baseline, real, placebo in comparison_specs:
            base = lookup.loc[(classifier, baseline)]
            real_row = lookup.loc[(classifier, real)]
            placebo_row = lookup.loc[(classifier, placebo)]
            base_scores = np.asarray([base[f"fold_{fold}"] for fold in range(1, 6)])
            real_scores = np.asarray([real_row[f"fold_{fold}"] for fold in range(1, 6)])
            placebo_scores = np.asarray(
                [placebo_row[f"fold_{fold}"] for fold in range(1, 6)]
            )
            comparisons.append({
                "classifier": classifier,
                "scope": scope,
                "delta_square_vs_baseline_pp": 100 * float((real_scores - base_scores).mean()),
                "p_square_vs_baseline": float(ttest_rel(real_scores, base_scores).pvalue),
                "delta_square_vs_placebo_pp": 100 * float((real_scores - placebo_scores).mean()),
                "p_square_vs_placebo": float(ttest_rel(real_scores, placebo_scores).pvalue),
                **{
                    f"square_minus_baseline_fold_{fold}_pp": (
                        100 * float(real_scores[fold - 1] - base_scores[fold - 1])
                    )
                    for fold in range(1, 6)
                },
            })
    comparison_path = args.output.with_name(args.output.stem + "_comparisons.csv")
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"saved={args.output} comparisons={comparison_path}")


if __name__ == "__main__":
    main()
