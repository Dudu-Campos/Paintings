#!/usr/bin/env python3
"""Paired 2x2 MLP ablation of line- and square-graph features.

The experiment keeps the canonical 258 visual features fixed and toggles two
derived SLIC-RAG blocks: the 63-dimensional line graph and the 64-dimensional
strict distance-2 shell of G^2.  All four conditions use identical outer folds,
internal validation splits, optimization, and random seeds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_super_resolution_pilot import load_original  # noqa: E402
from model_evaluation import DEVICE, evaluate_mlp  # noqa: E402


CONDITION_COLUMNS = {
    "Visual 258": (False, False),
    "Visual 258 + line graph": (True, False),
    "Visual 258 + square": (False, True),
    "Visual 258 + line + square": (True, True),
}


def fold_scores(row: pd.Series) -> np.ndarray:
    return np.asarray([row[f"fold_{fold}"] for fold in range(1, 6)], dtype=np.float64)


def paired_row(
    lookup: pd.DataFrame, effect: str, baseline: str, augmented: str,
) -> dict[str, float | str]:
    base = fold_scores(lookup.loc[baseline])
    added = fold_scores(lookup.loc[augmented])
    delta = 100 * (added - base)
    return {
        "effect": effect,
        "baseline": baseline,
        "augmented": augmented,
        "delta_bacc_pp": float(delta.mean()),
        "p_paired": float(ttest_rel(added, base).pvalue),
        "positive_folds": int(np.sum(delta > 0)),
        **{f"delta_fold_{fold}_pp": float(delta[fold - 1]) for fold in range(1, 6)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=Path("results/super_resolution_3k_sample.csv")
    )
    parser.add_argument(
        "--line-features", type=Path,
        default=Path("results/line_graph_3k_features.parquet"),
    )
    parser.add_argument(
        "--square-features", type=Path,
        default=Path("results/square_graph_3k_features.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/line_square_mlp_factorial_3k_results.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    visual, visual_columns = load_original(
        Path("outputs/wikiart_slic_features"), Path("outputs/wikiart_features")
    )
    if len(visual_columns) != 258:
        raise ValueError(f"expected 258 canonical visual features, got {len(visual_columns)}")

    line_raw = pd.read_parquet(args.line_features)
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    line_columns = list(line_rename.values())
    if len(line_columns) != 63:
        raise ValueError(f"expected 63 line-graph features, got {len(line_columns)}")

    square_raw = pd.read_parquet(args.square_features)
    square_raw_columns = [
        column for column in square_raw if column not in {"stem", "style"}
    ]
    square_rename = {column: f"square_{column}" for column in square_raw_columns}
    square = square_raw[["stem"] + square_raw_columns].rename(columns=square_rename)
    square_columns = list(square_rename.values())
    if len(square_columns) != 64:
        raise ValueError(f"expected 64 square-graph features, got {len(square_columns)}")

    frame = sample.merge(visual, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(line, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(square, on="stem", how="inner", validate="one_to_one")
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)

    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    print(
        f"cohort={len(frame):,}; styles={len(np.unique(labels))}; "
        f"visual={len(visual_columns)}, line={len(line_columns)}, "
        f"square={len(square_columns)}; device={DEVICE}",
        flush=True,
    )

    rows = []
    for condition, (use_line, use_square) in CONDITION_COLUMNS.items():
        columns = list(visual_columns)
        if use_line:
            columns.extend(line_columns)
        if use_square:
            columns.extend(square_columns)
        started = time.perf_counter()
        scores = np.asarray(
            evaluate_mlp(
                frame[columns].to_numpy(dtype=np.float32), labels, splits
            ),
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
            f"{condition:<34} {scores.mean():.4f} +/- {scores.std():.4f} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    result = pd.DataFrame(rows)
    lookup = result.set_index("condition")
    comparisons = [
        paired_row(lookup, "line | no square", "Visual 258", "Visual 258 + line graph"),
        paired_row(lookup, "square | no line", "Visual 258", "Visual 258 + square"),
        paired_row(
            lookup, "square | line", "Visual 258 + line graph",
            "Visual 258 + line + square",
        ),
        paired_row(
            lookup, "line | square", "Visual 258 + square",
            "Visual 258 + line + square",
        ),
        paired_row(
            lookup, "line + square | baseline", "Visual 258",
            "Visual 258 + line + square",
        ),
    ]

    base = fold_scores(lookup.loc["Visual 258"])
    line_scores = fold_scores(lookup.loc["Visual 258 + line graph"])
    square_scores = fold_scores(lookup.loc["Visual 258 + square"])
    both = fold_scores(lookup.loc["Visual 258 + line + square"])
    interaction = 100 * (both - line_scores - square_scores + base)
    comparisons.append({
        "effect": "line x square interaction",
        "baseline": "additive expectation",
        "augmented": "observed line + square",
        "delta_bacc_pp": float(interaction.mean()),
        "p_paired": float(ttest_1samp(interaction, 0).pvalue),
        "positive_folds": int(np.sum(interaction > 0)),
        **{
            f"delta_fold_{fold}_pp": float(interaction[fold - 1])
            for fold in range(1, 6)
        },
    })
    comparison = pd.DataFrame(comparisons)
    comparison_path = args.output.with_name(args.output.stem + "_comparisons.csv")
    comparison.to_csv(comparison_path, index=False)
    print("\nFactorial comparisons:", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"saved={args.output} comparisons={comparison_path}", flush=True)


if __name__ == "__main__":
    main()
