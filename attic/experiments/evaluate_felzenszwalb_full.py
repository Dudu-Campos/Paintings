#!/usr/bin/env python3
"""Evaluate full Felzenszwalb multigraph fusion on the canonical cohort."""

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
from compare_segmentations_pilot import FEATURE_COLUMNS  # noqa: E402
from evaluate_big_lama_visual_pilot import lda_scores  # noqa: E402
from evaluate_line_graph_3k import nonredundant_columns  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402


BASELINE_CONDITIONS = (
    "SLIC 71",
    "SLIC 71 + line graph",
    "Visual 258",
    "Visual 258 + line graph",
)


def selected_baselines(path: Path, classifiers: list[str]) -> pd.DataFrame:
    previous = pd.read_csv(path)
    selected = previous[
        previous["classifier"].isin(classifiers)
        & previous["condition"].isin(BASELINE_CONDITIONS)
    ].copy()
    expected = len(classifiers) * len(BASELINE_CONDITIONS)
    if len(selected) != expected:
        raise ValueError(f"missing cached baselines: {len(selected)}/{expected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/line_graph_full_sample.csv"))
    parser.add_argument(
        "--felz-features", type=Path,
        default=Path("outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet"),
    )
    parser.add_argument(
        "--line-features", type=Path,
        default=Path("paper/line_graph_full_features.parquet"),
    )
    parser.add_argument(
        "--baseline-results", type=Path,
        default=Path("paper/line_graph_full_results.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/felzenszwalb_full_results.csv"))
    parser.add_argument(
        "--classifiers", nargs="+", choices=("LDA", "XGBoost", "MLP"),
        default=("LDA", "XGBoost", "MLP"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    classifiers = list(args.classifiers)

    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    visual, visual_columns = load_original(
        Path("outputs/wikiart_slic_features"), Path("outputs/wikiart_features")
    )
    slic_columns = [column for column in visual_columns if "_slic_" in column]
    if len(slic_columns) != 71 or len(visual_columns) != 258:
        raise ValueError(
            f"unexpected canonical dimensions: SLIC={len(slic_columns)}, visual={len(visual_columns)}"
        )

    line_raw = pd.read_parquet(args.line_features)
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    line_columns = list(line_rename.values())
    if len(line_columns) != 63:
        raise ValueError(f"unexpected line-graph dimension: {len(line_columns)}")

    felz_raw = pd.read_parquet(args.felz_features)
    missing = set(FEATURE_COLUMNS) - set(felz_raw.columns)
    if missing:
        raise ValueError(f"missing Felzenszwalb features: {sorted(missing)}")
    felz_rename = {column: f"felz_{column}" for column in FEATURE_COLUMNS}
    felz = felz_raw[["stem"] + FEATURE_COLUMNS].drop_duplicates("stem").rename(columns=felz_rename)
    felz_columns = list(felz_rename.values())
    if len(felz_columns) != 71:
        raise ValueError(f"unexpected Felzenszwalb dimension: {len(felz_columns)}")

    frame = sample.merge(visual, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(line, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(felz, on="stem", how="inner", validate="one_to_one")
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(frame))
    shuffled_columns = [f"shuffled_{column}" for column in felz_columns]
    shuffled = frame[felz_columns].iloc[permutation].reset_index(drop=True)
    shuffled.columns = shuffled_columns
    frame = pd.concat([frame, shuffled], axis=1)

    graph_columns = slic_columns + line_columns
    visual_line_columns = visual_columns + line_columns
    unique_slic = nonredundant_columns(frame, felz_columns, slic_columns)
    unique_graph = nonredundant_columns(frame, felz_columns, graph_columns)
    unique_visual = nonredundant_columns(frame, felz_columns, visual_columns)
    unique_visual_line = nonredundant_columns(frame, felz_columns, visual_line_columns)
    shuffled_lookup = {column: f"shuffled_{column}" for column in felz_columns}

    print(
        "Nonredundant Felzenszwalb features at r2<0.90: "
        f"vs SLIC={len(unique_slic)}, vs SLIC+line={len(unique_graph)}, "
        f"vs visual={len(unique_visual)}, vs visual+line={len(unique_visual_line)}",
        flush=True,
    )

    conditions = {
        "Felzenszwalb 71": felz_columns,
        "SLIC 71 + Felzenszwalb": slic_columns + felz_columns,
        "SLIC 71 + shuffled Felzenszwalb": slic_columns + shuffled_columns,
        "SLIC 71 + unique Felzenszwalb": slic_columns + unique_slic,
        "SLIC 71 + shuffled unique Felzenszwalb": (
            slic_columns + [shuffled_lookup[column] for column in unique_slic]
        ),
        "SLIC 71 + line + Felzenszwalb": graph_columns + felz_columns,
        "SLIC 71 + line + shuffled Felzenszwalb": graph_columns + shuffled_columns,
        "SLIC 71 + line + unique Felzenszwalb": graph_columns + unique_graph,
        "SLIC 71 + line + shuffled unique Felzenszwalb": (
            graph_columns + [shuffled_lookup[column] for column in unique_graph]
        ),
        "Visual 258 + Felzenszwalb": visual_columns + felz_columns,
        "Visual 258 + shuffled Felzenszwalb": visual_columns + shuffled_columns,
        "Visual 258 + unique Felzenszwalb": visual_columns + unique_visual,
        "Visual 258 + shuffled unique Felzenszwalb": (
            visual_columns + [shuffled_lookup[column] for column in unique_visual]
        ),
        "Visual 258 + line + Felzenszwalb": visual_line_columns + felz_columns,
        "Visual 258 + line + shuffled Felzenszwalb": visual_line_columns + shuffled_columns,
        "Visual 258 + line + unique Felzenszwalb": visual_line_columns + unique_visual_line,
        "Visual 258 + line + shuffled unique Felzenszwalb": (
            visual_line_columns
            + [shuffled_lookup[column] for column in unique_visual_line]
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

    baselines = selected_baselines(args.baseline_results, classifiers)
    if not (baselines["n_images"] == len(frame)).all():
        raise ValueError("cached baseline cohort differs from current cohort")
    if args.output.exists() and not args.overwrite:
        existing = pd.read_csv(args.output)
        existing = existing[existing["classifier"].isin(classifiers)].copy()
        rows = existing.to_dict("records")
        print(f"resume={len(existing)} rows from {args.output}", flush=True)
    else:
        rows = baselines.to_dict("records")
    completed = {(str(row["classifier"]), str(row["condition"])) for row in rows}

    for classifier in classifiers:
        for condition, columns in conditions.items():
            if (classifier, condition) in completed:
                continue
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
                **{f"fold_{index + 1}": float(score) for index, score in enumerate(scores)},
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(
                f"{classifier:<8} {condition:<52} {scores.mean():.4f} +/- {scores.std():.4f}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    lookup = result.set_index(["classifier", "condition"])
    comparisons = []
    specs = (
        ("SLIC", "SLIC 71", "SLIC 71 + Felzenszwalb", "SLIC 71 + shuffled Felzenszwalb"),
        (
            "SLIC unique", "SLIC 71", "SLIC 71 + unique Felzenszwalb",
            "SLIC 71 + shuffled unique Felzenszwalb",
        ),
        (
            "SLIC+line", "SLIC 71 + line graph", "SLIC 71 + line + Felzenszwalb",
            "SLIC 71 + line + shuffled Felzenszwalb",
        ),
        (
            "SLIC+line unique", "SLIC 71 + line graph",
            "SLIC 71 + line + unique Felzenszwalb",
            "SLIC 71 + line + shuffled unique Felzenszwalb",
        ),
        (
            "Visual", "Visual 258", "Visual 258 + Felzenszwalb",
            "Visual 258 + shuffled Felzenszwalb",
        ),
        (
            "Visual unique", "Visual 258", "Visual 258 + unique Felzenszwalb",
            "Visual 258 + shuffled unique Felzenszwalb",
        ),
        (
            "Visual+line", "Visual 258 + line graph",
            "Visual 258 + line + Felzenszwalb",
            "Visual 258 + line + shuffled Felzenszwalb",
        ),
        (
            "Visual+line unique", "Visual 258 + line graph",
            "Visual 258 + line + unique Felzenszwalb",
            "Visual 258 + line + shuffled unique Felzenszwalb",
        ),
    )
    for classifier in classifiers:
        for scope, baseline, real, placebo in specs:
            base = lookup.loc[(classifier, baseline)]
            real_row = lookup.loc[(classifier, real)]
            placebo_row = lookup.loc[(classifier, placebo)]
            base_scores = np.asarray([base[f"fold_{fold}"] for fold in range(1, 6)])
            real_scores = np.asarray([real_row[f"fold_{fold}"] for fold in range(1, 6)])
            placebo_scores = np.asarray([placebo_row[f"fold_{fold}"] for fold in range(1, 6)])
            comparisons.append({
                "classifier": classifier,
                "scope": scope,
                "delta_felz_vs_baseline_pp": 100 * float((real_scores - base_scores).mean()),
                "p_felz_vs_baseline": float(ttest_rel(real_scores, base_scores).pvalue),
                "delta_felz_vs_placebo_pp": 100 * float((real_scores - placebo_scores).mean()),
                "p_felz_vs_placebo": float(ttest_rel(real_scores, placebo_scores).pvalue),
                **{
                    f"felz_minus_baseline_fold_{fold}_pp": (
                        100 * float(real_scores[fold - 1] - base_scores[fold - 1])
                    )
                    for fold in range(1, 6)
                },
            })
    comparison_path = args.output.with_name(args.output.stem + "_comparisons.csv")
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    print(f"saved={args.output} comparisons={comparison_path}", flush=True)


if __name__ == "__main__":
    main()
