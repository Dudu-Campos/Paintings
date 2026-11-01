"""Fuse global-plus-quadrant visual features with line and Felzenszwalb graphs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from compare_segmentations_pilot import FEATURE_COLUMNS
from evaluate_robustness_mlp_aug import train_one
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


CONDITIONS = {
    "Visual GIST multiscale + line graph": (True, False),
    "Visual GIST multiscale + Felzenszwalb": (False, True),
    "Visual GIST multiscale + line graph + Felzenszwalb": (True, True),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_graph_blocks(
    frame: pd.DataFrame, line_path: Path, felz_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    line_raw = pd.read_parquet(line_path)
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    line_columns = list(line_rename.values())
    if len(line_columns) != 63 or line["stem"].duplicated().any():
        raise ValueError(f"Invalid line-graph block: {len(line_columns)} dimensions")

    felz_raw = pd.read_parquet(felz_path)
    missing = set(FEATURE_COLUMNS) - set(felz_raw)
    if missing:
        raise ValueError(f"Missing Felzenszwalb columns: {sorted(missing)}")
    felz_rename = {column: f"felz_{column}" for column in FEATURE_COLUMNS}
    felz = (
        felz_raw[["stem"] + FEATURE_COLUMNS]
        .drop_duplicates("stem")
        .rename(columns=felz_rename)
    )
    felz_columns = list(felz_rename.values())
    if len(felz_columns) != 71:
        raise ValueError(f"Invalid Felzenszwalb block: {len(felz_columns)} dimensions")

    expected_stems = frame["stem"].to_numpy(dtype=str)
    result = frame.merge(line, on="stem", how="inner", validate="one_to_one")
    result = result.merge(felz, on="stem", how="inner", validate="one_to_one")
    result = result.sort_values("stem").reset_index(drop=True)
    if len(result) != len(frame) or not np.array_equal(result["stem"], expected_stems):
        raise ValueError(f"Graph merge changed the canonical cohort: {len(result)}/{len(frame)}")
    return result, line_columns, felz_columns


def cached_baseline(path: Path, n_images: int) -> list[dict]:
    cached = pd.read_csv(path)
    cached = cached[cached["condition"].eq("Visual GIST multiscale")].copy()
    if (
        len(cached) != N_FOLDS
        or cached["fold"].nunique() != N_FOLDS
        or not cached["n_images"].eq(n_images).all()
    ):
        raise ValueError("Cached multiscale baseline is incomplete or uses another cohort")
    cached["dimensions"] = 1290
    cached["branches"] = 15
    return cached.to_dict("records")


def save_results(rows: list[dict], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["fold", "condition"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby("condition", as_index=False).agg(
        folds_completed=("fold", "nunique"), n_images=("n_images", "first"),
        dimensions=("dimensions", "first"), branches=("branches", "first"),
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"),
        best_epoch_mean=("best_epoch", "mean"), elapsed_seconds=("elapsed_seconds", "sum"),
    )
    baseline_score = summary.loc[
        summary.condition.eq("Visual GIST multiscale"), "bacc_mean"
    ].iat[0]
    summary["delta_from_multiscale_pp"] = 100 * (summary.bacc_mean - baseline_score)
    summary["delta_from_50_pp"] = 100 * (summary.bacc_mean - 0.50)
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")

    baseline = detail.loc[
        detail.condition.eq("Visual GIST multiscale"), ["fold", "bacc"]
    ].set_index("fold")
    comparisons = []
    for condition in CONDITIONS:
        candidate = detail.loc[detail.condition.eq(condition), ["fold", "bacc"]].set_index("fold")
        paired = baseline.join(candidate, lsuffix="_base", rsuffix="_candidate", how="inner")
        statistic, pvalue = np.nan, np.nan
        if len(paired) >= 2:
            statistic, pvalue = ttest_rel(paired.bacc_candidate, paired.bacc_base)
        comparisons.append({
            "condition": condition, "paired_folds": len(paired),
            "delta_from_multiscale_pp": 100 * (paired.bacc_candidate - paired.bacc_base).mean(),
            "positive_folds": int((paired.bacc_candidate > paired.bacc_base).sum()),
            "paired_t": statistic, "paired_p": pvalue,
            **{
                f"delta_fold_{fold}_pp": 100 * float(
                    paired.loc[fold, "bacc_candidate"] - paired.loc[fold, "bacc_base"]
                )
                for fold in paired.index
            },
        })
    pd.DataFrame(comparisons).to_csv(
        output_dir / "comparisons.csv", index=False, float_format="%.6f"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features"
    )
    parser.add_argument(
        "--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet"
    )
    parser.add_argument(
        "--felz-features", type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument(
        "--baseline-folds", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp_multiscale/fold_results.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/multiscale_graph_fusion"
    )
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features
    )
    visual_groups = list(groups.values())
    visual_columns = [column for columns in visual_groups for column in columns]
    if len(visual_groups) != 15 or len(visual_columns) != 1290:
        raise ValueError(
            f"Unexpected multiscale representation: {len(visual_groups)} branches, "
            f"{len(visual_columns)} dimensions"
        )
    blocks = {
        "visual": frame[visual_columns].to_numpy(dtype=np.float32),
        "line": frame[line_columns].to_numpy(dtype=np.float32),
        "felz": frame[felz_columns].to_numpy(dtype=np.float32),
    }
    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(dtype=str))
    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    if result_path.exists() and not args.restart:
        rows = pd.read_csv(result_path).to_dict("records")
    else:
        rows = cached_baseline(args.baseline_folds, len(frame))
    completed = {(int(row["fold"]), str(row["condition"])) for row in rows}
    log(
        f"Multiscale graph fusion: {len(frame):,} images, {len(np.unique(labels))} styles; "
        f"visual=1290/15 branches, line=63, Felzenszwalb=71"
    )

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
        )
        for condition, (use_line, use_felz) in CONDITIONS.items():
            if (fold, condition) in completed:
                log(f"fold {fold}/{N_FOLDS} {condition}: already complete")
                continue
            names = ["visual"] + (["line"] if use_line else []) + (["felz"] if use_felz else [])
            selected = [blocks[name] for name in names]
            values = np.hstack(selected)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(values[fit_train]).astype(np.float32)
            validation_x = scaler.transform(values[validation]).astype(np.float32)
            test_x = scaler.transform(values[outer_test]).astype(np.float32)
            branch_sizes = [len(columns) for columns in visual_groups]
            if use_line:
                branch_sizes.append(len(line_columns))
            if use_felz:
                branch_sizes.append(len(felz_columns))
            started = time.perf_counter()
            metrics, _ = train_one(
                train_x, labels[fit_train], validation_x, labels[validation],
                test_x, labels[outer_test], branch_sizes, len(np.unique(labels)),
                SEED + fold, args.epochs, args.patience,
            )
            rows.append({
                "fold": fold, "condition": condition, "n_images": len(frame),
                "n_fit_train": len(fit_train), "n_validation": len(validation),
                "n_test": len(outer_test), "dimensions": values.shape[1],
                "branches": len(branch_sizes), "elapsed_seconds": time.perf_counter() - started,
                **metrics,
            })
            completed.add((fold, condition))
            save_results(rows, args.output_dir)
            log(
                f"fold {fold}/{N_FOLDS} {condition}: BAcc={metrics['bacc']:.4f}; "
                f"val={metrics['validation_bacc']:.4f}; epoch={metrics['best_epoch']}"
            )
            del values, train_x, validation_x, test_x
    if rows:
        save_results(rows, args.output_dir)
        print(pd.read_csv(args.output_dir / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
