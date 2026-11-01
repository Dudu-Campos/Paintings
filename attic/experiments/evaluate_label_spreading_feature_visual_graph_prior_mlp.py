"""MLP test for the text-free visual similarity graph feature."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402
from evaluate_full_factorial_ablation_text import selected_blocks  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_prior import (  # noqa: E402
    VISUAL_FULL_MASK,
    node_feature_scores,
    prior_correct,
    scale_blocks,
)
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402

torch.set_num_threads(int(THREADS_PER_WORKER))
OUTPUT_DIR = ROOT / "outputs/label_spreading_feature_visual_graph_prior_mlp"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_comparison(detail: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline_name = "1: visual-only MLP"
    baseline = detail[detail.condition.eq(baseline_name)].set_index("fold")["bacc"]
    rows = []
    for condition in sorted(c for c in detail["condition"].unique() if c != baseline_name):
        paired = detail[detail.condition.eq(condition)].set_index("fold")["bacc"]
        common_folds = baseline.index.intersection(paired.index)
        delta = paired.loc[common_folds] - baseline.loc[common_folds]
        if len(delta) > 1:
            statistic, p_value = ttest_rel(paired.loc[common_folds], baseline.loc[common_folds])
        else:
            statistic, p_value = float("nan"), float("nan")
        rows.append(
            {
                "condition": condition,
                "folds": len(common_folds),
                "base_bacc_mean": 100 * float(baseline.loc[common_folds].mean()),
                "graph_bacc_mean": 100 * float(paired.loc[common_folds].mean()),
                "delta_bacc_pp": 100 * float(delta.mean()),
                "delta_std_pp": 100 * float(delta.std()),
                "t_statistic": float(statistic),
                "p_paired": float(p_value),
                "positive_folds": int((delta > 0).sum()),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False, float_format="%.6f")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument(
        "--felz-features",
        type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    visual_blocks, visual_names = selected_blocks(VISUAL_FULL_MASK, group_blocks, fixed_extra_blocks)
    log(
        f"visual-only MLP similarity graph: {len(frame):,} images, {n_classes} classes, "
        f"{sum(block.shape[1] for block in visual_blocks)} visual dims, blocks={visual_names}"
    )

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(row["condition"]), int(row["fold"])) for row in rows}

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        pipeline_scaled = scale_blocks(visual_blocks, outer_train)
        started = time.perf_counter()
        oof_fit, target_scores = node_feature_scores(
            pipeline_scaled[outer_train],
            labels[outer_train],
            pipeline_scaled[outer_test],
            n_classes,
            args.n_neighbors,
            args.alpha,
            args.max_iter,
            SEED + fold,
        )
        elapsed_graph = time.perf_counter() - started
        node_feature = np.zeros((len(frame), n_classes), dtype=np.float32)
        node_feature[outer_train] = oof_fit
        node_feature[outer_test] = target_scores
        corrected_feature = prior_correct(node_feature, labels[outer_train], n_classes)
        log(f"fold={fold}/{N_FOLDS} visual graph calculated in {elapsed_graph:.1f}s")

        conditions = {
            "1: visual-only MLP": visual_blocks,
            "2: visual-only MLP + similarity graph": visual_blocks + [node_feature],
            "3: visual-only MLP + similarity graph (prior-corrected)": visual_blocks + [corrected_feature],
        }
        for condition, condition_blocks in conditions.items():
            if (condition, fold) in completed:
                continue
            dimensions = sum(block.shape[1] for block in condition_blocks)
            started = time.perf_counter()
            metrics = evaluate_mlp(condition_blocks, labels, outer_train, outer_test, fold, args.epochs, args.patience)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "dimensions": dimensions,
                    "elapsed_seconds": elapsed + (elapsed_graph if "similarity graph" in condition else 0.0),
                    **metrics,
                }
            )
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            log(f"fold={fold}/{N_FOLDS} {condition:<58} dims={dimensions:4d} BAcc={metrics['bacc']:.4f}")

    detail = pd.DataFrame(rows)
    comparison = write_comparison(detail, args.output_dir)
    log(f"saved: {args.output_dir}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
