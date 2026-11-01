"""Lightweight placebo check for the 8-factor pipeline (mask=255).

Not part of the exact-Shapley factorial sweep -- that would need a full
2^9-mask restart. Instead, a cheap paired comparison: does stacking a
row-shuffled copy of an existing block (same marginal distribution and
dimensionality, label association destroyed) on top of the full pipeline
change LDA balanced accuracy? Establishes a noise floor for the smaller
Shapley contributions (Fourier, Multiscale) found in the factorial run,
using the "shuffled" placebo convention already used elsewhere in this repo
(`evaluate_felzenszwalb_full.py`, the Fourier branch shuffled control).
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from evaluate_full_factorial_ablation import evaluate_lda
from evaluate_full_factorial_ablation_text import FULL_MASK, load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


PLACEBO_TEMPLATES = ("Text", "SLIC")  # small (27d) and moderate (71d) reference sizes


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
        "--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet"
    )
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/placebo_modality")
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features
    )
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()
    }
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    log(f"Placebo check: {len(frame):,} images, {n_classes} styles, templates={PLACEBO_TEMPLATES}")

    rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra_blocks = {**fixed_extra_blocks, "Text": text_block}
        lookup = {**group_blocks, **extra_blocks}
        blocks, block_names = selected_blocks(FULL_MASK, group_blocks, extra_blocks)

        rng = np.random.default_rng(SEED + fold)
        conditions = {"Pipeline completo (8 fatores)": blocks}
        for template in PLACEBO_TEMPLATES:
            source = lookup[template]
            shuffled = source[rng.permutation(len(source))]
            label = f"+ placebo embaralhado ({template}, {source.shape[1]}d)"
            conditions[label] = blocks + [shuffled]

        for condition, condition_blocks in conditions.items():
            started = time.perf_counter()
            metrics = evaluate_lda(condition_blocks, labels, outer_train, outer_test)
            elapsed = time.perf_counter() - started
            rows.append({
                "fold": fold, "condition": condition,
                "dimensions": sum(block.shape[1] for block in condition_blocks),
                "elapsed_seconds": elapsed, **metrics,
            })
            log(f"fold={fold}/{N_FOLDS} {condition:<45} BAcc={metrics['bacc']:.4f}")

    detail = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)

    baseline = detail[detail.condition.eq("Pipeline completo (8 fatores)")].set_index("fold")["bacc"]
    comparisons = []
    for condition, condition_rows in detail.groupby("condition"):
        if condition == "Pipeline completo (8 fatores)":
            continue
        paired = condition_rows.set_index("fold")["bacc"]
        delta = paired - baseline
        statistic, p_value = ttest_rel(paired, baseline)
        comparisons.append({
            "condition": condition, "delta_bacc_pp": 100 * float(delta.mean()),
            "delta_std_pp": 100 * float(delta.std()), "p_paired": float(p_value),
            "positive_folds": int((delta > 0).sum()),
        })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(args.output_dir / "comparisons.csv", index=False, float_format="%.6f")

    log(f"Saved to {args.output_dir}")
    print(comparisons.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
