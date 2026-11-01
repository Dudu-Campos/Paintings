"""Does Multiscale-Fourier (4 quadrant Fourier spectra) help, on top of the
baseline pipeline without Multiscale? Same paired-comparison pattern as
`evaluate_multiscale_by_family.py`'s per-family tests, extended to Fourier
now that quadrant Fourier features exist
(`outputs/wikiart_quadrant_features/<region>/fourier_full_features.parquet`).

LDA and MLP, but only these 2 conditions (not the full 5-condition sweep from
`evaluate_multiscale_by_family.py`) -- MLP on margarita has no GPU (falls
back to CPU, ~300-500s/fit), so scope is kept to exactly this comparison.
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(int(THREADS_PER_WORKER))

from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import FEATURE_NAMES, load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, REGIONS, ROOT, SEED, load_dataset


BASELINE_MASK = 127  # 7 fatores originais (SLIC..Text), sem Multiscale
QUADRANT_DIR = ROOT / "outputs/wikiart_quadrant_features"
OUTPUT_DIR = ROOT / "outputs/multiscale_fourier"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_quadrant_fourier(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    region_columns: dict[str, list[str]] = {}
    for region in REGIONS:
        path = QUADRANT_DIR / region / "fourier_full_features.parquet"
        frame, generic_columns = load_fourier(frame, path)
        renamed = {column: f"{region}_{column}" for column in generic_columns}
        frame = frame.rename(columns=renamed)
        region_columns[region] = list(renamed.values())
    return frame, region_columns


def main() -> None:
    frame, groups = load_dataset(ROOT / "outputs/wikiart_quadrant_features")
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, ROOT / "results/line_graph_full_features.parquet",
        ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    frame, fourier_columns = load_fourier(frame, ROOT / "results/fourier_full_features.parquet")
    global_fourier_rename = {column: f"global_{column}" for column in fourier_columns}
    frame = frame.rename(columns=global_fourier_rename)
    fourier_columns = list(global_fourier_rename.values())
    frame, fourier_region_columns = load_quadrant_fourier(frame)
    log(f"Coorte: {len(frame):,} imagens, {len(FEATURE_NAMES)} features Fourier/quadrante")

    group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()
    }
    fourier_quadrant_blocks = [
        frame[fourier_region_columns[region]].to_numpy(dtype=np.float32) for region in REGIONS
    ]
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(row["classifier"]), str(row["condition"]), int(row["fold"])) for row in rows}

    fold_data = {fold: split for fold, split in enumerate(splits, start=1)}

    # LDA (rápido) primeiro, fecha os 5 folds; MLP (lento, CPU aqui) depois --
    # evita perder resultados prontos se o job for interrompido no meio do MLP.
    for classifier in ("lda", "mlp"):
        for fold, (outer_train, outer_test) in fold_data.items():
            text_block = load_text_block(ROOT / "outputs/tfidf_text_cache", fold, frame, n_classes)
            extra_blocks = {**fixed_extra_blocks, "Text": text_block}
            baseline_blocks, _ = selected_blocks(BASELINE_MASK, group_blocks, extra_blocks)

            conditions = {
                "1: baseline (sem Multiscale)": baseline_blocks,
                "2: + Multiscale-Fourier": baseline_blocks + fourier_quadrant_blocks,
            }
            for condition, condition_blocks in conditions.items():
                if (classifier, condition, fold) in completed:
                    continue
                dimensions = sum(block.shape[1] for block in condition_blocks)
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(condition_blocks, labels, outer_train, outer_test)
                else:
                    metrics = evaluate_mlp(
                        condition_blocks, labels, outer_train, outer_test, fold,
                        epochs=150, patience=15,
                    )
                elapsed = time.perf_counter() - started
                rows.append({
                    "fold": fold, "classifier": classifier, "condition": condition,
                    "dimensions": dimensions, "elapsed_seconds": elapsed, **metrics,
                })
                completed.add((classifier, condition, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(
                    f"fold={fold}/{N_FOLDS} {classifier:<4} {condition:<28} dims={dimensions:4d} "
                    f"BAcc={metrics['bacc']:.4f} seconds={elapsed:.1f}"
                )

    detail = pd.DataFrame(rows)
    comparisons = []
    for classifier, classifier_rows in detail.groupby("classifier"):
        baseline = classifier_rows[classifier_rows.condition.eq("1: baseline (sem Multiscale)")].set_index("fold")["bacc"]
        paired = classifier_rows[classifier_rows.condition.eq("2: + Multiscale-Fourier")].set_index("fold")["bacc"]
        delta = paired - baseline
        statistic, p_value = ttest_rel(paired, baseline)
        comparisons.append({
            "classifier": classifier, "condition": "2: + Multiscale-Fourier",
            "delta_bacc_pp": 100 * float(delta.mean()), "delta_std_pp": 100 * float(delta.std()),
            "p_paired": float(p_value), "positive_folds": int((delta > 0).sum()),
        })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUTPUT_DIR / "comparison.csv", index=False, float_format="%.6f")
    log(f"Saved to {OUTPUT_DIR}")
    print(comparisons.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
