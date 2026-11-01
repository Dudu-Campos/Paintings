"""Are features extracted from a pooled (coarser) version of the image
complementary to the same features at normal resolution?

Pools by averaging 2x2 blocks (`pool_image`, factor=2) before each
extractor's own resize step, so only fine detail beyond the pooled
resolution is lost -- everything downstream (TARGET/N_SIZE/SIZE constants,
descriptor formulas) is untouched. Pilot scale (12,148 images), cheap
modalities only (Descriptors, Hilbert, Fourier -- all under 0.3s/image),
sonda LDA.

Normal-resolution blocks are subset from the existing full-cohort caches
(no need to re-extract); pooled blocks come from the fresh factor=2
extraction in outputs/pooled_pilot_half/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from evaluate_full_factorial_ablation import evaluate_lda
from fusion_curve_slic import ROOT, SEED


PILOT_SAMPLE = ROOT / "results/fourier_12k_canonical_sample.csv"
NORMAL_FEATURE_DIR = ROOT / "outputs/wikiart_features"
NORMAL_FOURIER = ROOT / "results/fourier_full_features.parquet"
N_FOLDS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_npy_block(features_path: Path, stems_path: Path) -> pd.DataFrame:
    values = np.load(features_path)
    stems = np.load(stems_path, allow_pickle=True).astype(str)
    columns = [f"f{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "stem", stems)
    return frame


def load_fourier_block(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    numeric_columns = [c for c in frame.columns if c not in ("stem", "style")]
    return frame[["stem", *numeric_columns]]


def block_matrix(frame: pd.DataFrame, pilot: pd.DataFrame) -> np.ndarray:
    merged = pilot[["stem"]].merge(frame, on="stem", how="left", validate="one_to_one")
    if merged.drop(columns="stem").isna().any().any():
        raise ValueError("missing rows after aligning block to pilot stems")
    return merged.drop(columns="stem").to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-dir", type=Path, default=ROOT / "outputs/pooled_pilot_half")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pooled_image_complementarity")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir = args.pooled_dir

    pilot = pd.read_csv(PILOT_SAMPLE)[["stem", "style"]].drop_duplicates("stem")
    pilot = pilot.sort_values("stem").reset_index(drop=True)
    labels = LabelEncoder().fit_transform(pilot["style"].to_numpy(dtype=str))
    log(f"Pooling complementarity: {len(pilot):,} imagens, {len(np.unique(labels))} estilos")

    normal_descriptors = load_npy_block(
        NORMAL_FEATURE_DIR / "desc_full_features.npy", NORMAL_FEATURE_DIR / "desc_full_stems.npy"
    )
    normal_hilbert = load_npy_block(
        NORMAL_FEATURE_DIR / "hilbert_features.npy", NORMAL_FEATURE_DIR / "hilbert_stems.npy"
    )
    normal_fourier = load_fourier_block(NORMAL_FOURIER)

    pooled_descriptors = load_npy_block(
        pooled_dir / "descriptors/desc_full_features.npy", pooled_dir / "descriptors/desc_full_stems.npy"
    )
    pooled_hilbert = load_npy_block(
        pooled_dir / "hilbert/hilbert_features.npy", pooled_dir / "hilbert/hilbert_stems.npy"
    )
    pooled_fourier = load_fourier_block(pooled_dir / "fourier_full_features.parquet")

    baseline_blocks = [
        block_matrix(normal_descriptors, pilot),
        block_matrix(normal_hilbert, pilot),
        block_matrix(normal_fourier, pilot),
    ]
    pooled_blocks = {
        "Descriptors": block_matrix(pooled_descriptors, pilot),
        "Hilbert": block_matrix(pooled_hilbert, pilot),
        "Fourier": block_matrix(pooled_fourier, pilot),
    }
    log(
        "dims baseline="
        f"{sum(b.shape[1] for b in baseline_blocks)} "
        f"(Desc {baseline_blocks[0].shape[1]}, Hilb {baseline_blocks[1].shape[1]}, "
        f"Four {baseline_blocks[2].shape[1]})"
    )

    conditions = {
        "1: baseline": baseline_blocks,
        "2: + pooled Descriptors": baseline_blocks + [pooled_blocks["Descriptors"]],
        "3: + pooled Hilbert": baseline_blocks + [pooled_blocks["Hilbert"]],
        "4: + pooled Fourier": baseline_blocks + [pooled_blocks["Fourier"]],
        "5: + pooled todas as três": baseline_blocks + list(pooled_blocks.values()),
    }

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["condition"]), int(r["fold"])) for r in rows}

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        for condition, blocks in conditions.items():
            if (condition, fold) in completed:
                continue
            dims = sum(block.shape[1] for block in blocks)
            started = time.perf_counter()
            metrics = evaluate_lda(blocks, labels, outer_train, outer_test)
            elapsed = time.perf_counter() - started
            rows.append({
                "fold": fold, "condition": condition, "dimensions": dims,
                "elapsed_seconds": elapsed, **metrics,
            })
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            log(f"fold={fold}/{N_FOLDS} {condition:<28} dims={dims:4d} BAcc={metrics['bacc']:.4f}")

    detail = pd.DataFrame(rows)
    baseline = detail[detail.condition.eq("1: baseline")].set_index("fold")["bacc"]
    summary_rows = []
    for condition in conditions:
        if condition == "1: baseline":
            continue
        paired = detail[detail.condition.eq(condition)].set_index("fold")["bacc"]
        common_folds = baseline.index.intersection(paired.index)
        delta = paired.loc[common_folds] - baseline.loc[common_folds]
        if len(delta) > 1:
            _, p_value = ttest_rel(paired.loc[common_folds], baseline.loc[common_folds])
        else:
            p_value = float("nan")
        summary_rows.append({
            "condition": condition, "folds": len(common_folds),
            "delta_bacc_pp": 100 * float(delta.mean()), "delta_std_pp": 100 * float(delta.std()),
            "p_paired": float(p_value), "positive_folds": int((delta > 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"Saved to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
