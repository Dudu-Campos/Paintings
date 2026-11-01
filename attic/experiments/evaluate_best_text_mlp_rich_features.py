"""Paired MLP test of the best TF-IDF model with the new rich feature blocks.

The previous best text-aware interpretable model in
``outputs/full_factorial_ablation_text`` was mask 251:

    SLIC + Descriptors + Line graph + Felzenszwalb + Fourier + Text + Multiscale

This script reruns that condition and compares it to the same branch layout
after replacing the global Descriptor and Fourier blocks by their rich
versions.  Quadrant blocks remain the previously extracted compact blocks.
"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_full_factorial_ablation import evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset


BEST_TEXT_MASK = 251
BEST_TEXT_PLUS_HILBERT_MASK = 255


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_aligned_npy(values_path: Path, stems_path: Path, stems: np.ndarray) -> np.ndarray:
    values = np.load(values_path)
    value_stems = np.load(stems_path, allow_pickle=True).astype(str)
    if values.shape[0] != len(value_stems):
        raise ValueError(f"Shape/stem mismatch for {values_path}")
    lookup = pd.DataFrame({"stem": value_stems, "_row": np.arange(len(value_stems))})
    if lookup["stem"].duplicated().any():
        raise ValueError(f"Duplicate stems in {stems_path}")
    aligned = pd.DataFrame({"stem": stems}).merge(lookup, on="stem", how="left", validate="one_to_one")
    if aligned["_row"].isna().any():
        raise ValueError(f"{aligned['_row'].isna().sum():,} missing stems in {values_path}")
    return values[aligned["_row"].to_numpy(dtype=np.int64)].astype(np.float32, copy=False)


def load_aligned_parquet(path: Path, stems: np.ndarray, prefix: str) -> np.ndarray:
    table = pd.read_parquet(path)
    columns = [column for column in table.columns if column not in {"stem", "style"}]
    if table["stem"].duplicated().any():
        raise ValueError(f"Duplicate stems in {path}")
    renamed = {column: f"{prefix}_{index:03d}" for index, column in enumerate(columns)}
    table = table[["stem", *columns]].rename(columns=renamed)
    aligned = pd.DataFrame({"stem": stems}).merge(table, on="stem", how="left", validate="one_to_one")
    feature_columns = list(renamed.values())
    if aligned[feature_columns].isna().any().any():
        missing = int(aligned[feature_columns[0]].isna().sum())
        raise ValueError(f"{missing:,} missing stems in {path}")
    return aligned[feature_columns].to_numpy(dtype=np.float32)


def save(rows: list[dict], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["fold", "condition"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby("condition", as_index=False).agg(
        folds=("fold", "nunique"),
        dimensions=("dimensions", "first"),
        branches=("branches", "first"),
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    )
    baseline = detail.loc[detail.condition.eq("compact best text MLP"), ["fold", "bacc"]].set_index("fold")
    comparisons = []
    for condition in sorted(set(detail["condition"]) - {"compact best text MLP"}):
        candidate = detail.loc[detail.condition.eq(condition), ["fold", "bacc"]].set_index("fold")
        paired = baseline.join(candidate, how="inner", lsuffix="_base", rsuffix="_candidate")
        if len(paired) < 2:
            continue
        delta = 100 * (paired["bacc_candidate"] - paired["bacc_base"])
        statistic, pvalue = ttest_rel(paired["bacc_candidate"], paired["bacc_base"])
        comparisons.append({
            "condition": condition,
            "paired_folds": len(paired),
            "delta_pp_mean": float(delta.mean()),
            "delta_pp_std": float(delta.std()),
            "positive_folds": int((delta > 0).sum()),
            "paired_t": float(statistic),
            "paired_p": float(pvalue),
            **{f"delta_fold_{int(fold)}_pp": float(value) for fold, value in delta.items()},
        })
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(comparisons).to_csv(output_dir / "comparisons.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-features", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/best_text_mlp_rich_features")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, old_fourier_columns = load_fourier(frame, args.fourier_features)
    stems = frame["stem"].to_numpy(dtype=str)

    old_group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32)
        for name, columns in groups.items()
    }
    rich_group_blocks = dict(old_group_blocks)
    rich_group_blocks["Descriptors"] = load_aligned_npy(
        args.rich_dir / "descriptors" / "desc_full_features.npy",
        args.rich_dir / "descriptors" / "desc_full_stems.npy",
        stems,
    )
    rich_group_blocks["Hilbert"] = load_aligned_npy(
        args.rich_dir / "hilbert" / "hilbert_features.npy",
        args.rich_dir / "hilbert" / "hilbert_stems.npy",
        stems,
    )
    old_fourier = frame[old_fourier_columns].to_numpy(dtype=np.float32)
    rich_fourier = load_aligned_parquet(args.rich_fourier, stems, "fourier_rich")

    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    completed = {(str(row["condition"]), int(row["fold"])) for row in rows}

    fixed_extra_old = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": old_fourier,
    }
    fixed_extra_rich = {**fixed_extra_old, "Fourier": rich_fourier}
    log(
        f"Best text MLP rich test: {len(frame):,} images, {n_classes} styles; "
        f"old Fourier={old_fourier.shape[1]}d, rich Fourier={rich_fourier.shape[1]}d"
    )
    log(
        "Rich globals: Descriptors="
        f"{rich_group_blocks['Descriptors'].shape[1]}d, Hilbert={rich_group_blocks['Hilbert'].shape[1]}d"
    )

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        text = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        conditions = [
            ("compact best text MLP", BEST_TEXT_MASK, old_group_blocks, {**fixed_extra_old, "Text": text}),
            ("rich Descriptor+Fourier", BEST_TEXT_MASK, rich_group_blocks, {**fixed_extra_rich, "Text": text}),
            ("rich Descriptor+Fourier+Hilbert", BEST_TEXT_PLUS_HILBERT_MASK, rich_group_blocks, {**fixed_extra_rich, "Text": text}),
        ]
        for condition, mask, group_blocks, extra_blocks in conditions:
            if (condition, fold) in completed:
                continue
            blocks, names = selected_blocks(mask, group_blocks, extra_blocks)
            dimensions = int(sum(block.shape[1] for block in blocks))
            started = time.perf_counter()
            metrics = evaluate_mlp(blocks, labels, outer_train, outer_test, fold, args.epochs, args.patience)
            elapsed = time.perf_counter() - started
            rows.append({
                "fold": fold,
                "condition": condition,
                "mask": mask,
                "dimensions": dimensions,
                "branches": len(blocks),
                "block_names": " | ".join(names),
                "elapsed_seconds": elapsed,
                **metrics,
            })
            completed.add((condition, fold))
            save(rows, args.output_dir)
            log(
                f"fold={fold}/{N_FOLDS} {condition:<30} "
                f"dims={dimensions:<4d} branches={len(blocks):<2d} "
                f"BAcc={metrics['bacc']:.4f} val={metrics['validation_bacc']:.4f} "
                f"ep={metrics['best_epoch']}"
            )
    save(rows, args.output_dir)
    summary = pd.read_csv(args.output_dir / "summary.csv")
    log("\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
