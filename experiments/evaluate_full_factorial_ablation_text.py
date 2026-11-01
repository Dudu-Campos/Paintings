"""Exact 2^8 factorial ablation: the original seven visual/structural factors
plus TF-IDF caption text as an eighth modality.

Reuses the cached, leak-free, nested-OOF text scores from
`cache_tfidf_text_scores.py` (must be built first) instead of recomputing the
LinearSVC/TF-IDF fit for every mask. Text is treated like Line graph,
Felzenszwalb and Fourier: a simple block toggled on/off, not subject to the
Multiscale bit (there is only one caption per painting, no quadrant text).

Classifiers restricted to LDA and MLP (dropped knn/random_forest, the slow
ones per-sample) to keep the 256-mask x 5-fold sweep tractable.
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import argparse
import gc
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(int(THREADS_PER_WORKER))

from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


FACTORS = (
    "SLIC", "Descriptors", "Hilbert", "Line graph", "Felzenszwalb",
    "Fourier", "Text", "Multiscale",
)
CLASSIFIERS = ("lda", "mlp")
VISUAL_FACTORS = FACTORS[:3]
MULTISCALE_BIT = 1 << FACTORS.index("Multiscale")
SIMPLE_FACTORS = FACTORS[3:FACTORS.index("Multiscale")]
VISUAL_MASK = sum(1 << FACTORS.index(name) for name in VISUAL_FACTORS)
FULL_MASK = (1 << len(FACTORS)) - 1


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def canonical_mask(mask: int) -> int:
    return mask & ~MULTISCALE_BIT if mask & VISUAL_MASK == 0 else mask


def mask_record(mask: int) -> dict[str, object]:
    active = [name for index, name in enumerate(FACTORS) if mask & (1 << index)]
    canonical = canonical_mask(mask)
    return {
        "mask": mask,
        "mask_binary": format(mask, f"0{len(FACTORS)}b"),
        "canonical_mask": canonical,
        "is_alias": mask != canonical,
        "n_active_factors": len(active),
        "condition": " + ".join(active) if active else "Empty (chance)",
        **{f"use_{name.lower().replace(' ', '_')}": bool(mask & (1 << index))
           for index, name in enumerate(FACTORS)},
    }


def selected_blocks(
    mask: int,
    group_blocks: dict[str, np.ndarray],
    extra_blocks: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[str]]:
    active_visual = {
        name for index, name in enumerate(VISUAL_FACTORS) if mask & (1 << index)
    }
    use_multiscale = bool(mask & MULTISCALE_BIT)
    names = []
    for name in group_blocks:
        family = name.split(" (", maxsplit=1)[0]
        is_global = " (" not in name
        if family in active_visual and (is_global or use_multiscale):
            names.append(name)
    for factor in SIMPLE_FACTORS:
        if mask & (1 << FACTORS.index(factor)):
            names.append(factor)
    lookup = {**group_blocks, **extra_blocks}
    return [lookup[name] for name in names], names


def load_text_block(cache_dir: Path, fold: int, frame: pd.DataFrame, n_classes: int) -> np.ndarray:
    table = pd.read_parquet(cache_dir / f"fold_{fold}.parquet")
    score_columns = [f"score_{index}" for index in range(n_classes)]
    merged = frame[["stem"]].merge(table[["stem", *score_columns]], on="stem", how="left", validate="one_to_one")
    if merged[score_columns].isna().any().any():
        raise ValueError(f"Text cache for fold {fold} is missing rows for the canonical cohort")
    return merged[score_columns].to_numpy(dtype=np.float32)


def expand_aliases(unique: pd.DataFrame) -> pd.DataFrame:
    records = []
    lookup = {
        (str(row.classifier), int(row.fold), int(row.mask)): row
        for row in unique.itertuples(index=False)
    }
    classifiers = sorted(unique.classifier.astype(str).unique()) if len(unique) else []
    for classifier in classifiers:
        folds = sorted(unique.loc[unique.classifier.eq(classifier), "fold"].astype(int).unique())
        for fold in folds:
            for mask in range(1 << len(FACTORS)):
                source = lookup.get((classifier, fold, canonical_mask(mask)))
                if source is None:
                    continue
                record = source._asdict()
                record.update(mask_record(mask))
                record["source_mask"] = canonical_mask(mask)
                records.append(record)
    return pd.DataFrame(records).sort_values(["classifier", "fold", "mask"])


def exact_shapley(values: dict[int, float]) -> dict[str, float]:
    n = len(FACTORS)
    result = {}
    denominator = math.factorial(n)
    for index, factor in enumerate(FACTORS):
        bit = 1 << index
        contribution = 0.0
        for subset in range(1 << n):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
            contribution += weight * (values[subset | bit] - values[subset])
        result[factor] = contribution
    return result


def write_analysis(expanded: pd.DataFrame, output_dir: Path) -> None:
    if expanded.empty:
        return
    summaries = expanded.groupby(["classifier", "mask"], as_index=False).agg(
        folds_completed=("fold", "nunique"), bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"), accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"), dimensions=("dimensions", "first"),
        branches=("branches", "first"), condition=("condition", "first"),
        canonical_mask=("canonical_mask", "first"), is_alias=("is_alias", "first"),
        n_active_factors=("n_active_factors", "first"),
    )
    summaries.to_csv(output_dir / "configuration_summary.csv", index=False, float_format="%.8f")
    configuration_columns = [
        "mask", "mask_binary", "canonical_mask", "is_alias", "n_active_factors", "condition"
    ] + [f"use_{name.lower().replace(' ', '_')}" for name in FACTORS]
    pd.DataFrame([mask_record(mask) for mask in range(1 << len(FACTORS))])[
        configuration_columns
    ].to_csv(output_dir / "configuration_table.csv", index=False)

    shapley_rows, main_rows = [], []
    for classifier, classifier_rows in expanded.groupby("classifier"):
        complete_folds = [
            fold for fold, fold_rows in classifier_rows.groupby("fold")
            if fold_rows["mask"].nunique() == (1 << len(FACTORS))
        ]
        for fold in complete_folds:
            fold_rows = classifier_rows[classifier_rows.fold.eq(fold)]
            values = dict(zip(fold_rows["mask"].astype(int), fold_rows["bacc"].astype(float)))
            for factor, contribution in exact_shapley(values).items():
                shapley_rows.append({
                    "classifier": classifier, "fold": fold, "factor": factor,
                    "shapley_bacc": contribution, "shapley_pp": 100 * contribution,
                })
            for index, factor in enumerate(FACTORS):
                bit = 1 << index
                on = [value for mask, value in values.items() if mask & bit]
                off = [value for mask, value in values.items() if not mask & bit]
                main_rows.append({
                    "classifier": classifier, "fold": fold, "factor": factor,
                    "main_effect_bacc": float(np.mean(on) - np.mean(off)),
                    "main_effect_pp": 100 * float(np.mean(on) - np.mean(off)),
                })
    shapley = pd.DataFrame(shapley_rows)
    main = pd.DataFrame(main_rows)
    shapley.to_csv(output_dir / "shapley_by_fold.csv", index=False, float_format="%.8f")
    main.to_csv(output_dir / "main_effects_by_fold.csv", index=False, float_format="%.8f")
    if len(shapley):
        shapley.groupby(["classifier", "factor"], as_index=False).agg(
            folds=("fold", "nunique"), shapley_pp_mean=("shapley_pp", "mean"),
            shapley_pp_std=("shapley_pp", "std"),
        ).to_csv(output_dir / "shapley_summary.csv", index=False, float_format="%.8f")
    if len(main):
        main.groupby(["classifier", "factor"], as_index=False).agg(
            folds=("fold", "nunique"), main_effect_pp_mean=("main_effect_pp", "mean"),
            main_effect_pp_std=("main_effect_pp", "std"),
        ).to_csv(output_dir / "main_effects_summary.csv", index=False, float_format="%.8f")


def save(unique_rows: list[dict], output_dir: Path) -> None:
    unique = pd.DataFrame(unique_rows).sort_values(["classifier", "fold", "mask"])
    unique.to_csv(output_dir / "unique_fold_results.csv", index=False)
    expanded = expand_aliases(unique)
    expanded.to_csv(output_dir / "factorial_fold_results.csv", index=False)
    write_analysis(expanded, output_dir)


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
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/full_factorial_ablation_text"
    )
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--only-mask", type=int, choices=range(1 << len(FACTORS)))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "unique_fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() and not args.restart else []
    completed = {
        (str(row["classifier"]), int(row["fold"]), int(row["mask"])) for row in rows
    }

    all_masks = sorted({canonical_mask(mask) for mask in range(1 << len(FACTORS))})
    if args.only_mask is not None:
        all_masks = [canonical_mask(args.only_mask)]
    log(
        f"Full factorial + text: {len(frame):,} images, {n_classes} styles, "
        f"{1 << len(FACTORS)} logical masks / {len(all_masks)} unique; "
        f"classifiers={','.join(args.classifiers)}"
    )

    fold_indices = {
        fold: (outer_train, outer_test)
        for fold, (outer_train, outer_test) in enumerate(splits, start=1)
        if args.only_fold is None or fold == args.only_fold
    }
    text_blocks: dict[int, np.ndarray] = {}

    # Classifier-outer, fold-inner: cheap classifiers (e.g. LDA) finish all
    # five folds -- and therefore get a full five-fold Shapley estimate --
    # long before slow classifiers (e.g. MLP) get through a single fold.
    for classifier in args.classifiers:
        for fold, (outer_train, outer_test) in fold_indices.items():
            if all((classifier, fold, mask) in completed for mask in all_masks):
                continue
            if fold not in text_blocks:
                text_blocks[fold] = load_text_block(args.text_cache_dir, fold, frame, n_classes)
            extra_blocks = {**fixed_extra_blocks, "Text": text_blocks[fold]}

            for mask in all_masks:
                key = classifier, fold, mask
                if key in completed:
                    continue
                blocks, block_names = selected_blocks(mask, group_blocks, extra_blocks)
                dimensions = sum(block.shape[1] for block in blocks)
                record_base = {
                    **mask_record(mask), "classifier": classifier,
                    "dimensions": dimensions, "branches": len(blocks),
                    "block_names": " | ".join(block_names), "source_mask": mask,
                }
                if mask == 0:
                    metrics = {
                        "bacc": 1.0 / n_classes, "accuracy": np.nan, "macro_f1": np.nan,
                        "validation_bacc": np.nan, "best_epoch": np.nan,
                    }
                    elapsed = 0.0
                else:
                    started = time.perf_counter()
                    if classifier == "lda":
                        metrics = evaluate_lda(blocks, labels, outer_train, outer_test)
                    elif classifier == "mlp":
                        metrics = evaluate_mlp(
                            blocks, labels, outer_train, outer_test, fold,
                            args.epochs, args.patience,
                        )
                    else:  # pragma: no cover
                        raise ValueError(classifier)
                    elapsed = time.perf_counter() - started
                rows.append({
                    **record_base, "fold": fold, "n_images": len(frame),
                    "n_train": len(outer_train), "n_test": len(outer_test),
                    "elapsed_seconds": elapsed, **metrics,
                })
                completed.add(key)
                save(rows, args.output_dir)
                log(
                    f"{classifier:<5} mask={mask:03d}/{FULL_MASK} fold={fold}/{N_FOLDS} "
                    f"dims={dimensions:4d} BAcc={metrics['bacc']:.4f} seconds={elapsed:.1f}"
                )
                gc.collect()
    save(rows, args.output_dir)
    log(f"Saved factorial+text study to {args.output_dir}")


if __name__ == "__main__":
    main()
