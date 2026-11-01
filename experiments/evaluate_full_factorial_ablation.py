"""Exact 2^7 factorial ablation for the final interpretable image pipeline.

Factors toggle SLIC, local descriptors, Hilbert statistics, line graph,
Felzenszwalb graph, Fourier spectrum, and multiscale views.  Multiscale means
global plus four fixed quadrants for every active visual family.  When all
three visual families are off, the multiscale bit is an alias and is evaluated
only once.  The empty representation is assigned balanced chance (1 / 27).
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from evaluate_robustness_mlp_aug import train_one
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, REGIONS, load_dataset
# Splits e sondas moram em src/pipeline/protocol.py desde 06/set. Nasceram
# neste arquivo e eram importados daqui por 51 scripts -- seguem re-exportados.
from protocol import (  # noqa: E402,F401
    scaled_splits, prediction_metrics, evaluate_lda, evaluate_mlp,
)


FACTORS = (
    "SLIC",
    "Descriptors",
    "Hilbert",
    "Line graph",
    "Felzenszwalb",
    "Fourier",
    "Multiscale",
)
CLASSIFIERS = ("knn", "random_forest", "mlp", "lda")
VISUAL_FACTORS = FACTORS[:3]
MULTISCALE_BIT = 1 << FACTORS.index("Multiscale")
VISUAL_MASK = sum(1 << FACTORS.index(name) for name in VISUAL_FACTORS)
FULL_MASK = (1 << len(FACTORS)) - 1
K_VALUES = (5, 15, 31, 63)
KNN_WEIGHTS = ("uniform", "distance")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def canonical_mask(mask: int) -> int:
    """Collapse the irrelevant multiscale bit when no visual family is active."""
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
    for factor in FACTORS[3:6]:
        if mask & (1 << FACTORS.index(factor)):
            names.append(factor)
    lookup = {**group_blocks, **extra_blocks}
    return [lookup[name] for name in names], names


def neighbor_query(
    train_x: np.ndarray, query_x: np.ndarray, workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(
        n_neighbors=max(K_VALUES), metric="minkowski", p=2, n_jobs=workers
    )
    model.fit(train_x)
    return model.kneighbors(query_x, return_distance=True)


def neighbor_prediction(
    distances: np.ndarray,
    indices: np.ndarray,
    train_labels: np.ndarray,
    n_classes: int,
    k: int,
    weights: str,
) -> np.ndarray:
    labels = train_labels[indices[:, :k]]
    vote_weights = (
        np.ones_like(distances[:, :k], dtype=np.float64)
        if weights == "uniform"
        else 1.0 / np.maximum(distances[:, :k], 1e-8)
    )
    votes = np.zeros((len(labels), n_classes), dtype=np.float64)
    rows = np.repeat(np.arange(len(labels)), k)
    np.add.at(votes, (rows, labels.ravel()), vote_weights.ravel())
    return votes.argmax(axis=1)


def evaluate_knn(
    blocks: list[np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    workers: int,
) -> dict[str, object]:
    fit, validation = train_test_split(
        outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
    )
    fit_x, validation_x, _ = scaled_splits(blocks, fit, validation, outer_test)
    assert validation_x is not None
    distances, indices = neighbor_query(fit_x, validation_x, workers)
    candidates = []
    n_classes = len(np.unique(labels))
    for k in K_VALUES:
        for weights in KNN_WEIGHTS:
            prediction = neighbor_prediction(distances, indices, labels[fit], n_classes, k, weights)
            candidates.append(
                (float(balanced_accuracy_score(labels[validation], prediction)), k, weights)
            )
    validation_bacc, selected_k, selected_weights = max(candidates)
    del fit_x, validation_x, distances, indices

    train_x, _, test_x = scaled_splits(blocks, outer_train, None, outer_test)
    distances, indices = neighbor_query(train_x, test_x, workers)
    prediction = neighbor_prediction(
        distances, indices, labels[outer_train], n_classes, selected_k, selected_weights
    )
    return {
        **prediction_metrics(labels[outer_test], prediction),
        "validation_bacc": validation_bacc,
        "selected_k": selected_k,
        "selected_weights": selected_weights,
        "best_epoch": np.nan,
    }


def evaluate_random_forest(
    blocks: list[np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    workers: int,
) -> dict[str, object]:
    train_x = np.hstack([block[outer_train] for block in blocks]).astype(np.float32, copy=False)
    test_x = np.hstack([block[outer_test] for block in blocks]).astype(np.float32, copy=False)
    model = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced_subsample", n_jobs=workers, random_state=SEED + fold,
    )
    model.fit(train_x, labels[outer_train])
    prediction = model.predict(test_x)
    return {
        **prediction_metrics(labels[outer_test], prediction),
        "validation_bacc": np.nan, "selected_k": np.nan,
        "selected_weights": "", "best_epoch": np.nan,
    }


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

    shapley_rows, main_rows, interaction_rows = [], [], []
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
            for first, second in combinations(range(len(FACTORS)), 2):
                first_bit, second_bit = 1 << first, 1 << second
                differences = []
                for subset in range(1 << len(FACTORS)):
                    if subset & (first_bit | second_bit):
                        continue
                    differences.append(
                        values[subset | first_bit | second_bit]
                        - values[subset | first_bit]
                        - values[subset | second_bit]
                        + values[subset]
                    )
                interaction_rows.append({
                    "classifier": classifier, "fold": fold,
                    "factor_a": FACTORS[first], "factor_b": FACTORS[second],
                    "mean_second_difference_bacc": float(np.mean(differences)),
                    "mean_second_difference_pp": 100 * float(np.mean(differences)),
                })

    shapley = pd.DataFrame(shapley_rows)
    main = pd.DataFrame(main_rows)
    interactions = pd.DataFrame(interaction_rows)
    shapley.to_csv(output_dir / "shapley_by_fold.csv", index=False, float_format="%.8f")
    main.to_csv(output_dir / "main_effects_by_fold.csv", index=False, float_format="%.8f")
    interactions.to_csv(
        output_dir / "pairwise_interactions_by_fold.csv", index=False, float_format="%.8f"
    )
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
    if len(interactions):
        interactions.groupby(["classifier", "factor_a", "factor_b"], as_index=False).agg(
            folds=("fold", "nunique"),
            interaction_pp_mean=("mean_second_difference_pp", "mean"),
            interaction_pp_std=("mean_second_difference_pp", "std"),
        ).to_csv(
            output_dir / "pairwise_interactions_summary.csv", index=False, float_format="%.8f"
        )


def save(unique_rows: list[dict], output_dir: Path) -> None:
    unique = pd.DataFrame(unique_rows).sort_values(["classifier", "fold", "mask"])
    unique.to_csv(output_dir / "unique_fold_results.csv", index=False)
    expanded = expand_aliases(unique)
    expanded.to_csv(output_dir / "factorial_fold_results.csv", index=False)
    write_analysis(expanded, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS,
    )
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
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/full_factorial_ablation"
    )
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--only-mask", type=int, choices=range(1 << len(FACTORS)))
    parser.add_argument("--workers", type=int, default=12)
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
    extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "unique_fold_results.csv"
    if result_path.exists() and not args.restart:
        rows = pd.read_csv(result_path).to_dict("records")
    else:
        rows = []
    completed = {
        (str(row["classifier"]), int(row["fold"]), int(row["mask"])) for row in rows
    }

    all_masks = sorted({canonical_mask(mask) for mask in range(1 << len(FACTORS))})
    if args.only_mask is not None:
        all_masks = [canonical_mask(args.only_mask)]
    log(
        f"Full factorial: {len(frame):,} images, {n_classes} styles, "
        f"128 logical masks / {len(set(canonical_mask(mask) for mask in range(128)))} unique; "
        f"classifiers={','.join(args.classifiers)}"
    )

    for classifier in args.classifiers:
        for mask in all_masks:
            blocks, block_names = selected_blocks(mask, group_blocks, extra_blocks)
            dimensions = sum(block.shape[1] for block in blocks)
            record_base = {
                **mask_record(mask), "classifier": classifier,
                "dimensions": dimensions, "branches": len(blocks),
                "block_names": " | ".join(block_names), "source_mask": mask,
            }
            for fold, (outer_train, outer_test) in enumerate(splits, start=1):
                if args.only_fold is not None and fold != args.only_fold:
                    continue
                key = classifier, fold, mask
                if key in completed:
                    continue
                if mask == 0:
                    metrics = {
                        "bacc": 1.0 / n_classes, "accuracy": np.nan, "macro_f1": np.nan,
                        "validation_bacc": np.nan, "selected_k": np.nan,
                        "selected_weights": "", "best_epoch": np.nan,
                    }
                    elapsed = 0.0
                else:
                    started = time.perf_counter()
                    if classifier == "knn":
                        metrics = evaluate_knn(
                            blocks, labels, outer_train, outer_test, fold, args.workers
                        )
                    elif classifier == "random_forest":
                        metrics = evaluate_random_forest(
                            blocks, labels, outer_train, outer_test, fold, args.workers
                        )
                    elif classifier == "mlp":
                        metrics = evaluate_mlp(
                            blocks, labels, outer_train, outer_test, fold,
                            args.epochs, args.patience,
                        )
                    elif classifier == "lda":
                        metrics = evaluate_lda(blocks, labels, outer_train, outer_test)
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
                    f"{classifier:<13} mask={mask:03d}/{FULL_MASK} fold={fold}/{N_FOLDS} "
                    f"dims={dimensions:4d} BAcc={metrics['bacc']:.4f} "
                    f"seconds={elapsed:.1f}"
                )
                gc.collect()
    save(rows, args.output_dir)
    log(f"Saved factorial study to {args.output_dir}")


if __name__ == "__main__":
    main()
