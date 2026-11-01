#!/usr/bin/env python3
"""MLP probes on selected rich-representation top-k budgets.

Feature selection is still fold-local and supervised by the same LDA structure
coefficient used in the LDA top-k curve.  Each selected subset is then evaluated
with the project branch MLP.  The grid is intentionally sparse because each
point is a neural-network training run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

THREADS = os.environ.get("MLP_TOPK_THREADS", "8")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, THREADS)

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_robustness_mlp_aug import DEVICE, train_one  # noqa: E402

SEED = 42
N_FOLDS = 5
FAMILIES = ("Descriptors", "Hilbert", "Fourier")
K_GRID = (32, 64, 130, 256, 425, 608, 768, 1024, 1450)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def finite_copy(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    result = np.array(result, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes, dtype=np.float64) / n_classes,
        )),
    ])


def structure_importance(model: Pipeline, train_values: np.ndarray) -> np.ndarray:
    standardized = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(train_values)
    ).astype(np.float64)
    scores = model.named_steps["lda"].transform(standardized)
    scores -= scores.mean(axis=0)
    score_std = scores.std(axis=0)
    valid = score_std > 1e-12
    scores[:, valid] /= score_std[valid]
    scores[:, ~valid] = 0.0
    correlations = (standardized.T @ scores) / len(standardized)
    weights = model.named_steps["lda"].explained_variance_ratio_.astype(np.float64)
    weights = weights[: correlations.shape[1]]
    weights /= weights.sum()
    return np.sqrt(np.sum(correlations ** 2 * weights[None, :], axis=1))


def ranked_order(values: np.ndarray, labels: np.ndarray, train: np.ndarray) -> np.ndarray:
    model = make_lda(len(np.unique(labels)), values.shape[1])
    model.fit(values[train], labels[train])
    return np.argsort(-structure_importance(model, values[train]))


def labels_by_stem(meta_path: Path) -> dict[str, str]:
    meta = pd.read_csv(meta_path)
    stems = meta["filename"].astype(str).map(lambda value: Path(value).stem)
    return dict(zip(stems, meta["style"].astype(str), strict=False))


def sort_block(
    stems: np.ndarray, values: np.ndarray, labels: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stems = np.asarray(stems, dtype=str)
    keep = np.array([stem in labels for stem in stems], dtype=bool)
    stems = stems[keep]
    values = values[keep]
    order = np.argsort(stems)
    stems = stems[order]
    values = finite_copy(values[order])
    label_values = np.array([labels[stem] for stem in stems], dtype=str)
    return stems, label_values, values


def load_npy_block(
    values_path: Path, stems_path: Path, labels: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sort_block(
        np.load(stems_path, allow_pickle=True),
        np.load(values_path),
        labels,
    )


def load_fourier(path: Path, labels: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)
    feature_columns = [
        column for column in frame.columns
        if column not in {"stem", "style", "path", "filename"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return sort_block(
        frame["stem"].to_numpy(dtype=str),
        frame[feature_columns].to_numpy(dtype=np.float32),
        labels,
    )


def load_blocks(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = labels_by_stem(args.meta)
    rich = args.rich_dir
    loaded = {
        "Descriptors": load_npy_block(
            rich / "descriptors" / "desc_full_features.npy",
            rich / "descriptors" / "desc_full_stems.npy",
            labels,
        ),
        "Hilbert": load_npy_block(
            rich / "hilbert" / "hilbert_features.npy",
            rich / "hilbert" / "hilbert_stems.npy",
            labels,
        ),
        "Fourier": load_fourier(args.fourier_features, labels),
    }
    common = set(loaded["Descriptors"][0])
    for name in FAMILIES[1:]:
        common &= set(loaded[name][0])
    stems = np.array(sorted(common), dtype=str)
    label_values: np.ndarray | None = None
    blocks: dict[str, np.ndarray] = {}
    for name, (block_stems, block_labels, block_values) in loaded.items():
        index = {stem: row for row, stem in enumerate(block_stems)}
        rows = np.array([index[stem] for stem in stems], dtype=np.int64)
        current_labels = block_labels[rows]
        label_values = current_labels if label_values is None else label_values
        if not np.array_equal(label_values, current_labels):
            raise ValueError(f"Label mismatch after aligning {name}")
        blocks[name] = block_values[rows].astype(np.float32, copy=False)
        log(f"{name:<12} {blocks[name].shape[0]:,} images, {blocks[name].shape[1]}d")
    assert label_values is not None
    return label_values, blocks


def split_selection(
    family_orders: dict[str, np.ndarray],
    family_offsets: dict[str, int],
    family_widths: dict[str, int],
    k: int,
) -> np.ndarray:
    quotas = {name: k // len(FAMILIES) for name in FAMILIES}
    for name in FAMILIES[: k % len(FAMILIES)]:
        quotas[name] += 1
    overflow = 0
    for name in FAMILIES:
        if quotas[name] > family_widths[name]:
            overflow += quotas[name] - family_widths[name]
            quotas[name] = family_widths[name]
    while overflow:
        moved = False
        for name in FAMILIES:
            if quotas[name] < family_widths[name]:
                quotas[name] += 1
                overflow -= 1
                moved = True
                if overflow == 0:
                    break
        if not moved:
            break
    selected = []
    for name in FAMILIES:
        selected.extend((family_offsets[name] + family_orders[name][:quotas[name]]).tolist())
    return np.asarray(selected, dtype=np.int64)


def impute_scale(
    values: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median").fit(values[fit])
    fit_i = imputer.transform(values[fit])
    scaler = StandardScaler().fit(fit_i)
    return (
        scaler.transform(fit_i).astype(np.float32),
        scaler.transform(imputer.transform(values[validation])).astype(np.float32),
        scaler.transform(imputer.transform(values[test])).astype(np.float32),
    )


def evaluate_mlp_subset(
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    fold: int,
    epochs: int,
    patience: int,
) -> dict[str, float]:
    fit, validation = train_test_split(
        train, test_size=0.1, stratify=labels[train], random_state=SEED + fold
    )
    train_x, validation_x, test_x = impute_scale(values, fit, validation, test)
    metrics, prediction = train_one(
        train_x, labels[fit], validation_x, labels[validation], test_x, labels[test],
        [values.shape[1]], len(np.unique(labels)), SEED + fold, epochs, patience,
    )
    return {
        "bacc": float(metrics["bacc"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "validation_bacc": float(metrics["validation_bacc"]),
        "best_epoch": int(metrics["best_epoch"]),
    }


def save(rows: list[dict[str, object]], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["curve", "fold", "k"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby(["curve", "k", "total_features"], as_index=False).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        validation_bacc_mean=("validation_bacc", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        folds=("fold", "nunique"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    ).sort_values(["curve", "k"])
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rich_mlp_topk_curves")
    parser.add_argument("--k-grid", type=int, nargs="+", default=list(K_GRID))
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--curves", nargs="+", choices=("descriptors", "pooled", "split"), default=["descriptors", "pooled", "split"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_labels, blocks = load_blocks(args)
    labels = LabelEncoder().fit_transform(raw_labels)
    descriptor_values = blocks["Descriptors"]
    pooled_values = np.hstack([blocks[name] for name in FAMILIES]).astype(np.float32)
    family_widths = {name: blocks[name].shape[1] for name in FAMILIES}
    family_offsets: dict[str, int] = {}
    offset = 0
    for name in FAMILIES:
        family_offsets[name] = offset
        offset += family_widths[name]
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels), dtype=np.uint8), labels))
    result_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    done = {(str(row["curve"]), int(row["fold"]), int(row["k"])) for row in rows}
    log(f"MLP top-k curves: {len(labels):,} images, {len(np.unique(labels))} styles, device={DEVICE}")

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        desc_order = ranked_order(descriptor_values, labels, train)
        pooled_order = ranked_order(pooled_values, labels, train)
        family_orders = {name: ranked_order(blocks[name], labels, train) for name in FAMILIES}

        curve_specs: list[tuple[str, np.ndarray, dict[int, np.ndarray]]] = []
        if "descriptors" in args.curves:
            curve_specs.append((
                "Descriptors rich",
                descriptor_values,
                {k: desc_order[:k] for k in args.k_grid if k <= descriptor_values.shape[1]},
            ))
        if "pooled" in args.curves:
            curve_specs.append((
                "D+H+F pooled top-k",
                pooled_values,
                {k: pooled_order[:k] for k in args.k_grid if k <= pooled_values.shape[1]},
            ))
        if "split" in args.curves:
            curve_specs.append((
                "D+H+F split top-k",
                pooled_values,
                {k: split_selection(family_orders, family_offsets, family_widths, k) for k in args.k_grid if k <= pooled_values.shape[1]},
            ))

        for curve, values, selections in curve_specs:
            for k, selected in selections.items():
                if (curve, fold, k) in done:
                    continue
                started = time.perf_counter()
                subset = values[:, selected]
                metrics = evaluate_mlp_subset(subset, labels, train, test, fold, args.epochs, args.patience)
                rows.append({
                    "curve": curve,
                    "fold": fold,
                    "k": k,
                    "selected_features": len(selected),
                    "total_features": values.shape[1],
                    "elapsed_seconds": time.perf_counter() - started,
                    **metrics,
                })
                done.add((curve, fold, k))
                save(rows, args.output_dir)
                log(
                    f"fold={fold}/{N_FOLDS} {curve:<22} k={k:<4d} "
                    f"BAcc={metrics['bacc']:.4f} val={metrics['validation_bacc']:.4f} "
                    f"ep={metrics['best_epoch']}"
                )

    save(rows, args.output_dir)
    summary = pd.read_csv(args.output_dir / "summary.csv")
    log(f"Saved: {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
