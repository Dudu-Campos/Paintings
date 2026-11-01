#!/usr/bin/env python3
"""Evaluate compact-vs-rich feature ladders for single-source representations.

The rich vectors are intentionally nested: the first columns reproduce the
historical compact representation.  This script measures the effect of the
added within-family measurements with the canonical WikiArt 5-fold split.

Outputs are written incrementally so the run can be resumed while extractors
finish producing Hilbert/Fourier rich features.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS = os.environ.get("LADDER_THREADS", "8")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, THREADS)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_robustness_mlp_aug import DEVICE, train_one  # noqa: E402

SEED = 42
N_FOLDS = 5
COMPACT_DIMS = {"Descriptors": 130, "Hilbert": 57, "Fourier": 152}
DEFAULT_FAMILIES = ("Descriptors", "Hilbert", "Fourier")
PCA_GRID = (1, 2, 4, 8, 16, 32, 57, 64, 96, 128, 130, 152, 192, 256, 384, 417, 425, 512, 608, 768, 1024, 1450)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def finite_copy(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    result = np.array(result, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


def load_labels(meta_path: Path) -> dict[str, str]:
    meta = pd.read_csv(meta_path)
    stems = meta["filename"].astype(str).map(lambda value: Path(value).stem)
    return dict(zip(stems, meta["style"].astype(str), strict=False))


def sort_and_label(
    stems: np.ndarray, values: np.ndarray, labels_by_stem: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stems = np.asarray(stems, dtype=str)
    keep = np.array([stem in labels_by_stem for stem in stems], dtype=bool)
    stems = stems[keep]
    values = values[keep]
    order = np.argsort(stems)
    stems = stems[order]
    values = finite_copy(values[order])
    labels = np.array([labels_by_stem[stem] for stem in stems], dtype=str)
    return stems, values, labels


def load_npy_family(
    name: str, feature_path: Path, stems_path: Path, labels_by_stem: dict[str, str]
) -> dict[str, object] | None:
    if not feature_path.exists() or not stems_path.exists():
        log(f"{name}: missing {feature_path} or {stems_path}; skipping for now")
        return None
    values = np.load(feature_path)
    stems = np.load(stems_path, allow_pickle=True)
    stems, values, labels = sort_and_label(stems, values, labels_by_stem)
    return {"name": name, "stems": stems, "values": values, "labels": labels}


def load_fourier(path: Path, labels_by_stem: dict[str, str]) -> dict[str, object] | None:
    if not path.exists():
        log(f"Fourier: missing {path}; skipping for now")
        return None
    frame = pd.read_parquet(path)
    feature_columns = [
        column for column in frame.columns
        if column not in {"stem", "style", "path", "filename"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise ValueError(f"No numeric Fourier feature columns found in {path}")
    stems, values, labels = sort_and_label(
        frame["stem"].to_numpy(dtype=str),
        frame[feature_columns].to_numpy(dtype=np.float32),
        labels_by_stem,
    )
    return {"name": "Fourier", "stems": stems, "values": values, "labels": labels}


def load_families(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    labels_by_stem = load_labels(args.meta)
    rich = args.rich_dir
    loaders = {
        "Descriptors": lambda: load_npy_family(
            "Descriptors",
            rich / "descriptors" / "desc_full_features.npy",
            rich / "descriptors" / "desc_full_stems.npy",
            labels_by_stem,
        ),
        "Hilbert": lambda: load_npy_family(
            "Hilbert",
            rich / "hilbert" / "hilbert_features.npy",
            rich / "hilbert" / "hilbert_stems.npy",
            labels_by_stem,
        ),
        "Fourier": lambda: load_fourier(args.fourier_features, labels_by_stem),
    }
    families: dict[str, dict[str, object]] = {}
    for name in args.families:
        loaded = loaders[name]()
        if loaded is None:
            continue
        expected = COMPACT_DIMS[name]
        n_features = int(loaded["values"].shape[1])
        if n_features <= expected:
            log(f"{name}: only {n_features}d available; compact width is {expected}d")
        log(f"{name}: {len(loaded['stems']):,} images, {n_features}d ({expected}d compact)")
        families[name] = loaded
    return families


def aligned_single_family(data: dict[str, object], compact: bool) -> tuple[str, np.ndarray, np.ndarray, list[int]]:
    name = str(data["name"])
    width = COMPACT_DIMS[name] if compact else int(data["values"].shape[1])
    label = f"{name} {'compact' if compact else 'rich'}"
    return label, data["values"][:, :width], data["labels"], [width]


def aligned_fusion(
    families: dict[str, dict[str, object]], names: list[str], compact: bool
) -> tuple[str, np.ndarray, np.ndarray, list[int]]:
    common = set(families[names[0]]["stems"].astype(str))
    for name in names[1:]:
        common &= set(families[name]["stems"].astype(str))
    stems = np.array(sorted(common), dtype=str)
    blocks, branch_sizes = [], []
    labels = None
    for name in names:
        data = families[name]
        index = {stem: i for i, stem in enumerate(data["stems"].astype(str))}
        rows = np.array([index[stem] for stem in stems], dtype=np.int64)
        width = COMPACT_DIMS[name] if compact else int(data["values"].shape[1])
        blocks.append(data["values"][rows, :width])
        branch_sizes.append(width)
        current_labels = data["labels"][rows]
        labels = current_labels if labels is None else labels
        if not np.array_equal(labels, current_labels):
            raise ValueError(f"Label mismatch while aligning {name}")
    label = f"{'+'.join(names)} {'compact' if compact else 'rich'}"
    return label, np.hstack(blocks).astype(np.float32), labels, branch_sizes


def make_splits(labels: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    encoded = LabelEncoder().fit_transform(labels)
    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(encoded), dtype=np.uint8), encoded
        )
    )
    return encoded, splits


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def lda_model(n_classes: int, n_features: int) -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(
        n_components=min(n_classes - 1, n_features),
        priors=np.ones(n_classes, dtype=np.float64) / n_classes,
    )


def evaluate_lda_full(values: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict[str, object]:
    n_classes = len(np.unique(labels))
    imputer = SimpleImputer(strategy="median").fit(values[train])
    train_i = imputer.transform(values[train])
    test_i = imputer.transform(values[test])
    scaler = StandardScaler().fit(train_i)
    train_x = scaler.transform(train_i)
    test_x = scaler.transform(test_i)
    model = lda_model(n_classes, values.shape[1])
    model.fit(train_x, labels[train])
    return {**prediction_metrics(labels[test], model.predict(test_x)), "k": values.shape[1]}


def pca_grid(max_k: int) -> list[int]:
    grid = [k for k in PCA_GRID if k <= max_k]
    if not grid or grid[-1] != max_k:
        grid.append(max_k)
    return sorted(set(grid))


def evaluate_lda_pca(
    values: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray
) -> list[dict[str, object]]:
    n_classes = len(np.unique(labels))
    max_k = min(values.shape[1], len(train) - n_classes)
    imputer = SimpleImputer(strategy="median").fit(values[train])
    train_i = imputer.transform(values[train])
    test_i = imputer.transform(values[test])
    scaler = StandardScaler().fit(train_i)
    train_x = scaler.transform(train_i)
    test_x = scaler.transform(test_i)
    reducer = PCA(n_components=max_k, svd_solver="randomized", random_state=SEED).fit(train_x)
    train_p = reducer.transform(train_x)
    test_p = reducer.transform(test_x)
    rows = []
    for k in pca_grid(max_k):
        model = lda_model(n_classes, k)
        model.fit(train_p[:, :k], labels[train])
        rows.append({
            **prediction_metrics(labels[test], model.predict(test_p[:, :k])),
            "k": k,
            "explained_variance": float(reducer.explained_variance_ratio_[:k].sum()),
        })
    return rows


def impute_scale_blocks(
    values: np.ndarray,
    branch_sizes: list[int],
    fit: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts, validation_parts, test_parts = [], [], []
    start = 0
    for size in branch_sizes:
        block = values[:, start:start + size]
        start += size
        imputer = SimpleImputer(strategy="median").fit(block[fit])
        fit_i = imputer.transform(block[fit])
        scaler = StandardScaler().fit(fit_i)
        train_parts.append(scaler.transform(fit_i).astype(np.float32))
        validation_parts.append(scaler.transform(imputer.transform(block[validation])).astype(np.float32))
        test_parts.append(scaler.transform(imputer.transform(block[test])).astype(np.float32))
    return (
        np.hstack(train_parts).astype(np.float32),
        np.hstack(validation_parts).astype(np.float32),
        np.hstack(test_parts).astype(np.float32),
    )


def evaluate_mlp(
    values: np.ndarray,
    labels: np.ndarray,
    branch_sizes: list[int],
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    epochs: int,
    patience: int,
) -> dict[str, object]:
    fit, validation = train_test_split(
        outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
    )
    train_x, validation_x, test_x = impute_scale_blocks(values, branch_sizes, fit, validation, outer_test)
    metrics, _ = train_one(
        train_x, labels[fit], validation_x, labels[validation], test_x, labels[outer_test],
        branch_sizes, len(np.unique(labels)), SEED + fold, epochs, patience,
    )
    return {**metrics, "k": values.shape[1]}


def save_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["classifier", "condition", "fold", "k"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby(
        ["classifier", "condition", "representation_set", "variant", "dimensions", "k"],
        as_index=False,
    ).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        folds=("fold", "nunique"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    peaks = detail.loc[detail.groupby(["classifier", "condition", "fold"])["bacc"].idxmax()]
    peaks.to_csv(output_dir / "peaks.csv", index=False, float_format="%.6f")
    peak_summary = peaks.groupby(
        ["classifier", "condition", "representation_set", "variant", "dimensions"],
        as_index=False,
    ).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        k_mean=("k", "mean"),
        folds=("fold", "nunique"),
    )
    peak_summary.to_csv(output_dir / "peak_summary.csv", index=False, float_format="%.6f")


def add_result(
    rows: list[dict[str, object]],
    output_dir: Path,
    classifier: str,
    representation_set: str,
    variant: str,
    condition: str,
    dimensions: int,
    fold: int,
    metrics: dict[str, object],
    elapsed: float,
) -> None:
    rows.append({
        "classifier": classifier,
        "representation_set": representation_set,
        "variant": variant,
        "condition": condition,
        "dimensions": dimensions,
        "fold": fold,
        "elapsed_seconds": elapsed,
        **metrics,
    })
    save_outputs(rows, output_dir)


def completed_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, int]]:
    return {
        (str(row["classifier"]), str(row["condition"]), int(row["fold"]))
        for row in rows
        if str(row["classifier"]) in {"lda_full", "mlp"}
    }


def completed_pca_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, int, int]]:
    return {
        (str(row["classifier"]), str(row["condition"]), int(row["fold"]), int(row["k"]))
        for row in rows
        if str(row["classifier"]) == "lda_pca"
    }


def evaluate_condition(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    condition: str,
    representation_set: str,
    variant: str,
    values: np.ndarray,
    raw_labels: np.ndarray,
    branch_sizes: list[int],
) -> None:
    labels, splits = make_splits(raw_labels)
    log(
        f"=== {condition}: {len(labels):,} images, {len(np.unique(labels))} styles, "
        f"{values.shape[1]}d, branches={branch_sizes}"
    )
    done_simple = completed_keys(rows)
    done_pca = completed_pca_keys(rows)
    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        if "lda_full" in args.classifiers and ("lda_full", condition, fold) not in done_simple:
            started = time.perf_counter()
            metrics = evaluate_lda_full(values, labels, train, test)
            add_result(rows, args.output_dir, "lda_full", representation_set, variant, condition, values.shape[1], fold, metrics, time.perf_counter() - started)
            log(f"lda_full fold={fold}/{N_FOLDS} {condition:<32} BAcc={metrics['bacc']:.4f}")
        if "lda_pca" in args.classifiers:
            started = time.perf_counter()
            pca_rows = evaluate_lda_pca(values, labels, train, test)
            for metrics in pca_rows:
                key = ("lda_pca", condition, fold, int(metrics["k"]))
                if key in done_pca:
                    continue
                add_result(rows, args.output_dir, "lda_pca", representation_set, variant, condition, values.shape[1], fold, metrics, time.perf_counter() - started)
            best = max(row["bacc"] for row in pca_rows)
            log(f"lda_pca  fold={fold}/{N_FOLDS} {condition:<32} peak={best:.4f}")
        if "mlp" in args.classifiers and ("mlp", condition, fold) not in done_simple:
            started = time.perf_counter()
            metrics = evaluate_mlp(values, labels, branch_sizes, train, test, fold, args.epochs, args.patience)
            add_result(rows, args.output_dir, "mlp", representation_set, variant, condition, values.shape[1], fold, metrics, time.perf_counter() - started)
            log(
                f"mlp      fold={fold}/{N_FOLDS} {condition:<32} "
                f"BAcc={metrics['bacc']:.4f} val={metrics['validation_bacc']:.4f} "
                f"ep={metrics['best_epoch']} device={DEVICE}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rich_feature_ladders")
    parser.add_argument("--families", nargs="+", choices=DEFAULT_FAMILIES, default=list(DEFAULT_FAMILIES))
    parser.add_argument("--classifiers", nargs="+", choices=("lda_full", "lda_pca", "mlp"), default=["lda_full", "lda_pca"])
    parser.add_argument("--fusion", action="store_true", help="Also evaluate compact/rich fusion of all currently available families.")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    families = load_families(args)
    if not families:
        raise SystemExit("No requested rich feature families are available yet.")

    for name, data in families.items():
        for compact in (True, False):
            condition, values, labels, branch_sizes = aligned_single_family(data, compact)
            evaluate_condition(
                args, rows, condition, name, "compact" if compact else "rich",
                values, labels, branch_sizes,
            )

    if args.fusion and len(families) >= 2:
        names = [name for name in args.families if name in families]
        for compact in (True, False):
            condition, values, labels, branch_sizes = aligned_fusion(families, names, compact)
            evaluate_condition(
                args, rows, condition, "+".join(names), "compact" if compact else "rich",
                values, labels, branch_sizes,
            )

    save_outputs(rows, args.output_dir)
    peak_summary = pd.read_csv(args.output_dir / "peak_summary.csv")
    log("Peak summary:")
    print(peak_summary.sort_values(["classifier", "representation_set", "variant"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
