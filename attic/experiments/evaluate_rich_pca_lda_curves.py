#!/usr/bin/env python3
"""Fold-local PCA + LDA curves for the rich single-source representations.

This is the budget-matched counterpart to the supervised top-k curves.  The
horizontal axis is the number of PCA dimensions made available to LDA.

Curves:
  1. Descriptors rich only.
  2. Descriptors + Hilbert + Fourier rich compressed as one pooled matrix.
  3. The same three families compressed separately, splitting the k-budget
     across families and concatenating the fold-local PCA scores.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS = os.environ.get("PCA_LDA_THREADS", "12")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, THREADS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

SEED = 42
N_FOLDS = 5
FAMILIES = ("Descriptors", "Hilbert", "Fourier")
K_GRID = (1, 2, 4, 8, 16, 32, 64, 96, 130, 152, 192, 256, 384, 425)
COLORS = {
    "Descriptors PCA": "#2a78d6",
    "D+H+F pooled PCA": "#e99b00",
    "D+H+F split PCA": "#e84b4b",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def finite_copy(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    result = np.array(result, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


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


def fit_transform_pca(
    values: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    imputer = SimpleImputer(strategy="median").fit(values[train])
    train_i = imputer.transform(values[train])
    test_i = imputer.transform(values[test])
    scaler = StandardScaler().fit(train_i)
    train_s = scaler.transform(train_i)
    test_s = scaler.transform(test_i)
    reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
    train_p = reducer.fit_transform(train_s).astype(np.float32)
    test_p = reducer.transform(test_s).astype(np.float32)
    return train_p, test_p, float(reducer.explained_variance_ratio_.sum())


def split_quotas(k: int, widths: dict[str, int]) -> dict[str, int]:
    quotas = {name: k // len(FAMILIES) for name in FAMILIES}
    for name in FAMILIES[: k % len(FAMILIES)]:
        quotas[name] += 1
    overflow = 0
    for name in FAMILIES:
        if quotas[name] > widths[name]:
            overflow += quotas[name] - widths[name]
            quotas[name] = widths[name]
    while overflow:
        moved = False
        for name in FAMILIES:
            if quotas[name] < widths[name]:
                quotas[name] += 1
                overflow -= 1
                moved = True
                if overflow == 0:
                    break
        if not moved:
            break
    return quotas


def evaluate_lda(train_x: np.ndarray, test_x: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    n_classes = len(np.unique(labels))
    model = LinearDiscriminantAnalysis(
        n_components=min(n_classes - 1, train_x.shape[1]),
        priors=np.ones(n_classes, dtype=np.float64) / n_classes,
    )
    model.fit(train_x, labels[train])
    return float(balanced_accuracy_score(labels[test], model.predict(test_x)))


def save(rows: list[dict[str, object]], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["curve", "fold", "k"])
    detail.to_csv(output_dir / "curve_fold_results.csv", index=False)
    summary = detail.groupby(["curve", "k", "total_features"], as_index=False).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        explained_variance_mean=("explained_variance", "mean"),
        folds=("fold", "nunique"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    ).sort_values(["curve", "k"])
    summary.to_csv(output_dir / "curve_summary.csv", index=False, float_format="%.6f")
    plot(summary, output_dir)


def plot(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 7.2))
    for curve, color in COLORS.items():
        rows = summary[summary["curve"].eq(curve)].sort_values("k")
        if rows.empty:
            continue
        axis.plot(rows["k"], rows["bacc_mean"], "o-", color=color, linewidth=2.4, markersize=6.8, label=curve)
        axis.fill_between(
            rows["k"],
            rows["bacc_mean"] - rows["bacc_std"].fillna(0),
            rows["bacc_mean"] + rows["bacc_std"].fillna(0),
            color=color,
            alpha=0.12,
        )
    axis.set_xscale("log")
    axis.set_xlabel("k (componentes PCA, escala log)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("WikiArt (27 estilos, LDA) - PCA rico com orçamento ate 425")
    axis.grid(True, which="both", color="#d8d7d0", alpha=0.55, linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(loc="lower right", frameon=False, title="Curvas")
    legend.get_title().set_color("#52514e")
    figure.tight_layout()
    figure.savefig(output_dir / "rich_pca_lda_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rich_pca_lda_curves")
    parser.add_argument("--k-grid", type=int, nargs="+", default=list(K_GRID))
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_labels, blocks = load_blocks(args)
    labels = LabelEncoder().fit_transform(raw_labels)
    descriptor_values = blocks["Descriptors"]
    pooled_values = np.hstack([blocks[name] for name in FAMILIES]).astype(np.float32)
    family_widths = {name: blocks[name].shape[1] for name in FAMILIES}
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels), dtype=np.uint8), labels))
    rows: list[dict[str, object]] = []
    log(f"PCA+LDA curves: {len(labels):,} images, {len(np.unique(labels))} styles")

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        for k in args.k_grid:
            if k <= descriptor_values.shape[1]:
                started = time.perf_counter()
                train_p, test_p, variance = fit_transform_pca(descriptor_values, train, test, k)
                bacc = evaluate_lda(train_p, test_p, labels, train, test)
                rows.append({
                    "curve": "Descriptors PCA", "fold": fold, "k": k,
                    "total_features": descriptor_values.shape[1], "bacc": bacc,
                    "explained_variance": variance, "elapsed_seconds": time.perf_counter() - started,
                })
                log(f"fold={fold}/{N_FOLDS} Descriptors PCA       k={k:<4d} BAcc={bacc:.4f}")

            if k <= pooled_values.shape[1]:
                started = time.perf_counter()
                train_p, test_p, variance = fit_transform_pca(pooled_values, train, test, k)
                bacc = evaluate_lda(train_p, test_p, labels, train, test)
                rows.append({
                    "curve": "D+H+F pooled PCA", "fold": fold, "k": k,
                    "total_features": pooled_values.shape[1], "bacc": bacc,
                    "explained_variance": variance, "elapsed_seconds": time.perf_counter() - started,
                })
                log(f"fold={fold}/{N_FOLDS} D+H+F pooled PCA     k={k:<4d} BAcc={bacc:.4f}")

                started = time.perf_counter()
                train_parts, test_parts, variance_parts = [], [], []
                quotas = split_quotas(k, family_widths)
                for name in FAMILIES:
                    quota = quotas[name]
                    if quota <= 0:
                        continue
                    train_part, test_part, variance_part = fit_transform_pca(blocks[name], train, test, quota)
                    train_parts.append(train_part)
                    test_parts.append(test_part)
                    variance_parts.append(variance_part * quota / k)
                train_split = np.hstack(train_parts).astype(np.float32)
                test_split = np.hstack(test_parts).astype(np.float32)
                bacc = evaluate_lda(train_split, test_split, labels, train, test)
                rows.append({
                    "curve": "D+H+F split PCA", "fold": fold, "k": k,
                    "total_features": pooled_values.shape[1], "bacc": bacc,
                    "explained_variance": float(np.sum(variance_parts)),
                    "elapsed_seconds": time.perf_counter() - started,
                })
                log(f"fold={fold}/{N_FOLDS} D+H+F split PCA      k={k:<4d} BAcc={bacc:.4f}")
        save(rows, args.output_dir)

    save(rows, args.output_dir)
    summary = pd.read_csv(args.output_dir / "curve_summary.csv")
    log(f"Saved: {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
