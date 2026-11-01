#!/usr/bin/env python3
"""Fold-local PCA + MLP curves for rich representations plus graph blocks.

PCA is fit only on the fit split inside each outer fold; validation is reserved
for early stopping.  This keeps the neural probe aligned with the production
MLP protocol while comparing matched PCA budgets.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS = os.environ.get("PCA_MLP_THREADS", "8")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, THREADS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_robustness_mlp_aug import DEVICE, train_one  # noqa: E402

SEED = 42
N_FOLDS = 5
BASE_FAMILIES = ("Descriptors", "Hilbert", "Fourier")
GRAPH_FAMILIES = ("SLIC", "Line graph", "Felzenszwalb")
FAMILIES = BASE_FAMILIES + GRAPH_FAMILIES
K_GRID = (32, 64, 130, 256, 425)
COLORS = {
    "Descriptors PCA": "#2a78d6",
    "D+H+F+Graphs pooled PCA": "#1baf7a",
    "D+H+F+Graphs split PCA": "#4a3aa7",
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


def load_parquet_block(path: Path, labels: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)
    metadata = {
        "stem", "style", "segmentation", "slic_requested_segments",
        "slic_actual_segments", "slic_compactness", "slic_sigma",
        "slic_max_side", "elapsed_seconds_all_graphs",
    }
    feature_columns = [
        column for column in frame.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(frame[column])
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
        "SLIC": load_parquet_block(args.slic_features, labels),
        "Line graph": load_parquet_block(args.line_features, labels),
        "Felzenszwalb": load_parquet_block(args.felz_features, labels),
    }
    common = set(loaded["Descriptors"][0])
    for name in FAMILIES:
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
        log(f"{name:<13} {blocks[name].shape[0]:,} images, {blocks[name].shape[1]}d")
    assert label_values is not None
    return label_values, blocks


def quotas(k: int, widths: dict[str, int]) -> dict[str, int]:
    result = {name: k // len(FAMILIES) for name in FAMILIES}
    for name in FAMILIES[: k % len(FAMILIES)]:
        result[name] += 1
    overflow = 0
    for name in FAMILIES:
        if result[name] > widths[name]:
            overflow += result[name] - widths[name]
            result[name] = widths[name]
    while overflow:
        moved = False
        for name in FAMILIES:
            if result[name] < widths[name]:
                result[name] += 1
                overflow -= 1
                moved = True
                if overflow == 0:
                    break
        if not moved:
            break
    return result


def fit_pca_parts(
    values: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    imputer = SimpleImputer(strategy="median").fit(values[fit])
    fit_i = imputer.transform(values[fit])
    validation_i = imputer.transform(values[validation])
    test_i = imputer.transform(values[test])
    scaler = StandardScaler().fit(fit_i)
    fit_s = scaler.transform(fit_i)
    validation_s = scaler.transform(validation_i)
    test_s = scaler.transform(test_i)
    reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
    fit_p = reducer.fit_transform(fit_s).astype(np.float32)
    validation_p = reducer.transform(validation_s).astype(np.float32)
    test_p = reducer.transform(test_s).astype(np.float32)
    return fit_p, validation_p, test_p, float(reducer.explained_variance_ratio_.sum())


def evaluate_curve(
    values: np.ndarray,
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    k: int,
    epochs: int,
    patience: int,
) -> tuple[dict[str, object], float]:
    fit, validation = train_test_split(
        outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
    )
    train_x, validation_x, test_x, explained = fit_pca_parts(values, fit, validation, outer_test, k)
    metrics, _ = train_one(
        train_x, labels[fit], validation_x, labels[validation], test_x, labels[outer_test],
        [k], len(np.unique(labels)), SEED + fold, epochs, patience,
    )
    return metrics, explained


def evaluate_split(
    blocks: dict[str, np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    k: int,
    epochs: int,
    patience: int,
) -> tuple[dict[str, object], float, list[int]]:
    fit, validation = train_test_split(
        outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
    )
    widths = {name: blocks[name].shape[1] for name in FAMILIES}
    allocation = quotas(k, widths)
    train_parts, validation_parts, test_parts, explained_parts, branch_sizes = [], [], [], [], []
    for name in FAMILIES:
        n_components = allocation[name]
        if n_components <= 0:
            continue
        train_p, validation_p, test_p, explained = fit_pca_parts(
            blocks[name], fit, validation, outer_test, n_components
        )
        train_parts.append(train_p)
        validation_parts.append(validation_p)
        test_parts.append(test_p)
        explained_parts.append(explained * n_components / k)
        branch_sizes.append(n_components)
    train_x = np.hstack(train_parts).astype(np.float32)
    validation_x = np.hstack(validation_parts).astype(np.float32)
    test_x = np.hstack(test_parts).astype(np.float32)
    metrics, _ = train_one(
        train_x, labels[fit], validation_x, labels[validation], test_x, labels[outer_test],
        branch_sizes, len(np.unique(labels)), SEED + fold, epochs, patience,
    )
    return metrics, float(np.sum(explained_parts)), branch_sizes


def save(rows: list[dict[str, object]], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["curve", "fold", "k"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby(["curve", "k", "total_features"], as_index=False).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        validation_bacc_mean=("validation_bacc", "mean"),
        explained_variance_mean=("explained_variance", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        folds=("fold", "nunique"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    ).sort_values(["curve", "k"])
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
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
    axis.set_title("WikiArt (27 estilos, MLP) - PCA rico + grafos, orcamento ate 425")
    axis.grid(True, which="both", color="#d8d7d0", alpha=0.55, linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(loc="lower right", frameon=False, title="Curvas")
    legend.get_title().set_color("#52514e")
    figure.tight_layout()
    figure.savefig(output_dir / "rich_pca_mlp_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic-features", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-features", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rich_pca_mlp_curves_graphs")
    parser.add_argument("--k-grid", type=int, nargs="+", default=list(K_GRID))
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_labels, blocks = load_blocks(args)
    labels = LabelEncoder().fit_transform(raw_labels)
    descriptors = blocks["Descriptors"]
    pooled = np.hstack([blocks[name] for name in FAMILIES]).astype(np.float32)
    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    result_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    done = {(str(row["curve"]), int(row["fold"]), int(row["k"])) for row in rows}
    log(f"PCA+MLP curves: {len(labels):,} images, {len(np.unique(labels))} styles, device={DEVICE}")

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        for k in args.k_grid:
            specs = [
                ("Descriptors PCA", descriptors, descriptors.shape[1]),
                ("D+H+F+Graphs pooled PCA", pooled, pooled.shape[1]),
            ]
            for curve, values, total_features in specs:
                if (curve, fold, k) in done:
                    continue
                started = time.perf_counter()
                metrics, explained = evaluate_curve(
                    values, labels, outer_train, outer_test, fold, k, args.epochs, args.patience
                )
                rows.append({
                    "curve": curve, "fold": fold, "k": k, "total_features": total_features,
                    "selected_features": k, "branch_sizes": str([k]),
                    "explained_variance": explained, "elapsed_seconds": time.perf_counter() - started,
                    **metrics,
                })
                done.add((curve, fold, k))
                save(rows, args.output_dir)
                log(
                    f"fold={fold}/{N_FOLDS} {curve:<25} k={k:<4d} "
                    f"BAcc={metrics['bacc']:.4f} val={metrics['validation_bacc']:.4f} "
                    f"ep={metrics['best_epoch']}"
                )
            curve = "D+H+F+Graphs split PCA"
            if (curve, fold, k) in done:
                continue
            started = time.perf_counter()
            metrics, explained, branch_sizes = evaluate_split(
                blocks, labels, outer_train, outer_test, fold, k, args.epochs, args.patience
            )
            rows.append({
                "curve": curve, "fold": fold, "k": k, "total_features": pooled.shape[1],
                "selected_features": k, "branch_sizes": str(branch_sizes),
                "explained_variance": explained, "elapsed_seconds": time.perf_counter() - started,
                **metrics,
            })
            done.add((curve, fold, k))
            save(rows, args.output_dir)
            log(
                f"fold={fold}/{N_FOLDS} {curve:<25} k={k:<4d} "
                f"BAcc={metrics['bacc']:.4f} val={metrics['validation_bacc']:.4f} "
                f"ep={metrics['best_epoch']} branches={branch_sizes}"
            )

    save(rows, args.output_dir)
    summary = pd.read_csv(args.output_dir / "summary.csv")
    log(f"Saved: {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
