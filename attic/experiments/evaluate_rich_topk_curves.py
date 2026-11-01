#!/usr/bin/env python3
"""Supervised LDA top-k curves for the rich single-source representations.

Curves:
  1. Descriptors rich only.
  2. Descriptors + Hilbert + Fourier rich, ranked as one pooled image vector.
  3. The same three representations, but the k-budget is split across
     representations by within-family ranks.

All ranks are fit inside each training fold only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS = os.environ.get("TOPK_THREADS", "12")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, THREADS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

SEED = 42
N_FOLDS = 5
FAMILIES = ("Descriptors", "Hilbert", "Fourier")
COLORS = {
    "Descriptors rich": "#2a78d6",
    "D+H+F pooled top-k": "#e99b00",
    "D+H+F split top-k": "#e84b4b",
}


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


def checkpoints(n_features: int) -> list[int]:
    base = {min(n_features, 2 ** exponent) for exponent in range(0, 12)}
    base |= {8, 16, 32, 57, 64, 96, 128, 130, 152, 192, 256, 384, 417, 425, 512, 608, 768, 1024, 1450, n_features}
    return [value for value in sorted(base) if 1 <= value <= n_features]


def labels_by_stem(meta_path: Path) -> dict[str, str]:
    meta = pd.read_csv(meta_path)
    stems = meta["filename"].astype(str).map(lambda value: Path(value).stem)
    return dict(zip(stems, meta["style"].astype(str), strict=False))


def sort_frame(stems: np.ndarray, values: np.ndarray, labels: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame({"stem": np.asarray(stems, dtype=str)})
    keep = frame["stem"].isin(labels)
    frame = frame.loc[keep].copy()
    values = values[keep.to_numpy()]
    order = np.argsort(frame["stem"].to_numpy(dtype=str))
    frame = frame.iloc[order].reset_index(drop=True)
    values = finite_copy(values[order])
    frame["label"] = frame["stem"].map(labels)
    for index in range(values.shape[1]):
        frame[f"x{index:04d}"] = values[:, index]
    return frame


def load_npy_block(values_path: Path, stems_path: Path, labels: dict[str, str]) -> pd.DataFrame:
    return sort_frame(
        np.load(stems_path, allow_pickle=True),
        np.load(values_path),
        labels,
    )


def load_fourier(path: Path, labels: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    feature_columns = [
        column for column in frame.columns
        if column not in {"stem", "style", "path", "filename"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return sort_frame(
        frame["stem"].to_numpy(dtype=str),
        frame[feature_columns].to_numpy(dtype=np.float32),
        labels,
    )


def load_blocks(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = labels_by_stem(args.meta)
    rich = args.rich_dir
    frames = {
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
    common = set(frames["Descriptors"]["stem"])
    for name in FAMILIES[1:]:
        common &= set(frames[name]["stem"])
    stems = np.array(sorted(common), dtype=str)
    label_values: np.ndarray | None = None
    blocks: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        frame = frame.set_index("stem").loc[stems].reset_index()
        current_labels = frame["label"].to_numpy(dtype=str)
        label_values = current_labels if label_values is None else label_values
        if not np.array_equal(label_values, current_labels):
            raise ValueError(f"Label mismatch after aligning {name}")
        columns = [column for column in frame.columns if column.startswith("x")]
        blocks[name] = frame[columns].to_numpy(dtype=np.float32)
        log(f"{name:<12} {blocks[name].shape[0]:,} images, {blocks[name].shape[1]}d")
    assert label_values is not None
    return label_values, blocks


def ranked_order(values: np.ndarray, labels: np.ndarray, train: np.ndarray) -> np.ndarray:
    model = make_lda(len(np.unique(labels)), values.shape[1])
    model.fit(values[train], labels[train])
    return np.argsort(-structure_importance(model, values[train]))


def split_selection(family_orders: dict[str, np.ndarray], family_offsets: dict[str, int], family_widths: dict[str, int], k: int) -> np.ndarray:
    quotas = {name: k // len(FAMILIES) for name in FAMILIES}
    for name in FAMILIES[: k % len(FAMILIES)]:
        quotas[name] += 1
    overflow = 0
    for name in FAMILIES:
        if quotas[name] > family_widths[name]:
            overflow += quotas[name] - family_widths[name]
            quotas[name] = family_widths[name]
    while overflow:
        changed = False
        for name in FAMILIES:
            if quotas[name] < family_widths[name]:
                quotas[name] += 1
                overflow -= 1
                changed = True
                if overflow == 0:
                    break
        if not changed:
            break
    selected = []
    for name in FAMILIES:
        selected.extend((family_offsets[name] + family_orders[name][:quotas[name]]).tolist())
    return np.asarray(selected, dtype=np.int64)


def evaluate_curve(
    rows: list[dict[str, object]],
    curve: str,
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    selections: dict[int, np.ndarray],
    fold: int,
) -> None:
    n_classes = len(np.unique(labels))
    for k, selected in selections.items():
        started = time.perf_counter()
        model = make_lda(n_classes, len(selected))
        model.fit(values[train][:, selected], labels[train])
        prediction = model.predict(values[test][:, selected])
        bacc = float(balanced_accuracy_score(labels[test], prediction))
        rows.append({
            "fold": fold,
            "curve": curve,
            "k": k,
            "selected_features": len(selected),
            "total_features": values.shape[1],
            "bacc": bacc,
            "elapsed_seconds": time.perf_counter() - started,
        })
        log(f"fold={fold}/{N_FOLDS} {curve:<22} k={k:<4d} BAcc={bacc:.4f}")


def save(rows: list[dict[str, object]], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["curve", "fold", "k"])
    detail.to_csv(output_dir / "curve_fold_results.csv", index=False)
    summary = detail.groupby(["curve", "k", "total_features"], as_index=False).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
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
        axis.plot(rows["k"], rows["bacc_mean"], "o-", color=color, linewidth=2.3, markersize=6.5, label=curve)
        axis.fill_between(
            rows["k"],
            rows["bacc_mean"] - rows["bacc_std"].fillna(0),
            rows["bacc_mean"] + rows["bacc_std"].fillna(0),
            color=color,
            alpha=0.10,
        )
    axis.set_xscale("log")
    axis.set_xlabel("k (features selecionadas, escala log)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("WikiArt (27 estilos, LDA) - rich single-source representations")
    axis.grid(True, which="both", color="#d8d7d0", alpha=0.5, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(loc="lower right", frameon=False, title="Curvas")
    legend.get_title().set_color("#52514e")
    figure.tight_layout()
    figure.savefig(output_dir / "rich_topk_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rich_topk_curves")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
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
    rows: list[dict[str, object]] = []
    log(f"Top-k curves: {len(labels):,} images, {len(np.unique(labels))} styles")

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        desc_order = ranked_order(descriptor_values, labels, train)
        desc_selections = {k: desc_order[:k] for k in checkpoints(descriptor_values.shape[1])}
        evaluate_curve(rows, "Descriptors rich", descriptor_values, labels, train, test, desc_selections, fold)

        pooled_order = ranked_order(pooled_values, labels, train)
        pooled_selections = {k: pooled_order[:k] for k in checkpoints(pooled_values.shape[1])}
        evaluate_curve(rows, "D+H+F pooled top-k", pooled_values, labels, train, test, pooled_selections, fold)

        family_orders = {
            name: ranked_order(blocks[name], labels, train)
            for name in FAMILIES
        }
        split_selections = {
            k: split_selection(family_orders, family_offsets, family_widths, k)
            for k in checkpoints(pooled_values.shape[1])
        }
        evaluate_curve(rows, "D+H+F split top-k", pooled_values, labels, train, test, split_selections, fold)
        save(rows, args.output_dir)

    save(rows, args.output_dir)
    summary = pd.read_csv(args.output_dir / "curve_summary.csv")
    log(f"Saved: {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
