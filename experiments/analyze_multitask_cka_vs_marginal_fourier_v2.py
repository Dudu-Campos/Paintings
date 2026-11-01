"""Multitask CKA-vs-marginal analysis comparing old Fourier and Fourier-v2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from registry import feature_columns, load_blocks, log, sanitize  # noqa: E402

SEED = 42
HAND8 = ["Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
         "Hilbert", "Fourier", "Gabor"]
COLOURS = {
    "Descriptors": "#2a78d6",
    "SLIC": "#1baf7a",
    "Felzenszwalb": "#e34948",
    "Line graph": "#eda100",
    "L(G_felz)": "#9c6ade",
    "Hilbert": "#00a884",
    "Fourier": "#555555",
    "Fourier v2": "#eb6834",
    "Gabor": "#7c3fa8",
}


def centre(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (x - x.mean(axis=0)) / sd


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x, y = centre(x), centre(y)
    xy = np.linalg.norm(x.T @ y, "fro") ** 2
    xx = np.linalg.norm(x.T @ x, "fro")
    yy = np.linalg.norm(y.T @ y, "fro")
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


def whiten(x: np.ndarray) -> np.ndarray:
    x = centre(np.asarray(x, dtype=np.float64))
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    keep = s > s.max() * 1e-8
    z = x @ vt[keep].T
    return z / np.maximum(z.std(axis=0), 1e-12)


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (16, 32, 64, 128, 192, 256, 384, 512, 768, 1024)
            if k <= max_k]
    if not grid or grid[-1] != max_k:
        grid.append(max_k)
    return grid


def probe(values: np.ndarray, labels: np.ndarray, splits, n_classes: int, *, pca_grid: bool = False) -> float:
    scores = []
    for train, test in splits:
        scaler = StandardScaler().fit(values[train])
        train_x = scaler.transform(values[train])
        test_x = scaler.transform(values[test])
        if not pca_grid:
            model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            model.fit(train_x, labels[train])
            scores.append(balanced_accuracy_score(labels[test], model.predict(test_x)))
            continue
        limit = min(train_x.shape[1], len(train) - n_classes)
        reducer = PCA(n_components=limit, svd_solver="randomized",
                      random_state=SEED).fit(train_x)
        train_p = reducer.transform(train_x)
        test_p = reducer.transform(test_x)
        best = 0.0
        for k in k_grid(limit):
            model = LinearDiscriminantAnalysis(
                n_components=min(n_classes - 1, k),
                priors=np.ones(n_classes) / n_classes)
            model.fit(train_p[:, :k], labels[train])
            best = max(best, balanced_accuracy_score(labels[test],
                                                     model.predict(test_p[:, :k])))
        scores.append(best)
    return float(np.mean(scores))


def load_fourier_v2(path: Path, stems: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path)
    cols = feature_columns(table)
    index = pd.Series(np.arange(len(table)), index=table["stem"].astype(str)).reindex(stems)
    if index.isna().any():
        raise ValueError(f"Fourier-v2: {int(index.isna().sum())} stems missing")
    return sanitize(table.iloc[index.to_numpy(dtype=int)][cols].to_numpy(np.float32),
                    "Fourier-v2")


def run(blocks: dict[str, np.ndarray], labels: np.ndarray,
        sample: int, folds: int, version: str, *, pca_grid: bool = False) -> pd.DataFrame:
    names = list(blocks)
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:folds]
    standard = {name: standardise(blocks[name]) for name in names}
    rng = np.random.default_rng(SEED)
    index = rng.choice(len(labels), size=min(sample, len(labels)), replace=False)
    whitened = {name: whiten(standard[name][index]) for name in names}

    everything = probe(np.hstack([standard[n] for n in names]), labels, splits, n_classes, pca_grid=pca_grid)
    log(f"{version}: all {len(names)} blocks -> BAcc {100 * everything:.2f}%")
    rows = []
    for name in names:
        others = [n for n in names if n != name]
        rest = np.hstack([standard[n] for n in others])
        rest_w = np.hstack([whitened[n] for n in others])
        solo = probe(standard[name], labels, splits, n_classes, pca_grid=pca_grid)
        without = probe(rest, labels, splits, n_classes, pca_grid=pca_grid)
        row = {
            "version": version,
            "block": name,
            "dims": standard[name].shape[1],
            "cka_raw": linear_cka(standard[name][index], rest[index]),
            "cka_whitened": linear_cka(whitened[name], rest_w),
            "solo_bacc": 100 * solo,
            "all_bacc": 100 * everything,
            "without_bacc": 100 * without,
            "loo_marginal": 100 * (everything - without),
        }
        rows.append(row)
        log(f"  {name:<12} CKA {row['cka_raw']:.3f}/{row['cka_whitened']:.3f} "
            f"solo {row['solo_bacc']:.2f} LOO {row['loo_marginal']:+.2f}pp")
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, output: Path, column: str = "cka_raw") -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8), facecolor="white")
    ax.axhline(0, color="#999999", linewidth=1.0, zorder=1)
    for row in table.itertuples(index=False):
        marker = "D" if row.version == "Fourier v2" and row.block == "Fourier v2" else "o"
        edge = "#111111" if "Fourier" in row.block else "white"
        ax.scatter(getattr(row, column), row.loo_marginal,
                   s=16 * np.sqrt(row.dims), color=COLOURS.get(row.block, "#777777"),
                   marker=marker, alpha=0.86, edgecolor=edge, linewidth=0.9, zorder=3)
        label = row.block
        if row.version == "old Fourier" and row.block == "Fourier":
            label = "Fourier old"
        ax.annotate(label, (getattr(row, column), row.loo_marginal),
                    textcoords="offset points", xytext=(0, 7 + 0.28 * np.sqrt(row.dims)),
                    ha="center", fontsize=8.2, color="#3f3f3f")
    ax.set_xscale("log")
    rho, pvalue = spearmanr(table[column], table.loo_marginal)
    ax.set_title(f"Multitask125: CKA with the rest vs. LOO marginal  ·  Spearman {rho:+.2f} (p={pvalue:.2f})",
                 loc="left", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("linear CKA with the rest")
    ax.set_ylabel("Δ balanced accuracy when removed (pp, LDA)")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.margins(x=0.18, y=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")
    print(f"saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--fourier-v2", type=Path,
                        default=ROOT / "results/fourier_v2_multitask_features.parquet")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-grid", action="store_true", help="use the slower original PCA-grid probe")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cka_vs_marginal_multitask_fourier_v2.csv")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "results/cka_vs_marginal_multitask_fourier_v2.png")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    old_blocks = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    new_blocks = dict(old_blocks)
    new_blocks.pop("Fourier")
    new_blocks["Fourier v2"] = load_fourier_v2(args.fourier_v2, cohort.stems)

    old = run(old_blocks, labels, args.sample, args.folds, "old Fourier", pca_grid=args.pca_grid)
    new = run(new_blocks, labels, args.sample, args.folds, "Fourier v2", pca_grid=args.pca_grid)
    # For the plot, use the new bank plus the old Fourier point as a direct
    # visual reference. The old point's LOO is with the old bank, by design.
    table = pd.concat([new, old[old.block.eq("Fourier")]], ignore_index=True)
    table.to_csv(args.output, index=False, float_format="%.6f")
    plot(table, args.plot)
    plot(table, args.plot.with_name(args.plot.stem + "_whitened.png"), "cka_whitened")
    print("\n" + table[["version", "block", "dims", "cka_raw", "cka_whitened",
                       "solo_bacc", "all_bacc", "without_bacc", "loo_marginal"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
