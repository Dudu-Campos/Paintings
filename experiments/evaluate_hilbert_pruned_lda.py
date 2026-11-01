"""Test whether the compact Hilbert block is diluted by weak feature groups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "experiments"))

from registry import load_blocks, log  # noqa: E402
from view_names import display  # noqa: E402
import extract_hilbert_features as hilbert  # noqa: E402

SEED = 42
PREFIX = ["Felzenszwalb", "SLIC", "Descriptors", "L(G_felz)", "Line graph", "Gabor"]
HAND8 = ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
         "Hilbert", "Fourier", "Gabor"]


def hilbert_slices() -> dict[str, list[int]]:
    names = hilbert.compact_feature_names()
    keep_means = [i for i, n in enumerate(names)
                  if "_band_mean_" in n or n.endswith("_mean")]
    keep_global = [i for i, n in enumerate(names)
                   if n.endswith("_mean") or n.endswith("_std")]
    keep_pruned = sorted(set(keep_means + keep_global))
    keep_no_corr = [i for i, n in enumerate(names) if not n.endswith("_corr")]
    return {
        "Hilbert full": list(range(len(names))),
        "Hilbert no correlations": keep_no_corr,
        "Hilbert mean/global": keep_pruned,
        "Hilbert means only": keep_means,
    }


def lda(values: np.ndarray, labels: np.ndarray, splits, condition: str) -> pd.DataFrame:
    n_classes = len(np.unique(labels))
    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        scaler = StandardScaler().fit(values[train])
        z = scaler.transform(values).astype(np.float32)
        model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
        model.fit(z[train], labels[train])
        bacc = 100 * balanced_accuracy_score(labels[test], model.predict(z[test]))
        rows.append({"condition": condition, "fold": fold,
                     "dimensions": values.shape[1], "bacc": bacc})
        log(f"{condition:<30} fold {fold}: BAcc={bacc:.2f} ({values.shape[1]}d)")
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    summary = frame.groupby("condition").bacc.agg(["mean", "std"]).sort_values("mean")
    colours = ["#2a78d6" if "full" in c else "#eb6834" for c in summary.index]
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], color=colours, alpha=0.9)
    ax.set_xlabel("balanced accuracy (%)")
    ax.set_title("Hilbert pruning under LDA", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/hilbert_pruned_lda.csv")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "results/hilbert_pruned_lda.png")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    base = {n: cohort.blocks[n].astype(np.float32) for n in HAND8}
    slices = hilbert_slices()

    rows = []
    # Standalone Hilbert variants.
    for name, keep in slices.items():
        rows.append(lda(base["Hilbert"][:, keep], labels, splits, name))

    # Cumulative context: up to Gabor, then add Hilbert variant.
    prefix = np.hstack([base[n] for n in PREFIX])
    rows.append(lda(prefix, labels, splits, "Prefix without Hilbert"))
    for name, keep in slices.items():
        rows.append(lda(np.hstack([prefix, base["Hilbert"][:, keep]]), labels, splits,
                        "Prefix + " + name))

    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(args.output, index=False, float_format="%.6f")
    plot(frame, args.plot)

    print("\nMean BAcc")
    print(frame.groupby("condition").bacc.agg(["mean", "std"]).sort_values("mean", ascending=False).round(3).to_string())
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    pairs = [
        ("Hilbert full", "Hilbert mean/global"),
        ("Prefix + Hilbert full", "Prefix + Hilbert mean/global"),
        ("Prefix without Hilbert", "Prefix + Hilbert full"),
        ("Prefix without Hilbert", "Prefix + Hilbert mean/global"),
    ]
    print("\nPaired deltas")
    for a, b in pairs:
        delta = pivot[b] - pivot[a]
        _, p = ttest_rel(pivot[b], pivot[a])
        print(f"{a:28s} -> {b:28s}: {delta.mean():+.3f} pp "
              f"(sd {delta.std(ddof=1):.3f}, p={p:.4f}, +folds {(delta > 0).sum()}/{len(delta)})")
    log(f"saved: {args.output}")
    log(f"saved: {args.plot}")


if __name__ == "__main__":
    main()
