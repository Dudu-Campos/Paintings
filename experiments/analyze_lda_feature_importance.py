"""LDA feature importance for the visual PRISM bank.

Importance is computed on standardized features as the L2 norm of the
multiclass LDA coefficients for each feature. Each fold is normalized to sum to
one before aggregation, so the table reads as a share of supervised linear
importance rather than raw coefficient magnitude.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from registry import load_blocks, log  # noqa: E402

SEED = 42
VISUAL = [
    "Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
    "Hilbert", "Fourier", "Gabor",
]
FAMILY = {
    "Descriptors": "Image statistics",
    "Felzenszwalb": "Region graphs",
    "SLIC": "Region graphs",
    "Line graph": "Region graphs",
    "L(G_felz)": "Region graphs",
    "Hilbert": "Space-filling curve",
    "Fourier": "Spectral",
    "Gabor": "Spectral",
}
COLOURS = {
    "Image statistics": "#2a78d6",
    "Region graphs": "#eb6834",
    "Space-filling curve": "#00a884",
    "Spectral": "#5b3fb4",
}


def short_feature(name: str) -> str:
    for prefix in VISUAL:
        marker = f"{prefix}_"
        if name.startswith(marker):
            return name[len(marker):]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--blocks", nargs="+", default=VISUAL)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "results/lda_feature_importance_visual_125")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, args.blocks, allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))

    matrices, names, blocks, families = [], [], [], []
    for block in args.blocks:
        if block not in cohort.blocks:
            continue
        matrices.append(cohort.blocks[block])
        names.extend(cohort.columns[block])
        blocks.extend([block] * cohort.blocks[block].shape[1])
        families.extend([FAMILY.get(block, block)] * cohort.blocks[block].shape[1])
    values = np.hstack(matrices).astype(np.float32)

    rows = []
    splits = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splits.split(np.zeros(len(labels)), labels), start=1):
        scaler = StandardScaler().fit(values[train])
        train_x = scaler.transform(values[train])
        model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
        model.fit(train_x, labels[train])
        coef = np.asarray(model.coef_, dtype=np.float64)
        score = np.linalg.norm(coef, axis=0)
        score /= score.sum()
        for index, importance in enumerate(score):
            rows.append({
                "fold": fold,
                "feature": names[index],
                "display_name": short_feature(names[index]),
                "block": blocks[index],
                "family": families[index],
                "importance_share": importance,
            })
        log(f"fold {fold}: fitted LDA on {train_x.shape[0]:,} x {train_x.shape[1]} "
            f"({n_classes} classes)")

    per_fold = pd.DataFrame(rows)
    features = (per_fold.groupby(["feature", "display_name", "block", "family"], as_index=False)
                .importance_share.agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "importance_share_mean",
                                 "std": "importance_share_std"}))
    features["importance_percent"] = 100 * features["importance_share_mean"]
    features = features.sort_values("importance_share_mean", ascending=False)
    features["rank"] = np.arange(1, len(features) + 1)

    family = (features.groupby("family", as_index=False)
              .agg(features=("feature", "count"),
                   total_importance=("importance_share_mean", "sum"),
                   mean_feature_importance=("importance_share_mean", "mean")))
    family["total_importance_percent"] = 100 * family["total_importance"]
    family["mean_feature_importance_percent"] = 100 * family["mean_feature_importance"]
    family = family.sort_values("total_importance", ascending=False)

    block = (features.groupby(["family", "block"], as_index=False)
             .agg(features=("feature", "count"),
                  total_importance=("importance_share_mean", "sum"),
                  mean_feature_importance=("importance_share_mean", "mean")))
    block["total_importance_percent"] = 100 * block["total_importance"]
    block["mean_feature_importance_percent"] = 100 * block["mean_feature_importance"]
    block = block.sort_values("total_importance", ascending=False)

    prefix = args.output_prefix
    features.to_csv(prefix.with_suffix(".features.csv"), index=False, float_format="%.8f")
    family.to_csv(prefix.with_suffix(".families.csv"), index=False, float_format="%.8f")
    block.to_csv(prefix.with_suffix(".blocks.csv"), index=False, float_format="%.8f")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.patch.set_facecolor("white")
    family_plot = family.sort_values("total_importance")
    axes[0].barh(family_plot["family"], family_plot["total_importance_percent"],
                 color=[COLOURS.get(v, "#777777") for v in family_plot["family"]])
    axes[0].set_xlabel("share of LDA coefficient norm (%)")
    axes[0].set_title("Total importance by family", fontweight="bold", loc="left")

    top = features.head(25).iloc[::-1]
    axes[1].barh(top["display_name"], top["importance_percent"],
                 color=[COLOURS.get(v, "#777777") for v in top["family"]])
    axes[1].set_xlabel("share of LDA coefficient norm (%)")
    axes[1].set_title("Top individual features", fontweight="bold", loc="left")
    for axis in axes:
        axis.grid(axis="x", color="#dddddd", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("LDA feature importance without text or global similarity graph",
                 fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(prefix.with_suffix(".png"), dpi=220, facecolor="white")
    log(f"saved: {prefix.with_suffix('.features.csv')}")
    log(f"saved: {prefix.with_suffix('.families.csv')}")
    log(f"saved: {prefix.with_suffix('.blocks.csv')}")
    log(f"saved: {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
