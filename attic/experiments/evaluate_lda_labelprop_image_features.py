#!/usr/bin/env python3
"""LDA curves after redefining image features by r^2 label propagation.

The baseline uses all rich descriptor features, whose origin is the image
statistics extractor.  The main test keeps only descriptor features that the
r^2 graph label-propagation step predicts as "Image statistics".
"""

from __future__ import annotations

import argparse
import os
import re
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_rich_topk_curves import (  # noqa: E402
    N_FOLDS,
    SEED,
    checkpoints,
    evaluate_curve,
    load_blocks,
    log,
    ranked_order,
)


COLORS = {
    "All image-origin descriptors": "#2a78d6",
    "LP-image descriptors": "#1baf7a",
    "LP-non-image descriptors": "#eb6834",
}


def descriptor_index(feature: str) -> int:
    match = re.fullmatch(r"descriptor_rich_(\d+)", feature)
    if match is None:
        raise ValueError(f"Not a rich descriptor feature: {feature}")
    return int(match.group(1))


def labelprop_descriptor_sets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    table = table[table["family"].eq("Image statistics")].copy()
    table["descriptor_index"] = table["feature"].map(descriptor_index)
    lp_image = table.loc[table["predicted_family"].eq("Image statistics"), "descriptor_index"].to_numpy(dtype=np.int64)
    lp_non_image = table.loc[~table["predicted_family"].eq("Image statistics"), "descriptor_index"].to_numpy(dtype=np.int64)
    return np.sort(lp_image), np.sort(lp_non_image)


def plot(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 6.8))
    for curve, color in COLORS.items():
        rows = summary[summary["curve"].eq(curve)].sort_values("k")
        if rows.empty:
            continue
        axis.plot(rows["k"], rows["bacc_mean"], "o-", color=color, linewidth=2.4, markersize=6.5, label=curve)
        axis.fill_between(
            rows["k"],
            rows["bacc_mean"] - rows["bacc_std"].fillna(0),
            rows["bacc_mean"] + rows["bacc_std"].fillna(0),
            color=color,
            alpha=0.12,
        )
    axis.set_xscale("log")
    axis.set_xlabel("k selected features (log scale)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("WikiArt LDA after label-propagation image-feature filtering")
    axis.grid(True, which="both", color="#d8d7d0", alpha=0.55, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower right", frameon=False, title="Feature definition")
    figure.tight_layout()
    figure.savefig(output_dir / "lda_labelprop_image_features.png", dpi=220, facecolor="white")
    figure.savefig(output_dir / "lda_labelprop_image_features_paper.png", dpi=220, facecolor="white")
    plt.close(figure)


def save(rows: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    detail = pd.DataFrame(rows).sort_values(["curve", "fold", "k"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby(["curve", "k", "total_features"], as_index=False).agg(
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        folds=("fold", "nunique"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    ).sort_values(["curve", "k"])
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    plot(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--labelprop", type=Path, default=ROOT / "results/rich_feature_label_propagation_r2/feature_probabilities.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lda_labelprop_image_features")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_labels, blocks = load_blocks(args)
    labels = LabelEncoder().fit_transform(raw_labels)
    descriptors = blocks["Descriptors"]
    lp_image, lp_non_image = labelprop_descriptor_sets(args.labelprop)

    rows: list[dict[str, object]] = []
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels), dtype=np.uint8), labels))
    log(
        "Label-propagation descriptor split: "
        f"{len(lp_image)} LP-image, {len(lp_non_image)} LP-non-image, {descriptors.shape[1]} total"
    )

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        started = time.perf_counter()
        all_order = ranked_order(descriptors, labels, train)
        evaluate_curve(
            rows,
            "All image-origin descriptors",
            descriptors,
            labels,
            train,
            test,
            {k: all_order[:k] for k in checkpoints(descriptors.shape[1])},
            fold,
        )

        lp_values = descriptors[:, lp_image]
        lp_order = ranked_order(lp_values, labels, train)
        evaluate_curve(
            rows,
            "LP-image descriptors",
            lp_values,
            labels,
            train,
            test,
            {k: lp_order[:k] for k in checkpoints(len(lp_image))},
            fold,
        )

        non_values = descriptors[:, lp_non_image]
        non_order = ranked_order(non_values, labels, train)
        evaluate_curve(
            rows,
            "LP-non-image descriptors",
            non_values,
            labels,
            train,
            test,
            {k: non_order[:k] for k in checkpoints(len(lp_non_image))},
            fold,
        )
        summary = save(rows, args.output_dir)
        log(f"fold {fold} completed in {time.perf_counter() - started:.1f}s")

    summary = save(rows, args.output_dir)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    log(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
