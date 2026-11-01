"""Paired LDA comparison of graph features from image segmenters."""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src" / "pipeline"))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from compare_slic_multigraph import EDGE_FEATURES, RAG_FEATURES
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log, make_model


PILOT_ROOT = ROOT / "outputs/wikiart_segmentation_pilot"
METHOD_PATHS = {
    "SLIC": ROOT / "outputs/wikiart_slic_features/combined_features.parquet",
    "SLICO": PILOT_ROOT / "slico_n1800/combined_features.parquet",
    "Watershed": PILOT_ROOT / "watershed_n1800_c0/combined_features.parquet",
    "Felzenszwalb": (
        PILOT_ROOT / "felzenszwalb_s50_m50/combined_features.parquet"
    ),
    "Quickshift": PILOT_ROOT / "quickshift_k3_d10/combined_features.parquet",
}
from corpora import FEATURE_COLUMNS  # noqa: E402,F401
assert len(FEATURE_COLUMNS) == 71

OUTPUT_CSV = ROOT / "paper/segmentation_methods_pilot_results.csv"
OUTPUT_PNG = ROOT / "paper/segmentation_methods_pilot.png"
WINNERS_CSV = ROOT / "paper/segmentation_winners_12k_results.csv"
WINNERS_PNG = ROOT / "paper/segmentation_winners_12k.png"


def load_methods(
    method_paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    frames = {}
    columns_by_method = {}
    diagnostics = []
    for method, path in method_paths.items():
        available = pd.read_parquet(path)
        missing = set(["stem"] + FEATURE_COLUMNS) - set(available.columns)
        if missing:
            raise ValueError(f"{method} is missing columns: {sorted(missing)}")
        frame = available[["stem"] + FEATURE_COLUMNS].drop_duplicates("stem")
        rename = {
            column: f"{method.lower()}__{column}"
            for column in FEATURE_COLUMNS
        }
        frames[method] = frame.rename(columns=rename)
        columns_by_method[method] = list(rename.values())
        diagnostics.append({
            "method": method,
            "available_images": len(frame),
            "nodes_mean_available": available["n_nodes"].mean(),
            "nodes_median_available": available["n_nodes"].median(),
            "nodes_p05_available": available["n_nodes"].quantile(0.05),
            "nodes_p95_available": available["n_nodes"].quantile(0.95),
            "elapsed_mean_available": available[
                "elapsed_seconds_all_graphs"
            ].mean(),
            "elapsed_median_available": available[
                "elapsed_seconds_all_graphs"
            ].median(),
        })

    method_order = list(method_paths)
    combined = frames[method_order[0]]
    for method in method_order[1:]:
        combined = combined.merge(
            frames[method], on="stem", how="inner", validate="one_to_one"
        )
    metadata = pd.read_csv(ROOT / "data/artemis/wikiart_meta.csv")
    label_map = dict(
        zip(metadata["stem"].astype(str), metadata["style"].astype(str))
    )
    combined["label"] = combined["stem"].astype(str).map(label_map)
    combined = (
        combined.dropna(subset=["label"])
        .sort_values("stem")
        .reset_index(drop=True)
    )
    counts = combined["label"].value_counts()
    combined = combined[
        combined["label"].isin(counts[counts >= N_FOLDS].index)
    ].reset_index(drop=True)
    return combined, columns_by_method, pd.DataFrame(diagnostics)


def evaluate(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    values = (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )
    scores = []
    for fold, (train, test) in enumerate(splits, 1):
        estimator = make_model(len(np.unique(y)), len(columns))
        estimator.fit(values[train], y[train])
        score = balanced_accuracy_score(
            y[test], estimator.predict(values[test])
        )
        scores.append(float(score))
        log(f"  fold {fold}/{N_FOLDS}: {score:.4f}")
    return scores


def make_plot(result: pd.DataFrame, output_png: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(result))
    colors = ["#2a78d6", "#54a24b", "#eeca3b", "#e45756", "#b279a2"]

    axes[0].bar(
        x, result["bacc_mean"], yerr=result["bacc_std"],
        capsize=4, color=colors,
    )
    axes[0].axhline(1 / 27, color="#888888", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("71-feature multigraph")

    axes[1].bar(
        x,
        result["nodes_median_available"],
        yerr=np.vstack([
            result["nodes_median_available"] - result["nodes_p05_available"],
            result["nodes_p95_available"] - result["nodes_median_available"],
        ]),
        capsize=4,
        color=colors,
    )
    axes[1].set_ylabel("Regions per image")
    axes[1].set_title("Median and 5–95% interval")

    axes[2].bar(
        x, result["elapsed_median_available"], color=colors
    )
    axes[2].set_ylabel("Seconds per image")
    axes[2].set_title("Observed feature-extraction time")

    labels = result["method"].tolist()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle(
        f"Segmentation ablation — paired {int(result['n_images'].iloc[0]):,} images"
    )
    figure.tight_layout()
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--winners-only",
        action="store_true",
        help="Compare SLIC, Felzenszwalb, and Quickshift on the expanded cohort.",
    )
    args = parser.parse_args()
    if args.winners_only:
        method_paths = {
            method: METHOD_PATHS[method]
            for method in ("SLIC", "Felzenszwalb", "Quickshift")
        }
        output_csv, output_png = WINNERS_CSV, WINNERS_PNG
    else:
        method_paths = METHOD_PATHS
        output_csv, output_png = OUTPUT_CSV, OUTPUT_PNG

    frame, columns_by_method, diagnostics = load_methods(method_paths)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(
        splitter.split(np.zeros(len(y), dtype=np.uint8), y)
    )
    log(
        f"Paired pilot: {len(frame):,} images, "
        f"{len(encoder.classes_)} styles, {len(FEATURE_COLUMNS)} features"
    )

    rows = []
    for method, columns in columns_by_method.items():
        log(f"{method}:")
        scores = np.asarray(evaluate(frame, columns, y, splits))
        rows.append({
            "method": method,
            "n_images": len(frame),
            "n_styles": len(encoder.classes_),
            "n_features": len(columns),
            "bacc_mean": scores.mean(),
            "bacc_std": scores.std(),
            **{
                f"fold_{index + 1}": score
                for index, score in enumerate(scores)
            },
        })
        log(f"{method}: {scores.mean():.4f} ± {scores.std():.4f}")

    result = pd.DataFrame(rows).merge(
        diagnostics, on="method", how="left", validate="one_to_one"
    )
    result.to_csv(output_csv, index=False)
    make_plot(result, output_png)
    log("\n" + result[[
        "method", "bacc_mean", "bacc_std",
        "nodes_median_available", "elapsed_median_available",
    ]].to_string(index=False))
    log(f"Saved: {output_csv}")
    log(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
