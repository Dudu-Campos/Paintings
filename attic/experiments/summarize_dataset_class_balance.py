#!/usr/bin/env python3
"""Summarize image counts, classes, and class imbalance for paper datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    class_column: str
    label: str


DEFAULT_DATASETS = (
    DatasetSpec("WikiArt", ROOT / "data/artemis/wikiart_meta.csv", "style", "style"),
    DatasetSpec("Pandora18k", ROOT / "data/pandora_meta.csv", "style", "style"),
    DatasetSpec("Rijksmuseum", ROOT / "data/rijksmuseum_labels.csv", "period", "period"),
)


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64))
    if len(values) == 0 or values.sum() <= 0:
        return float("nan")
    index = np.arange(1, len(values) + 1, dtype=np.float64)
    return float((2 * np.sum(index * values) / (len(values) * values.sum())) - ((len(values) + 1) / len(values)))


def normalized_entropy(values: np.ndarray) -> float:
    values = values.astype(np.float64)
    total = values.sum()
    if total <= 0 or len(values) <= 1:
        return float("nan")
    probabilities = values / total
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(values)))


def summarize(spec: DatasetSpec, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(spec.path)
    if spec.class_column not in frame.columns:
        raise ValueError(f"{spec.path} has no column {spec.class_column!r}; columns={frame.columns.tolist()}")

    labels = frame[spec.class_column].map(normalize_label)
    labels = labels[labels.ne("")]
    counts = labels.value_counts().sort_index()
    values = counts.to_numpy(dtype=np.int64)
    total = int(values.sum())
    majority = int(values.max())
    minority = int(values.min())

    class_counts = pd.DataFrame({
        "dataset": spec.name,
        "class_axis": spec.label,
        "class": counts.index,
        "n_images": values,
        "fraction": values / total,
    }).sort_values("n_images", ascending=False)
    class_counts.to_csv(output_dir / f"{spec.name.lower()}_class_counts.csv", index=False, float_format="%.8f")

    return {
        "dataset": spec.name,
        "class_axis": spec.label,
        "metadata_rows": int(len(frame)),
        "images_with_class": total,
        "n_classes": int(len(counts)),
        "min_class_size": minority,
        "median_class_size": float(np.median(values)),
        "mean_class_size": float(np.mean(values)),
        "max_class_size": majority,
        "max_min_ratio": float(majority / minority) if minority else float("inf"),
        "majority_fraction": float(majority / total),
        "minority_fraction": float(minority / total),
        "class_count_cv": float(np.std(values, ddof=0) / np.mean(values)),
        "gini": gini(values),
        "normalized_entropy": normalized_entropy(values),
        "largest_class": str(counts.idxmax()),
        "smallest_class": str(counts.idxmin()),
    }


def latex_table(summary: pd.DataFrame) -> str:
    display = summary.copy()
    display["Images"] = display["images_with_class"].map(lambda value: f"{int(value):,}")
    display["Classes"] = display["n_classes"].map(lambda value: f"{int(value):,}")
    display["Min"] = display["min_class_size"].map(lambda value: f"{int(value):,}")
    display["Median"] = display["median_class_size"].map(lambda value: f"{value:,.0f}")
    display["Max"] = display["max_class_size"].map(lambda value: f"{int(value):,}")
    display["Max/min"] = display["max_min_ratio"].map(lambda value: f"{value:.1f}")
    display["Majority"] = display["majority_fraction"].map(lambda value: f"{100 * value:.1f}\\%")
    display["Entropy"] = display["normalized_entropy"].map(lambda value: f"{value:.3f}")
    display = display[["dataset", "class_axis", "Images", "Classes", "Min", "Median", "Max", "Max/min", "Majority", "Entropy"]]
    display = display.rename(columns={"dataset": "Dataset", "class_axis": "Class axis"})
    return display.to_latex(index=False, escape=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dataset_class_balance")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize(spec, args.output_dir) for spec in DEFAULT_DATASETS]
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False, float_format="%.8f")
    (args.output_dir / "dataset_summary.tex").write_text(latex_table(summary), encoding="utf-8")

    compact = summary[[
        "dataset",
        "class_axis",
        "images_with_class",
        "n_classes",
        "min_class_size",
        "median_class_size",
        "max_class_size",
        "max_min_ratio",
        "majority_fraction",
        "normalized_entropy",
    ]]
    print(compact.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved: {args.output_dir / 'dataset_summary.csv'}")
    print(f"saved: {args.output_dir / 'dataset_summary.tex'}")


if __name__ == "__main__":
    main()
