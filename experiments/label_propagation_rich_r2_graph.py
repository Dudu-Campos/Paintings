"""Semi-supervised label propagation on the rich-feature r^2 graph.

The graph nodes are individual features and edge weights are pairwise r^2
values above the threshold used for the redundancy layout.  We repeatedly hide
a stratified subset of family labels, propagate the remaining labels through
the graph, and average the predicted class distribution for each hidden
feature.  This gives an out-of-sample "representational identity" estimate for
each feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
FAMILY_ORDER = ["Region graphs", "Image statistics", "Space-filling curve", "Spectral"]
PORTUGUESE = {
    "Region graphs": "Grafos de regiao",
    "Image statistics": "Estatisticas de imagem",
    "Space-filling curve": "Curva de preenchimento",
    "Spectral": "Espectral",
}


def build_transition(index: pd.DataFrame, pairs: pd.DataFrame, weight_power: float) -> sparse.csr_matrix:
    feature_to_index = dict(zip(index["feature"], index["feature_index"]))
    rows = pairs["feature_a"].map(feature_to_index).to_numpy(dtype=np.int32)
    cols = pairs["feature_b"].map(feature_to_index).to_numpy(dtype=np.int32)
    weights = np.power(pairs["r2"].to_numpy(dtype=np.float64), weight_power)

    n_nodes = len(index)
    adjacency = sparse.coo_matrix(
        (
            np.concatenate([weights, weights]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degrees = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    return sparse.diags(inv_degrees) @ adjacency


def propagate(
    transition: sparse.csr_matrix,
    labels: np.ndarray,
    labeled_mask: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n_nodes = len(labels)
    n_classes = len(FAMILY_ORDER)
    y0 = np.zeros((n_nodes, n_classes), dtype=np.float64)
    y0[labeled_mask, labels[labeled_mask]] = 1.0

    distributions = y0.copy()
    for _ in range(max_iter):
        updated = alpha * transition.dot(distributions) + (1.0 - alpha) * y0
        delta = np.abs(updated - distributions).max()
        distributions = updated
        if delta < tol:
            break

    row_sums = distributions.sum(axis=1, keepdims=True)
    distributions = np.divide(
        distributions,
        row_sums,
        out=np.full_like(distributions, 1.0 / n_classes),
        where=row_sums > 0,
    )
    return distributions


def stratified_holdout(labels: np.ndarray, holdout_fraction: float, rng: np.random.Generator) -> np.ndarray:
    heldout = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        n_holdout = max(1, int(round(len(candidates) * holdout_fraction)))
        heldout[rng.choice(candidates, size=n_holdout, replace=False)] = True
    return heldout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/rich_feature_redundancy_r2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rich_feature_label_propagation_r2")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = pd.read_csv(args.input_dir / "feature_index.csv")
    pairs = pd.read_csv(args.input_dir / "pairs_ge_threshold.csv")
    labels = index["family"].map({family: i for i, family in enumerate(FAMILY_ORDER)}).to_numpy(dtype=np.int32)
    transition = build_transition(index, pairs, args.weight_power).tocsr()

    rng = np.random.default_rng(args.seed)
    sums = np.zeros((len(index), len(FAMILY_ORDER)), dtype=np.float64)
    counts = np.zeros(len(index), dtype=np.int32)

    for repeat in range(args.repeats):
        heldout = stratified_holdout(labels, args.holdout_fraction, rng)
        labeled = ~heldout
        distributions = propagate(transition, labels, labeled, args.alpha, args.max_iter, args.tol)
        sums[heldout] += distributions[heldout]
        counts[heldout] += 1
        print(f"repeat {repeat + 1:03d}/{args.repeats}: held out {heldout.sum()} features", flush=True)

    probabilities = np.divide(
        sums,
        counts[:, None],
        out=np.full_like(sums, np.nan),
        where=counts[:, None] > 0,
    )
    predicted = np.nanargmax(probabilities, axis=1)
    true_probability = probabilities[np.arange(len(labels)), labels]
    sorted_prob = np.sort(probabilities, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]

    output = index.copy()
    output["family_pt"] = output["family"].map(PORTUGUESE)
    output["heldout_count"] = counts
    for class_index, family in enumerate(FAMILY_ORDER):
        slug = family.lower().replace(" ", "_").replace("-", "_")
        output[f"prob_{slug}"] = probabilities[:, class_index]
    output["predicted_family"] = [FAMILY_ORDER[i] for i in predicted]
    output["predicted_family_pt"] = output["predicted_family"].map(PORTUGUESE)
    output["predicted_probability"] = probabilities[np.arange(len(labels)), predicted]
    output["true_family_probability"] = true_probability
    output["prediction_margin"] = margin
    output["mismatch"] = output["predicted_family"] != output["family"]
    output["cross_family_probability"] = 1.0 - output["true_family_probability"]

    summary = (
        output.groupby("family", sort=False)
        .agg(
            features=("feature", "size"),
            accuracy=("mismatch", lambda values: float((~values).mean())),
            mismatch_rate=("mismatch", "mean"),
            mean_true_family_probability=("true_family_probability", "mean"),
            median_true_family_probability=("true_family_probability", "median"),
            mean_cross_family_probability=("cross_family_probability", "mean"),
            median_margin=("prediction_margin", "median"),
        )
        .reset_index()
    )

    confusion = pd.crosstab(
        output["family"],
        output["predicted_family"],
        rownames=["true_family"],
        colnames=["predicted_family"],
    ).reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER, fill_value=0)

    image_features = output[output["family"].eq("Image statistics")].copy()
    image_features = image_features.sort_values(
        ["mismatch", "cross_family_probability", "predicted_probability"],
        ascending=[False, False, False],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "feature_probabilities.csv", index=False, float_format="%.8f")
    summary.to_csv(args.output_dir / "family_summary.csv", index=False, float_format="%.8f")
    confusion.to_csv(args.output_dir / "confusion.csv")
    image_features.to_csv(args.output_dir / "image_statistics_ranked.csv", index=False, float_format="%.8f")
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "input_dir": str(args.input_dir),
                "features": int(len(index)),
                "edges": int(len(pairs)),
                "repeats": int(args.repeats),
                "holdout_fraction": float(args.holdout_fraction),
                "alpha": float(args.alpha),
                "weight_power": float(args.weight_power),
            },
            handle,
            indent=2,
        )

    print(f"saved: {args.output_dir / 'feature_probabilities.csv'}")
    print(f"saved: {args.output_dir / 'image_statistics_ranked.csv'}")
    print(summary.to_string(index=False))
    print(confusion.to_string())


if __name__ == "__main__":
    main()
