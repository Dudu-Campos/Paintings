"""Paired feature-fusion curves for legacy and SLIC multigraphs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from compare_slic_multigraph import EDGE_FEATURES, RAG_FEATURES
from fusion_curve_slic import (
    BASELINE_DIR, N_FOLDS, ROOT, SEED, SENTIMENT_FEATURES,
    evaluate, load_npy, log,
)

OUTPUT_CSV = ROOT / "paper/fusion_curves_slic_vs_legacy.csv"
OUTPUT_PNG = ROOT / "paper/fusion_curves_slic_vs_legacy.png"


def load_multigraph(directory: Path, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    files = {
        "rag": "graph_features.parquet", "knn": "knn_features.parquet",
        "del": "delaunay_features.parquet", "mst": "mst_features.parquet",
    }
    rag = pd.read_parquet(directory / files["rag"], columns=["stem"] + RAG_FEATURES)
    rag = rag.drop_duplicates("stem")
    rename = {column: f"{prefix}_rag_{column}" for column in RAG_FEATURES}
    combined = rag.rename(columns=rename)
    columns = list(rename.values())
    for graph in ("knn", "del", "mst"):
        part = pd.read_parquet(
            directory / files[graph], columns=["stem"] + EDGE_FEATURES
        ).drop_duplicates("stem")
        part_rename = {
            column: f"{prefix}_{graph}_{column}" for column in EDGE_FEATURES
        }
        combined = combined.merge(
            part.rename(columns=part_rename), on="stem", how="inner",
            validate="one_to_one",
        )
        columns.extend(part_rename.values())
    assert len(columns) == 71
    return combined, columns


def main() -> None:
    legacy, legacy_columns = load_multigraph(BASELINE_DIR, "legacy")
    slic, slic_columns = load_multigraph(ROOT / "outputs/wikiart_slic_features", "slic")
    hilbert, hilbert_columns = load_npy(
        "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
    )
    random_walk, random_walk_columns = load_npy(
        "rw_features.npy", "rw_stems.npy", "rw"
    )
    hybrid_walk, hybrid_walk_columns = load_npy(
        "hybrid_walk_features.npy", "hybrid_walk_stems.npy", "hybrid"
    )
    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    sentiment = pd.read_parquet(
        BASELINE_DIR / "sentiment_features.parquet",
        columns=["stem"] + SENTIMENT_FEATURES,
    ).drop_duplicates("stem")

    frame = legacy.merge(slic, on="stem", how="inner", validate="one_to_one")
    for addition in (hilbert, random_walk, hybrid_walk, descriptors, sentiment):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
    metadata = pd.read_csv(ROOT / "data/artemis/wikiart_meta.csv")
    labels = dict(zip(metadata["stem"].astype(str), metadata["style"].astype(str)))
    frame["label"] = frame["stem"].astype(str).map(labels)
    frame = frame.dropna(subset=["label"]).sort_values("stem").reset_index(drop=True)
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= N_FOLDS].index)].reset_index(drop=True)
    y = frame["label"].to_numpy(dtype=str)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y)), y))
    log(f"Paired dataset: {len(frame):,} images, {len(np.unique(y))} styles")

    additions = [
        ("Descriptors", descriptor_columns), ("Hilbert", hilbert_columns),
        ("Sentiment", SENTIMENT_FEATURES), ("Random walk", random_walk_columns),
        ("Hybrid walk", hybrid_walk_columns),
    ]
    rows = []
    for method, graph_columns in (
        ("Legacy percolation", legacy_columns), ("SLIC-2048", slic_columns),
    ):
        active = list(graph_columns)
        sequence = [("Multigraph", [])] + additions
        for step, (name, columns) in enumerate(sequence, 1):
            active.extend(columns)
            mean, std, scores = evaluate(frame, active, y, splits)
            rows.append({
                "method": method, "step": step, "added_group": name,
                "n_features": len(active), "bacc_mean": mean, "bacc_std": std,
                **{f"fold_{index + 1}": value for index, value in enumerate(scores)},
            })
            log(f"{method:<18} {name:<12}: {mean:.4f} ± {std:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = {"Legacy percolation": "#777777", "SLIC-2048": "#2a78d6"}
    for method, group in result.groupby("method", sort=False):
        x = group["step"].to_numpy()
        mean = group["bacc_mean"].to_numpy()
        std = group["bacc_std"].to_numpy()
        axis.plot(x, mean, "o-", linewidth=2.3, markersize=7,
                  color=colors[method], label=method)
        axis.fill_between(x, mean - std, mean + std,
                          color=colors[method], alpha=0.14)
    labels = ["Multigraph"] + [f"+ {name}" for name, _ in additions]
    axis.axhline(1 / len(np.unique(y)), color="#999999", linestyle=":", linewidth=1)
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=18, ha="right")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Feature-fusion curve — paired legacy vs SLIC segmentation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
