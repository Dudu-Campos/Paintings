"""Same-fold MLP controls for the 32 RAG and 71 SLIC multigraph summaries.

This is paired with ``train_slic_gnn_pilot.py``: it uses the exact exported
2,693-graph cohort, outer fold and inner validation split.  It therefore
separates the contribution of the aggregate features from a GNN embedding.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_robustness_mlp_aug import train_one
from train_slic_gnn_pilot import N_FOLDS, ROOT, SEED


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-index", type=Path, default=ROOT / "outputs/slic_gnn_pilot/index.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/slic_gnn_pilot/aggregate_controls")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1), default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    index = pd.read_csv(args.graph_index, dtype={"stem": str, "style": str})
    encoder = LabelEncoder()
    y = encoder.fit_transform(index["style"].to_numpy())
    aggregate = pd.read_parquet(ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    metadata = {
        "stem", "style", "segmentation", "reference_n_nodes", "slic_requested_segments",
        "slic_actual_segments", "slic_compactness", "slic_sigma", "slic_max_side",
        "slic_enforce_connectivity", "elapsed_seconds", "elapsed_seconds_all_graphs",
    }
    all_columns = [column for column in aggregate if column not in metadata]
    if len(all_columns) != 71:
        raise ValueError(f"Expected 71 multigraph features, found {len(all_columns)}")
    frame = index[["stem", "style"]].merge(
        aggregate[["stem", "style"] + all_columns], on=["stem", "style"], how="left", validate="one_to_one",
    )
    if frame[all_columns].isna().any().any():
        raise ValueError("An exported pilot graph is missing aggregate features")
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(splitter.split(index, y), start=1):
        if fold != args.only_fold:
            continue
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold,
        )
        for name, columns in (("SLIC RAG 32 + MLP", all_columns[:32]), ("SLIC multigraph 71 + MLP", all_columns)):
            values = frame[columns].to_numpy(dtype=np.float32)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(values[fit_train]).astype(np.float32)
            validation_x = scaler.transform(values[validation]).astype(np.float32)
            test_x = scaler.transform(values[outer_test]).astype(np.float32)
            started = time.perf_counter()
            metrics, _ = train_one(
                train_x, y[fit_train], validation_x, y[validation], test_x, y[outer_test],
                [len(columns)], len(encoder.classes_), SEED + fold, args.epochs, args.patience,
            )
            rows.append({"fold": fold, "condition": name, "feature_dimensions": len(columns),
                         "elapsed_seconds": time.perf_counter() - started, **metrics})
            log(f"fold {fold}/{N_FOLDS} {name}: BAcc={metrics['bacc']:.4f}, val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}")
    detail = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    detail.groupby("condition", as_index=False).agg(
        feature_dimensions=("feature_dimensions", "first"), bacc_mean=("bacc", "mean"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"), best_epoch=("best_epoch", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    ).to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"Saved controls to {args.output_dir}")


if __name__ == "__main__":
    main()
