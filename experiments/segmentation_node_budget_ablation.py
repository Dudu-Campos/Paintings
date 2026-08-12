"""Test how much segmentation accuracy comes from direct region-count cues."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from compare_segmentations_pilot import (
    FEATURE_COLUMNS,
    METHOD_PATHS,
    evaluate,
    load_methods,
)
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log


OUTPUT_CSV = ROOT / "paper/segmentation_node_budget_ablation.csv"
DIRECT_BUDGET_FEATURES = {
    "n_nodes", "area_mean", "n_edges", "density",
    "knn_n_edges", "knn_density",
    "del_n_edges", "del_density",
    "mst_n_edges", "mst_density",
}


def main() -> None:
    method_paths = {
        method: METHOD_PATHS[method]
        for method in ("SLIC", "Felzenszwalb", "Quickshift")
    }
    frame, columns_by_method, _ = load_methods(method_paths)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(
        splitter.split(np.zeros(len(y), dtype=np.uint8), y)
    )

    rows = []
    for method, renamed_columns in columns_by_method.items():
        mapping = dict(zip(FEATURE_COLUMNS, renamed_columns))
        configurations = {
            "Nodes only": [mapping["n_nodes"]],
            "Without direct budget": [
                renamed for original, renamed in mapping.items()
                if original not in DIRECT_BUDGET_FEATURES
            ],
            "Full multigraph": renamed_columns,
        }
        for configuration, columns in configurations.items():
            scores = np.asarray(
                evaluate(frame, columns, y, splits)
            )
            rows.append({
                "method": method,
                "configuration": configuration,
                "n_features": len(columns),
                "bacc_mean": scores.mean(),
                "bacc_std": scores.std(),
                **{
                    f"fold_{index + 1}": score
                    for index, score in enumerate(scores)
                },
            })
            log(
                f"{method:<13} | {configuration:<21} "
                f"{scores.mean():.4f} ± {scores.std():.4f}"
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
