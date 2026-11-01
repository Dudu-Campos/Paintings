"""Classifier-robustness check for the expanded segmentation pilot."""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from model_evaluation import DEVICE, evaluate_mlp, evaluate_xgb
from compare_segmentations_pilot import (
    METHOD_PATHS,
    WINNERS_CSV,
    load_methods,
)
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log


OUTPUT_CSV = ROOT / "paper/segmentation_winners_12k_3clf.csv"
OUTPUT_PNG = ROOT / "paper/segmentation_winners_12k_3clf.png"


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
    log(
        f"Cached segmentation cohort: {len(frame):,} images, "
        f"{len(encoder.classes_)} styles, device={DEVICE}"
    )

    lda = pd.read_csv(WINNERS_CSV)
    rows = []
    for row in lda.itertuples(index=False):
        rows.append({
            "method": row.method,
            "classifier": "LDA",
            "bacc_mean": row.bacc_mean,
            "bacc_std": row.bacc_std,
            "elapsed_seconds": np.nan,
            **{
                f"fold_{index}": getattr(row, f"fold_{index}")
                for index in range(1, N_FOLDS + 1)
            },
        })

    for method, columns in columns_by_method.items():
        values = (
            frame[columns]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32)
        )
        for classifier, evaluator in (
            ("XGBoost", evaluate_xgb),
            ("MLP", evaluate_mlp),
        ):
            log(f"{method} | {classifier}")
            started = time.perf_counter()
            scores = np.asarray(evaluator(values, y, splits))
            rows.append({
                "method": method,
                "classifier": classifier,
                "bacc_mean": scores.mean(),
                "bacc_std": scores.std(),
                "elapsed_seconds": time.perf_counter() - started,
                **{
                    f"fold_{index + 1}": score
                    for index, score in enumerate(scores)
                },
            })
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
            log(
                f"{method} | {classifier}: "
                f"{scores.mean():.4f} ± {scores.std():.4f}"
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    methods = ["SLIC", "Felzenszwalb", "Quickshift"]
    classifiers = ["LDA", "XGBoost", "MLP"]
    x = np.arange(len(classifiers))
    width = 0.24
    colors = ["#2a78d6", "#e45756", "#b279a2"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, (method, color) in enumerate(zip(methods, colors)):
        group = (
            result[result["method"] == method]
            .set_index("classifier")
            .loc[classifiers]
        )
        axis.bar(
            x + (index - 1) * width,
            group["bacc_mean"],
            width,
            yerr=group["bacc_std"],
            capsize=4,
            label=method,
            color=color,
        )
    axis.set_xticks(x, classifiers)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title(
        "Segmentation ablation — 12,213 paired images, 71 graph features"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)
    log("\n" + result[[
        "method", "classifier", "bacc_mean", "bacc_std"
    ]].to_string(index=False))
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
