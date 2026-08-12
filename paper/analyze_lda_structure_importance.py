"""Collinearity-robust complement to canonical LDA feature importance.

The ordinary LDA coefficient and permutation rankings are distorted when
large opposing coefficients cancel across correlated variables.  This script
therefore reports structure coefficients: correlations between standardized
input features and held-in discriminant scores, combined across discriminant
axes using LDA's explained-variance ratios.  It also retrains LDA after
removing or isolating each complete representation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


PAPER = Path(__file__).resolve().parent
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from analyze_visual_feature_importance_lda_xgb import (  # noqa: E402
    N_FOLDS,
    PREFIX,
    SEED,
    load_canonical_visual,
    make_lda,
)
from analyze_visual_feature_redundancy_r2 import feature_metadata  # noqa: E402


OUT_PREFIX = PAPER / "visual_feature_importance_lda_structure"


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def structure_importance(model, train_values: np.ndarray) -> np.ndarray:
    standardized = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(train_values)
    ).astype(np.float64)
    scores = model.named_steps["lda"].transform(standardized)
    scores -= scores.mean(axis=0)
    score_std = scores.std(axis=0)
    valid = score_std > 1e-12
    scores[:, valid] /= score_std[valid]
    scores[:, ~valid] = 0.0
    correlations = (standardized.T @ scores) / len(standardized)
    weights = model.named_steps["lda"].explained_variance_ratio_.astype(np.float64)
    weights = weights[: correlations.shape[1]]
    weights /= weights.sum()
    result = np.sqrt(np.sum(correlations ** 2 * weights[None, :], axis=1))
    return result


def plot_structure(summary: pd.DataFrame, output: Path, top_n: int = 35):
    top = summary.head(top_n).sort_values("importance_mean")
    palette = {
        "SLIC multigraph": "#2E86AB",
        "Descriptors": "#9B59B6",
        "Hilbert": "#DD6E42",
    }
    figure, axis = plt.subplots(figsize=(11, 11))
    axis.barh(
        np.arange(len(top)),
        top["importance_mean"],
        xerr=top["importance_std"],
        color=[palette[item] for item in top["group"]],
        alpha=0.9,
    )
    axis.set_yticks(np.arange(len(top)), top["display_name"], fontsize=8)
    axis.set_xlabel("Importância estrutural LDA")
    axis.set_title(
        "LDA — correlação com os eixos discriminantes\n"
        "média ± desvio nos cinco folds canônicos"
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_ablation(summary: pd.DataFrame, output: Path):
    plot = summary.sort_values("bacc_mean")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    bars = axis.barh(
        np.arange(len(plot)),
        plot["bacc_mean"],
        xerr=plot["bacc_std"],
        color=["#2a78d6" if item == "All 258" else "#7f8c8d" for item in plot["condition"]],
        alpha=0.9,
    )
    axis.set_yticks(np.arange(len(plot)), plot["condition"])
    axis.set_xlabel("Balanced accuracy OOF")
    axis.set_title("LDA retreinado — ablação das representações visuais")
    axis.grid(axis="x", alpha=0.2)
    for bar, score in zip(bars, plot["bacc_mean"]):
        axis.text(score + 0.002, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main():
    started = time.perf_counter()
    frame, groups, columns, values, corrected = load_canonical_visual()
    metadata = feature_metadata(columns, groups)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].astype(str))
    splits = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    index = {feature: position for position, feature in enumerate(columns)}
    conditions = {"All 258": columns}
    for group, members in groups.items():
        conditions[f"Without {group}"] = [item for item in columns if item not in set(members)]
        conditions[f"Only {group}"] = list(members)

    structure_rows = []
    ablation_rows = []
    for fold, (train, test) in enumerate(splits, 1):
        full = make_lda(len(encoder.classes_), len(columns))
        full.fit(values[train], labels[train])
        importance = structure_importance(full, values[train])
        normalized = importance / importance.sum()
        for feature_index, feature in enumerate(columns):
            structure_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance": importance[feature_index],
                    "importance_normalized": normalized[feature_index],
                }
            )

        for condition, members in conditions.items():
            member_indices = np.asarray([index[item] for item in members])
            if condition == "All 258":
                model = full
            else:
                model = make_lda(len(encoder.classes_), len(members))
                model.fit(values[train][:, member_indices], labels[train])
            prediction = model.predict(values[test][:, member_indices])
            ablation_rows.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "n_features": len(members),
                    "bacc": balanced_accuracy_score(labels[test], prediction),
                }
            )
        log(f"fold={fold}/{N_FOLDS} complete")

    structure_folds = pd.DataFrame(structure_rows)
    structure_folds.to_csv(f"{OUT_PREFIX}_folds.csv", index=False)
    structure = structure_folds.groupby("feature").agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_normalized_mean=("importance_normalized", "mean"),
        importance_normalized_std=("importance_normalized", "std"),
    ).reset_index().merge(metadata, on="feature", validate="one_to_one")
    structure = structure.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    structure.insert(0, "rank", np.arange(1, len(structure) + 1))
    structure.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)

    ablation_folds = pd.DataFrame(ablation_rows)
    ablation_folds.to_csv(f"{OUT_PREFIX}_ablation_folds.csv", index=False)
    ablation = ablation_folds.groupby(["condition", "n_features"], sort=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std")
    ).reset_index()
    baseline = float(ablation.loc[ablation["condition"] == "All 258", "bacc_mean"].iloc[0])
    ablation["delta_vs_all"] = ablation["bacc_mean"] - baseline
    ablation.to_csv(f"{OUT_PREFIX}_ablation_summary.csv", index=False)

    # Quantify why coefficient/permutation magnitudes are unstable.
    standard_deviations = values.std(axis=0)
    active = standard_deviations > 1e-12
    standardized = (values[:, active] - values[:, active].mean(axis=0)) / standard_deviations[active]
    covariance = (standardized.T @ standardized) / len(standardized)
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[eigenvalues > 1e-12]
    condition_number = float(positive.max() / positive.min())

    native = pd.read_csv(f"{PREFIX}_native_summary.csv")
    lda_native = native[native["model"] == "LDA"].set_index("feature")
    common_features = structure.set_index("feature")
    rank_correlation = float(
        lda_native.loc[common_features.index, "importance_normalized_mean"].corr(
            common_features["importance_normalized_mean"], method="spearman"
        )
    )
    summary = {
        "method": "LDA structure coefficients weighted by discriminant explained variance",
        "cohort_images": len(frame),
        "features": len(columns),
        "corrected_labels": corrected,
        "nonconstant_covariance_rank": int(np.sum(eigenvalues > 1e-12)),
        "nonconstant_features": int(active.sum()),
        "standardized_covariance_condition_number": condition_number,
        "coefficient_vs_structure_spearman": rank_correlation,
        "baseline_bacc": baseline,
        "ablation": ablation.to_dict(orient="records"),
        "top_structure_features": structure.head(30).to_dict(orient="records"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(f"{OUT_PREFIX}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_structure(structure, Path(f"{OUT_PREFIX}_top.png"))
    plot_ablation(ablation, Path(f"{OUT_PREFIX}_ablation.png"))
    log("summary:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
