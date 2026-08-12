"""Label-aware importance audit for canonical non-text visual features.

Models and outer folds match the current canonical WikiArt evaluation.  Three
complementary notions of importance are reported:

* LDA standardized coefficient magnitude (RMS over 27 classes);
* XGBoost normalized total split gain;
* held-out balanced-accuracy permutation drop for both models.

Pairwise-r² clusters at r² >= .99 are permuted jointly, so near-duplicate
features cannot mask one another by acting as substitutes.  Complete visual
representations and the entire visual matrix are also permuted as explicit
group controls.  Text/sentiment features are excluded.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from analyze_visual_feature_redundancy_r2 import feature_metadata  # noqa: E402
from fusion_curve_slic import N_FOLDS, SEED, SLIC_DIR  # noqa: E402
from slic_oof_stacking import load_dataset  # noqa: E402


CLUSTERS = PAPER / "visual_feature_redundancy_r2_clusters.csv"
PREFIX = PAPER / "visual_feature_importance_lda_xgb"
R2_THRESHOLD = 0.99
PERMUTATION_REPEATS = 3


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    n_components=min(n_classes - 1, n_features),
                    priors=np.ones(n_classes) / n_classes,
                ),
            ),
        ]
    )


def make_xgb(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="multi:softmax",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=16,
        random_state=SEED,
    )


def load_canonical_visual():
    frame, all_groups = load_dataset()
    groups = {
        name: columns for name, columns in all_groups.items() if name != "Sentiment"
    }
    columns = [column for values in groups.values() for column in values]
    if len(columns) != 258 or len(set(columns)) != 258:
        raise RuntimeError(f"Expected 258 unique visual features, found {len(columns)}")

    graph_labels = pd.read_parquet(
        SLIC_DIR / "combined_features.parquet", columns=["stem", "style"]
    ).drop_duplicates("stem")
    frame = frame.merge(
        graph_labels,
        on="stem",
        how="left",
        validate="one_to_one",
    )
    if frame["style"].isna().any():
        raise RuntimeError("Canonical visual rows without path-qualified SLIC style")
    corrected = int(frame["label"].ne(frame["style"]).sum())
    frame["label"] = frame["style"]
    frame = frame.drop(columns="style")
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Visual matrix contains non-finite values")
    return frame, groups, columns, values, corrected


def load_r2_units(
    columns: list[str], groups: dict[str, list[str]], metadata: pd.DataFrame
) -> list[dict]:
    clusters = pd.read_csv(CLUSTERS)
    selected = clusters[np.isclose(clusters["threshold"], R2_THRESHOLD)]
    cluster_members = []
    assigned = set()
    metadata_lookup = metadata.set_index("feature")
    for row in selected.sort_values("cluster_id").itertuples(index=False):
        members = str(row.members).split(" | ")
        unknown = set(members) - set(columns)
        overlap = set(members) & assigned
        if unknown or overlap:
            raise RuntimeError(
                f"Invalid r² cluster; unknown={sorted(unknown)}, overlap={sorted(overlap)}"
            )
        assigned.update(members)
        displays = metadata_lookup.loc[members, "display_name"].tolist()
        represented = sorted(set(metadata_lookup.loc[members, "group"]))
        cluster_members.append(
            {
                "unit_id": f"r2_cluster_{int(row.cluster_id):02d}",
                "unit_type": "r2_cluster",
                "display_name": " ↔ ".join(displays),
                "members": members,
                "represented_groups": " | ".join(represented),
            }
        )

    units = list(cluster_members)
    for feature in columns:
        if feature in assigned:
            continue
        item = metadata_lookup.loc[feature]
        units.append(
            {
                "unit_id": f"feature::{feature}",
                "unit_type": "single_feature",
                "display_name": item["display_name"],
                "members": [feature],
                "represented_groups": item["group"],
            }
        )
    for group, members in groups.items():
        units.append(
            {
                "unit_id": f"representation::{group}",
                "unit_type": "representation",
                "display_name": group,
                "members": list(members),
                "represented_groups": group,
            }
        )
    units.append(
        {
            "unit_id": "all_visual_features",
            "unit_type": "all_features",
            "display_name": "All 258 visual features",
            "members": list(columns),
            "represented_groups": " | ".join(groups),
        }
    )
    expected_atomic = len(columns) - sum(len(item["members"]) - 1 for item in cluster_members)
    observed_atomic = sum(
        item["unit_type"] in {"r2_cluster", "single_feature"} for item in units
    )
    if observed_atomic != expected_atomic:
        raise RuntimeError(f"Permutation unit mismatch: {observed_atomic} != {expected_atomic}")
    return units


def xgb_native_importance(model: XGBClassifier, columns: list[str]):
    booster = model.get_booster()
    total_gain = booster.get_score(importance_type="total_gain")
    weight = booster.get_score(importance_type="weight")
    gains = np.zeros(len(columns), dtype=np.float64)
    splits = np.zeros(len(columns), dtype=np.float64)
    for index in range(len(columns)):
        key = f"f{index}"
        gains[index] = float(total_gain.get(key, 0.0))
        splits[index] = float(weight.get(key, 0.0))
    gain_sum = gains.sum()
    return gains, gains / gain_sum if gain_sum else gains, splits


def summarize_native(
    native: pd.DataFrame,
    metadata: pd.DataFrame,
    classes: np.ndarray,
    class_coefficients: np.ndarray,
) -> pd.DataFrame:
    aggregation = native.groupby(["model", "feature"], sort=False).agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_normalized_mean=("importance_normalized", "mean"),
        importance_normalized_std=("importance_normalized", "std"),
        split_count_mean=("split_count", "mean"),
    ).reset_index()
    aggregation = aggregation.merge(metadata, on="feature", validate="many_to_one")

    coefficient_mean = class_coefficients.mean(axis=0)
    feature_index = {feature: index for index, feature in enumerate(metadata["feature"])}
    dominant_class, dominant_coefficient = [], []
    for row in aggregation.itertuples(index=False):
        if row.model != "LDA":
            dominant_class.append("")
            dominant_coefficient.append(np.nan)
            continue
        index = feature_index[row.feature]
        coefficients = coefficient_mean[:, index]
        class_index = int(np.argmax(np.abs(coefficients)))
        dominant_class.append(str(classes[class_index]))
        dominant_coefficient.append(float(coefficients[class_index]))
    aggregation["lda_dominant_class"] = dominant_class
    aggregation["lda_dominant_coefficient"] = dominant_coefficient
    aggregation["rank_within_model"] = aggregation.groupby("model")[
        "importance_normalized_mean"
    ].rank(method="first", ascending=False).astype(int)
    return aggregation.sort_values(["model", "rank_within_model"])


def plot_native(summary: pd.DataFrame, output: Path, top_n: int = 25):
    models = ["LDA", "XGBoost"]
    figure, axes = plt.subplots(1, 2, figsize=(16, 10))
    palette = {
        "SLIC multigraph": "#2E86AB",
        "Descriptors": "#9B59B6",
        "Hilbert": "#DD6E42",
    }
    for axis, model in zip(axes, models):
        top = (
            summary[summary["model"] == model]
            .nsmallest(top_n, "rank_within_model")
            .sort_values("importance_normalized_mean")
        )
        labels = top["display_name"].tolist()
        colors = [palette[group] for group in top["group"]]
        axis.barh(
            np.arange(len(top)),
            top["importance_normalized_mean"],
            xerr=top["importance_normalized_std"].fillna(0),
            color=colors,
            alpha=0.88,
        )
        axis.set_yticks(np.arange(len(top)), labels, fontsize=8)
        axis.set_xlabel("Importância nativa normalizada")
        axis.set_title(
            "LDA — RMS dos coeficientes padronizados"
            if model == "LDA"
            else "XGBoost — ganho total das divisões"
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Feature importance visual — média dos cinco folds canônicos",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_permutation(summary: pd.DataFrame, output: Path, top_n: int = 25):
    atomic = summary[summary["unit_type"].isin(["single_feature", "r2_cluster"])]
    order = (
        atomic.groupby("unit_id")["bacc_drop_mean"]
        .max()
        .nlargest(top_n)
        .index.tolist()
    )
    plot = atomic[atomic["unit_id"].isin(order)].copy()
    display_order = (
        plot.groupby(["unit_id", "display_name"])["bacc_drop_mean"]
        .max()
        .sort_values()
        .reset_index()
    )
    labels = display_order["display_name"].tolist()
    ids = display_order["unit_id"].tolist()
    y = np.arange(len(ids))
    figure, axis = plt.subplots(figsize=(14, 11))
    width = 0.36
    colors = {"LDA": "#2a78d6", "XGBoost": "#e67e22"}
    for offset, model in [(-width / 2, "LDA"), (width / 2, "XGBoost")]:
        indexed = plot[plot["model"] == model].set_index("unit_id")
        values = np.array([indexed.loc[item, "bacc_drop_mean"] for item in ids])
        errors = np.array([indexed.loc[item, "bacc_drop_std"] for item in ids])
        axis.barh(y + offset, values, height=width, xerr=errors, label=model, color=colors[model])
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8)
    axis.set_xlabel("Queda de balanced accuracy após permutação OOF")
    axis.set_title(
        "Importância por permutação — duplicatas r²≥0,99 embaralhadas juntas"
    )
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_representations(summary: pd.DataFrame, output: Path):
    data = summary[summary["unit_type"].isin(["representation", "all_features"])]
    labels = [
        "SLIC multigraph", "Descriptors", "Hilbert", "All 258 visual features"
    ]
    ids = [
        "representation::SLIC multigraph",
        "representation::Descriptors",
        "representation::Hilbert",
        "all_visual_features",
    ]
    y = np.arange(len(ids))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for offset, model, color in [
        (-width / 2, "LDA", "#2a78d6"),
        (width / 2, "XGBoost", "#e67e22"),
    ]:
        indexed = data[data["model"] == model].set_index("unit_id")
        values = np.array([indexed.loc[item, "bacc_drop_mean"] for item in ids])
        errors = np.array([indexed.loc[item, "bacc_drop_std"] for item in ids])
        axis.barh(y + offset, values, height=width, xerr=errors, label=model, color=color)
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Queda de balanced accuracy após permutação OOF")
    axis.set_title("Importância conjunta das representações visuais")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main():
    started = time.perf_counter()
    frame, groups, columns, values, corrected = load_canonical_visual()
    metadata = feature_metadata(columns, groups)
    units = load_r2_units(columns, groups, metadata)
    column_index = {column: index for index, column in enumerate(columns)}
    unit_indices = {
        unit["unit_id"]: np.asarray(
            [column_index[item] for item in unit["members"]], dtype=np.int64
        )
        for unit in units
    }
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].astype(str).to_numpy())
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(labels), dtype=np.uint8), labels))
    log(
        f"cohort={len(frame):,}; visual features={len(columns)}; "
        f"classes={len(encoder.classes_)}; corrected labels={corrected}; "
        f"permutation units={len(units)} x {PERMUTATION_REPEATS} repeats"
    )

    native_rows = []
    class_coefficients = []
    permutation_rows = []
    fold_rows = []
    for fold, (train, test) in enumerate(splits, 1):
        fold_started = time.perf_counter()
        train_values, test_values = values[train], values[test]
        train_labels, test_labels = labels[train], labels[test]
        lda = make_lda(len(encoder.classes_), len(columns))
        xgb = make_xgb(len(encoder.classes_))
        lda.fit(train_values, train_labels)
        xgb.fit(
            train_values,
            train_labels,
            sample_weight=compute_sample_weight("balanced", train_labels),
        )
        predictions = {
            "LDA": lda.predict(test_values),
            "XGBoost": xgb.predict(test_values),
        }
        baselines = {
            model: float(balanced_accuracy_score(test_labels, prediction))
            for model, prediction in predictions.items()
        }
        for model, score in baselines.items():
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "baseline_bacc": score,
                    "train_images": len(train),
                    "test_images": len(test),
                }
            )

        coefficients = lda.named_steps["lda"].coef_.astype(np.float64)
        class_coefficients.append(coefficients)
        lda_raw = np.sqrt(np.mean(coefficients ** 2, axis=0))
        lda_normalized = lda_raw / lda_raw.sum()
        xgb_gain, xgb_normalized, xgb_splits = xgb_native_importance(xgb, columns)
        for index, feature in enumerate(columns):
            native_rows.extend(
                [
                    {
                        "fold": fold,
                        "model": "LDA",
                        "feature": feature,
                        "importance": lda_raw[index],
                        "importance_normalized": lda_normalized[index],
                        "split_count": np.nan,
                    },
                    {
                        "fold": fold,
                        "model": "XGBoost",
                        "feature": feature,
                        "importance": xgb_gain[index],
                        "importance_normalized": xgb_normalized[index],
                        "split_count": xgb_splits[index],
                    },
                ]
            )

        for unit_number, unit in enumerate(units):
            indices = unit_indices[unit["unit_id"]]
            for repeat in range(PERMUTATION_REPEATS):
                rng = np.random.default_rng(
                    SEED + fold * 1_000_000 + unit_number * 100 + repeat
                )
                order = rng.permutation(len(test_values))
                permuted = test_values.copy()
                permuted[:, indices] = test_values[order[:, None], indices]
                for model_name, model in (("LDA", lda), ("XGBoost", xgb)):
                    prediction = model.predict(permuted)
                    score = float(balanced_accuracy_score(test_labels, prediction))
                    permutation_rows.append(
                        {
                            "fold": fold,
                            "repeat": repeat + 1,
                            "model": model_name,
                            "unit_id": unit["unit_id"],
                            "unit_type": unit["unit_type"],
                            "display_name": unit["display_name"],
                            "represented_groups": unit["represented_groups"],
                            "n_members": len(unit["members"]),
                            "members": " | ".join(unit["members"]),
                            "baseline_bacc": baselines[model_name],
                            "permuted_bacc": score,
                            "bacc_drop": baselines[model_name] - score,
                        }
                    )
            if (unit_number + 1) % 50 == 0 or unit_number + 1 == len(units):
                log(
                    f"fold={fold}/{N_FOLDS} permutation units="
                    f"{unit_number + 1}/{len(units)}"
                )
        log(
            f"fold={fold} BAcc LDA={baselines['LDA']:.4f} "
            f"XGBoost={baselines['XGBoost']:.4f}; "
            f"elapsed={time.perf_counter() - fold_started:.1f}s"
        )
        pd.DataFrame(permutation_rows).to_csv(
            f"{PREFIX}_permutation_raw.csv", index=False
        )
        pd.DataFrame(native_rows).to_csv(f"{PREFIX}_native_folds.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(f"{PREFIX}_model_folds.csv", index=False)

    native = pd.DataFrame(native_rows)
    coefficients = np.stack(class_coefficients, axis=0)
    native_summary = summarize_native(
        native, metadata, encoder.classes_, coefficients
    )
    native_summary.to_csv(f"{PREFIX}_native_summary.csv", index=False)

    class_rows = []
    coefficient_mean = coefficients.mean(axis=0)
    coefficient_std = coefficients.std(axis=0)
    for class_index, class_name in enumerate(encoder.classes_):
        for feature_index, feature in enumerate(columns):
            class_rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "coefficient_mean": coefficient_mean[class_index, feature_index],
                    "coefficient_std": coefficient_std[class_index, feature_index],
                }
            )
    pd.DataFrame(class_rows).merge(
        metadata, on="feature", validate="many_to_one"
    ).to_csv(f"{PREFIX}_lda_class_coefficients.csv", index=False)

    permutation = pd.DataFrame(permutation_rows)
    permutation_summary = permutation.groupby(
        [
            "model", "unit_id", "unit_type", "display_name",
            "represented_groups", "n_members", "members",
        ],
        sort=False,
    ).agg(
        bacc_drop_mean=("bacc_drop", "mean"),
        bacc_drop_std=("bacc_drop", "std"),
        bacc_drop_median=("bacc_drop", "median"),
        bacc_drop_min=("bacc_drop", "min"),
        bacc_drop_max=("bacc_drop", "max"),
        positive_runs=("bacc_drop", lambda values: int(np.sum(values > 0))),
        total_runs=("bacc_drop", "size"),
    ).reset_index()
    permutation_summary["rank_within_model_and_type"] = permutation_summary.groupby(
        ["model", "unit_type"]
    )["bacc_drop_mean"].rank(method="first", ascending=False).astype(int)
    permutation_summary = permutation_summary.sort_values(
        ["model", "unit_type", "rank_within_model_and_type"]
    )
    permutation_summary.to_csv(f"{PREFIX}_permutation_summary.csv", index=False)

    plot_native(native_summary, Path(f"{PREFIX}_native_top.png"))
    plot_permutation(permutation_summary, Path(f"{PREFIX}_permutation_top.png"))
    plot_representations(
        permutation_summary, Path(f"{PREFIX}_representations.png")
    )

    top_native = {}
    top_permutation = {}
    for model in ("LDA", "XGBoost"):
        top_native[model] = native_summary[
            native_summary["model"] == model
        ].nsmallest(20, "rank_within_model")[
            [
                "feature", "display_name", "group",
                "importance_normalized_mean", "importance_normalized_std",
                "lda_dominant_class", "lda_dominant_coefficient",
            ]
        ].to_dict(orient="records")
        candidates = permutation_summary[
            (permutation_summary["model"] == model)
            & permutation_summary["unit_type"].isin(["single_feature", "r2_cluster"])
        ]
        top_permutation[model] = candidates.nlargest(20, "bacc_drop_mean")[
            [
                "unit_id", "display_name", "represented_groups", "n_members",
                "members", "bacc_drop_mean", "bacc_drop_std", "positive_runs",
            ]
        ].to_dict(orient="records")
    representation = permutation_summary[
        permutation_summary["unit_type"].isin(["representation", "all_features"])
    ].to_dict(orient="records")
    summary = {
        "method": {
            "labels": "path-qualified canonical SLIC style labels",
            "features": "258 visual only; no sentiment/text",
            "folds": N_FOLDS,
            "permutation_repeats_per_fold": PERMUTATION_REPEATS,
            "r2_duplicate_threshold": R2_THRESHOLD,
            "permutation_metric": "held-out balanced accuracy drop",
        },
        "cohort_images": len(frame),
        "classes": len(encoder.classes_),
        "corrected_ambiguous_labels": corrected,
        "permutation_units": len(units),
        "elapsed_seconds": time.perf_counter() - started,
        "baseline_bacc": pd.DataFrame(fold_rows).groupby("model")[
            "baseline_bacc"
        ].agg(["mean", "std"]).to_dict(orient="index"),
        "representation_permutation": representation,
        "top_native": top_native,
        "top_group_aware_permutation": top_permutation,
    }
    Path(f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log("summary:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
