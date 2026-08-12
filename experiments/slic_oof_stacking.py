"""Leakage-free late fusion of the SLIC feature representations.

Each representation is modeled by its own LDA.  Inside every outer training
fold, inner out-of-fold class probabilities train a linear logistic
meta-classifier.  The outer test fold remains untouched until final scoring.

Only cached SLIC and auxiliary features are loaded; no graph is regenerated.
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from fusion_curve_slic import (
    BASELINE_DIR,
    N_FOLDS,
    ROOT,
    SEED,
    SENTIMENT_FEATURES,
    SLIC_DIR,
    SLIC_METADATA,
    load_npy,
    log,
    make_model,
)


INNER_FOLDS = 5
OUTPUT_CSV = ROOT / "paper/slic_oof_stacking_results.csv"
OUTPUT_PNG = ROOT / "paper/slic_oof_stacking.png"


def load_dataset() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    graph = pd.read_parquet(SLIC_DIR / "combined_features.parquet")
    graph_columns = [
        column for column in graph.columns if column not in SLIC_METADATA
    ]
    if len(graph_columns) != 71:
        raise ValueError(
            f"Expected 71 SLIC multigraph features, found {len(graph_columns)}"
        )

    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    hilbert, hilbert_columns = load_npy(
        "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
    )
    sentiment = pd.read_parquet(
        BASELINE_DIR / "sentiment_features.parquet",
        columns=["stem"] + SENTIMENT_FEATURES,
    ).drop_duplicates("stem")

    frame = graph[["stem"] + graph_columns].drop_duplicates("stem")
    for addition in (descriptors, hilbert, sentiment):
        frame = frame.merge(
            addition, on="stem", how="inner", validate="one_to_one"
        )

    metadata = pd.read_csv(ROOT / "data/artemis/wikiart_meta.csv")
    label_map = dict(
        zip(metadata["stem"].astype(str), metadata["style"].astype(str))
    )
    frame["label"] = frame["stem"].astype(str).map(label_map)
    frame = (
        frame.dropna(subset=["label"])
        .sort_values("stem")
        .reset_index(drop=True)
    )
    counts = frame["label"].value_counts()
    frame = frame[
        frame["label"].isin(counts[counts >= N_FOLDS].index)
    ].reset_index(drop=True)

    groups = {
        "SLIC multigraph": graph_columns,
        "Descriptors": descriptor_columns,
        "Hilbert": hilbert_columns,
        "Sentiment": SENTIMENT_FEATURES,
    }
    return frame, groups


def predict_proba_aligned(
    estimator, values: np.ndarray, n_classes: int
) -> np.ndarray:
    probabilities = estimator.predict_proba(values)
    classes = estimator.named_steps["lda"].classes_.astype(int)
    aligned = np.zeros((len(values), n_classes), dtype=np.float64)
    aligned[:, classes] = probabilities
    return aligned


def summarize(
    scores: dict[str, list[float]],
    feature_counts: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    for strategy, fold_scores in scores.items():
        values = np.asarray(fold_scores)
        n_input, n_meta = feature_counts[strategy]
        rows.append({
            "strategy": strategy,
            "n_input_features": n_input,
            "n_meta_features": n_meta,
            "bacc_mean": values.mean(),
            "bacc_std": values.std(),
            **{
                f"fold_{index + 1}": score
                for index, score in enumerate(values)
            },
        })
    return pd.DataFrame(rows)


def make_plot(result: pd.DataFrame) -> None:
    order = [
        "SLIC multigraph",
        "Descriptors",
        "Hilbert",
        "Sentiment",
        "Mean probabilities",
        "OOF logistic stacking",
        "Early concatenation LDA",
    ]
    plot_data = result.set_index("strategy").loc[order].reset_index()
    colors = [
        "#4c78a8", "#72b7b2", "#54a24b", "#b5b5b5",
        "#f58518", "#e45756", "#7a5195",
    ]
    figure, axis = plt.subplots(figsize=(11, 5.7))
    x = np.arange(len(plot_data))
    bars = axis.bar(
        x,
        plot_data["bacc_mean"],
        yerr=plot_data["bacc_std"],
        capsize=4,
        color=colors,
        alpha=0.9,
    )
    axis.axhline(1 / 27, color="#888888", linestyle=":", linewidth=1)
    axis.set_xticks(
        x,
        [
            "SLIC graph", "Descriptors", "Hilbert", "Sentiment",
            "Mean prob.", "OOF stacking", "Early concat.",
        ],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("SLIC-only representation fusion — 5-fold outer CV")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, score in zip(bars, plot_data["bacc_mean"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame, groups = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    group_values = {
        name: frame[columns].to_numpy(dtype=np.float64)
        for name, columns in groups.items()
    }
    concatenated_columns = [
        column for columns in groups.values() for column in columns
    ]
    concatenated_values = frame[concatenated_columns].to_numpy(dtype=np.float64)

    outer_splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    outer_splits = list(
        outer_splitter.split(np.zeros(len(frame), dtype=np.uint8), y)
    )
    log(
        f"SLIC-only cohort: {len(frame):,} images, {n_classes} styles; "
        f"{N_FOLDS} outer x {INNER_FOLDS} inner folds"
    )
    log("Representations: " + ", ".join(
        f"{name}={len(columns)}" for name, columns in groups.items()
    ))

    strategy_names = list(groups) + [
        "Mean probabilities",
        "OOF logistic stacking",
        "Early concatenation LDA",
    ]
    scores = {name: [] for name in strategy_names}
    meta_importance_rows = []

    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        fold_started = time.perf_counter()
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + outer_fold,
        )
        inner_splits = list(
            inner_splitter.split(
                np.zeros(len(outer_train), dtype=np.uint8), y[outer_train]
            )
        )
        train_meta_blocks = []
        test_meta_blocks = []

        for branch_name, values in group_values.items():
            oof_probabilities = np.zeros(
                (len(outer_train), n_classes), dtype=np.float64
            )
            for inner_train_rel, inner_validation_rel in inner_splits:
                inner_train = outer_train[inner_train_rel]
                inner_validation = outer_train[inner_validation_rel]
                estimator = make_model(n_classes, values.shape[1])
                estimator.fit(values[inner_train], y[inner_train])
                oof_probabilities[inner_validation_rel] = (
                    predict_proba_aligned(
                        estimator, values[inner_validation], n_classes
                    )
                )

            final_estimator = make_model(n_classes, values.shape[1])
            final_estimator.fit(values[outer_train], y[outer_train])
            test_probabilities = predict_proba_aligned(
                final_estimator, values[outer_test], n_classes
            )
            branch_prediction = test_probabilities.argmax(axis=1)
            branch_score = balanced_accuracy_score(
                y[outer_test], branch_prediction
            )
            scores[branch_name].append(float(branch_score))
            train_meta_blocks.append(oof_probabilities)
            test_meta_blocks.append(test_probabilities)

        mean_probabilities = np.mean(test_meta_blocks, axis=0)
        mean_score = balanced_accuracy_score(
            y[outer_test], mean_probabilities.argmax(axis=1)
        )
        scores["Mean probabilities"].append(float(mean_score))

        train_meta = np.concatenate(train_meta_blocks, axis=1)
        test_meta = np.concatenate(test_meta_blocks, axis=1)
        meta_classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=SEED + outer_fold,
        )
        meta_classifier.fit(train_meta, y[outer_train])
        stacked_prediction = meta_classifier.predict(test_meta)
        stacked_score = balanced_accuracy_score(
            y[outer_test], stacked_prediction
        )
        scores["OOF logistic stacking"].append(float(stacked_score))

        absolute_coefficients = np.abs(meta_classifier.coef_).reshape(
            n_classes, len(groups), n_classes
        )
        branch_importance = absolute_coefficients.sum(axis=(0, 2))
        branch_importance /= branch_importance.sum()
        meta_importance_rows.append({
            "fold": outer_fold,
            **{
                name: float(weight)
                for name, weight in zip(groups, branch_importance)
            },
        })

        early_estimator = make_model(n_classes, len(concatenated_columns))
        early_estimator.fit(
            concatenated_values[outer_train], y[outer_train]
        )
        early_score = balanced_accuracy_score(
            y[outer_test],
            early_estimator.predict(concatenated_values[outer_test]),
        )
        scores["Early concatenation LDA"].append(float(early_score))

        elapsed = time.perf_counter() - fold_started
        log(
            f"Outer fold {outer_fold}/{N_FOLDS}: "
            f"mean={mean_score:.4f}, stacking={stacked_score:.4f}, "
            f"early={early_score:.4f} ({elapsed:.1f}s)"
        )

    feature_counts = {
        name: (len(columns), 0) for name, columns in groups.items()
    }
    feature_counts.update({
        "Mean probabilities": (
            sum(len(columns) for columns in groups.values()),
            len(groups) * n_classes,
        ),
        "OOF logistic stacking": (
            sum(len(columns) for columns in groups.values()),
            len(groups) * n_classes,
        ),
        "Early concatenation LDA": (len(concatenated_columns), 0),
    })
    result = summarize(scores, feature_counts)
    result.to_csv(OUTPUT_CSV, index=False)

    importance = pd.DataFrame(meta_importance_rows)
    importance_path = ROOT / "paper/slic_oof_stacking_importance.csv"
    importance.to_csv(importance_path, index=False)
    make_plot(result)

    for row in result.itertuples(index=False):
        log(
            f"{row.strategy:<25} "
            f"{row.bacc_mean:.4f} ± {row.bacc_std:.4f}"
        )
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {importance_path}")
    log(f"Saved: {OUTPUT_PNG}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
