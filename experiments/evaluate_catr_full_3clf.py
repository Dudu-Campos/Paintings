"""Paired full-cohort evaluation of numeric features with and without CATR.

The text branch is generated strictly out of fold inside each outer training
partition.  The untouched outer test partition is scored by a text model fitted
only on that fold's outer training data.  LDA, XGBoost and the current
BranchFusionMLP are evaluated on identical corrected labels and splits.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from compare_text_controls import first_complete_sentence
from compare_text_controls_mlp import nested_text_scores, run_mlp
from fusion_curve_slic import (
    N_FOLDS,
    ROOT,
    SEED,
    SLIC_DIR,
    log,
    make_model,
)
from slic_oof_stacking import load_dataset
from tfidf_caption_style import make_stop_words


CAPTIONS = ROOT / "outputs/affect_captions/catr_coco_full.jsonl"
OUTPUT_RESULTS = ROOT / "paper/catr_full_3clf_results.csv"
OUTPUT_FOLDS = ROOT / "paper/catr_full_3clf_folds.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/catr_full_3clf_predictions.csv"
OUTPUT_CLASS_DELTAS = ROOT / "paper/catr_full_3clf_class_deltas.csv"
OUTPUT_PNG = ROOT / "paper/catr_full_3clf.png"
CONDITION = "CATR COCO human full"


def load_paired() -> tuple[pd.DataFrame, dict[str, list[str]], int]:
    frame, groups = load_dataset()
    graph_labels = (
        pd.read_parquet(
            SLIC_DIR / "combined_features.parquet",
            columns=["stem", "style"],
        )
        .drop_duplicates("stem")
        .rename(columns={"style": "graph_style"})
    )
    frame = frame.merge(graph_labels, on="stem", how="left", validate="one_to_one")
    if frame["graph_style"].isna().any():
        raise ValueError("Some numeric rows lack the path-qualified graph style")
    corrected = int(frame["label"].ne(frame["graph_style"]).sum())
    frame["label"] = frame["graph_style"]
    frame = frame.drop(columns="graph_style")

    captions = pd.read_json(CAPTIONS, lines=True, dtype=False)
    if (
        len(captions) != 81_444
        or captions["image_id"].nunique() != 81_444
        or captions["caption"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Full CATR output failed completeness checks")
    captions = captions[["stem", "style", "caption"]]
    if captions.duplicated(["style", "stem"]).any():
        raise ValueError("CATR has duplicate (style, stem) pairs")
    frame = frame.merge(
        captions,
        left_on=["label", "stem"],
        right_on=["style", "stem"],
        how="left",
        validate="one_to_one",
    )
    if frame["caption"].isna().any():
        raise ValueError(f"Missing paired captions: {int(frame['caption'].isna().sum())}")
    frame["text"] = frame["caption"].map(first_complete_sentence)
    frame = frame.drop(columns="style")
    return frame, groups, corrected


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


def metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "bacc_oof": float(balanced_accuracy_score(true, predicted)),
        "accuracy_oof": float(accuracy_score(true, predicted)),
        "macro_f1_oof": float(f1_score(true, predicted, average="macro")),
    }


def plot_results(result: pd.DataFrame) -> None:
    order = ["LDA", "XGBoost", "BranchFusionMLP"]
    base = result[result["condition"] == "Numeric 274"].set_index("classifier")
    catr = result[result["condition"] == "Numeric 274 + CATR"].set_index("classifier")
    x = np.arange(len(order))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.bar(
        x - width / 2,
        [base.loc[name, "bacc_oof"] for name in order],
        width,
        label="Numeric 274",
        color="#7f8c8d",
    )
    bars = axis.bar(
        x + width / 2,
        [catr.loc[name, "bacc_oof"] for name in order],
        width,
        label="+ CATR COCO",
        color="#238b75",
    )
    for bar, name in zip(bars, order):
        delta = 100 * catr.loc[name, "delta_bacc_vs_numeric"]
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{delta:+.2f} pp",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_xticks(x, order)
    axis.set_ylabel("Balanced accuracy OOF")
    axis.set_ylim(0, max(catr["bacc_oof"].max(), base["bacc_oof"].max()) + 0.07)
    axis.set_title("CATR COCO no pipeline completo — 79.207 imagens")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame, groups, corrected_labels = load_paired()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_branch_sizes = [len(columns) for columns in groups.values()]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    stop_words = make_stop_words(encoder.classes_)
    stems = frame["stem"].to_numpy(dtype=str)
    texts = frame["text"].to_numpy(dtype=str)
    splits = list(
        StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED
        ).split(np.zeros(len(frame), dtype=np.uint8), labels)
    )
    log(
        f"CATR paired full cohort={len(frame):,}; numeric={len(numeric_columns)}; "
        f"styles={n_classes}; corrected ambiguous metadata labels={corrected_labels}"
    )

    prediction = {
        classifier: {
            condition: np.full(len(frame), -1, dtype=np.int64)
            for condition in ("Numeric 274", "Numeric 274 + CATR")
        }
        for classifier in ("LDA", "XGBoost", "BranchFusionMLP")
    }
    text_prediction = np.full(len(frame), -1, dtype=np.int64)
    fold_rows = []

    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        fold_started = time.perf_counter()
        train_text, test_text = nested_text_scores(
            CONDITION,
            fold,
            outer_train,
            outer_test,
            stems,
            texts,
            labels,
            stop_words,
            n_classes,
        )
        text_prediction[outer_test] = np.argmax(test_text, axis=1)
        augmented_train = np.concatenate(
            [numeric_values[outer_train], train_text], axis=1
        )
        augmented_test = np.concatenate(
            [numeric_values[outer_test], test_text], axis=1
        )

        lda_base = make_model(n_classes, numeric_values.shape[1])
        lda_base.fit(numeric_values[outer_train], labels[outer_train])
        prediction["LDA"]["Numeric 274"][outer_test] = lda_base.predict(
            numeric_values[outer_test]
        )
        lda_catr = make_model(n_classes, augmented_train.shape[1])
        lda_catr.fit(augmented_train, labels[outer_train])
        prediction["LDA"]["Numeric 274 + CATR"][outer_test] = lda_catr.predict(
            augmented_test
        )

        weights = compute_sample_weight("balanced", labels[outer_train])
        xgb_base = make_xgb(n_classes)
        xgb_base.fit(
            numeric_values[outer_train],
            labels[outer_train],
            sample_weight=weights,
        )
        prediction["XGBoost"]["Numeric 274"][outer_test] = xgb_base.predict(
            numeric_values[outer_test]
        )
        xgb_catr = make_xgb(n_classes)
        xgb_catr.fit(
            augmented_train,
            labels[outer_train],
            sample_weight=weights,
        )
        prediction["XGBoost"]["Numeric 274 + CATR"][outer_test] = xgb_catr.predict(
            augmented_test
        )

        fit_rel, validation_rel = train_test_split(
            np.arange(len(outer_train)),
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=SEED + fold,
        )
        fit_train = outer_train[fit_rel]
        validation = outer_train[validation_rel]
        seed = SEED + fold + 10_000
        base_mlp_prediction, base_epoch, base_elapsed = run_mlp(
            numeric_values,
            numeric_branch_sizes,
            labels,
            fit_train,
            validation,
            outer_test,
            n_classes,
            seed,
        )
        prediction["BranchFusionMLP"]["Numeric 274"][outer_test] = (
            base_mlp_prediction
        )
        augmented = np.empty(
            (len(frame), numeric_values.shape[1] + n_classes), dtype=np.float32
        )
        augmented[:, :numeric_values.shape[1]] = numeric_values
        augmented[outer_train, numeric_values.shape[1]:] = train_text
        augmented[outer_test, numeric_values.shape[1]:] = test_text
        catr_mlp_prediction, catr_epoch, catr_elapsed = run_mlp(
            augmented,
            numeric_branch_sizes + [n_classes],
            labels,
            fit_train,
            validation,
            outer_test,
            n_classes,
            seed,
        )
        prediction["BranchFusionMLP"]["Numeric 274 + CATR"][outer_test] = (
            catr_mlp_prediction
        )

        for classifier, conditions in prediction.items():
            for condition, values in conditions.items():
                fold_rows.append({
                    "fold": fold,
                    "classifier": classifier,
                    "condition": condition,
                    "bacc": balanced_accuracy_score(
                        labels[outer_test], values[outer_test]
                    ),
                    "n_train": len(outer_train),
                    "n_test": len(outer_test),
                    "best_epoch": (
                        base_epoch
                        if classifier == "BranchFusionMLP"
                        and condition == "Numeric 274"
                        else catr_epoch
                        if classifier == "BranchFusionMLP"
                        else np.nan
                    ),
                    "mlp_elapsed_seconds": (
                        base_elapsed
                        if classifier == "BranchFusionMLP"
                        and condition == "Numeric 274"
                        else catr_elapsed
                        if classifier == "BranchFusionMLP"
                        else np.nan
                    ),
                })
        pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)
        summary = {
            f"{classifier} {condition}": balanced_accuracy_score(
                labels[outer_test], values[outer_test]
            )
            for classifier, conditions in prediction.items()
            for condition, values in conditions.items()
        }
        log(
            f"Fold {fold}/{N_FOLDS}: "
            + "; ".join(f"{key}={value:.4f}" for key, value in summary.items())
            + f"; elapsed={time.perf_counter() - fold_started:.1f}s"
        )

    if np.any(text_prediction < 0) or any(
        np.any(values < 0)
        for conditions in prediction.values()
        for values in conditions.values()
    ):
        raise RuntimeError("Incomplete OOF predictions")

    detail = pd.DataFrame(fold_rows)
    result_rows = []
    class_rows = []
    for classifier, conditions in prediction.items():
        base_metrics = metrics(labels, conditions["Numeric 274"])
        for condition, values in conditions.items():
            subset = detail[
                (detail["classifier"] == classifier)
                & (detail["condition"] == condition)
            ].sort_values("fold")
            values_metrics = metrics(labels, values)
            base_correct = conditions["Numeric 274"] == labels
            current_correct = values == labels
            current_only = int(np.sum(current_correct & ~base_correct))
            base_only = int(np.sum(~current_correct & base_correct))
            discordant = current_only + base_only
            result_rows.append({
                "classifier": classifier,
                "condition": condition,
                "n_images": len(frame),
                "n_features": (
                    len(numeric_columns)
                    if condition == "Numeric 274"
                    else len(numeric_columns) + n_classes
                ),
                **values_metrics,
                "delta_bacc_vs_numeric": (
                    values_metrics["bacc_oof"] - base_metrics["bacc_oof"]
                ),
                "bacc_fold_mean": subset["bacc"].mean(),
                "bacc_fold_std": subset["bacc"].std(ddof=0),
                "catr_only_correct": current_only,
                "numeric_only_correct": base_only,
                "mcnemar_exact_p": (
                    float(
                        binomtest(
                            min(current_only, base_only), discordant, 0.5
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                ),
                **{
                    f"fold_{index + 1}": score
                    for index, score in enumerate(subset["bacc"])
                },
            })
        for style_index, style in enumerate(encoder.classes_):
            mask = labels == style_index
            base_recall = float(
                np.mean(conditions["Numeric 274"][mask] == style_index)
            )
            catr_recall = float(
                np.mean(conditions["Numeric 274 + CATR"][mask] == style_index)
            )
            class_rows.append({
                "classifier": classifier,
                "style": style,
                "support": int(mask.sum()),
                "numeric_recall": base_recall,
                "catr_recall": catr_recall,
                "delta_recall": catr_recall - base_recall,
            })

    text_metrics = metrics(labels, text_prediction)
    result_rows.append({
        "classifier": "Text TF-IDF SVM",
        "condition": "CATR only",
        "n_images": len(frame),
        "n_features": n_classes,
        **text_metrics,
        "delta_bacc_vs_numeric": np.nan,
        "bacc_fold_mean": np.nan,
        "bacc_fold_std": np.nan,
        "catr_only_correct": np.nan,
        "numeric_only_correct": np.nan,
        "mcnemar_exact_p": np.nan,
    })
    results = pd.DataFrame(result_rows)
    results.to_csv(OUTPUT_RESULTS, index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUT_CLASS_DELTAS, index=False)

    output = frame[["stem", "label"]].copy()
    output["text_prediction"] = encoder.inverse_transform(text_prediction)
    for classifier, conditions in prediction.items():
        slug = classifier.lower().replace(" ", "_")
        output[f"{slug}_numeric"] = encoder.inverse_transform(
            conditions["Numeric 274"]
        )
        output[f"{slug}_catr"] = encoder.inverse_transform(
            conditions["Numeric 274 + CATR"]
        )
    output.to_csv(OUTPUT_PREDICTIONS, index=False)
    plot_results(results)
    log("Results:\n" + results.to_string(index=False))
    log(f"Saved {OUTPUT_RESULTS}, {OUTPUT_FOLDS}, {OUTPUT_PNG}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
