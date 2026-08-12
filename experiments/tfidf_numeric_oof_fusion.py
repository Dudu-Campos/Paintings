"""Leakage-free late fusion of interpretable caption text and numeric features.

For every outer fold, inner OOF predictions train a linear meta-classifier.
The numeric branch is the current 274-feature SLIC/descriptor/Hilbert/sentiment
representation; the text branch is the sparse caption TF-IDF LinearSVC.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log, make_model
from slic_oof_stacking import load_dataset
from tfidf_caption_style import (
    clean_caption,
    load_caption_map,
    make_classifier,
    make_stop_words,
    make_vectorizer,
)


INNER_FOLDS = 5
OUTPUT_RESULTS = ROOT / "paper/tfidf_numeric_oof_fusion_results.csv"
OUTPUT_FOLDS = ROOT / "paper/tfidf_numeric_oof_fusion_folds.csv"
OUTPUT_IMPORTANCE = ROOT / "paper/tfidf_numeric_oof_fusion_importance.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/tfidf_numeric_oof_fusion_predictions.csv"


def aligned_text_decision(classifier, values, classes: np.ndarray) -> np.ndarray:
    decision = classifier.decision_function(values)
    aligned = np.zeros((values.shape[0], len(classes)), dtype=np.float64)
    index = {label: position for position, label in enumerate(classes)}
    for local, label in enumerate(classifier.classes_):
        aligned[:, index[label]] = decision[:, local]
    return aligned


def aligned_numeric_probability(estimator, values, classes: np.ndarray) -> np.ndarray:
    probability = estimator.predict_proba(values)
    local_classes = estimator.named_steps["lda"].classes_
    aligned = np.zeros((len(values), len(classes)), dtype=np.float64)
    index = {label: position for position, label in enumerate(classes)}
    for local, label in enumerate(local_classes):
        aligned[:, index[label]] = probability[:, local]
    return aligned


def fit_text(train_texts: np.ndarray, train_labels: np.ndarray, stop_words):
    vectorizer = make_vectorizer(stop_words)
    train_values = vectorizer.fit_transform(train_texts)
    classifier = make_classifier()
    classifier.fit(train_values, train_labels)
    return vectorizer, classifier


def make_meta() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1_000,
            random_state=SEED,
        )),
    ])


def main() -> None:
    started = time.perf_counter()
    frame, groups = load_dataset()
    caption_map = load_caption_map()
    frame["caption"] = frame["stem"].astype(str).map(caption_map)
    if frame["caption"].isna().any():
        raise ValueError(f"Missing captions: {int(frame['caption'].isna().sum())}")
    texts = frame["caption"].map(clean_caption).to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels)
    stop_words = make_stop_words(classes)
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float64)
    outer_splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    outer_splits = list(
        outer_splitter.split(np.zeros(len(frame), dtype=np.uint8), labels)
    )
    log(
        f"Cohort={len(frame):,}; numeric={len(numeric_columns)}; "
        f"styles={len(classes)}; outer={N_FOLDS} x inner={INNER_FOLDS}"
    )

    predictions = {
        "Numeric 274 LDA": np.empty(len(frame), dtype=object),
        "Text TF-IDF SVM": np.empty(len(frame), dtype=object),
        "OOF text + numeric": np.empty(len(frame), dtype=object),
    }
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    fold_rows = []
    importance_rows = []

    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        fold_started = time.perf_counter()
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + outer_fold,
        )
        inner_splits = list(
            inner_splitter.split(
                np.zeros(len(outer_train), dtype=np.uint8), labels[outer_train]
            )
        )
        train_meta = np.zeros(
            (len(outer_train), 2 * len(classes)), dtype=np.float64
        )
        for inner_fold, (inner_train_rel, inner_validation_rel) in enumerate(
            inner_splits, 1
        ):
            inner_train = outer_train[inner_train_rel]
            inner_validation = outer_train[inner_validation_rel]
            vectorizer, text_classifier = fit_text(
                texts[inner_train], labels[inner_train], stop_words
            )
            validation_text = vectorizer.transform(texts[inner_validation])
            train_meta[inner_validation_rel, :len(classes)] = (
                aligned_text_decision(text_classifier, validation_text, classes)
            )
            numeric_classifier = make_model(len(classes), len(numeric_columns))
            numeric_classifier.fit(
                numeric_values[inner_train], labels[inner_train]
            )
            train_meta[inner_validation_rel, len(classes):] = (
                aligned_numeric_probability(
                    numeric_classifier, numeric_values[inner_validation], classes
                )
            )
            log(
                f"Outer {outer_fold}/{N_FOLDS}, inner {inner_fold}/{INNER_FOLDS} done"
            )

        final_vectorizer, final_text_classifier = fit_text(
            texts[outer_train], labels[outer_train], stop_words
        )
        outer_text_values = final_vectorizer.transform(texts[outer_test])
        test_text_meta = aligned_text_decision(
            final_text_classifier, outer_text_values, classes
        )
        text_prediction = final_text_classifier.predict(outer_text_values)

        final_numeric = make_model(len(classes), len(numeric_columns))
        final_numeric.fit(numeric_values[outer_train], labels[outer_train])
        test_numeric_meta = aligned_numeric_probability(
            final_numeric, numeric_values[outer_test], classes
        )
        numeric_prediction = final_numeric.predict(numeric_values[outer_test])
        test_meta = np.concatenate(
            [test_text_meta, test_numeric_meta], axis=1
        )
        meta = make_meta()
        meta.fit(train_meta, labels[outer_train])
        fused_prediction = meta.predict(test_meta)

        predictions["Numeric 274 LDA"][outer_test] = numeric_prediction
        predictions["Text TF-IDF SVM"][outer_test] = text_prediction
        predictions["OOF text + numeric"][outer_test] = fused_prediction
        fold_assignment[outer_test] = outer_fold
        scores = {
            name: balanced_accuracy_score(labels[outer_test], prediction[outer_test])
            for name, prediction in predictions.items()
        }
        for name, score in scores.items():
            fold_rows.append({
                "fold": outer_fold,
                "strategy": name,
                "bacc": score,
                "n_train": len(outer_train),
                "n_test": len(outer_test),
            })

        coefficients = np.abs(meta.named_steps["logistic"].coef_)
        branch_importance = coefficients.sum(axis=(0,))
        text_importance = branch_importance[:len(classes)].sum()
        numeric_importance = branch_importance[len(classes):].sum()
        total = text_importance + numeric_importance
        importance_rows.append({
            "fold": outer_fold,
            "text_importance": text_importance / total,
            "numeric_importance": numeric_importance / total,
        })
        pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)
        pd.DataFrame(importance_rows).to_csv(OUTPUT_IMPORTANCE, index=False)
        log(
            f"Outer {outer_fold}/{N_FOLDS}: "
            f"numeric={scores['Numeric 274 LDA']:.4f}; "
            f"text={scores['Text TF-IDF SVM']:.4f}; "
            f"fusion={scores['OOF text + numeric']:.4f}; "
            f"{time.perf_counter() - fold_started:.1f}s"
        )

    if np.any(fold_assignment == 0):
        raise RuntimeError("Incomplete OOF predictions")
    detail = pd.DataFrame(fold_rows)
    result_rows = []
    for name, prediction in predictions.items():
        fold_scores = detail[detail["strategy"] == name].sort_values("fold")["bacc"]
        result_rows.append({
            "strategy": name,
            "n_images": len(frame),
            "bacc_oof": balanced_accuracy_score(labels, prediction),
            "bacc_fold_mean": fold_scores.mean(),
            "bacc_fold_std": fold_scores.std(ddof=0),
            **{f"fold_{i + 1}": score for i, score in enumerate(fold_scores)},
        })
    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT_RESULTS, index=False)
    prediction_frame = pd.DataFrame({
        "stem": frame["stem"].astype(str),
        "fold": fold_assignment,
        "true_style": labels,
        "numeric_prediction": predictions["Numeric 274 LDA"],
        "text_prediction": predictions["Text TF-IDF SVM"],
        "fused_prediction": predictions["OOF text + numeric"],
    })
    prediction_frame.to_csv(OUTPUT_PREDICTIONS, index=False)
    log("Results:\n" + result.to_string(index=False))
    log("Importance:\n" + pd.DataFrame(importance_rows).to_string(index=False))
    for path in (OUTPUT_RESULTS, OUTPUT_FOLDS, OUTPUT_IMPORTANCE, OUTPUT_PREDICTIONS):
        log(f"Saved: {path}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
