"""Current branch MLP versus a fifth interpretable TF-IDF text-score branch.

Text scores for every outer-training image are produced out of sample by five
inner folds.  The final text model only scores the untouched outer test fold.
Both MLPs use the same outer/internal-validation splits and early stopping.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_branch_mlp_confusion import train_and_predict
from slic_branch_mlp_fusion import BranchFusionMLP, parameter_count
from slic_oof_stacking import load_dataset
from tfidf_caption_style import (
    clean_caption,
    load_caption_map,
    make_classifier,
    make_stop_words,
    make_vectorizer,
)


INNER_FOLDS = 5
OUTPUT_RESULTS = ROOT / "paper/tfidf_branch_mlp_fusion_results.csv"
OUTPUT_FOLDS = ROOT / "paper/tfidf_branch_mlp_fusion_folds.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/tfidf_branch_mlp_fusion_predictions.csv"
OUTPUT_TEXT_DECISIONS = ROOT / "paper/tfidf_caption_nested_oof_decisions.npy"
CACHE_DIR = ROOT / "paper/cache_tfidf_nested"


def aligned_decision(classifier, values, n_classes: int) -> np.ndarray:
    decision = classifier.decision_function(values)
    aligned = np.zeros((values.shape[0], n_classes), dtype=np.float32)
    aligned[:, classifier.classes_.astype(int)] = decision
    return aligned


def fit_text(train_texts: np.ndarray, train_labels: np.ndarray, stop_words):
    vectorizer = make_vectorizer(stop_words)
    train_values = vectorizer.fit_transform(train_texts)
    classifier = make_classifier()
    classifier.set_params(max_iter=20_000)
    classifier.fit(train_values, train_labels)
    return vectorizer, classifier


def nested_text_scores(
    outer_fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    texts: np.ndarray,
    labels: np.ndarray,
    stop_words: list[str],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"outer_fold_{outer_fold}.npz"
    if cache.exists():
        saved = np.load(cache)
        if (
            np.array_equal(saved["outer_train"], outer_train)
            and np.array_equal(saved["outer_test"], outer_test)
        ):
            log(f"Outer {outer_fold}: loading cached nested text scores")
            return saved["train_decisions"], saved["test_decisions"]

    train_decisions = np.zeros(
        (len(outer_train), n_classes), dtype=np.float32
    )
    inner_splitter = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + outer_fold,
    )
    inner_splits = inner_splitter.split(
        np.zeros(len(outer_train), dtype=np.uint8), labels[outer_train]
    )
    for inner_fold, (inner_train_rel, inner_validation_rel) in enumerate(
        inner_splits, 1
    ):
        inner_train = outer_train[inner_train_rel]
        inner_validation = outer_train[inner_validation_rel]
        vectorizer, classifier = fit_text(
            texts[inner_train], labels[inner_train], stop_words
        )
        validation_values = vectorizer.transform(texts[inner_validation])
        train_decisions[inner_validation_rel] = aligned_decision(
            classifier, validation_values, n_classes
        )
        log(
            f"Outer {outer_fold}/{N_FOLDS}, text inner "
            f"{inner_fold}/{INNER_FOLDS} done"
        )

    vectorizer, classifier = fit_text(
        texts[outer_train], labels[outer_train], stop_words
    )
    test_values = vectorizer.transform(texts[outer_test])
    test_decisions = aligned_decision(classifier, test_values, n_classes)
    np.savez_compressed(
        cache,
        outer_train=outer_train,
        outer_test=outer_test,
        train_decisions=train_decisions,
        test_decisions=test_decisions,
    )
    return train_decisions, test_decisions


def main() -> None:
    started = time.perf_counter()
    frame, groups = load_dataset()
    caption_map = load_caption_map()
    frame["caption"] = frame["stem"].astype(str).map(caption_map)
    if frame["caption"].isna().any():
        raise ValueError(f"Missing captions: {int(frame['caption'].isna().sum())}")
    texts = frame["caption"].map(clean_caption).to_numpy(dtype=str)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    stop_words = make_stop_words(encoder.classes_)
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float32)
    numeric_branch_sizes = [len(columns) for columns in groups.values()]
    augmented_branch_sizes = numeric_branch_sizes + [n_classes]
    parameter_counts = {
        "Current branch MLP": parameter_count(
            BranchFusionMLP(numeric_branch_sizes, n_classes)
        ),
        "Branch MLP + text": parameter_count(
            BranchFusionMLP(augmented_branch_sizes, n_classes)
        ),
    }
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    outer_splits = list(
        splitter.split(np.zeros(len(frame), dtype=np.uint8), labels)
    )
    log(
        f"Cohort={len(frame):,}; numeric branches={numeric_branch_sizes}; "
        f"text branch={n_classes}; parameters={parameter_counts}"
    )

    predictions = {
        name: np.empty(len(frame), dtype=np.int64)
        for name in parameter_counts
    }
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    text_oof_decisions = np.zeros(
        (len(frame), n_classes), dtype=np.float32
    )
    fold_rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        fold_started = time.perf_counter()
        train_text, test_text = nested_text_scores(
            outer_fold,
            outer_train,
            outer_test,
            texts,
            labels,
            stop_words,
            n_classes,
        )
        text_oof_decisions[outer_test] = test_text
        augmented_values = np.empty(
            (len(frame), numeric_values.shape[1] + n_classes), dtype=np.float32
        )
        augmented_values[:, :numeric_values.shape[1]] = numeric_values
        augmented_values[outer_train, numeric_values.shape[1]:] = train_text
        augmented_values[outer_test, numeric_values.shape[1]:] = test_text

        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=SEED + outer_fold,
        )
        seed = SEED + outer_fold + 10_000
        conditions = (
            (
                "Current branch MLP",
                numeric_values,
                numeric_branch_sizes,
            ),
            (
                "Branch MLP + text",
                augmented_values,
                augmented_branch_sizes,
            ),
        )
        for name, values, branch_sizes in conditions:
            scaler = StandardScaler()
            train_values = scaler.fit_transform(values[fit_train]).astype(np.float32)
            validation_values = scaler.transform(values[validation]).astype(np.float32)
            test_values = scaler.transform(values[outer_test]).astype(np.float32)
            prediction, best_epoch, elapsed = train_and_predict(
                branch_sizes,
                train_values,
                labels[fit_train],
                validation_values,
                labels[validation],
                test_values,
                n_classes,
                seed,
            )
            predictions[name][outer_test] = prediction
            score = balanced_accuracy_score(labels[outer_test], prediction)
            fold_rows.append({
                "fold": outer_fold,
                "architecture": name,
                "parameters": parameter_counts[name],
                "bacc": score,
                "best_epoch": best_epoch,
                "mlp_elapsed_seconds": elapsed,
                "outer_elapsed_seconds_so_far": time.perf_counter() - fold_started,
                "n_outer_train": len(outer_train),
                "n_fit_train": len(fit_train),
                "n_internal_validation": len(validation),
                "n_outer_test": len(outer_test),
            })
            pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)
            log(
                f"Outer {outer_fold}/{N_FOLDS} | {name:<20} "
                f"BAcc={score:.4f}; epoch={best_epoch}; MLP={elapsed:.1f}s"
            )
        fold_assignment[outer_test] = outer_fold
        np.save(OUTPUT_TEXT_DECISIONS, text_oof_decisions)

    result_rows = []
    detail = pd.DataFrame(fold_rows)
    for name, prediction in predictions.items():
        subset = detail[detail["architecture"] == name].sort_values("fold")
        scores = subset["bacc"].to_numpy(dtype=np.float64)
        result_rows.append({
            "architecture": name,
            "parameters": parameter_counts[name],
            "n_images": len(frame),
            "bacc_oof": balanced_accuracy_score(labels, prediction),
            "bacc_fold_mean": scores.mean(),
            "bacc_fold_std": scores.std(),
            "best_epoch_mean": subset["best_epoch"].mean(),
            **{f"fold_{index + 1}": score for index, score in enumerate(scores)},
        })
    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT_RESULTS, index=False)
    pd.DataFrame({
        "stem": frame["stem"].astype(str),
        "fold": fold_assignment,
        "true_style": encoder.inverse_transform(labels),
        "current_mlp_prediction": encoder.inverse_transform(
            predictions["Current branch MLP"]
        ),
        "text_mlp_prediction": encoder.inverse_transform(
            predictions["Branch MLP + text"]
        ),
    }).to_csv(OUTPUT_PREDICTIONS, index=False)
    log("Results:\n" + result.to_string(index=False))
    for path in (
        OUTPUT_RESULTS, OUTPUT_FOLDS, OUTPUT_PREDICTIONS, OUTPUT_TEXT_DECISIONS
    ):
        log(f"Saved: {path}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
