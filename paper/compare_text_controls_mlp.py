"""Nested-CV branch-MLP comparison for four paired text representations.

For each outer fold, text scores for outer-training paintings are generated
strictly out of sample by inner folds.  A model fitted on the complete outer
training partition scores the untouched outer test partition.  Gaussian noise
branches with the same 27-dimensional width test whether any gain is merely
caused by the larger branch-fusion architecture.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from compare_text_controls import CONDITIONS, load_cohort
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_branch_mlp_confusion import train_and_predict
from slic_branch_mlp_fusion import BranchFusionMLP, parameter_count
from slic_oof_stacking import load_dataset
from tfidf_caption_style import make_classifier, make_stop_words, make_vectorizer


INNER_FOLDS = 5
CACHE_DIR = ROOT / "paper/cache_text_controls_nested"
OUTPUT_RESULTS = ROOT / "paper/text_controls_mlp_results.csv"
OUTPUT_FOLDS = ROOT / "paper/text_controls_mlp_folds.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/text_controls_mlp_predictions.csv"
OUTPUT_PLACEBOS = ROOT / "paper/text_controls_mlp_placebos.csv"
OUTPUT_PAIRWISE = ROOT / "paper/text_controls_mlp_pairwise.csv"


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def aligned_decision(classifier, values, n_classes: int) -> np.ndarray:
    decision = classifier.decision_function(values)
    aligned = np.zeros((values.shape[0], n_classes), dtype=np.float32)
    aligned[:, classifier.classes_.astype(int)] = decision
    return aligned


def fit_text(texts: np.ndarray, labels: np.ndarray, stop_words: list[str]):
    vectorizer = make_vectorizer(stop_words)
    values = vectorizer.fit_transform(texts)
    classifier = make_classifier()
    classifier.set_params(max_iter=20_000)
    classifier.fit(values, labels)
    return vectorizer, classifier


def nested_text_scores(
    condition: str,
    outer_fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    stems: np.ndarray,
    texts: np.ndarray,
    labels: np.ndarray,
    stop_words: list[str],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    condition_dir = CACHE_DIR / slug(condition)
    condition_dir.mkdir(parents=True, exist_ok=True)
    cache = condition_dir / f"outer_fold_{outer_fold}.npz"
    if cache.exists():
        saved = np.load(cache)
        if (
            np.array_equal(saved["outer_train"], outer_train)
            and np.array_equal(saved["outer_test"], outer_test)
            and np.array_equal(saved["stems"], stems)
        ):
            log(f"Outer {outer_fold}: loading cached {condition} scores")
            return saved["train_decisions"], saved["test_decisions"]

    train_decisions = np.zeros((len(outer_train), n_classes), dtype=np.float32)
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
            f"Outer {outer_fold}/{N_FOLDS}, {condition}, "
            f"inner {inner_fold}/{INNER_FOLDS}"
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
        stems=stems,
        train_decisions=train_decisions,
        test_decisions=test_decisions,
    )
    return train_decisions, test_decisions


def run_mlp(
    values: np.ndarray,
    branch_sizes: list[int],
    labels: np.ndarray,
    fit_train: np.ndarray,
    validation: np.ndarray,
    outer_test: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    scaler = StandardScaler()
    train_values = scaler.fit_transform(values[fit_train]).astype(np.float32)
    validation_values = scaler.transform(values[validation]).astype(np.float32)
    test_values = scaler.transform(values[outer_test]).astype(np.float32)
    return train_and_predict(
        branch_sizes,
        train_values,
        labels[fit_train],
        validation_values,
        labels[validation],
        test_values,
        n_classes,
        seed,
    )


def prediction_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "bacc_oof": balanced_accuracy_score(true, predicted),
        "accuracy_oof": accuracy_score(true, predicted),
        "macro_f1_oof": f1_score(true, predicted, average="macro"),
    }


def paired_comparison(
    labels: np.ndarray,
    base: np.ndarray,
    condition: str,
    prediction: np.ndarray,
) -> dict:
    base_correct = base == labels
    text_correct = prediction == labels
    text_only = int(np.sum(text_correct & ~base_correct))
    base_only = int(np.sum(~text_correct & base_correct))
    discordant = text_only + base_only
    p_value = (
        float(binomtest(min(text_only, base_only), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return {
        "condition": condition,
        "text_only_correct": text_only,
        "base_only_correct": base_only,
        "net_correct": text_only - base_only,
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placebos", type=int, default=20)
    args = parser.parse_args()
    if args.placebos < 1:
        raise ValueError("--placebos must be positive")

    started = time.perf_counter()
    text_frame = load_cohort()
    numeric_frame, groups = load_dataset()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_frame = numeric_frame[["stem", "label"] + numeric_columns]
    frame = text_frame.merge(
        numeric_frame,
        on="stem",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_numeric"),
    )
    if len(frame) != len(text_frame):
        raise ValueError(
            f"Numeric intersection changed cohort: {len(text_frame)} -> {len(frame)}"
        )
    if not frame["label"].equals(frame["label_numeric"]):
        raise ValueError("Text/numeric labels differ")
    frame = frame.drop(columns="label_numeric")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    stop_words = make_stop_words(encoder.classes_)
    stems = frame["stem"].to_numpy(dtype=str)
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float32)
    numeric_branch_sizes = [len(columns) for columns in groups.values()]
    augmented_branch_sizes = numeric_branch_sizes + [n_classes]
    parameters = {
        "Numeric MLP": parameter_count(
            BranchFusionMLP(numeric_branch_sizes, n_classes)
        ),
        "Augmented MLP": parameter_count(
            BranchFusionMLP(augmented_branch_sizes, n_classes)
        ),
    }
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(
        splitter.split(np.zeros(len(frame), dtype=np.uint8), labels)
    )
    log(
        f"MLP paired cohort={len(frame):,}; numeric={numeric_values.shape[1]}; "
        f"text={n_classes}; device cohort has {len(encoder.classes_)} styles; "
        f"Gaussian placebos={args.placebos}"
    )

    real_predictions = {
        "Numeric MLP": np.full(len(frame), -1, dtype=np.int64),
        **{
            condition: np.full(len(frame), -1, dtype=np.int64)
            for condition in CONDITIONS
        },
    }
    placebo_predictions = np.full(
        (args.placebos, len(frame)), -1, dtype=np.int64
    )
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    fold_rows = []

    placebo_values = []
    for permutation in range(1, args.placebos + 1):
        rng = np.random.default_rng(SEED + 100_000 * permutation)
        noise = rng.standard_normal((len(frame), n_classes)).astype(np.float32)
        placebo_values.append(np.concatenate([numeric_values, noise], axis=1))

    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        fold_started = time.perf_counter()
        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=SEED + outer_fold,
        )
        seed = SEED + outer_fold + 10_000

        prediction, best_epoch, elapsed = run_mlp(
            numeric_values,
            numeric_branch_sizes,
            labels,
            fit_train,
            validation,
            outer_test,
            n_classes,
            seed,
        )
        real_predictions["Numeric MLP"][outer_test] = prediction
        base_score = balanced_accuracy_score(labels[outer_test], prediction)
        fold_rows.append({
            "fold": outer_fold,
            "kind": "base",
            "condition": "Numeric MLP",
            "permutation": 0,
            "bacc": base_score,
            "best_epoch": best_epoch,
            "mlp_elapsed_seconds": elapsed,
        })
        log(
            f"Outer {outer_fold}/{N_FOLDS} | Numeric MLP "
            f"BAcc={base_score:.4f}; epoch={best_epoch}; {elapsed:.1f}s"
        )

        for condition in CONDITIONS:
            train_text, test_text = nested_text_scores(
                condition,
                outer_fold,
                outer_train,
                outer_test,
                stems,
                frame[condition].to_numpy(dtype=str),
                labels,
                stop_words,
                n_classes,
            )
            augmented = np.empty(
                (len(frame), numeric_values.shape[1] + n_classes),
                dtype=np.float32,
            )
            augmented[:, :numeric_values.shape[1]] = numeric_values
            augmented[outer_train, numeric_values.shape[1]:] = train_text
            augmented[outer_test, numeric_values.shape[1]:] = test_text
            prediction, best_epoch, elapsed = run_mlp(
                augmented,
                augmented_branch_sizes,
                labels,
                fit_train,
                validation,
                outer_test,
                n_classes,
                seed,
            )
            real_predictions[condition][outer_test] = prediction
            score = balanced_accuracy_score(labels[outer_test], prediction)
            fold_rows.append({
                "fold": outer_fold,
                "kind": "real",
                "condition": condition,
                "permutation": 0,
                "bacc": score,
                "best_epoch": best_epoch,
                "mlp_elapsed_seconds": elapsed,
            })
            log(
                f"Outer {outer_fold}/{N_FOLDS} | {condition:<20} "
                f"BAcc={score:.4f}; delta={100 * (score - base_score):+.3f} pp"
            )

        for permutation, values in enumerate(placebo_values, 1):
            prediction, best_epoch, elapsed = run_mlp(
                values,
                augmented_branch_sizes,
                labels,
                fit_train,
                validation,
                outer_test,
                n_classes,
                seed,
            )
            placebo_predictions[permutation - 1, outer_test] = prediction
            score = balanced_accuracy_score(labels[outer_test], prediction)
            fold_rows.append({
                "fold": outer_fold,
                "kind": "placebo",
                "condition": "Gaussian noise",
                "permutation": permutation,
                "bacc": score,
                "best_epoch": best_epoch,
                "mlp_elapsed_seconds": elapsed,
            })
            if permutation == 1 or permutation % 5 == 0:
                log(
                    f"Outer {outer_fold}/{N_FOLDS} | placebo "
                    f"{permutation}/{args.placebos}: BAcc={score:.4f}"
                )
        fold_assignment[outer_test] = outer_fold
        pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)

    if any(np.any(values < 0) for values in real_predictions.values()):
        raise RuntimeError("Incomplete real OOF predictions")
    if np.any(placebo_predictions < 0) or np.any(fold_assignment == 0):
        raise RuntimeError("Incomplete placebo OOF predictions")

    base_prediction = real_predictions["Numeric MLP"]
    base_metrics = prediction_metrics(labels, base_prediction)
    detail = pd.DataFrame(fold_rows)
    result_rows = []
    for condition, prediction in real_predictions.items():
        kind = "base" if condition == "Numeric MLP" else "real"
        subset = detail[
            (detail["kind"] == kind) & (detail["condition"] == condition)
        ].sort_values("fold")
        scores = subset["bacc"].to_numpy(dtype=float)
        metrics = prediction_metrics(labels, prediction)
        result_rows.append({
            "condition": condition,
            "kind": kind,
            "parameters": (
                parameters["Numeric MLP"] if kind == "base"
                else parameters["Augmented MLP"]
            ),
            "n_images": len(frame),
            **metrics,
            "delta_bacc_vs_numeric": metrics["bacc_oof"] - base_metrics["bacc_oof"],
            "bacc_fold_mean": scores.mean(),
            "bacc_fold_std": scores.std(),
            "best_epoch_mean": subset["best_epoch"].mean(),
            **{f"fold_{index + 1}": score for index, score in enumerate(scores)},
        })
    results = pd.DataFrame(result_rows).sort_values("bacc_oof", ascending=False)
    results.to_csv(OUTPUT_RESULTS, index=False)

    placebo_rows = []
    for permutation, prediction in enumerate(placebo_predictions, 1):
        metrics = prediction_metrics(labels, prediction)
        placebo_rows.append({
            "permutation": permutation,
            **metrics,
            "delta_bacc_vs_numeric": metrics["bacc_oof"] - base_metrics["bacc_oof"],
        })
    placebos = pd.DataFrame(placebo_rows)
    placebos.to_csv(OUTPUT_PLACEBOS, index=False)

    pairwise = pd.DataFrame([
        paired_comparison(labels, base_prediction, condition, prediction)
        for condition, prediction in real_predictions.items()
        if condition != "Numeric MLP"
    ])
    real_deltas = results.set_index("condition")["delta_bacc_vs_numeric"]
    placebo_deltas = placebos["delta_bacc_vs_numeric"].to_numpy(dtype=float)
    pairwise["real_delta_bacc"] = pairwise["condition"].map(real_deltas)
    pairwise["placebo_delta_mean"] = placebo_deltas.mean()
    pairwise["placebo_delta_std"] = placebo_deltas.std()
    pairwise["placebo_delta_p025"] = np.quantile(placebo_deltas, 0.025)
    pairwise["placebo_delta_p975"] = np.quantile(placebo_deltas, 0.975)
    pairwise["placebo_delta_max"] = placebo_deltas.max()
    pairwise["empirical_p_placebo_ge_real"] = pairwise["real_delta_bacc"].map(
        lambda delta: (1 + np.sum(placebo_deltas >= delta)) / (len(placebo_deltas) + 1)
    )
    pairwise.to_csv(OUTPUT_PAIRWISE, index=False)

    prediction_frame = frame[["stem", "label"]].copy()
    prediction_frame["fold"] = fold_assignment
    for condition, prediction in real_predictions.items():
        prediction_frame[f"prediction_{slug(condition)}"] = (
            encoder.inverse_transform(prediction)
        )
    prediction_frame.to_csv(OUTPUT_PREDICTIONS, index=False)
    log("MLP results:\n" + results.to_string(index=False))
    log("Paired/placebo comparison:\n" + pairwise.to_string(index=False))
    log(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
