"""Permutation placebo for adding the new TF-IDF text scores to numeric LDA.

The 27 text decision scores were generated strictly OOF inside each outer
training fold.  A placebo independently permutes whole text-score vectors in
the outer train and test partitions, preserving dimensionality and joint
distribution while destroying their association with each painting.
"""

from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log, make_model
from slic_oof_stacking import load_dataset
from tfidf_branch_mlp_fusion import CACHE_DIR


OUTPUT_PERMUTATIONS = ROOT / "paper/tfidf_lda_placebo_permutations.csv"
OUTPUT_SUMMARY = ROOT / "paper/tfidf_lda_placebo_summary.csv"
OUTPUT_PNG = ROOT / "paper/tfidf_lda_placebo_test.png"


def evaluate_fold(
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
    train_text: np.ndarray,
    test_text: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
) -> float:
    train_values = np.concatenate([train_numeric, train_text], axis=1)
    test_values = np.concatenate([test_numeric, test_text], axis=1)
    estimator = make_model(n_classes, train_values.shape[1])
    estimator.fit(train_values, train_labels)
    prediction = estimator.predict(test_values)
    return float(balanced_accuracy_score(test_labels, prediction))


def evaluate_base_fold(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
) -> float:
    estimator = make_model(n_classes, train_values.shape[1])
    estimator.fit(train_values, train_labels)
    return float(
        balanced_accuracy_score(test_labels, estimator.predict(test_values))
    )


def make_plot(placebo_delta: np.ndarray, real_delta: float) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.hist(
        100 * placebo_delta,
        bins=min(20, max(8, len(placebo_delta) // 3)),
        color="#8fa9bd",
        edgecolor="white",
        alpha=0.9,
    )
    axis.axvline(
        100 * placebo_delta.mean(),
        color="#425f73",
        linestyle="--",
        linewidth=2,
        label=f"Média placebo: {100 * placebo_delta.mean():+.3f} pp",
    )
    axis.axvline(
        100 * real_delta,
        color="#d1495b",
        linewidth=2.5,
        label=f"Texto real: {100 * real_delta:+.3f} pp",
    )
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_xlabel("Ganho sobre LDA-274 (pontos percentuais)")
    axis.set_ylabel("Número de permutações")
    axis.set_title(
        "Nova representação textual contra placebo na LDA\n"
        "27 escores TF-IDF/SVM reais ou embaralhados"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=50)
    args = parser.parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")

    started = time.perf_counter()
    frame, groups = load_dataset()
    labels = frame["label"].to_numpy(dtype=str)
    n_classes = len(np.unique(labels))
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(splitter.split(np.zeros(len(frame), dtype=np.uint8), labels))
    cached_text = []
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        cache_path = CACHE_DIR / f"outer_fold_{fold}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing nested text cache: {cache_path}")
        cache = np.load(cache_path)
        if not (
            np.array_equal(cache["outer_train"], outer_train)
            and np.array_equal(cache["outer_test"], outer_test)
        ):
            raise ValueError(f"Fold indices differ from cache: fold {fold}")
        cached_text.append((cache["train_decisions"], cache["test_decisions"]))
    log(
        f"Cohort={len(frame):,}; numeric={numeric_values.shape[1]}; "
        f"text={cached_text[0][0].shape[1]}; permutations={args.permutations}"
    )

    base_scores = []
    real_scores = []
    for fold, ((outer_train, outer_test), (train_text, test_text)) in enumerate(
        zip(splits, cached_text), 1
    ):
        base_scores.append(evaluate_base_fold(
            numeric_values[outer_train],
            numeric_values[outer_test],
            labels[outer_train],
            labels[outer_test],
            n_classes,
        ))
        real_scores.append(evaluate_fold(
            numeric_values[outer_train],
            numeric_values[outer_test],
            train_text,
            test_text,
            labels[outer_train],
            labels[outer_test],
            n_classes,
        ))
        log(
            f"Real fold {fold}/{N_FOLDS}: base={base_scores[-1]:.4f}; "
            f"text={real_scores[-1]:.4f}; "
            f"delta={100 * (real_scores[-1] - base_scores[-1]):+.3f} pp"
        )
    base_scores = np.asarray(base_scores)
    real_scores = np.asarray(real_scores)
    base_mean = float(base_scores.mean())
    real_mean = float(real_scores.mean())

    rows = []
    for permutation in range(1, args.permutations + 1):
        scores = []
        for fold, ((outer_train, outer_test), (train_text, test_text)) in enumerate(
            zip(splits, cached_text), 1
        ):
            rng = np.random.default_rng(
                SEED + 100_000 * permutation + fold
            )
            shuffled_train = train_text[rng.permutation(len(train_text))]
            shuffled_test = test_text[rng.permutation(len(test_text))]
            scores.append(evaluate_fold(
                numeric_values[outer_train],
                numeric_values[outer_test],
                shuffled_train,
                shuffled_test,
                labels[outer_train],
                labels[outer_test],
                n_classes,
            ))
        scores = np.asarray(scores)
        rows.append({
            "permutation": permutation,
            "bacc_mean": scores.mean(),
            "bacc_std": scores.std(),
            "delta_vs_base": scores.mean() - base_mean,
            **{f"fold_{index + 1}": score for index, score in enumerate(scores)},
        })
        pd.DataFrame(rows).to_csv(OUTPUT_PERMUTATIONS, index=False)
        if permutation == 1 or permutation % 5 == 0:
            log(
                f"Placebo {permutation}/{args.permutations}: "
                f"delta={100 * (scores.mean() - base_mean):+.3f} pp"
            )

    placebo = pd.DataFrame(rows)
    placebo_delta = placebo["delta_vs_base"].to_numpy(dtype=np.float64)
    real_delta = real_mean - base_mean
    empirical_p = (1 + np.sum(placebo_delta >= real_delta)) / (
        len(placebo_delta) + 1
    )
    summary = pd.DataFrame([{
        "n_images": len(frame),
        "n_folds": N_FOLDS,
        "n_permutations": len(placebo),
        "base_lda_274_bacc": base_mean,
        "real_text_lda_301_bacc": real_mean,
        "real_delta": real_delta,
        "placebo_delta_mean": placebo_delta.mean(),
        "placebo_delta_std": placebo_delta.std(),
        "placebo_delta_p025": np.quantile(placebo_delta, 0.025),
        "placebo_delta_p975": np.quantile(placebo_delta, 0.975),
        "placebo_delta_max": placebo_delta.max(),
        "empirical_p_placebo_ge_real": empirical_p,
        **{f"base_fold_{i + 1}": score for i, score in enumerate(base_scores)},
        **{f"real_fold_{i + 1}": score for i, score in enumerate(real_scores)},
    }])
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    make_plot(placebo_delta, real_delta)
    log("Summary:\n" + summary.to_string(index=False))
    for path in (OUTPUT_PERMUTATIONS, OUTPUT_SUMMARY, OUTPUT_PNG):
        log(f"Saved: {path}")
    log(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
