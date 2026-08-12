"""Paired permutation-placebo test for the sentiment fusion gain.

Each placebo preserves all 16 sentiment columns and their joint distribution,
but assigns whole sentiment vectors to different images.  Thus dimensionality
and covariance are preserved while image-specific information is destroyed.
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


OUTPUT_CSV = ROOT / "paper/sentiment_placebo_permutations.csv"
OUTPUT_SUMMARY = ROOT / "paper/sentiment_placebo_summary.csv"
OUTPUT_PNG = ROOT / "paper/sentiment_placebo_test.png"


def evaluate(
    values: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    scores = []
    n_classes = len(np.unique(labels))
    for train, test in splits:
        estimator = make_model(n_classes, values.shape[1])
        estimator.fit(values[train], labels[train])
        prediction = estimator.predict(values[test])
        scores.append(balanced_accuracy_score(labels[test], prediction))
    return np.asarray(scores, dtype=np.float64)


def make_plot(
    placebo: pd.DataFrame,
    base_mean: float,
    real_mean: float,
) -> None:
    placebo_delta = placebo["bacc_mean"].to_numpy() - base_mean
    real_delta = real_mean - base_mean
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.hist(
        100 * placebo_delta,
        bins=min(20, max(8, len(placebo) // 3)),
        color="#8fa9bd",
        edgecolor="white",
        alpha=0.9,
    )
    axis.axvline(
        100 * placebo_delta.mean(), color="#425f73", linestyle="--",
        linewidth=2, label=f"Média placebo: {100 * placebo_delta.mean():+.3f} pp",
    )
    axis.axvline(
        100 * real_delta, color="#d1495b", linewidth=2.5,
        label=f"Sentimento real: {100 * real_delta:+.3f} pp",
    )
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_xlabel("Ganho sobre a base sem sentimento (pontos percentuais)")
    axis.set_ylabel("Número de permutações")
    axis.set_title(
        "Sentimento real contra placebo embaralhado\n"
        "Mesmo número de features, distribuição e covariância"
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
    sentiment_columns = groups["Sentiment"]
    base_columns = [
        column
        for name, columns in groups.items()
        if name != "Sentiment"
        for column in columns
    ]
    base_values = frame[base_columns].to_numpy(dtype=np.float64)
    sentiment_values = frame[sentiment_columns].to_numpy(dtype=np.float64)
    real_values = np.concatenate([base_values, sentiment_values], axis=1)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(splitter.split(np.zeros(len(frame), dtype=np.uint8), labels))
    log(
        f"Cohort={len(frame):,}; base={base_values.shape[1]} features; "
        f"sentiment={sentiment_values.shape[1]}; permutations={args.permutations}"
    )

    base_scores = evaluate(base_values, labels, splits)
    real_scores = evaluate(real_values, labels, splits)
    base_mean = float(base_scores.mean())
    real_mean = float(real_scores.mean())
    log(
        f"Base={base_mean:.6f}; real={real_mean:.6f}; "
        f"delta={100 * (real_mean - base_mean):+.3f} pp"
    )

    rows = []
    for permutation in range(1, args.permutations + 1):
        rng = np.random.default_rng(SEED + 10_000 + permutation)
        order = rng.permutation(len(frame))
        placebo_values = np.concatenate(
            [base_values, sentiment_values[order]], axis=1
        )
        scores = evaluate(placebo_values, labels, splits)
        rows.append({
            "permutation": permutation,
            "seed": SEED + 10_000 + permutation,
            "bacc_mean": scores.mean(),
            "bacc_std": scores.std(),
            "delta_vs_base": scores.mean() - base_mean,
            **{
                f"fold_{fold + 1}": score
                for fold, score in enumerate(scores)
            },
        })
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
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
        "base_bacc": base_mean,
        "real_sentiment_bacc": real_mean,
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
    make_plot(placebo, base_mean, real_mean)
    log("Summary:\n" + summary.to_string(index=False))
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {OUTPUT_SUMMARY}")
    log(f"Saved: {OUTPUT_PNG}")
    log(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
