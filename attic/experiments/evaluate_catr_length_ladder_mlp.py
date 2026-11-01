"""CATR caption-length ladder on the fixed 12,210-image text cohort.

CATR emits one short COCO-style sentence, so its cumulative ladder is defined
by word prefixes rather than sentence prefixes.  Every level keeps the same
caption prefix, outer folds, nested text-score protocol and branch-fusion MLP.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluate_catr_coco import load_frame
from evaluate_qwen_sentence_ladder_mlp import (
    exact_paired_p,
    nested_text_scores,
    run_mlp,
)
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_oof_stacking import load_dataset
from tfidf_caption_style import make_stop_words


LEVELS: tuple[tuple[str, int | None], ...] = (
    ("3 words", 3),
    ("5 words", 5),
    ("7 words", 7),
    ("9 words", 9),
    ("11 words", 11),
    ("full caption", None),
)
CACHE_DIR = ROOT / "paper/cache_catr_length_ladder_nested"
BASE_PREDICTIONS = ROOT / "paper/qwen_sentence_ladder_mlp_predictions.csv"
PLACEBO_FILE = ROOT / "paper/text_controls_mlp_placebos.csv"
REFERENCE_CATR_RESULT = ROOT / "paper/catr_coco_mlp_results.csv"
OUTPUT_RESULTS = ROOT / "paper/catr_length_ladder_mlp_results.csv"
OUTPUT_FOLDS = ROOT / "paper/catr_length_ladder_mlp_folds.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/catr_length_ladder_mlp_predictions.csv"
OUTPUT_PLOT = ROOT / "paper/catr_length_ladder_mlp.png"


def word_prefix(text: str, limit: int | None) -> str:
    words = str(text).strip().split()
    if not words:
        raise ValueError("Empty CATR caption")
    return " ".join(words if limit is None else words[:limit])


def load_cohort() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    text = load_frame()
    if len(text) != 12_210 or text["stem"].nunique() != 12_210:
        raise ValueError("Unexpected fixed CATR cohort")
    for index, (_, limit) in enumerate(LEVELS, 1):
        text[f"text_{index}"] = text["text"].map(
            lambda caption, limit=limit: word_prefix(caption, limit)
        )

    numeric, groups = load_dataset()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric = numeric[["stem", "label"] + numeric_columns]
    frame = text.merge(
        numeric,
        on="stem",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_numeric"),
    )
    if len(frame) != len(text):
        raise ValueError(f"CATR/numeric cohort changed: {len(text)} -> {len(frame)}")
    mismatches = int(frame["label"].ne(frame["label_numeric"]).sum())
    if mismatches:
        log(
            f"Keeping {mismatches} path-qualified cohort labels over ambiguous "
            "stem-only metadata labels"
        )
    return frame, groups


def load_numeric_baseline(
    frame: pd.DataFrame,
    encoder: LabelEncoder,
    expected_folds: np.ndarray,
) -> np.ndarray:
    saved = pd.read_csv(BASE_PREDICTIONS, dtype=str)
    if saved["stem"].duplicated().any():
        raise ValueError("Duplicate stems in saved numeric predictions")
    saved = saved.set_index("stem")
    predictions = frame["stem"].map(saved["prediction_numeric"])
    folds = frame["stem"].map(saved["fold"])
    if predictions.isna().any() or folds.isna().any():
        raise ValueError("Incomplete saved numeric baseline")
    if not np.array_equal(folds.to_numpy(dtype=np.int8), expected_folds):
        raise ValueError("Saved baseline fold assignment differs")
    return encoder.transform(predictions.to_numpy(dtype=str))


def make_plot(results: pd.DataFrame) -> None:
    x = results["mean_words"].to_numpy(dtype=float)
    labels = results["level"].to_numpy(dtype=str)
    figure, (accuracy, gain) = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True
    )
    accuracy.axhline(
        results["numeric_bacc_oof"].iloc[0],
        color="#6b7280",
        linestyle="--",
        label="MLP numérico",
    )
    accuracy.plot(
        x, results["text_bacc_oof"], marker="o", linewidth=2.2,
        label="CATR isolado",
    )
    accuracy.plot(
        x, results["fusion_bacc_oof"], marker="o", linewidth=2.2,
        label="Numérico + CATR",
    )
    accuracy.set_xlabel("Palavras médias efetivamente usadas")
    accuracy.set_ylabel("Acurácia balanceada OOF")
    accuracy.grid(alpha=0.25)
    accuracy.legend()
    accuracy.set_title("Desempenho")

    gain.plot(
        x,
        100 * results["delta_bacc_vs_numeric"],
        marker="o",
        linewidth=2.2,
        color="#0f9d76",
    )
    for x_value, y_value, label in zip(
        x, 100 * results["delta_bacc_vs_numeric"], labels
    ):
        gain.annotate(
            label.replace(" words", "p").replace("full caption", "completa"),
            (x_value, y_value),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    gain.set_xlabel("Palavras médias efetivamente usadas")
    gain.set_ylabel("Ganho sobre o MLP numérico (p.p.)")
    gain.grid(alpha=0.25)
    gain.set_title("Ganho × comprimento")
    figure.suptitle("Escadinha de comprimento — CATR treinado somente em COCO")
    figure.savefig(OUTPUT_PLOT, dpi=180)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame, groups = load_cohort()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float32)
    numeric_branch_sizes = [len(columns) for columns in groups.values()]

    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    stems = frame["stem"].to_numpy(dtype=str)
    stop_words = make_stop_words(encoder.classes_)
    augmented_branch_sizes = numeric_branch_sizes + [n_classes]
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(
        splitter.split(np.zeros(len(frame), dtype=np.uint8), labels)
    )
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    for fold, (_, test) in enumerate(outer_splits, 1):
        fold_assignment[test] = fold
    numeric_prediction = load_numeric_baseline(
        frame, encoder, fold_assignment
    )
    numeric_bacc = balanced_accuracy_score(labels, numeric_prediction)

    text_predictions = {
        index: np.full(len(frame), -1, dtype=np.int64)
        for index in range(1, len(LEVELS) + 1)
    }
    fusion_predictions = {
        index: np.full(len(frame), -1, dtype=np.int64)
        for index in range(1, len(LEVELS) + 1)
    }
    fold_rows = []
    log(
        f"CATR length ladder cohort={len(frame):,}; styles={n_classes}; "
        f"numeric={numeric_values.shape[1]}; baseline={numeric_bacc:.4f}"
    )

    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=SEED + outer_fold,
        )
        seed = SEED + outer_fold + 10_000
        base_score = balanced_accuracy_score(
            labels[outer_test], numeric_prediction[outer_test]
        )
        for level_index, (level, limit) in enumerate(LEVELS, 1):
            texts = frame[f"text_{level_index}"].to_numpy(dtype=str)
            train_text, test_text = nested_text_scores(
                level_index,
                outer_fold,
                outer_train,
                outer_test,
                stems,
                texts,
                labels,
                stop_words,
                n_classes,
                cache_root=CACHE_DIR,
            )
            text_prediction = test_text.argmax(axis=1)
            text_predictions[level_index][outer_test] = text_prediction
            text_score = balanced_accuracy_score(
                labels[outer_test], text_prediction
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
            fusion_predictions[level_index][outer_test] = prediction
            fusion_score = balanced_accuracy_score(labels[outer_test], prediction)
            fold_rows.append({
                "fold": outer_fold,
                "level_index": level_index,
                "level": level,
                "word_limit": limit,
                "numeric_bacc": base_score,
                "text_bacc": text_score,
                "fusion_bacc": fusion_score,
                "delta_bacc_vs_numeric": fusion_score - base_score,
                "fusion_best_epoch": best_epoch,
                "fusion_elapsed_seconds": elapsed,
            })
            log(
                f"Outer {outer_fold}/{N_FOLDS} | {level}: "
                f"text={text_score:.4f}; fusion={fusion_score:.4f}; "
                f"delta={100 * (fusion_score - base_score):+.3f} pp"
            )
        pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)

    if any(np.any(values < 0) for values in text_predictions.values()):
        raise RuntimeError("Incomplete text predictions")
    if any(np.any(values < 0) for values in fusion_predictions.values()):
        raise RuntimeError("Incomplete fusion predictions")

    detail = pd.DataFrame(fold_rows)
    original_lengths = frame["text"].map(lambda text: len(str(text).split())).to_numpy()
    rows = []
    previous_prediction = numeric_prediction
    previous_bacc = numeric_bacc
    for level_index, (level, limit) in enumerate(LEVELS, 1):
        texts = frame[f"text_{level_index}"].to_numpy(dtype=str)
        lengths = np.asarray([len(text.split()) for text in texts])
        text_bacc = balanced_accuracy_score(labels, text_predictions[level_index])
        fusion_bacc = balanced_accuracy_score(labels, fusion_predictions[level_index])
        subset = detail[detail["level_index"] == level_index].sort_values("fold")
        row = {
            "level_index": level_index,
            "level": level,
            "word_limit": limit,
            "n_images": len(frame),
            "n_styles": n_classes,
            "mean_words": float(lengths.mean()),
            "median_words": float(np.median(lengths)),
            "share_captions_truncated": (
                float(np.mean(original_lengths > limit)) if limit is not None else 0.0
            ),
            "numeric_bacc_oof": numeric_bacc,
            "text_bacc_oof": text_bacc,
            "text_bacc_fold_mean": subset["text_bacc"].mean(),
            "text_bacc_fold_std": subset["text_bacc"].std(ddof=0),
            "fusion_bacc_oof": fusion_bacc,
            "fusion_bacc_fold_mean": subset["fusion_bacc"].mean(),
            "fusion_bacc_fold_std": subset["fusion_bacc"].std(ddof=0),
            "delta_bacc_vs_numeric": fusion_bacc - numeric_bacc,
            "delta_bacc_vs_previous": fusion_bacc - previous_bacc,
            "mcnemar_vs_numeric_p": exact_paired_p(
                labels, numeric_prediction, fusion_predictions[level_index]
            ),
            "mcnemar_vs_previous_p": exact_paired_p(
                labels, previous_prediction, fusion_predictions[level_index]
            ),
            "fusion_best_epoch_mean": subset["fusion_best_epoch"].mean(),
        }
        for fold_index, score in enumerate(subset["fusion_bacc"], 1):
            row[f"fusion_fold_{fold_index}"] = score
        rows.append(row)
        previous_prediction = fusion_predictions[level_index]
        previous_bacc = fusion_bacc

    results = pd.DataFrame(rows)
    if PLACEBO_FILE.exists():
        placebo = pd.read_csv(PLACEBO_FILE)["delta_bacc_vs_numeric"].to_numpy()
        results["prior_placebo_delta_mean"] = placebo.mean()
        results["prior_placebo_delta_p975"] = np.quantile(placebo, 0.975)
        results["prior_placebo_delta_max"] = placebo.max()
        results["empirical_p_placebo_ge_gain"] = results[
            "delta_bacc_vs_numeric"
        ].map(lambda value: (1 + np.sum(placebo >= value)) / (len(placebo) + 1))
    results.to_csv(OUTPUT_RESULTS, index=False)

    predictions = frame[["stem", "label"]].copy()
    predictions["fold"] = fold_assignment
    predictions["prediction_numeric"] = encoder.inverse_transform(numeric_prediction)
    for level_index, _ in enumerate(LEVELS, 1):
        predictions[f"prediction_text_{level_index}"] = encoder.inverse_transform(
            text_predictions[level_index]
        )
        predictions[f"prediction_fusion_{level_index}"] = encoder.inverse_transform(
            fusion_predictions[level_index]
        )
    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    make_plot(results)

    if REFERENCE_CATR_RESULT.exists():
        expected = float(pd.read_csv(REFERENCE_CATR_RESULT)["bacc_oof"].iloc[0])
        observed = float(results.iloc[-1]["fusion_bacc_oof"])
        if not np.isclose(expected, observed, atol=1e-12):
            raise RuntimeError(
                f"Full-caption reproduction failed: expected={expected}, "
                f"observed={observed}"
            )
        log(f"Full-caption result reproduced exactly: {observed:.6f}")
    log("CATR length ladder results:\n" + results.to_string(index=False))
    log(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
