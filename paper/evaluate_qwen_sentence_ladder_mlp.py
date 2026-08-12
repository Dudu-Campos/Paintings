"""Evaluate how cumulative Qwen caption length changes textual and fused accuracy.

The six-line caption is generated once per image.  This experiment evaluates
the identical prefix after lines 1..6, so each step adds information without
changing any earlier text.  Text scores used by the branch-fusion MLP are
strictly nested out of sample inside each outer training fold.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fusion_curve_slic import N_FOLDS, ROOT, SEED, SLIC_DIR, log
from slic_branch_mlp_confusion import train_and_predict
from slic_branch_mlp_fusion import BranchFusionMLP, parameter_count
from slic_oof_stacking import load_dataset
from tfidf_caption_style import make_classifier, make_stop_words, make_vectorizer


INNER_FOLDS = 5
MAX_SENTENCES = 6
LADDER_FILE = (
    ROOT / "outputs/affect_captions/qwen_neutral_6sentence_ladder.jsonl"
)
SAMPLE_FILE = ROOT / "paper/qwen_prompt_ablation_sample.csv"
CACHE_DIR = ROOT / "paper/cache_qwen_sentence_ladder_nested"
OUTPUT_RESULTS = ROOT / "paper/qwen_sentence_ladder_mlp_results.csv"
OUTPUT_FOLDS = ROOT / "paper/qwen_sentence_ladder_mlp_folds.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/qwen_sentence_ladder_mlp_predictions.csv"
OUTPUT_PLOT = ROOT / "paper/qwen_sentence_ladder_mlp.png"
PLACEBO_FILE = ROOT / "paper/text_controls_mlp_placebos.csv"

NUMBERED_LINE_RE = re.compile(
    r"^\s*(?:line\s*)?(\d+)\s*[:.)-]\s*(.+?)\s*$", re.IGNORECASE
)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def parse_numbered_lines(caption: str) -> tuple[list[str], list[str]]:
    """Return both raw numbered lines and content without numbering."""
    raw_lines = [line.strip() for line in str(caption).splitlines() if line.strip()]
    if len(raw_lines) != MAX_SENTENCES:
        raise ValueError(f"Expected six non-empty lines, found {len(raw_lines)}")
    clean_lines = []
    for expected, line in enumerate(raw_lines, 1):
        match = NUMBERED_LINE_RE.match(line)
        if match is None or int(match.group(1)) != expected:
            raise ValueError(f"Malformed line {expected}: {line!r}")
        clean_lines.append(match.group(2).strip())
    return raw_lines, clean_lines


def load_ladder() -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    rows = []
    for item in read_jsonl(LADDER_FILE):
        raw_lines, clean_lines = parse_numbered_lines(str(item["caption"]))
        row = {
            "stem": str(item["stem"]),
            "ladder_style": str(item["style"]),
            "generated_tokens_6": int(item["generated_tokens"]),
            "generation_seconds_6": float(
                item["batch_generation_seconds_per_image"]
            ),
        }
        for count in range(1, MAX_SENTENCES + 1):
            row[f"text_{count}"] = " ".join(clean_lines[:count])
            row[f"raw_text_{count}"] = "\n".join(raw_lines[:count])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["stem"].duplicated().any():
        raise ValueError("Ladder contains duplicate stems")

    token_counts: dict[int, np.ndarray] = {}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", local_files_only=True
        )
        for count in range(1, MAX_SENTENCES + 1):
            token_counts[count] = np.asarray([
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in frame[f"raw_text_{count}"].to_numpy(dtype=str)
            ])
    except Exception as error:  # Token counts are descriptive, never model input.
        log(f"Tokenizer unavailable; output-token means will be NaN: {error}")
        for count in range(1, MAX_SENTENCES + 1):
            token_counts[count] = np.full(len(frame), np.nan)

    cost = {}
    for count in range(1, MAX_SENTENCES + 1):
        texts = frame[f"text_{count}"].to_numpy(dtype=str)
        words = np.asarray([len(WORD_RE.findall(text)) for text in texts])
        tokens = token_counts[count]
        cost[count] = {
            "mean_words": float(words.mean()),
            "mean_output_tokens": float(np.nanmean(tokens)),
            "mean_generation_seconds_6": float(
                frame["generation_seconds_6"].mean()
            ),
        }
    full_tokens = cost[MAX_SENTENCES]["mean_output_tokens"]
    for count in range(1, MAX_SENTENCES + 1):
        cost[count]["output_token_fraction_of_6"] = (
            cost[count]["mean_output_tokens"] / full_tokens
            if np.isfinite(full_tokens) and full_tokens > 0
            else np.nan
        )
    return frame, cost


def load_cohort() -> tuple[
    pd.DataFrame, dict[str, list[str]], dict[int, dict[str, float]]
]:
    sample = pd.read_csv(SAMPLE_FILE, dtype=str)
    if sample["stem"].duplicated().any():
        raise ValueError("Sample contains duplicate stems")
    sample = sample.sort_values(["style", "stem"]).reset_index(drop=True)
    ladder, cost = load_ladder()
    frame = sample[["stem", "style", "path"]].merge(
        ladder, on="stem", how="inner", validate="one_to_one"
    )
    if len(frame) != len(sample) or len(frame) != len(ladder):
        raise ValueError(
            f"Sample/ladder mismatch: sample={len(sample)}, ladder={len(ladder)}, "
            f"intersection={len(frame)}"
        )
    if not frame["style"].equals(frame["ladder_style"]):
        raise ValueError("Sample and ladder style labels differ")

    numeric, groups = load_dataset()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric = numeric[["stem", "label"] + numeric_columns]
    frame = frame.merge(numeric, on="stem", how="inner", validate="one_to_one")
    if len(frame) != len(sample):
        raise ValueError(
            f"Numeric intersection changed cohort: {len(sample)} -> {len(frame)}"
        )
    # ``load_dataset`` reconstructs labels through a stem-only ArtEmis map.
    # Six filenames occur under more than one WikiArt style, so that map's
    # last value differs from the actual path whose features were extracted.
    # The SLIC table preserves the extraction-time style and resolves them.
    graph_labels = pd.read_parquet(
        SLIC_DIR / "combined_features.parquet", columns=["stem", "style"]
    ).rename(columns={"style": "graph_style"})
    frame = frame.merge(
        graph_labels, on="stem", how="left", validate="one_to_one"
    )
    if frame["graph_style"].isna().any():
        raise ValueError("Missing extraction-time graph labels")
    if not np.array_equal(
        frame["style"].to_numpy(dtype=str),
        frame["graph_style"].to_numpy(dtype=str),
    ):
        raise ValueError("Sample labels differ from extraction-time graph labels")
    metadata_mismatches = int(np.sum(
        frame["style"].to_numpy(dtype=str) != frame["label"].to_numpy(dtype=str)
    ))
    if metadata_mismatches:
        log(
            f"Correcting {metadata_mismatches} ambiguous stem-only metadata "
            "labels from extraction-time graph style"
        )
    frame["label"] = frame["style"]
    return frame, groups, cost


def digest_texts(stems: np.ndarray, texts: np.ndarray) -> str:
    digest = hashlib.sha256()
    for stem, caption in zip(stems, texts):
        digest.update(stem.encode("utf-8"))
        digest.update(b"\0")
        digest.update(caption.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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
    sentence_count: int,
    outer_fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    stems: np.ndarray,
    texts: np.ndarray,
    labels: np.ndarray,
    stop_words: list[str],
    n_classes: int,
    cache_root: Path = CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = cache_root / f"sentences_{sentence_count}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"outer_fold_{outer_fold}.npz"
    text_digest = digest_texts(stems, texts)
    if cache.exists():
        saved = np.load(cache)
        if (
            np.array_equal(saved["outer_train"], outer_train)
            and np.array_equal(saved["outer_test"], outer_test)
            and np.array_equal(saved["stems"], stems)
            and str(saved["text_digest"].item()) == text_digest
        ):
            log(
                f"Outer {outer_fold}: loading cached {sentence_count}-sentence scores"
            )
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
            f"Outer {outer_fold}/{N_FOLDS}, sentences={sentence_count}, "
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
        text_digest=text_digest,
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


def exact_paired_p(
    labels: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    left_correct = left == labels
    right_correct = right == labels
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    return (
        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )


def make_plot(results: pd.DataFrame) -> None:
    figure, (axis_accuracy, axis_cost) = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True
    )
    x = results["sentences"].to_numpy()
    axis_accuracy.axhline(
        results["numeric_bacc_oof"].iloc[0],
        color="#6b7280",
        linestyle="--",
        label="MLP visual",
    )
    axis_accuracy.plot(
        x,
        results["text_bacc_oof"],
        marker="o",
        linewidth=2.2,
        label="Texto isolado",
    )
    axis_accuracy.plot(
        x,
        results["fusion_bacc_oof"],
        marker="o",
        linewidth=2.2,
        label="Visual + texto",
    )
    axis_accuracy.set_xlabel("Frases cumulativas")
    axis_accuracy.set_ylabel("Acurácia balanceada OOF")
    axis_accuracy.set_xticks(x)
    axis_accuracy.grid(alpha=0.25)
    axis_accuracy.legend()
    axis_accuracy.set_title("Desempenho")

    axis_cost.plot(
        results["mean_output_tokens"],
        100 * results["delta_bacc_vs_numeric"],
        marker="o",
        color="#0f9d76",
        linewidth=2.2,
    )
    for row in results.itertuples(index=False):
        axis_cost.annotate(
            f"{row.sentences}",
            (row.mean_output_tokens, 100 * row.delta_bacc_vs_numeric),
            xytext=(4, 5),
            textcoords="offset points",
        )
    axis_cost.set_xlabel("Tokens médios gerados (prefixo)")
    axis_cost.set_ylabel("Ganho sobre o MLP visual (p.p.)")
    axis_cost.grid(alpha=0.25)
    axis_cost.set_title("Ganho × tamanho textual")
    figure.suptitle("Escadinha de descrição neutra do Qwen", fontsize=14)
    figure.savefig(OUTPUT_PLOT, dpi=180)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame, groups, cost = load_cohort()
    numeric_columns = [column for columns in groups.values() for column in columns]
    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float32)
    numeric_branch_sizes = [len(columns) for columns in groups.values()]

    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    augmented_branch_sizes = numeric_branch_sizes + [n_classes]
    stop_words = make_stop_words(encoder.classes_)
    stems = frame["stem"].to_numpy(dtype=str)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(
        splitter.split(np.zeros(len(frame), dtype=np.uint8), labels)
    )

    numeric_prediction = np.full(len(frame), -1, dtype=np.int64)
    text_predictions = {
        count: np.full(len(frame), -1, dtype=np.int64)
        for count in range(1, MAX_SENTENCES + 1)
    }
    fusion_predictions = {
        count: np.full(len(frame), -1, dtype=np.int64)
        for count in range(1, MAX_SENTENCES + 1)
    }
    folds = np.zeros(len(frame), dtype=np.int8)
    fold_rows = []
    log(
        f"Sentence ladder cohort={len(frame):,}; styles={n_classes}; "
        f"numeric={numeric_values.shape[1]}; device MLP parameters="
        f"{parameter_count(BranchFusionMLP(numeric_branch_sizes, n_classes)):,}; "
        f"fused parameters="
        f"{parameter_count(BranchFusionMLP(augmented_branch_sizes, n_classes)):,}"
    )

    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        fit_train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=labels[outer_train],
            random_state=SEED + outer_fold,
        )
        seed = SEED + outer_fold + 10_000
        base_prediction, base_epoch, base_elapsed = run_mlp(
            numeric_values,
            numeric_branch_sizes,
            labels,
            fit_train,
            validation,
            outer_test,
            n_classes,
            seed,
        )
        numeric_prediction[outer_test] = base_prediction
        base_score = balanced_accuracy_score(labels[outer_test], base_prediction)
        log(
            f"Outer {outer_fold}/{N_FOLDS} | numeric BAcc={base_score:.4f}; "
            f"epoch={base_epoch}; {base_elapsed:.1f}s"
        )

        for count in range(1, MAX_SENTENCES + 1):
            texts = frame[f"text_{count}"].to_numpy(dtype=str)
            train_text, test_text = nested_text_scores(
                count,
                outer_fold,
                outer_train,
                outer_test,
                stems,
                texts,
                labels,
                stop_words,
                n_classes,
            )
            text_prediction = test_text.argmax(axis=1)
            text_predictions[count][outer_test] = text_prediction
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
            fusion_predictions[count][outer_test] = prediction
            fusion_score = balanced_accuracy_score(labels[outer_test], prediction)
            fold_rows.append({
                "fold": outer_fold,
                "sentences": count,
                "n_train": len(outer_train),
                "n_test": len(outer_test),
                "numeric_bacc": base_score,
                "text_bacc": text_score,
                "fusion_bacc": fusion_score,
                "delta_bacc_vs_numeric": fusion_score - base_score,
                "numeric_best_epoch": base_epoch,
                "fusion_best_epoch": best_epoch,
                "numeric_elapsed_seconds": base_elapsed,
                "fusion_elapsed_seconds": elapsed,
            })
            log(
                f"Outer {outer_fold}/{N_FOLDS} | sentences={count}: "
                f"text={text_score:.4f}; fusion={fusion_score:.4f}; "
                f"delta={100 * (fusion_score - base_score):+.3f} pp"
            )
        folds[outer_test] = outer_fold
        pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)

    if np.any(numeric_prediction < 0) or np.any(folds == 0):
        raise RuntimeError("Incomplete numeric OOF predictions")
    if any(np.any(values < 0) for values in text_predictions.values()):
        raise RuntimeError("Incomplete text OOF predictions")
    if any(np.any(values < 0) for values in fusion_predictions.values()):
        raise RuntimeError("Incomplete fusion OOF predictions")

    numeric_bacc = balanced_accuracy_score(labels, numeric_prediction)
    detail = pd.DataFrame(fold_rows)
    rows = []
    previous = numeric_prediction
    previous_bacc = numeric_bacc
    for count in range(1, MAX_SENTENCES + 1):
        text_bacc = balanced_accuracy_score(labels, text_predictions[count])
        fusion_bacc = balanced_accuracy_score(labels, fusion_predictions[count])
        subset = detail[detail["sentences"] == count].sort_values("fold")
        row = {
            "sentences": count,
            "n_images": len(frame),
            "n_styles": n_classes,
            **cost[count],
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
                labels, numeric_prediction, fusion_predictions[count]
            ),
            "mcnemar_vs_previous_p": exact_paired_p(
                labels, previous, fusion_predictions[count]
            ),
            "fusion_best_epoch_mean": subset["fusion_best_epoch"].mean(),
        }
        for fold_index, score in enumerate(subset["fusion_bacc"], 1):
            row[f"fusion_fold_{fold_index}"] = score
        rows.append(row)
        previous = fusion_predictions[count]
        previous_bacc = fusion_bacc

    results = pd.DataFrame(rows)
    if PLACEBO_FILE.exists():
        placebos = pd.read_csv(PLACEBO_FILE)
        deltas = placebos["delta_bacc_vs_numeric"].to_numpy(dtype=float)
        results["prior_placebo_delta_mean"] = deltas.mean()
        results["prior_placebo_delta_p975"] = np.quantile(deltas, 0.975)
        results["prior_placebo_delta_max"] = deltas.max()
        results["empirical_p_placebo_ge_gain"] = results[
            "delta_bacc_vs_numeric"
        ].map(lambda value: (1 + np.sum(deltas >= value)) / (len(deltas) + 1))
    results.to_csv(OUTPUT_RESULTS, index=False)

    prediction_frame = frame[["stem", "label"]].copy()
    prediction_frame["fold"] = folds
    prediction_frame["prediction_numeric"] = encoder.inverse_transform(
        numeric_prediction
    )
    for count in range(1, MAX_SENTENCES + 1):
        prediction_frame[f"prediction_text_{count}"] = encoder.inverse_transform(
            text_predictions[count]
        )
        prediction_frame[f"prediction_fusion_{count}"] = encoder.inverse_transform(
            fusion_predictions[count]
        )
    prediction_frame.to_csv(OUTPUT_PREDICTIONS, index=False)
    make_plot(results)
    log("Sentence ladder results:\n" + results.to_string(index=False))
    log(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
