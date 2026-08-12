"""Paired, interpretable comparison of four caption representations.

Every condition uses the same paintings, labels, outer folds, preprocessing,
vectorizer, and sparse linear classifier.  Only the source of the caption
changes.  A single complete sentence is retained to control text length and
remove generation-limit truncation.  Label words are excluded from every
fold-specific vocabulary.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from tfidf_caption_style import (
    CAPTION_FILES,
    make_classifier,
    make_stop_words,
    make_vectorizer,
)


QWEN_FILE = (
    ROOT / "outputs/affect_captions/qwen_prompt_ablation_500_per_style.jsonl"
)
QWEN_SAMPLE = ROOT / "paper/qwen_prompt_ablation_sample.csv"
COCO_FILE = ROOT / "outputs/affect_captions/coco_wikiart_captions.jsonl"
IMAGE_DIR = ROOT / "wikiart"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

OUTPUT_RESULTS = ROOT / "paper/text_controls_standalone_results.csv"
OUTPUT_FOLDS = ROOT / "paper/text_controls_standalone_folds.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/text_controls_standalone_predictions.csv"
OUTPUT_TERMS = ROOT / "paper/text_controls_terms.csv"
OUTPUT_PAIRWISE = ROOT / "paper/text_controls_pairwise.csv"
OUTPUT_COHORT = ROOT / "paper/text_controls_cohort.csv"
OUTPUT_DECISIONS = ROOT / "paper/text_controls_oof_decisions.npz"

CONDITIONS = (
    "ArtEmis fine-tuned",
    "COCO pretrained",
    "Qwen neutral",
    "Qwen emotional",
)
FIRST_SENTENCE_RE = re.compile(
    r"^.*?[.!?](?:[\"'\N{RIGHT DOUBLE QUOTATION MARK}]+)?(?=\s|$)",
    flags=re.DOTALL,
)
MAX_FALLBACK_WORDS = 50


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def first_complete_sentence(text: str) -> str:
    """Keep one complete sentence, with a bounded fallback if none terminates."""
    normalized = " ".join(str(text).strip().split())
    match = FIRST_SENTENCE_RE.search(normalized)
    if match:
        return match.group(0).strip()
    return " ".join(normalized.split()[:MAX_FALLBACK_WORDS])


def load_artemis() -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in CAPTION_FILES:
        for row in read_jsonl(path):
            stem = str(row["image_id"])
            if stem in captions:
                raise ValueError(f"Duplicate ArtEmis stem: {stem}")
            captions[stem] = str(row.get("caption", ""))
    return captions


def load_coco_by_path() -> dict[str, str]:
    """Recover the source path omitted by the overnight caption writer.

    ``caption_inference.py`` writes rows in the same sorted path order used to
    load images.  Matching this order disambiguates repeated stems that occur
    in multiple style directories.
    """
    image_paths = sorted(
        path for path in IMAGE_DIR.rglob("*")
        if path.suffix.lower() in EXTENSIONS
    )
    rows = list(read_jsonl(COCO_FILE))
    if len(rows) != len(image_paths):
        raise ValueError(
            f"Cannot recover COCO paths: {len(rows)} rows != "
            f"{len(image_paths)} image paths"
        )
    captions: dict[str, str] = {}
    mismatches = []
    for index, (path, row) in enumerate(zip(image_paths, rows)):
        if str(row["image_id"]) != path.stem or str(row["file_name"]) != path.name:
            mismatches.append(index)
            if len(mismatches) >= 5:
                break
        captions[str(path.resolve())] = str(row.get("caption", ""))
    if mismatches:
        raise ValueError(f"COCO row/path order mismatch at indices {mismatches}")
    return captions


def load_cohort() -> pd.DataFrame:
    sample = pd.read_csv(QWEN_SAMPLE, dtype=str)
    if sample["stem"].duplicated().any():
        raise ValueError("Qwen sample contains duplicate stems")
    sample = sample.sort_values(["style", "stem"]).reset_index(drop=True)

    qwen: dict[tuple[str, str], str] = {}
    qwen_styles: dict[str, str] = {}
    for row in read_jsonl(QWEN_FILE):
        stem = str(row["stem"])
        condition = str(row["condition"])
        key = (stem, condition)
        if key in qwen:
            raise ValueError(f"Duplicate Qwen pair: {key}")
        qwen[key] = str(row.get("caption", ""))
        qwen_styles[stem] = str(row["style"])

    artemis = load_artemis()
    coco_by_path = load_coco_by_path()
    rows = []
    for item in sample.itertuples(index=False):
        stem = str(item.stem)
        style = str(item.style)
        path = str(Path(item.path).resolve())
        if qwen_styles.get(stem) != style:
            raise ValueError(f"Qwen/sample style mismatch for {stem}")
        raw = {
            "ArtEmis fine-tuned": artemis.get(stem),
            "COCO pretrained": coco_by_path.get(path),
            "Qwen neutral": qwen.get((stem, "neutral")),
            "Qwen emotional": qwen.get((stem, "emotional")),
        }
        missing = [name for name, text in raw.items() if text is None]
        if missing:
            raise ValueError(f"Missing {missing} for {stem}")
        cleaned = {name: first_complete_sentence(text) for name, text in raw.items()}
        empty = [name for name, text in cleaned.items() if not text]
        if empty:
            raise ValueError(f"Empty cleaned {empty} for {stem}")
        rows.append({"stem": stem, "label": style, "path": path, **cleaned})
    frame = pd.DataFrame(rows)
    if frame["stem"].duplicated().any():
        raise RuntimeError("Final cohort contains duplicate stems")
    return frame


def aligned_decision(classifier, values, classes: np.ndarray) -> np.ndarray:
    decision = classifier.decision_function(values)
    aligned = np.zeros((values.shape[0], len(classes)), dtype=np.float32)
    class_to_index = {label: index for index, label in enumerate(classes)}
    for local_index, label in enumerate(classifier.classes_):
        aligned[:, class_to_index[label]] = decision[:, local_index]
    return aligned


def global_terms(
    condition: str,
    texts: np.ndarray,
    labels: np.ndarray,
    stop_words: list[str],
) -> list[dict]:
    vectorizer = make_vectorizer(stop_words)
    values = vectorizer.fit_transform(texts)
    classifier = make_classifier()
    classifier.fit(values, labels)
    terms = vectorizer.get_feature_names_out()
    rows = []
    for class_index, style in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        nonzero = np.flatnonzero(coefficients)
        selected = nonzero[np.argsort(coefficients[nonzero])[::-1]][:20]
        for rank, index in enumerate(selected, 1):
            rows.append({
                "condition": condition,
                "style": style,
                "rank": rank,
                "term": terms[index],
                "coefficient": float(coefficients[index]),
            })
    return rows


def pairwise_rows(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1:]:
            left_prediction = predictions[left]
            right_prediction = predictions[right]
            left_correct = left_prediction == labels
            right_correct = right_prediction == labels
            left_only = int(np.sum(left_correct & ~right_correct))
            right_only = int(np.sum(~left_correct & right_correct))
            discordant = left_only + right_only
            p_value = (
                float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                if discordant else 1.0
            )
            rows.append({
                "left": left,
                "right": right,
                "left_bacc": balanced_accuracy_score(labels, left_prediction),
                "right_bacc": balanced_accuracy_score(labels, right_prediction),
                "delta_left_minus_right": (
                    balanced_accuracy_score(labels, left_prediction)
                    - balanced_accuracy_score(labels, right_prediction)
                ),
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "net_correct_left_minus_right": left_only - right_only,
                "mcnemar_exact_p": p_value,
            })
    return rows


def main() -> None:
    started = time.perf_counter()
    frame = load_cohort()
    labels = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels)
    stop_words = make_stop_words(classes)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(frame), dtype=np.uint8), labels))
    log(
        f"Paired text cohort={len(frame):,}; styles={len(classes)}; "
        f"conditions={len(CONDITIONS)}; identical {N_FOLDS} folds"
    )

    predictions = {
        condition: np.empty(len(frame), dtype=object) for condition in CONDITIONS
    }
    decisions = {
        condition: np.zeros((len(frame), len(classes)), dtype=np.float32)
        for condition in CONDITIONS
    }
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    fold_rows = []
    term_rows = []
    for condition in CONDITIONS:
        texts = frame[condition].to_numpy(dtype=str)
        log(
            f"{condition}: mean words="
            f"{np.mean([len(text.split()) for text in texts]):.1f}"
        )
        for fold, (train, test) in enumerate(splits, 1):
            fold_started = time.perf_counter()
            vectorizer = make_vectorizer(stop_words)
            train_values = vectorizer.fit_transform(texts[train])
            test_values = vectorizer.transform(texts[test])
            classifier = make_classifier()
            classifier.fit(train_values, labels[train])
            prediction = classifier.predict(test_values)
            predictions[condition][test] = prediction
            decisions[condition][test] = aligned_decision(
                classifier, test_values, classes
            )
            fold_assignment[test] = fold
            score = balanced_accuracy_score(labels[test], prediction)
            fold_rows.append({
                "condition": condition,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "bacc": score,
                "accuracy": accuracy_score(labels[test], prediction),
                "macro_f1": f1_score(
                    labels[test], prediction, average="macro"
                ),
                "elapsed_seconds": time.perf_counter() - fold_started,
            })
            pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)
            log(
                f"{condition:<20} fold {fold}/{N_FOLDS}: "
                f"BAcc={score:.4f}; vocab={len(vectorizer.vocabulary_):,}"
            )
        term_rows.extend(global_terms(condition, texts, labels, stop_words))

    if np.any(fold_assignment == 0):
        raise RuntimeError("Incomplete fold assignment")
    folds = pd.DataFrame(fold_rows)
    result_rows = []
    for condition in CONDITIONS:
        subset = folds[folds["condition"] == condition].sort_values("fold")
        scores = subset["bacc"].to_numpy(dtype=float)
        prediction = predictions[condition]
        result_rows.append({
            "condition": condition,
            "n_images": len(frame),
            "n_styles": len(classes),
            "sentence_policy": "first_complete_sentence",
            "bacc_oof": balanced_accuracy_score(labels, prediction),
            "bacc_fold_mean": scores.mean(),
            "bacc_fold_std": scores.std(),
            "accuracy_oof": accuracy_score(labels, prediction),
            "macro_f1_oof": f1_score(labels, prediction, average="macro"),
            **{f"fold_{index + 1}": score for index, score in enumerate(scores)},
        })
    results = pd.DataFrame(result_rows).sort_values("bacc_oof", ascending=False)
    results.to_csv(OUTPUT_RESULTS, index=False)
    pd.DataFrame(term_rows).to_csv(OUTPUT_TERMS, index=False)
    pd.DataFrame(pairwise_rows(labels, predictions)).to_csv(
        OUTPUT_PAIRWISE, index=False
    )

    prediction_frame = frame[["stem", "label"]].copy()
    prediction_frame["fold"] = fold_assignment
    for condition in CONDITIONS:
        slug = condition.lower().replace(" ", "_").replace("-", "_")
        prediction_frame[f"prediction_{slug}"] = predictions[condition]
        prediction_frame[f"correct_{slug}"] = predictions[condition] == labels
    prediction_frame.to_csv(OUTPUT_PREDICTIONS, index=False)
    frame.to_csv(OUTPUT_COHORT, index=False)
    np.savez_compressed(
        OUTPUT_DECISIONS,
        stems=frame["stem"].to_numpy(dtype=str),
        labels=labels,
        classes=classes,
        fold=fold_assignment,
        **{
            condition.lower().replace(" ", "_").replace("-", "_"): values
            for condition, values in decisions.items()
        },
    )
    log("Results:\n" + results.to_string(index=False))
    log(f"Saved paired comparison artifacts under {ROOT / 'paper'}")
    log(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
