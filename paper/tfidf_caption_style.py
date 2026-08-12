"""Interpretable style classification from generated ArtEmis captions.

Captions are reduced to their first three distinct sentences.  Binary TF-IDF
ensures that loops do not increase a term's weight.  A sparse linear SVM makes
both global class coefficients and per-image token contributions auditable.
All vocabulary fitting happens inside each outer fold.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_oof_stacking import load_dataset


CAPTION_FILES = (
    ROOT / "outputs/affect_captions/artemis_train_captions.jsonl",
    ROOT / "outputs/affect_captions/artemis_test_captions.jsonl",
)
OUTPUT_RESULTS = ROOT / "paper/tfidf_caption_style_results.csv"
OUTPUT_FOLDS = ROOT / "paper/tfidf_caption_style_folds.csv"
OUTPUT_TERMS = ROOT / "paper/tfidf_caption_style_terms.csv"
OUTPUT_PREDICTIONS = ROOT / "paper/tfidf_caption_style_oof_predictions.csv"
OUTPUT_DECISIONS = ROOT / "paper/tfidf_caption_style_oof_decisions.npy"
OUTPUT_CLASSES = ROOT / "paper/tfidf_caption_style_classes.npy"
OUTPUT_STEMS = ROOT / "paper/tfidf_caption_style_stems.npy"

MAX_FEATURES = 50_000
MIN_DF = 10
MAX_DF = 0.90
NGRAM_RANGE = (1, 2)
MAX_UNIQUE_SENTENCES = 3
MAX_WORDS = 120
SVM_C = 0.1

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
BOILERPLATE = {
    "art", "artwork", "image", "images", "look", "looks", "looking",
    "overall", "painting", "paintings", "picture", "pictures", "piece",
    "scene", "seem", "seems", "feel", "feeling", "feels", "make", "makes",
}


def load_caption_map() -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in CAPTION_FILES:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                stem = str(row["image_id"])
                if stem in captions:
                    raise ValueError(f"Duplicate caption stem: {stem}")
                captions[stem] = str(row.get("caption", ""))
    return captions


def clean_caption(text: str) -> str:
    unique = []
    seen = set()
    for sentence in SENTENCE_SPLIT_RE.split(text.strip()):
        normalized = " ".join(WORD_RE.findall(sentence.lower()))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(sentence.strip())
        if len(unique) >= MAX_UNIQUE_SENTENCES:
            break
    cleaned = " ".join(unique)
    words = cleaned.split()
    if len(words) > MAX_WORDS:
        cleaned = " ".join(words[:MAX_WORDS])
    return cleaned


def make_stop_words(labels: np.ndarray) -> list[str]:
    stop_words = set(ENGLISH_STOP_WORDS) | BOILERPLATE
    # Conservative leakage guard: remove every lexical component of a label.
    for label in labels:
        stop_words.update(WORD_RE.findall(str(label).replace("_", " ").lower()))
    stop_words.update({"ukiyo", "modern", "late", "early", "high", "new"})
    return sorted(stop_words)


def make_vectorizer(stop_words: list[str]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=stop_words,
        ngram_range=NGRAM_RANGE,
        binary=True,
        use_idf=True,
        norm="l2",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )


def make_classifier() -> LinearSVC:
    return LinearSVC(
        C=SVM_C,
        penalty="l1",
        dual=False,
        class_weight="balanced",
        max_iter=5_000,
        random_state=SEED,
    )


def save_global_terms(
    texts: np.ndarray,
    labels: np.ndarray,
    stop_words: list[str],
) -> None:
    log("Fitting final full-cohort model for global term interpretation")
    vectorizer = make_vectorizer(stop_words)
    values = vectorizer.fit_transform(texts)
    classifier = make_classifier()
    classifier.fit(values, labels)
    terms = vectorizer.get_feature_names_out()
    document_frequency = np.rint(
        (len(texts) + 1) / np.exp(vectorizer.idf_ - 1) - 1
    ).astype(int)
    rows = []
    for class_index, style in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        nonzero = np.flatnonzero(coefficients)
        positive = nonzero[np.argsort(coefficients[nonzero])[::-1]][:30]
        negative = nonzero[np.argsort(coefficients[nonzero])][:15]
        for direction, indices in (("positive", positive), ("negative", negative)):
            for rank, index in enumerate(indices, 1):
                rows.append({
                    "style": style,
                    "direction": direction,
                    "rank": rank,
                    "term": terms[index],
                    "coefficient": float(coefficients[index]),
                    "document_frequency": int(document_frequency[index]),
                })
    pd.DataFrame(rows).to_csv(OUTPUT_TERMS, index=False)
    log(
        f"Final vocabulary={len(terms):,}; "
        f"nonzero coefficients={np.count_nonzero(classifier.coef_):,}/"
        f"{classifier.coef_.size:,}"
    )


def main() -> None:
    started = time.perf_counter()
    frame, _ = load_dataset()
    caption_map = load_caption_map()
    frame["caption"] = frame["stem"].astype(str).map(caption_map)
    missing = frame["caption"].isna()
    if missing.any():
        raise ValueError(f"Missing captions for {int(missing.sum())} cohort images")
    frame["clean_caption"] = frame["caption"].map(clean_caption)
    empty = frame["clean_caption"].str.len().eq(0)
    if empty.any():
        raise ValueError(f"Empty cleaned captions: {int(empty.sum())}")

    texts = frame["clean_caption"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    stop_words = make_stop_words(classes)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(splitter.split(np.zeros(len(frame), dtype=np.uint8), labels))
    predictions = np.empty(len(frame), dtype=object)
    decisions = np.zeros((len(frame), len(classes)), dtype=np.float32)
    fold_assignment = np.zeros(len(frame), dtype=np.int8)
    fold_rows = []
    log(
        f"Cohort={len(frame):,}; styles={len(classes)}; "
        f"binary TF-IDF {NGRAM_RANGE}; max_features={MAX_FEATURES:,}"
    )

    for fold, (train, test) in enumerate(splits, 1):
        fold_started = time.perf_counter()
        vectorizer = make_vectorizer(stop_words)
        train_values = vectorizer.fit_transform(texts[train])
        test_values = vectorizer.transform(texts[test])
        classifier = make_classifier()
        classifier.fit(train_values, labels[train])
        fold_prediction = classifier.predict(test_values)
        fold_decision = classifier.decision_function(test_values)
        aligned_decision = np.zeros((len(test), len(classes)), dtype=np.float32)
        for local_index, label in enumerate(classifier.classes_):
            aligned_decision[:, class_to_index[label]] = fold_decision[:, local_index]
        predictions[test] = fold_prediction
        decisions[test] = aligned_decision
        fold_assignment[test] = fold
        bacc = balanced_accuracy_score(labels[test], fold_prediction)
        fold_rows.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "nonzero_coefficients": int(np.count_nonzero(classifier.coef_)),
            "total_coefficients": int(classifier.coef_.size),
            "bacc": bacc,
            "accuracy": accuracy_score(labels[test], fold_prediction),
            "macro_f1": f1_score(labels[test], fold_prediction, average="macro"),
            "elapsed_seconds": time.perf_counter() - fold_started,
        })
        pd.DataFrame(fold_rows).to_csv(OUTPUT_FOLDS, index=False)
        log(
            f"Fold {fold}/{N_FOLDS}: BAcc={bacc:.4f}; "
            f"vocab={len(vectorizer.vocabulary_):,}; "
            f"nonzero={np.count_nonzero(classifier.coef_):,}; "
            f"{time.perf_counter() - fold_started:.1f}s"
        )

    if np.any(fold_assignment == 0):
        raise RuntimeError("Incomplete OOF predictions")
    bacc = balanced_accuracy_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    fold_result = pd.DataFrame(fold_rows)
    result = pd.DataFrame([{
        "representation": "cleaned_binary_tfidf_unigram_bigram",
        "n_images": len(frame),
        "n_styles": len(classes),
        "max_features": MAX_FEATURES,
        "min_df": MIN_DF,
        "svm_c": SVM_C,
        "bacc_oof": bacc,
        "bacc_fold_mean": fold_result["bacc"].mean(),
        "bacc_fold_std": fold_result["bacc"].std(ddof=0),
        "accuracy_oof": accuracy,
        "macro_f1_oof": macro_f1,
    }])
    result.to_csv(OUTPUT_RESULTS, index=False)
    pd.DataFrame({
        "stem": frame["stem"].astype(str),
        "fold": fold_assignment,
        "true_style": labels,
        "predicted_style": predictions,
        "correct": labels == predictions,
        "clean_caption": texts,
    }).to_csv(OUTPUT_PREDICTIONS, index=False)
    np.save(OUTPUT_DECISIONS, decisions)
    np.save(OUTPUT_CLASSES, classes)
    np.save(OUTPUT_STEMS, frame["stem"].astype(str).to_numpy())
    save_global_terms(texts, labels, stop_words)

    log("Result:\n" + result.to_string(index=False))
    for path in (
        OUTPUT_RESULTS, OUTPUT_FOLDS, OUTPUT_TERMS, OUTPUT_PREDICTIONS,
        OUTPUT_DECISIONS, OUTPUT_CLASSES, OUTPUT_STEMS,
    ):
        log(f"Saved: {path}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
