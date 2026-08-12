"""Diagnostics for the 16 cached sentiment features from ArtEmis captions."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parent.parent
CAPTIONS = ROOT / "outputs/affect_captions/artemis_train_captions.jsonl"
FEATURES = ROOT / "outputs/wikiart_features/sentiment_features.parquet"
GROUP_CSV = ROOT / "paper/artemis_sentiment_feature_group_cv.csv"
DIAGNOSTIC_CSV = ROOT / "paper/artemis_sentiment_feature_diagnostics.csv"
CORRELATION_CSV = ROOT / "paper/artemis_sentiment_repetition_correlations.csv"
REPETITION_CSV = ROOT / "paper/artemis_sentiment_repetition_performance.csv"

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")
SEED = 42
N_FOLDS = 5

VADER = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
TEXTBLOB = ["tb_polarity", "tb_subjectivity"]
NRC = [
    "nrc_anger", "nrc_anticipation", "nrc_disgust", "nrc_fear", "nrc_joy",
    "nrc_negative", "nrc_positive", "nrc_sadness", "nrc_surprise", "nrc_trust",
]
ALL_FEATURES = VADER + TEXTBLOB + NRC


def caption_diagnostics(text: str) -> dict[str, float]:
    tokens = TOKEN_RE.findall(text.lower())
    content_tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    token_counts = Counter(content_tokens)
    sentences = [
        " ".join(TOKEN_RE.findall(sentence.lower()))
        for sentence in SENTENCE_RE.split(text)
    ]
    sentences = [sentence for sentence in sentences if sentence]
    sentence_counts = Counter(sentences)
    n_tokens = len(tokens)
    n_content = len(content_tokens)
    n_sentences = len(sentences)
    return {
        "n_words": n_tokens,
        "unique_content_ratio": (
            len(token_counts) / n_content if n_content else 0.0
        ),
        "max_content_word_share": (
            max(token_counts.values()) / n_content if n_content else 0.0
        ),
        "n_sentences": n_sentences,
        "unique_sentence_ratio": (
            len(sentence_counts) / n_sentences if n_sentences else 0.0
        ),
        "max_sentence_repetitions": (
            max(sentence_counts.values()) if sentence_counts else 0
        ),
    }


def load_diagnostics() -> pd.DataFrame:
    rows = []
    with CAPTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append({
                "stem": str(record["image_id"]),
                **caption_diagnostics(str(record.get("caption", ""))),
            })
    return pd.DataFrame(rows)


def cross_validated_predictions(
    values: np.ndarray, labels: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[float]]:
    prediction = np.empty(len(labels), dtype=labels.dtype)
    scores = []
    n_classes = len(np.unique(labels))
    for train, test in splits:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(
                priors=np.ones(n_classes) / n_classes
            )),
        ])
        model.fit(values[train], labels[train])
        fold_prediction = model.predict(values[test])
        prediction[test] = fold_prediction
        scores.append(balanced_accuracy_score(labels[test], fold_prediction))
    return prediction, scores


def main() -> None:
    features = pd.read_parquet(FEATURES)
    diagnostics = load_diagnostics()
    frame = features.merge(diagnostics, on="stem", how="inner", validate="one_to_one")
    labels = frame["style"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(frame), dtype=np.uint8), labels))

    groups = {
        "VADER": VADER,
        "TextBlob": TEXTBLOB,
        "NRC": NRC,
        "VADER + TextBlob": VADER + TEXTBLOB,
        "All 16": ALL_FEATURES,
    }
    group_rows = []
    all_prediction = None
    for name, columns in groups.items():
        prediction, scores = cross_validated_predictions(
            frame[columns].to_numpy(dtype=np.float64), labels, splits
        )
        group_rows.append({
            "feature_group": name,
            "n_features": len(columns),
            "bacc_mean": np.mean(scores),
            "bacc_std": np.std(scores),
            **{f"fold_{index + 1}": score for index, score in enumerate(scores)},
        })
        if name == "All 16":
            all_prediction = prediction
    group_result = pd.DataFrame(group_rows)
    group_result.to_csv(GROUP_CSV, index=False)

    diagnostic_rows = []
    for column in ALL_FEATURES:
        values = frame[column].to_numpy(dtype=np.float64)
        diagnostic_rows.append({
            "feature": column,
            "mean": values.mean(),
            "std": values.std(),
            "zero_rate": np.mean(values == 0),
            "unique_values": len(np.unique(values)),
            "abs_ge_0_95_rate": np.mean(np.abs(values) >= 0.95),
            "abs_ge_0_99_rate": np.mean(np.abs(values) >= 0.99),
        })
    diagnostic_result = pd.DataFrame(diagnostic_rows)
    diagnostic_result.to_csv(DIAGNOSTIC_CSV, index=False)

    repetition_columns = [
        "n_words", "unique_content_ratio", "max_content_word_share",
        "n_sentences", "unique_sentence_ratio", "max_sentence_repetitions",
    ]
    correlations = frame[ALL_FEATURES + repetition_columns].corr(method="spearman")
    correlation_rows = []
    for feature in ALL_FEATURES:
        for measure in repetition_columns:
            correlation_rows.append({
                "feature": feature,
                "repetition_measure": measure,
                "spearman_r": correlations.loc[feature, measure],
            })
    correlation_result = pd.DataFrame(correlation_rows).sort_values(
        "spearman_r", key=lambda series: series.abs(), ascending=False
    )
    correlation_result.to_csv(CORRELATION_CSV, index=False)

    if all_prediction is None:
        raise RuntimeError("Missing full-feature OOF predictions")
    frame["repetition_quartile"] = pd.qcut(
        frame["unique_content_ratio"], 4, labels=False, duplicates="drop"
    )
    repetition_rows = []
    for quartile, subset in frame.groupby("repetition_quartile"):
        indices = subset.index.to_numpy()
        repetition_rows.append({
            "quartile": int(quartile) + 1,
            "n_images": len(indices),
            "unique_content_ratio_mean": subset["unique_content_ratio"].mean(),
            "n_words_mean": subset["n_words"].mean(),
            "bacc_all_16_oof": balanced_accuracy_score(
                labels[indices], all_prediction[indices]
            ),
        })
    repetition_result = pd.DataFrame(repetition_rows)
    repetition_result.to_csv(REPETITION_CSV, index=False)

    print(f"Merged train captions: {len(frame):,}")
    print("\nFeature-group OOF CV:")
    print(group_result.to_string(index=False))
    print("\nCaption degeneration:")
    print(frame[repetition_columns].describe(percentiles=[.1, .25, .5, .75, .9]).T)
    print("\nFeature pathologies:")
    print(diagnostic_result.to_string(index=False))
    print("\nStrongest repetition correlations:")
    print(correlation_result.head(15).to_string(index=False))
    print("\nPerformance by content-diversity quartile:")
    print(repetition_result.to_string(index=False))
    for path in (GROUP_CSV, DIAGNOSTIC_CSV, CORRELATION_CSV, REPETITION_CSV):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
