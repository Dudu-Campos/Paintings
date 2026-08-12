"""Audit label leakage and style-associated words in ArtEmis train captions.

The association analysis uses document presence: a word counts at most once
per caption.  This prevents degenerate repetition inside generated captions
from artificially dominating the vocabulary.  Words are ranked with a
smoothed log-odds z-score comparing one style against all other styles.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).parent.parent
INPUT = ROOT / "outputs/affect_captions/artemis_train_captions.jsonl"
LEAKAGE_CSV = ROOT / "paper/artemis_caption_realism_leakage.csv"
WORDS_CSV = ROOT / "paper/artemis_caption_style_words.csv"
SUMMARY_TXT = ROOT / "paper/artemis_caption_lexicon_summary.txt"

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
REALISM_FORMS = ("realismo", "realism", "realist", "realists", "realistic", "realistically")

# Prompt/model boilerplate is not linguistic evidence about artistic style.
# Keep this separate from the standard English stopword list for transparency.
BOILERPLATE = {
    "art", "artwork", "image", "images", "look", "looks", "looking",
    "overall", "painting", "paintings", "picture", "pictures", "piece",
    "scene", "seem", "seems", "feel", "feeling", "feels", "make", "makes",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | BOILERPLATE | {"nt", "ve", "ll", "re"}
PRIOR_STRENGTH = 1_000.0
TOP_K = 20


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_captions() -> tuple[list[dict], dict[str, Counter], Counter]:
    rows: list[dict] = []
    style_document_frequency: dict[str, Counter] = defaultdict(Counter)
    style_sizes: Counter = Counter()
    with INPUT.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            style = str(row["split"])
            text = str(row.get("caption", ""))
            tokens = tokenize(text)
            content_words = {
                token for token in tokens
                if len(token) >= 3 and token not in STOP_WORDS
            }
            rows.append({
                "image_id": str(row["image_id"]),
                "style": style,
                "text": text,
                "tokens": tokens,
                "content_words": content_words,
            })
            style_document_frequency[style].update(content_words)
            style_sizes[style] += 1
    return rows, style_document_frequency, style_sizes


def analyze_realism_leakage(rows: list[dict]) -> pd.DataFrame:
    groups = {
        "Realism": [row for row in rows if row["style"] == "Realism"],
        "Other_styles": [row for row in rows if row["style"] != "Realism"],
    }
    result = []
    for term in (*REALISM_FORMS, "ANY_REALISM_FORM"):
        per_group = {}
        for group_name, group_rows in groups.items():
            if term == "ANY_REALISM_FORM":
                occurrences = [
                    sum(row["tokens"].count(form) for form in REALISM_FORMS)
                    for row in group_rows
                ]
            else:
                occurrences = [row["tokens"].count(term) for row in group_rows]
            caption_count = sum(value > 0 for value in occurrences)
            per_group[group_name] = {
                "n_captions": len(group_rows),
                "captions_with_term": caption_count,
                "caption_rate": caption_count / len(group_rows),
                "occurrences": sum(occurrences),
            }
        real = per_group["Realism"]
        other = per_group["Other_styles"]
        smoothed_lift = (
            (real["captions_with_term"] + 0.5) / (real["n_captions"] + 1)
        ) / (
            (other["captions_with_term"] + 0.5) / (other["n_captions"] + 1)
        )
        result.append({
            "term": term,
            "realism_n_captions": real["n_captions"],
            "realism_captions_with_term": real["captions_with_term"],
            "realism_caption_rate": real["caption_rate"],
            "realism_occurrences": real["occurrences"],
            "other_n_captions": other["n_captions"],
            "other_captions_with_term": other["captions_with_term"],
            "other_caption_rate": other["caption_rate"],
            "other_occurrences": other["occurrences"],
            "smoothed_lift_realism_vs_other": smoothed_lift,
        })
    frame = pd.DataFrame(result)
    frame.to_csv(LEAKAGE_CSV, index=False)
    return frame


def analyze_style_words(
    style_df: dict[str, Counter], style_sizes: Counter
) -> pd.DataFrame:
    global_df = sum(style_df.values(), Counter())
    vocabulary = sorted(global_df)
    global_total = sum(global_df.values())
    n_documents = sum(style_sizes.values())
    prior = {
        word: PRIOR_STRENGTH * global_df[word] / global_total
        for word in vocabulary
    }
    rows = []
    for style in sorted(style_sizes):
        in_counts = style_df[style]
        out_counts = global_df - in_counts
        in_total = sum(in_counts.values())
        out_total = global_total - in_total
        candidates = []
        # Require evidence in at least 0.5% of the style, with a floor of 5.
        minimum_df = max(5, math.ceil(style_sizes[style] * 0.005))
        for word, count_in in in_counts.items():
            if count_in < minimum_df:
                continue
            count_out = out_counts[word]
            alpha = prior[word]
            numerator_in = count_in + alpha
            denominator_in = in_total + PRIOR_STRENGTH - numerator_in
            numerator_out = count_out + alpha
            denominator_out = out_total + PRIOR_STRENGTH - numerator_out
            delta = math.log(numerator_in / denominator_in) - math.log(
                numerator_out / denominator_out
            )
            variance = 1 / numerator_in + 1 / numerator_out
            z_score = delta / math.sqrt(variance)
            rate_in = count_in / style_sizes[style]
            out_documents = n_documents - style_sizes[style]
            rate_out = count_out / out_documents
            lift = (rate_in + 0.5 / style_sizes[style]) / (
                rate_out + 0.5 / out_documents
            )
            candidates.append((z_score, word, count_in, count_out, rate_in, rate_out, lift))
        candidates.sort(reverse=True)
        for rank, candidate in enumerate(candidates[:TOP_K], 1):
            z_score, word, count_in, count_out, rate_in, rate_out, lift = candidate
            rows.append({
                "style": style,
                "rank": rank,
                "word": word,
                "association_z": z_score,
                "style_captions_with_word": count_in,
                "style_caption_rate": rate_in,
                "other_captions_with_word": count_out,
                "other_caption_rate": rate_out,
                "lift_style_vs_other": lift,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(WORDS_CSV, index=False)
    return frame


def write_summary(
    leakage: pd.DataFrame,
    words: pd.DataFrame,
    style_sizes: Counter,
) -> str:
    any_row = leakage.set_index("term").loc["ANY_REALISM_FORM"]
    lines = [
        "ArtEmis train caption lexical audit",
        "===================================",
        f"Captions: {sum(style_sizes.values()):,}",
        f"Styles: {len(style_sizes)}",
        "",
        "Realism-family label leakage",
        "----------------------------",
        (
            f"Realism: {int(any_row.realism_captions_with_term):,}/"
            f"{int(any_row.realism_n_captions):,} captions "
            f"({any_row.realism_caption_rate:.3%}), "
            f"{int(any_row.realism_occurrences):,} occurrences"
        ),
        (
            f"Other styles: {int(any_row.other_captions_with_term):,}/"
            f"{int(any_row.other_n_captions):,} captions "
            f"({any_row.other_caption_rate:.3%}), "
            f"{int(any_row.other_occurrences):,} occurrences"
        ),
        f"Smoothed lift: {any_row.smoothed_lift_realism_vs_other:.2f}x",
        "",
        "Top distinctive words by style",
        "------------------------------",
    ]
    for style in sorted(style_sizes):
        subset = words[words["style"] == style].sort_values("rank")
        terms = ", ".join(
            f"{row.word} ({row.style_caption_rate:.1%}, {row.lift_style_vs_other:.1f}x)"
            for row in subset.itertuples()
        )
        lines.append(f"{style} [{style_sizes[style]:,}]: {terms}")
    summary = "\n".join(lines) + "\n"
    SUMMARY_TXT.write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    rows, style_df, style_sizes = load_captions()
    leakage = analyze_realism_leakage(rows)
    words = analyze_style_words(style_df, style_sizes)
    summary = write_summary(leakage, words, style_sizes)
    print(summary)
    print(f"Saved: {LEAKAGE_CSV}")
    print(f"Saved: {WORDS_CSV}")
    print(f"Saved: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
