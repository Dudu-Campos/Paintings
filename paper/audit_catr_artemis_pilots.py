"""Paired audit of the three small CATR/ArtEmis anti-collapse pilots."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/catr_artemis_multicaption_pilot/test_paintings.jsonl"
CONCAT = ROOT / "outputs/catr_artemis_rich_pilot/test_generations.jsonl"
CONSERVATIVE = ROOT / "outputs/catr_artemis_multicaption_pilot/test_generations.jsonl"
HUMAN_ONLY = ROOT / "outputs/catr_artemis_human_only_pilot/test_generations_on_common_cohort.jsonl"
OLD = ROOT / "outputs/affect_captions/artemis_test_captions.jsonl"
OUTPUT = ROOT / "paper/catr_artemis_pilot_comparison.csv"
EXAMPLES = ROOT / "paper/catr_artemis_pilot_examples.jsonl"

AFFECT = re.compile(
    r"\b(?:feel|feels|feeling|felt|sad|sadness|happy|happiness|calm|content|"
    r"awe|fear|scary|afraid|anger|angry|disgust|excited|excitement|peaceful|"
    r"beautiful|confus\w*|lonely|uncomfortable|curious\w*|mood|emotion\w*|"
    r"relax\w*|joy|pleasant|unpleasant|disturb\w*)\b",
    re.I,
)
EXPLICIT_REACTION = re.compile(
    r"\b(?:i feel|makes? me|reminds? me|i am|i'm|i like|i love|i hate)\b", re.I
)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def repeated_ngram(text: str, size: int = 3) -> bool:
    items = tokens(text)
    grams = [tuple(items[i : i + size]) for i in range(len(items) - size + 1)]
    return len(grams) != len(set(grams))


def metrics(condition: str, rows: list[dict]) -> dict:
    texts = [row["caption"] for row in rows]
    sentence_lists = [sentences(text) for text in texts]
    normalized_sentences = [[normalize(item) for item in items] for items in sentence_lists]
    global_frequency = Counter(item for items in normalized_sentences for item in items)
    repeated_sentence = [len(items) != len(set(items)) for items in normalized_sentences]
    result = {
        "condition": condition,
        "images": len(rows),
        "mean_words": float(np.mean([len(tokens(text)) for text in texts])),
        "mean_sentences": float(np.mean([len(items) for items in sentence_lists])),
        "affect_cue_image_rate": float(np.mean([bool(AFFECT.search(text)) for text in texts])),
        "explicit_reaction_image_rate": float(
            np.mean([bool(EXPLICIT_REACTION.search(text)) for text in texts])
        ),
        "repeated_exact_sentence_image_rate": float(np.mean(repeated_sentence)),
        "repeated_trigram_image_rate": float(
            np.mean([repeated_ngram(text) for text in texts])
        ),
        "cross_image_reused_sentence_rate": float(
            np.mean(
                [
                    any(global_frequency[item] > 1 for item in items)
                    for items in normalized_sentences
                ]
            )
        ),
        "unique_caption_rate": len({normalize(text) for text in texts}) / len(texts),
    }
    if "terminated_sep" in rows[0]:
        result["sep_termination_rate"] = float(
            np.mean([row["terminated_sep"] for row in rows])
        )
    return result


def main() -> None:
    cohort = read(COHORT)
    keys = [(row["style"], row["stem"]) for row in cohort]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate paired cohort keys")

    baseline = [{**row, "caption": row["catr_caption"]} for row in cohort]
    concat = read(CONCAT)
    conservative = read(CONSERVATIVE)
    human_only = read(HUMAN_ONLY)
    for name, rows in (
        ("concat", concat),
        ("conservative", conservative),
        ("human_only", human_only),
    ):
        observed = [(row["style"], row["stem"]) for row in rows]
        if observed != keys:
            raise RuntimeError(f"{name} does not match the paired cohort")

    old_map = {
        (row["split"], row["image_id"]): row["caption"] for row in read(OLD)
    }
    old = [
        {**row, "caption": old_map[key]}
        for row, key in zip(cohort, keys)
        if key in old_map
    ]
    result = [
        metrics("CATR COCO baseline", baseline),
        metrics("Long concatenated target", concat),
        metrics("Independent + conservative replay", conservative),
        metrics("Independent human-only full decoder", human_only),
        metrics("Old emotional model", old),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        fields = sorted({key for row in result for key in row})
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)

    selected = [0, 7, 16, 29, 44, 63, 87, 111, 144, 177]
    with EXAMPLES.open("w", encoding="utf-8") as target:
        for index in selected:
            row = {
                "style": cohort[index]["style"],
                "stem": cohort[index]["stem"],
                "catr": baseline[index]["caption"],
                "long_target": concat[index]["caption"],
                "conservative": conservative[index]["caption"],
                "human_only": human_only[index]["caption"],
                "human_reference_1": cohort[index]["human_1"],
                "human_reference_2": cohort[index]["human_2"],
            }
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(OUTPUT)
    for row in result:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
