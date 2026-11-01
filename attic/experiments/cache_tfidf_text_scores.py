"""Cache leak-free TF-IDF caption scores per canonical fold, for reuse across
experiments (e.g. `evaluate_full_factorial_ablation_text.py`) without paying
the nested-OOF LinearSVC cost more than once.

For each of the five canonical folds (same StratifiedKFold, SEED=42, as every
other full-cohort experiment in this repo), reproduces exactly the scoring
scheme from `final_tfidf_visual_mlp.fit_text_scores`: nested 5-fold OOF scores
for the 90% "fit" split, and fit-trained scores for the held-out 10%
"validation" split and the outer test split. The three pieces are stitched
back into one (n_images, n_styles) array in the frame's row order and saved
keyed by stem, so any consumer can just merge on `stem` without recomputing.
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, fit_text_scores, load_dataset
from tfidf_caption_style import make_stop_words


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame, _ = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    texts = frame["caption"].to_numpy(dtype=str)
    stems = frame["stem"].to_numpy(dtype=str)
    stop_words = make_stop_words(encoder.classes_)
    score_columns = [f"score_{index}" for index in range(n_classes)]

    log(f"TF-IDF cache: {len(frame):,} images, {n_classes} styles")

    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y), dtype=np.uint8), y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        out_path = args.output_dir / f"fold_{fold}.parquet"
        if out_path.exists() and not args.overwrite:
            log(f"fold {fold}/{N_FOLDS}: cache already exists, skipping")
            continue
        started = time.perf_counter()
        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold
        )
        target_texts = np.concatenate([texts[validation], texts[outer_test]])
        fit_text, target_text = fit_text_scores(
            texts[fit], y[fit], target_texts, stop_words, n_classes, SEED + fold,
        )
        validation_text = target_text[:len(validation)]
        test_text = target_text[len(validation):]

        scores = np.empty((len(frame), n_classes), dtype=np.float32)
        roles = np.empty(len(frame), dtype=object)
        scores[fit], roles[fit] = fit_text, "fit"
        scores[validation], roles[validation] = validation_text, "validation"
        scores[outer_test], roles[outer_test] = test_text, "test"

        table = pd.DataFrame(scores, columns=score_columns)
        table.insert(0, "role", roles)
        table.insert(0, "stem", stems)
        table.to_parquet(out_path, index=False)
        log(f"fold {fold}/{N_FOLDS}: saved {out_path} ({time.perf_counter() - started:.1f}s)")

    log(f"Done. Cache in {args.output_dir}")


if __name__ == "__main__":
    main()
