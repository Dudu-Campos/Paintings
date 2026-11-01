"""Evaluate any registered feature blocks under one reproducible protocol.

The script is deliberately agnostic to painting data.  Dataset identity and
model composition come from the benchmark TOML files; representation names come
from ``src/pipeline/registry.py``.  It is therefore the common evaluator for
the explicit pipeline, frozen encoders, their fusions, and future texture
benchmark representations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/paintings_mpl")
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from registry import load_blocks  # noqa: E402
sys.path.insert(0, str(ROOT / "src" / "benchmarking"))
from config import dataset_specs  # noqa: E402

SEED = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


def scale_blocks(blocks: list[np.ndarray], fit: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_parts, query_parts = [], []
    for block in blocks:
        scaler = StandardScaler().fit(block[fit])
        train_parts.append(scaler.transform(block[fit]).astype(np.float32))
        query_parts.append(scaler.transform(block[query]).astype(np.float32))
    return np.hstack(train_parts), np.hstack(query_parts)


def evaluate_lda(blocks: list[np.ndarray], labels: np.ndarray,
                 train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    train_x, test_x = scale_blocks(blocks, train, test)
    n_classes = len(np.unique(labels))
    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
    model.fit(train_x, labels[train])
    return metrics(labels[test], model.predict(test_x))


def evaluate_mlp(blocks: list[np.ndarray], labels: np.ndarray, train: np.ndarray,
                 test: np.ndarray, fold: int, epochs: int, patience: int) -> dict[str, float]:
    # Reuses the branch architecture used for the main pipeline; validation is
    # sampled only from the outer-training fold and never from the test fold.
    from evaluate_robustness_mlp_aug import train_one

    fit, validation = train_test_split(
        train, test_size=0.10, stratify=labels[train], random_state=SEED + fold,
    )
    fit_x, validation_x = scale_blocks(blocks, fit, validation)
    _, test_x = scale_blocks(blocks, fit, test)
    result, _ = train_one(
        fit_x, labels[fit], validation_x, labels[validation], test_x, labels[test],
        [block.shape[1] for block in blocks], len(np.unique(labels)),
        SEED + fold, epochs, patience,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(dataset_specs()))
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument("--classifier", choices=("lda", "mlp"), required=True)
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = dataset_specs()[args.dataset]
    cohort = load_blocks(args.dataset, args.blocks, allow_external=args.allow_external,
                         min_per_class=spec.min_per_class)
    missing = [name for name in args.blocks if name not in cohort.blocks]
    if missing:
        raise FileNotFoundError(
            f"{args.dataset}: blocos indisponíveis: {missing}. Rode o extrator correspondente primeiro."
        )
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    if args.folds < 2 or np.bincount(labels).min() < args.folds:
        raise ValueError("Not enough examples per class for requested cross-validation folds")
    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(cohort.stems, labels))
    blocks = [cohort.blocks[name] for name in args.blocks]
    condition = " + ".join(args.blocks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": args.dataset, "task": spec.task, "classifier": args.classifier,
        "blocks": args.blocks, "allow_external": args.allow_external,
        "folds": args.folds, "seed": SEED, "n_images": len(labels), "n_classes": n_classes,
        "dimensions": int(sum(block.shape[1] for block in blocks)),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    path = args.output_dir / "folds.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(str(row["condition"]), str(row["classifier"]), int(row["fold"])) for row in rows}
    log(f"{args.dataset}: {len(labels):,} examples, {n_classes} classes, {config['dimensions']}d")

    for fold, (train, test) in enumerate(splits, start=1):
        key = (condition, args.classifier, fold)
        if key in done:
            log(f"fold {fold}: already complete")
            continue
        started = time.perf_counter()
        if args.classifier == "lda":
            result = evaluate_lda(blocks, labels, train, test)
        else:
            result = evaluate_mlp(blocks, labels, train, test, fold, args.epochs, args.patience)
        rows.append({
            "dataset": args.dataset, "task": spec.task, "condition": condition,
            "classifier": args.classifier, "fold": fold, "n_images": len(labels),
            "n_classes": n_classes, "dimensions": config["dimensions"],
            "elapsed_seconds": time.perf_counter() - started, **result,
        })
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
        log(f"fold {fold}: BAcc={100 * result['bacc']:.2f}%")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["dataset", "task", "condition", "classifier"], as_index=False)
               .agg(folds=("fold", "nunique"), n_images=("n_images", "first"),
                    n_classes=("n_classes", "first"), dimensions=("dimensions", "first"),
                    bacc_mean=("bacc", "mean"), bacc_sd_sample=("bacc", "std"),
                    accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
                    elapsed_seconds=("elapsed_seconds", "sum")))
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log(f"saved {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
