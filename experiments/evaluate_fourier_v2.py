"""Compare the original Fourier block against Fourier-v2 with an LDA probe."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from registry import feature_columns, load_blocks, log, sanitize  # noqa: E402

SEED = 42
HAND8 = ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
         "Hilbert", "Fourier", "Gabor"]


def load_fourier_v2(path: Path, stems: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path)
    columns = feature_columns(table)
    index = pd.Series(np.arange(len(table)), index=table["stem"].astype(str)).reindex(stems)
    if index.isna().any():
        raise ValueError(f"Fourier-v2: {int(index.isna().sum())} stems missing")
    values = table.iloc[index.to_numpy(dtype=int)][columns].to_numpy(np.float32)
    return sanitize(values, "Fourier-v2")


def metrics(truth, pred) -> dict[str, float]:
    return {
        "bacc": 100 * balanced_accuracy_score(truth, pred),
        "accuracy": 100 * accuracy_score(truth, pred),
        "macro_f1": 100 * f1_score(truth, pred, average="macro"),
    }


def lda_probe(values: np.ndarray, labels: np.ndarray, splits, n_classes: int,
              condition: str) -> list[dict]:
    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        scaler = StandardScaler().fit(values[train])
        train_x = scaler.transform(values[train])
        test_x = scaler.transform(values[test])
        model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
        model.fit(train_x, labels[train])
        out = metrics(labels[test], model.predict(test_x))
        rows.append({"condition": condition, "fold": fold,
                     "dimensions": values.shape[1], **out})
        log(f"{condition:<24} fold {fold}: BAcc={out['bacc']:.2f} ({values.shape[1]}d)")
    return rows


def summarize(frame: pd.DataFrame) -> None:
    print("\nMean BAcc")
    print(frame.groupby("condition").bacc.agg(["mean", "std"]).sort_values("mean", ascending=False).round(3))
    pairs = [
        ("Fourier v1", "Fourier v2"),
        ("HAND8 v1", "HAND8 v2"),
        ("HAND8 no Fourier", "HAND8 v1"),
        ("HAND8 no Fourier", "HAND8 v2"),
    ]
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    print("\nPaired deltas")
    for a, b in pairs:
        if a not in pivot or b not in pivot:
            continue
        delta = pivot[b] - pivot[a]
        _, p = ttest_rel(pivot[b], pivot[a])
        print(f"{a:16s} -> {b:16s}: {delta.mean():+.3f} pp "
              f"(sd {delta.std(ddof=1):.3f}, p={p:.4f}, +folds {(delta > 0).sum()}/{len(delta)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--fourier-v2", type=Path,
                        default=ROOT / "results/fourier_v2_multitask_features.parquet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/fourier_v2_lda_comparison.csv")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    fourier_v2 = load_fourier_v2(args.fourier_v2, cohort.stems)

    hand_v1 = cohort.matrix(HAND8)
    hand_no_fourier = cohort.matrix([n for n in HAND8 if n != "Fourier"])
    hand_v2 = np.hstack([cohort.blocks[n] for n in HAND8 if n != "Fourier"] + [fourier_v2])

    conditions = {
        "Fourier v1": cohort.blocks["Fourier"],
        "Fourier v2": fourier_v2,
        "HAND8 no Fourier": hand_no_fourier,
        "HAND8 v1": hand_v1,
        "HAND8 v2": hand_v2,
    }
    log(f"{args.dataset}: {len(labels):,} images, {n_classes} classes")
    for name, values in conditions.items():
        log(f"  {name}: {values.shape[1]}d")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    rows = []
    for condition, values in conditions.items():
        rows.extend(lda_probe(values.astype(np.float32), labels, splits, n_classes, condition))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False, float_format="%.6f")
    summarize(frame)
    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
