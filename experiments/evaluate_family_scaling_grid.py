"""Paired scaling grid for cumulative representation families.

This is the small control behind the paper-style scaling figure:

* data axis: fix the selected-feature budget and vary the number of training
  images;
* dimensionality axis: fix the training set and vary the number of selected
  features;
* in both cases, compare cumulative provenance pools with 1, 2, 3 and 4 visual
  families under the same LDA probe and the same redundancy-first selector.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import VISUAL, clean  # noqa: E402
from evaluate_fusion_delta_across_corpora import load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_redundancy_selection_curve import r2_matrix, select  # noqa: E402
from protocol import SEED  # noqa: E402
from registry import feature_columns, sanitize  # noqa: E402


FAMILY_STEPS = [
    ("1 family", ["Descriptors"]),
    ("2 families", ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)"]),
    ("3 families", ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)", "Hilbert"]),
    ("4 families", ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)", "Hilbert", "Fourier", "Gabor"]),
]

DESCRIPTOR_BOUNDS = [(0, 30), (30, 60), (60, 90), (90, 99), (99, 117), (117, 130)]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def descriptor_groups() -> np.ndarray:
    return np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(DESCRIPTOR_BOUNDS)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--train-sizes", type=int, nargs="+",
                        default=[2500, 5000, 10000, 20000, 40000, 0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--order", default="max-r2-strat")
    parser.add_argument("--fourier-v2", type=Path, default=None,
                        help="optional parquet with the richer Fourier block; aligned by stem")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/family_scaling_grid_125.csv")
    args = parser.parse_args()

    blocks, raw_labels, frame = load_corpus(args.dataset, VISUAL + ["Gabor"])
    if args.fourier_v2 is not None:
        table = pd.read_parquet(args.fourier_v2)
        cols = feature_columns(table)
        index = pd.Series(np.arange(len(table)), index=table["stem"].astype(str)).reindex(
            frame["stem"].astype(str).to_numpy())
        if index.isna().any():
            raise ValueError(f"Fourier-v2: {int(index.isna().sum())} stems missing")
        blocks["Fourier"] = sanitize(
            table.iloc[index.to_numpy(dtype=int)][cols].to_numpy(np.float32),
            "Fourier-v2")
        log(f"Fourier replaced by Fourier-v2: {blocks['Fourier'].shape[1]} features")
    if args.classes_file and args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("class-file matching is incomplete")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))

    pools = {}
    for label, names in FAMILY_STEPS:
        missing = [n for n in names if n not in blocks]
        if missing:
            raise ValueError(f"{label} is missing blocks: {missing}")
        values = np.hstack([blocks[n] for n in names])
        if names == ["Descriptors"]:
            groups = descriptor_groups()
        else:
            groups = np.concatenate([np.full(blocks[n].shape[1], i) for i, n in enumerate(names)])
        pools[label] = (values, groups, "+".join(names))

    log(f"{args.dataset}: {len(labels):,} images, {n_classes} classes")
    for label, (values, groups, names) in pools.items():
        log(f"  {label}: {values.shape[1]} features, {len(np.unique(groups))} selector groups ({names})")

    rows, done = [], set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(int(r["fold"]), str(r["pool"]), int(r["budget"]),
                 int(r["train_size"]), int(r["seed"])) for r in rows}
        log(f"resuming: {len(done)} points already done")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    max_budget = max(args.budgets)
    for fold, (train_index, test_index) in enumerate(splits[:args.folds], start=1):
        for requested in args.train_sizes:
            if requested == 0 or requested > len(train_index) - n_classes:
                sub = train_index
            else:
                sub, _ = train_test_split(train_index, train_size=requested,
                                          stratify=labels[train_index],
                                          random_state=SEED + fold)
            for pool, (values, block_of, names) in pools.items():
                available_budgets = [b for b in args.budgets if b <= values.shape[1]]
                if all((fold, pool, b, int(len(sub)), seed) in done
                       for b in available_budgets for seed in range(args.seeds)):
                    continue
                scaler = StandardScaler().fit(values[sub])
                scaled = scaler.transform(values).astype(np.float32)
                r2 = r2_matrix(scaled[sub])
                for seed in range(args.seeds):
                    rng = np.random.default_rng(1000 * seed + fold)
                    picked = select(args.order, r2, block_of,
                                    min(max_budget, values.shape[1]), rng)
                    for budget in available_budgets:
                        key_done = (fold, pool, budget, int(len(sub)), seed)
                        if key_done in done:
                            continue
                        cols = np.array(picked[:budget])
                        present = np.unique(labels[sub])
                        model = LinearDiscriminantAnalysis(
                            priors=np.ones(len(present)) / len(present))
                        model.fit(scaled[sub][:, cols], labels[sub])
                        pred = model.predict(scaled[test_index][:, cols])
                        rows.append({
                            "fold": fold,
                            "pool": pool,
                            "n_families": int(pool.split()[0]),
                            "members": names,
                            "budget": int(budget),
                            "train_size": int(len(sub)),
                            "seed": seed,
                            "groups_used": int(len(np.unique(block_of[cols]))),
                            "bacc": 100 * balanced_accuracy_score(labels[test_index], pred),
                        })
                pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
            log(f"fold {fold} n={len(sub):>6}: done")

    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
