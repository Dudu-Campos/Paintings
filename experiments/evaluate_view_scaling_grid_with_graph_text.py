"""Scaling grid for cumulative views, adding Similarity graph and Text.

The first 8 views follow the visual Shapley order and use Fourier v2. The last
2 views are the per-fold cached Similarity graph and Text blocks from the exact
Shapley run, so the folds must match that protocol.
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
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from evaluate_redundancy_selection_curve import r2_matrix, select  # noqa: E402
from evaluate_shapley_twelve_views import HAND8, N_FOLDS, SEED  # noqa: E402
from registry import feature_columns, load_blocks, log, sanitize  # noqa: E402

VISUAL_ORDER = ["Felzenszwalb", "SLIC", "Descriptors", "L(G_felz)",
                "Line graph", "Gabor", "Hilbert", "Fourier"]
EXTENDED_ORDER = VISUAL_ORDER + ["Similarity graph", "Text"]


def view_steps() -> list[tuple[str, list[str]]]:
    return [(f"{i} views", EXTENDED_ORDER[:i]) for i in range(1, len(EXTENDED_ORDER) + 1)]


def load_fourier_v2(path: Path, stems: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path)
    cols = feature_columns(table)
    index = pd.Series(np.arange(len(table)), index=table["stem"].astype(str)).reindex(stems)
    if index.isna().any():
        raise ValueError(f"Fourier-v2: {int(index.isna().sum())} stems missing")
    return sanitize(table.iloc[index.to_numpy(dtype=int)][cols].to_numpy(np.float32), "Fourier-v2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[2500, 5000, 10000, 20000, 40000, 0])
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--order", default="max-r2-strat")
    parser.add_argument("--fourier-v2", type=Path, default=ROOT / "results/fourier_v2_multitask_features.parquet")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/shapley_twelve_views")
    parser.add_argument("--output", type=Path, default=ROOT / "results/view_scaling_grid_shapley_fourier_v2_graph_text.csv")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    static = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    static["Fourier"] = load_fourier_v2(args.fourier_v2, stems)
    log(f"{args.dataset}: {len(labels):,} images, {n_classes} classes")
    log(f"Fourier replaced by Fourier-v2: {static['Fourier'].shape[1]} features")

    rows, done = [], set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(int(r["fold"]), str(r["pool"]), int(r["budget"]), int(r["train_size"]), int(r["seed"])) for r in rows}
        log(f"resuming: {len(done)} points already done")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))[:args.folds]
    max_budget = max(args.budgets)
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        cache = args.cache_dir / f"fold_{fold}.npz"
        if not cache.exists():
            raise SystemExit(f"missing cache: {cache}")
        saved = np.load(cache)
        blocks = dict(static)
        blocks["Similarity graph"] = saved["graph"].astype(np.float32)
        blocks["Text"] = saved["text"].astype(np.float32)
        log(f"fold {fold}: graph {blocks['Similarity graph'].shape[1]}d, text {blocks['Text'].shape[1]}d")

        pools = {}
        for label, names in view_steps():
            values = np.hstack([blocks[n] for n in names])
            groups = np.concatenate([np.full(blocks[n].shape[1], i) for i, n in enumerate(names)])
            pools[label] = (values, groups, "+".join(names))

        for requested in args.train_sizes:
            if requested == 0 or requested > len(train_index) - n_classes:
                sub = train_index
            else:
                sub, _ = train_test_split(train_index, train_size=requested,
                                          stratify=labels[train_index], random_state=SEED + fold)
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
                    picked = select(args.order, r2, block_of, min(max_budget, values.shape[1]), rng)
                    for budget in available_budgets:
                        key_done = (fold, pool, budget, int(len(sub)), seed)
                        if key_done in done:
                            continue
                        cols = np.array(picked[:budget])
                        present = np.unique(labels[sub])
                        model = LinearDiscriminantAnalysis(priors=np.ones(len(present)) / len(present))
                        model.fit(scaled[sub][:, cols], labels[sub])
                        pred = model.predict(scaled[test_index][:, cols])
                        rows.append({
                            "fold": fold, "pool": pool, "n_views": int(pool.split()[0]),
                            "members": names, "budget": int(budget), "train_size": int(len(sub)),
                            "seed": seed, "groups_used": int(len(np.unique(block_of[cols]))),
                            "bacc": 100 * balanced_accuracy_score(labels[test_index], pred),
                        })
                pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
            log(f"fold {fold} n={len(sub):>6}: done")
    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
