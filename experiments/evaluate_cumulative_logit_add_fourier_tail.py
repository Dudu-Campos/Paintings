
"""Recompute k>=10 of cumulative_logit_add with the updated Fourier block."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src' / 'pipeline'))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_cumulative_twelve import ORDER  # noqa: E402
from evaluate_logit_add_multitask import build_oof, train_arm  # noqa: E402
from evaluate_shapley_twelve_views import DORMANT, HAND8, N_FOLDS, SEED  # noqa: E402
from registry import feature_columns, load_blocks, log, sanitize  # noqa: E402


def load_fourier(path: Path, stems: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path)
    cols = feature_columns(table)
    index = pd.Series(np.arange(len(table)), index=table['stem'].astype(str)).reindex(stems)
    if index.isna().any():
        raise ValueError(f"Fourier: {int(index.isna().sum())} stems missing")
    return sanitize(table.iloc[index.to_numpy(dtype=int)][cols].to_numpy(np.float32), 'Fourier')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fourier', type=Path, default=ROOT / 'results/fourier_v2_multitask_features.parquet')
    ap.add_argument('--k-neighbors', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--output', type=Path, default=ROOT / 'results/cumulative_logit_add_fourier_tail.csv')
    args = ap.parse_args()

    cohort = load_blocks('multitask', HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    blocks = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    blocks['Fourier'] = load_fourier(args.fourier, stems)
    for name in DORMANT:
        values_path, stems_path = EXTERNAL[name]
        values = np.load(ROOT / values_path)
        order = pd.Index(np.load(ROOT / stems_path, allow_pickle=True).astype(str)).get_indexer(stems)
        if (order < 0).any():
            raise SystemExit(f'{name}: {int((order < 0).sum())} stems ausentes')
        blocks[name] = values[order].astype(np.float32)
    log(f"multitask: {len(stems):,} images, {n_classes} classes; Fourier {blocks['Fourier'].shape[1]}d")

    if args.output.exists():
        args.output.unlink()
    cache_dir = ROOT / 'outputs/shapley_twelve_views'
    space = np.hstack([blocks[n] for n in HAND8]).astype(np.float32)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for fold, (outer, test) in enumerate(splits, start=1):
        cache = cache_dir / f'fold_{fold}.npz'
        per_fold = dict(blocks)
        per_fold['Text'] = np.load(cache)['text'].astype(np.float32)
        fit, validation = train_test_split(outer, test_size=0.1, stratify=labels[outer], random_state=SEED)
        scaler = StandardScaler().fit(space[outer])
        scaled = scaler.transform(space).astype(np.float32)
        oof, held = build_oof(scaled[outer], labels[outer], scaled[test], n_classes, args.k_neighbors)
        lift = np.zeros((len(labels), n_classes), dtype=np.float32)
        lift[outer], lift[test] = oof, held
        for k in range(10, len(ORDER) + 1):
            names = [n for n in ORDER[:k] if n != 'Similarity graph']
            start = time.time()
            metrics, lam = train_arm([per_fold[n] for n in names], lift, labels,
                                     fit, validation, test, n_classes,
                                     np.zeros(n_classes, dtype=np.int64),
                                     SEED, args.epochs, args.patience)
            row = pd.DataFrame([dict(k=k, classifier='BranchMLP+logit-add', fold=fold,
                                     added=ORDER[k-1], blocks='|'.join(names),
                                     dimensions=sum(per_fold[n].shape[1] for n in names),
                                     bacc=metrics['bacc'], lam=float(np.ravel(lam)[0]),
                                     seconds=round(time.time() - start, 1))])
            row.to_csv(args.output, mode='a', index=False, header=not args.output.exists(), float_format='%.4f')
            log(f"fold {fold} k={k:2d} | {row.dimensions.iloc[0]:4d}d | BAcc {metrics['bacc']:5.2f} | lambda {row.lam.iloc[0]:.4f}")
    log(f'saved: {args.output}')

if __name__ == '__main__':
    main()
