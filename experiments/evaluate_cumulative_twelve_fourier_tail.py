
"""Recompute k>=10 of cumulative_twelve with the updated Fourier block."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src' / 'pipeline'))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_cumulative_twelve import ORDER  # noqa: E402
from evaluate_redundancy_selection_curve import fit as plain_fit  # noqa: E402
from evaluate_shapley_twelve_views import DORMANT, HAND8, N_FOLDS, SEED  # noqa: E402
from registry import feature_columns, load_blocks, log, sanitize  # noqa: E402


def load_fourier_v2(path: Path, stems: np.ndarray) -> np.ndarray:
    table = pd.read_parquet(path)
    cols = feature_columns(table)
    index = pd.Series(np.arange(len(table)), index=table['stem'].astype(str)).reindex(stems)
    if index.isna().any():
        raise ValueError(f"Fourier: {int(index.isna().sum())} stems missing")
    return sanitize(table.iloc[index.to_numpy(dtype=int)][cols].to_numpy(np.float32), 'Fourier')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fourier', type=Path, default=ROOT / 'results/fourier_v2_multitask_features.parquet')
    ap.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs/shapley_twelve_views')
    ap.add_argument('--output', type=Path, default=ROOT / 'results/cumulative_twelve_fourier_tail.csv')
    args = ap.parse_args()

    cohort = load_blocks('multitask', HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    blocks = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    blocks['Fourier'] = load_fourier_v2(args.fourier, stems)
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
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        cache = args.cache_dir / f'fold_{fold}.npz'
        saved = np.load(cache)
        per_fold = dict(blocks)
        per_fold['Similarity graph'] = saved['graph'].astype(np.float32)
        per_fold['Text'] = saved['text'].astype(np.float32)
        scaled = {n: StandardScaler().fit(per_fold[n][train_idx]).transform(per_fold[n]).astype(np.float32)
                  for n in ORDER}
        for k in range(10, len(ORDER) + 1):
            names = ORDER[:k]
            stacked = np.hstack([scaled[n] for n in names])
            for classifier in ('LDA', 'MLP'):
                start = time.time()
                if classifier == 'LDA':
                    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
                    model.fit(stacked[train_idx], labels[train_idx])
                    bacc = 100 * balanced_accuracy_score(labels[test_idx], model.predict(stacked[test_idx]))
                else:
                    fit_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=labels[train_idx], random_state=SEED + fold)
                    bacc = plain_fit(stacked[fit_idx], labels[fit_idx], stacked[val_idx], labels[val_idx],
                                     stacked[test_idx], labels[test_idx], n_classes, seed=SEED + fold)['bacc']
                row = pd.DataFrame([dict(k=k, classifier=classifier, fold=fold, added=ORDER[k-1],
                                         blocks='|'.join(names), dimensions=stacked.shape[1], bacc=bacc,
                                         seconds=round(time.time() - start, 1))])
                row.to_csv(args.output, mode='a', index=False, header=not args.output.exists(), float_format='%.4f')
                log(f"fold {fold} k={k:2d} {classifier:3s} | {stacked.shape[1]:4d}d | BAcc {bacc:5.2f}")
    log(f'saved: {args.output}')

if __name__ == '__main__':
    main()
