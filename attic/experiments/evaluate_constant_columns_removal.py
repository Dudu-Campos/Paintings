"""LDA com vs. sem as 30 colunas de variância zero (achado exploratório: 12
`*_knn_n_leaf_frac`/`*_del_n_leaf_frac` + 18 `Fourier_log_power_r00-r02_a*`,
ambas estruturalmente constantes na coorte inteira, não bug de dado).

Mede quanto isso custa/ganha antes de remover as colunas do pipeline.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.pipeline.registry import load_blocks

SEED = 42
N_FOLDS = 5

CORE_BLOCKS = [
    "Descriptors", "SLIC", "Felzenszwalb", "Line graph",
    "L(G_felz)", "Hilbert", "Fourier", "Multiscale",
]


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def main() -> None:
    cohort = load_blocks("wikiart", CORE_BLOCKS, allow_external=False)
    all_columns = [c for name in CORE_BLOCKS for c in cohort.columns[name]]
    X_full = cohort.matrix(CORE_BLOCKS)
    assert X_full.shape[1] == len(all_columns)

    constant_mask = X_full.std(axis=0) < 1e-8
    print(f"{constant_mask.sum()} colunas constantes de {X_full.shape[1]}")
    dropped_names = [c for c, flag in zip(all_columns, constant_mask) if flag]
    print("colunas removidas:", dropped_names)

    X_reduced = X_full[:, ~constant_mask]

    y = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(y))

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_full, y))

    results = {"full": [], "reduced": []}
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        for tag, X in (("full", X_full), ("reduced", X_reduced)):
            model = make_lda(n_classes, X.shape[1])
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            bacc = balanced_accuracy_score(y[test_idx], pred)
            results[tag].append(bacc)
        print(f"fold {fold_index}: full={results['full'][-1]:.4f}  "
              f"reduced={results['reduced'][-1]:.4f}  "
              f"delta={results['reduced'][-1] - results['full'][-1]:+.4f}")

    full = np.array(results["full"])
    reduced = np.array(results["reduced"])
    delta = reduced - full
    stat, pvalue = ttest_rel(reduced, full)
    print()
    print(f"full ({X_full.shape[1]}d):    {full.mean():.4f} +/- {full.std(ddof=0):.4f}")
    print(f"reduced ({X_reduced.shape[1]}d): {reduced.mean():.4f} +/- {reduced.std(ddof=0):.4f}")
    print(f"delta: {delta.mean()*100:+.3f}pp +/- {delta.std(ddof=0)*100:.3f}pp  "
          f"p={pvalue:.4f}  {int((delta > 0).sum())}/{N_FOLDS} folds positivos")


if __name__ == "__main__":
    main()
