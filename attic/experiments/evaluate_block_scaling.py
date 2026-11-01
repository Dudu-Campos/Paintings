"""LDA: padronização por coluna (atual) vs. padronização + reequilíbrio por
bloco, pra testar se blocos com mais colunas (ex.: Multiscale ~1000d vs.
Hilbert 57d) dominam a covariância conjunta só por volume de dimensão, não
por sinal.

Duas variantes de reequilíbrio, aplicadas *depois* do StandardScaler por
coluna (que já resolve diferença de magnitude bruta entre blocos):
  - 1/sqrt(d_bloco): iguala a "energia" total (soma de variância) de cada
    bloco a 1, a normalização padrão em combinação multi-view/multi-kernel.
  - 1/d_bloco: mais agressiva, iguala a variância média por coluna do bloco
    ao invés da energia total.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.pipeline.registry import load_blocks

SEED = 42
N_FOLDS = 5

CORE_BLOCKS = [
    "Descriptors", "SLIC", "Felzenszwalb", "Line graph",
    "L(G_felz)", "Hilbert", "Fourier", "Multiscale",
]


def block_weights(cohort, mode: str) -> np.ndarray:
    """Um peso por coluna, constante (não depende do fold)."""
    weights = []
    for name in CORE_BLOCKS:
        d = cohort.blocks[name].shape[1]
        if mode == "column":
            w = 1.0
        elif mode == "sqrt":
            w = 1.0 / np.sqrt(d)
        elif mode == "linear":
            w = 1.0 / d
        else:
            raise ValueError(mode)
        weights.extend([w] * d)
    return np.asarray(weights, dtype=np.float64)


def fit_eval(X_train, y_train, X_test, y_test, n_classes) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    lda = LinearDiscriminantAnalysis(
        n_components=min(n_classes - 1, X_train.shape[1]),
        priors=np.ones(n_classes) / n_classes,
    )
    lda.fit(Xtr, y_train)
    pred = lda.predict(Xte)
    return balanced_accuracy_score(y_test, pred)


def main() -> None:
    cohort = load_blocks("wikiart", CORE_BLOCKS, allow_external=False)
    X = cohort.matrix(CORE_BLOCKS)
    y = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(y))

    print("dimensão por bloco:")
    for name in CORE_BLOCKS:
        print(f"  {name:14s} {cohort.blocks[name].shape[1]:5d}d")
    print(f"total: {X.shape[1]}d")

    modes = ["column", "sqrt", "linear"]
    weights = {mode: block_weights(cohort, mode) for mode in modes}

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    results = {mode: [] for mode in modes}
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        row = []
        for mode in modes:
            Xw = X * weights[mode][None, :]
            bacc = fit_eval(Xw[train_idx], y[train_idx], Xw[test_idx], y[test_idx], n_classes)
            results[mode].append(bacc)
            row.append(f"{mode}={bacc:.4f}")
        print(f"fold {fold_index}: " + "  ".join(row))

    print()
    baseline = np.array(results["column"])
    for mode in modes:
        scores = np.array(results[mode])
        line = f"{mode:8s}: {scores.mean()*100:.3f}% +/- {scores.std(ddof=0)*100:.3f}pp"
        if mode != "column":
            delta = scores - baseline
            stat, pvalue = ttest_rel(scores, baseline)
            line += (f"   delta={delta.mean()*100:+.3f}pp +/- {delta.std(ddof=0)*100:.3f}pp"
                     f"  p={pvalue:.4f}  {int((delta > 0).sum())}/{N_FOLDS} folds positivos")
        print(line)


if __name__ == "__main__":
    main()
