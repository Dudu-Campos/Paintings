"""Contribuicao marginal do Gabor (32d, descritor classico faltante, achado
27 do CLAUDE.md) sobre o pipeline de 8 blocos nativos do Pandora. LDA +
MLP de branches (mesma arquitetura ingenua de producao, achado 16).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from src.pipeline.registry import load_blocks  # noqa: E402
from balanced_branch_mlp import train_balanced  # noqa: E402

SEED = 42
N_FOLDS = 5
CORE_BLOCKS = [
    "Descriptors", "SLIC", "Felzenszwalb", "Line graph",
    "L(G_felz)", "Hilbert", "Fourier", "Multiscale",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def lda_eval(blocks: list[np.ndarray], train_idx, test_idx, y, n_classes) -> float:
    X = np.hstack(blocks)
    scaler = StandardScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
    model.fit(Xtr, y[train_idx])
    return balanced_accuracy_score(y[test_idx], model.predict(Xte))


def main() -> None:
    cohort = load_blocks("pandora", CORE_BLOCKS + ["Gabor"], allow_external=False)
    encoder = LabelEncoder()
    y = encoder.fit_transform(cohort.labels)
    n_classes = len(np.unique(y))

    multiscale_desc_only = cohort.select("Multiscale", lambda c: "_Descriptors_" in f"_{c}_")
    base_arrays = [cohort.blocks[name] for name in CORE_BLOCKS if name != "Multiscale"]
    base_names = [name for name in CORE_BLOCKS if name != "Multiscale"] + ["Multiscale_Descriptors"]
    base_arrays.append(multiscale_desc_only)
    gabor = cohort.blocks["Gabor"]

    log(f"{len(cohort.stems):,} imagens, {n_classes} classes; "
        f"base={sum(a.shape[1] for a in base_arrays)}d; gabor={gabor.shape[1]}d")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(y)), y))
    rows = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        for condition, blocks, names in (
            ("base", base_arrays, base_names),
            ("base+gabor", base_arrays + [gabor], base_names + ["Gabor"]),
        ):
            lda_bacc = lda_eval(blocks, train_idx, test_idx, y, n_classes)

            inner_train_idx, val_idx = train_test_split(
                train_idx, test_size=0.1, random_state=SEED, stratify=y[train_idx])
            X = np.hstack(blocks)
            mean = X[inner_train_idx].mean(axis=0, keepdims=True)
            std = X[inner_train_idx].std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            Xs = (X - mean) / std
            branch_sizes = [b.shape[1] for b in blocks]
            metrics, _pred, _dep = train_balanced(
                Xs[inner_train_idx], y[inner_train_idx], Xs[val_idx], y[val_idx],
                Xs[test_idx], y[test_idx], branch_sizes, n_classes, SEED,
                max_epochs=100, patience=12, branch_names=names,
            )
            rows.append({"fold": fold_index, "condition": condition,
                        "lda": lda_bacc, "mlp": metrics["bacc"]})
            log(f"fold={fold_index} {condition:12s} LDA={lda_bacc*100:.2f}%  MLP={metrics['bacc']*100:.2f}%")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "gabor_pandora_results.csv", index=False)
    print()
    for clf in ("lda", "mlp"):
        base = df[df.condition == "base"].sort_values("fold")[clf].to_numpy()
        with_g = df[df.condition == "base+gabor"].sort_values("fold")[clf].to_numpy()
        delta = with_g - base
        stat, p = ttest_rel(with_g, base)
        print(f"{clf}: base={base.mean()*100:.2f}+/-{base.std(ddof=0)*100:.2f}  "
              f"+gabor={with_g.mean()*100:.2f}+/-{with_g.std(ddof=0)*100:.2f}  "
              f"delta={delta.mean()*100:+.3f}pp +/- {delta.std(ddof=0)*100:.3f}pp  "
              f"p={p:.4f}  {int((delta>0).sum())}/5 folds")


if __name__ == "__main__":
    main()
