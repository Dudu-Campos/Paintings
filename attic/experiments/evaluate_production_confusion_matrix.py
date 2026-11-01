"""Matriz de confusao OOF (5-fold) da config de producao atual (7 blocos
nativos + Multiscale-Descriptors-only, ~1.105d pos-remocao das colunas
mortas), MLP de branches ingenuo (balanced_branch_mlp.py, aux_heads=False,
gating=False, branch_scale=False -- a mesma arquitetura usada como
instrumento de producao, achado 16 do CLAUDE.md). A matriz anterior
(slic_branch_mlp_confusion.png) era de uma config bem mais fraca (BAcc
0,458, quase perdendo pro LDA atual) -- essa aqui usa o modelo de verdade.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

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


def main() -> None:
    cohort = load_blocks("wikiart", CORE_BLOCKS, allow_external=False)
    encoder = LabelEncoder()
    y = encoder.fit_transform(cohort.labels)
    n_classes = len(np.unique(y))
    class_names = encoder.classes_

    multiscale_desc_only = cohort.select("Multiscale", lambda c: "_Descriptors_" in f"_{c}_")
    branch_arrays = [cohort.blocks[name] for name in CORE_BLOCKS if name != "Multiscale"]
    branch_names = [name for name in CORE_BLOCKS if name != "Multiscale"] + ["Multiscale_Descriptors"]
    branch_arrays.append(multiscale_desc_only)
    branch_sizes = [arr.shape[1] for arr in branch_arrays]
    X = np.hstack(branch_arrays).astype(np.float32)
    log(f"{X.shape[0]:,} imagens, {X.shape[1]}d em {len(branch_sizes)} branches: "
        f"{dict(zip(branch_names, branch_sizes))}")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    fold_scores = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        inner_train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=SEED, stratify=y[train_idx])

        # padronizacao por coluna, ajustada so no treino interno
        mean = X[inner_train_idx].mean(axis=0, keepdims=True)
        std = X[inner_train_idx].std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        Xs = (X - mean) / std

        metrics, prediction, _dependency = train_balanced(
            Xs[inner_train_idx], y[inner_train_idx], Xs[val_idx], y[val_idx],
            Xs[test_idx], y[test_idx], branch_sizes, n_classes, SEED,
            max_epochs=100, patience=12, branch_names=branch_names,
        )
        oof_pred[test_idx] = prediction
        fold_scores.append(metrics["bacc"])
        log(f"fold {fold_index}: BAcc={metrics['bacc']*100:.2f}% "
            f"(val={metrics['validation_bacc']*100:.2f}%, epoch={metrics['best_epoch']})")

    log(f"OOF BAcc media: {np.mean(fold_scores)*100:.2f} +/- {np.std(fold_scores)*100:.2f}")
    overall_bacc = balanced_accuracy_score(y, oof_pred)
    log(f"OOF BAcc (agregado): {overall_bacc*100:.2f}%")

    cm = confusion_matrix(y, oof_pred, labels=np.arange(n_classes))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    rows = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true_style": class_names[i], "predicted_style": class_names[j],
                    "count": int(cm[i, j]), "rate_within_true_style": float(cm_norm[i, j]),
                })
    pairs = pd.DataFrame(rows).sort_values("rate_within_true_style", ascending=False)

    out_dir = ROOT / "results"
    pairs.to_csv(out_dir / "production_mlp_confusion_pairs.csv", index=False)
    np.save(out_dir / "production_mlp_confusion_matrix.npy", cm)
    pd.Series(class_names).to_csv(out_dir / "production_mlp_confusion_classes.csv", index=False)

    print()
    print("=== top 20 pares mais confundidos ===")
    print(pairs.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
