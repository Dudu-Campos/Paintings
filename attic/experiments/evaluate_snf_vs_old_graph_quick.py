"""Comparacao rapida (10k, 1 split, nao 5-fold) entre o grafo antigo (k-NN
unico sobre o vetor concatenado, sklearn LabelSpreading kernel='knn',
transdutivo) e o novo grafo SNF (fusao por modalidade). So pra ter uma
direcao antes de rodar o experimento completo com CV de verdade.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from src.pipeline.registry import load_blocks  # noqa: E402
import evaluate_snf_similarity_graph as snf  # noqa: E402

SEED = 42
N_SAMPLE = 10_000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def make_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, max_iter=200,
                              early_stopping=True, n_iter_no_change=10, random_state=SEED)),
    ])


def evaluate(name: str, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, n_classes: int) -> None:
    for clf_name, make in (("lda", make_lda), ("mlp", make_mlp)):
        model = make(n_classes, xtr.shape[1]) if clf_name == "lda" else make()
        model.fit(xtr, ytr)
        pred = model.predict(xte)
        bacc = balanced_accuracy_score(yte, pred)
        log(f"  {name:18s} {clf_name:4s}: BAcc={bacc*100:.2f}%")


def main() -> None:
    rng = np.random.default_rng(SEED)
    cohort = load_blocks("wikiart", snf.CORE_BLOCKS, allow_external=False)
    n_total = len(cohort.stems)
    sample_idx = rng.choice(n_total, N_SAMPLE, replace=False)

    y = LabelEncoder().fit_transform(cohort.labels[sample_idx])
    n_classes = len(np.unique(y))
    log(f"amostra: {N_SAMPLE:,} imagens, {n_classes} classes")

    multiscale_desc_only = cohort.select("Multiscale", lambda c: "_Descriptors_" in f"_{c}_")[sample_idx]
    base_blocks = [cohort.blocks[name][sample_idx] for name in snf.CORE_BLOCKS if name != "Multiscale"]
    base = np.hstack(base_blocks + [multiscale_desc_only]).astype(np.float32)
    log(f"base de producao: {base.shape[1]}d")

    train_idx, test_idx = train_test_split(np.arange(N_SAMPLE), test_size=0.2, random_state=SEED, stratify=y)
    y_train, y_test = y[train_idx], y[test_idx]
    base_train, base_test = base[train_idx], base[test_idx]

    log("=== base sozinha ===")
    evaluate("base", base_train, y_train, base_test, y_test, n_classes)

    log("=== grafo ANTIGO (kNN unico, sklearn LabelSpreading transdutivo) ===")
    t0 = time.time()
    scaler = StandardScaler().fit(base_train)
    base_scaled_all = scaler.transform(base)
    y_masked = y.copy().astype(int)
    y_masked[test_idx] = -1
    spreader = LabelSpreading(kernel="knn", n_neighbors=30, alpha=0.4, max_iter=50)
    spreader.fit(base_scaled_all, y_masked)
    posterior_old = spreader.label_distributions_
    prior = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    prior /= prior.sum()
    old_feature = snf.log_lift(posterior_old, prior).astype(np.float32)
    log(f"grafo antigo: {time.time()-t0:.1f}s")
    evaluate("base+old_graph", np.hstack([base_train, old_feature[train_idx]]), y_train,
             np.hstack([base_test, old_feature[test_idx]]), y_test, n_classes)

    log("=== grafo NOVO (SNF, fundido por modalidade) ===")
    t0 = time.time()
    blocks_train = {name: cohort.blocks[name][sample_idx][train_idx].astype(np.float32) for name in snf.CORE_BLOCKS}
    blocks_test = {name: cohort.blocks[name][sample_idx][test_idx].astype(np.float32) for name in snf.CORE_BLOCKS}
    _s_mats, fused_train = snf.snf_fuse(blocks_train)
    onehot = np.zeros((len(train_idx), n_classes))
    onehot[np.arange(len(train_idx)), y_train] = 1.0
    train_posterior = snf.spread_labels(fused_train, onehot)
    test_posterior = snf.inductive_test_posterior(blocks_train, blocks_test, train_posterior)
    train_feature_new = snf.log_lift(train_posterior, prior).astype(np.float32)
    test_feature_new = snf.log_lift(test_posterior, prior).astype(np.float32)
    log(f"grafo novo: {time.time()-t0:.1f}s")
    evaluate("base+snf_graph", np.hstack([base_train, train_feature_new]), y_train,
             np.hstack([base_test, test_feature_new]), y_test, n_classes)

    log("done")


if __name__ == "__main__":
    main()
