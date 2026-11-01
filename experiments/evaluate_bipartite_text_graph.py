"""Grafo bipartido (obra, palavra) no lugar do LinearSVC do canal de texto.

O canal de texto do §5.18 usa TF-IDF -> LinearSVC one-vs-rest, e com 125 classes
isso custa ~66 min por passada OOF (x5 no aninhado). A alternativa e propagacao
de rotulo sobre o grafo bipartido, que em algebra e um matmul esparso:

    W = X_treino^T diag(w) Y_treino    (palavra x classe)
    escore = X_teste W

com `w_i = 1/n_classe(i)`, que e o equivalente do `class_weight="balanced"`, e
log-lift no fim (§2: todo bloco tipo-posterior nasce com log-lift).

**O bonus conceitual**: isso unifica o canal de texto com o grafo de similaridade
-- vira o MESMO operador (propagacao de rotulo, zero parametros aprendidos) sobre
um grafo bipartido em vez de um kNN imagem-imagem. Para um artigo sobre
decomposicao multiview isso le melhor que "e tambem usamos um SVM".

E o gargalo continua legivel: os pesos palavra->classe sao a explicacao, sem
precisar extrair coeficientes de 125 SVMs (versao "verificavel" do §7.2).

Alvo a bater: LinearSVC = 16,08 +- 0,57 nas mesmas 5 dobras.
CPU puro -- nao disputa GPU com a grade de fine-tuning.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_text_scores import (MAX_FEATURES, NGRAM_RANGE, clean_caption,  # noqa: E402
                               make_stop_words)
from evaluate_external_modalities import LABELS  # noqa: E402

SEED, N_FOLDS = 42, 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prior_lift(scores: np.ndarray, train_y: np.ndarray, n_classes: int) -> np.ndarray:
    prior = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    prior /= prior.sum()
    total = scores.sum(axis=1, keepdims=True)
    probs = scores / np.maximum(total, 1e-12)
    return (np.log(probs + 1e-8) - np.log(prior)).astype(np.float32)


def bipartite(train_x, train_y, test_x, n_classes, steps: int):
    """Propagacao obra->palavra->obra, `steps` vezes. steps=1 e um matmul."""
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts[train_y], 1.0)          # class_weight balanced
    y_hot = sp.csr_matrix((weights, (np.arange(len(train_y)), train_y)),
                          shape=(len(train_y), n_classes))
    signal = y_hot
    for _ in range(steps):
        word_class = train_x.T @ signal                        # palavra x classe
        word_class = np.asarray(word_class.todense() if sp.issparse(word_class) else word_class)
        word_class /= np.maximum(word_class.sum(axis=1, keepdims=True), 1e-12)
        if _ == steps - 1:
            break
        train_scores = np.asarray(train_x @ word_class)        # obra x classe
        signal = sp.csr_matrix(train_scores / np.maximum(
            train_scores.sum(axis=1, keepdims=True), 1e-12))
    return np.asarray(test_x @ word_class), word_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--captions", type=Path,
                    default=ROOT / "results/multitask_student_captions.jsonl")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--with-svm", action="store_true", default=True)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/bipartite_text_graph.csv")
    args = ap.parse_args()

    meta_path, column = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / meta_path, dtype={"stem": str}).dropna(subset=[column])
    meta = meta.drop_duplicates("stem", keep="last")
    table = {}
    with args.captions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            table[str(row["stem"])] = row["caption"]
    frame = meta[["stem", column]].rename(columns={column: "label"})
    frame["caption"] = frame["stem"].astype(str).map(table)
    frame = frame.dropna(subset=["caption"]).sort_values("stem").reset_index(drop=True)
    frame["clean"] = frame["caption"].map(clean_caption)
    frame = frame[frame["clean"].str.len() > 0].reset_index(drop=True)

    texts = frame["clean"].to_numpy(dtype=str)
    labels_str = frame["label"].to_numpy(dtype=str)
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    labels = np.array([index_of[v] for v in labels_str])
    n_classes = len(classes)
    stop_words = make_stop_words(classes)
    log(f"{args.dataset}: {len(frame):,} legendas, {n_classes} classes")

    rows = []
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(texts, labels)
    for fold, (train, test) in enumerate(folds, 1):
        t0 = time.perf_counter()
        vec = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                              binary=True, sublinear_tf=True, stop_words=stop_words, min_df=2)
        train_x = vec.fit_transform(texts[train])
        test_x = vec.transform(texts[test])
        t_vec = time.perf_counter() - t0
        log(f"fold {fold}: {train_x.shape[1]:,} termos, TF-IDF {t_vec:.0f}s")

        for steps in args.steps:
            t0 = time.perf_counter()
            scores, word_class = bipartite(train_x, labels[train], test_x, n_classes, steps)
            elapsed = time.perf_counter() - t0
            lifted = prior_lift(np.maximum(scores, 0), labels[train], n_classes)
            bacc = 100 * balanced_accuracy_score(labels[test], lifted.argmax(1))
            rows.append(dict(fold=fold, method=f"bipartido k={steps}", bacc=bacc,
                             seconds=elapsed, terms=train_x.shape[1]))
            log(f"  bipartido {steps} passo(s): BAcc {bacc:5.2f} | {elapsed:6.1f}s")

        if args.with_svm:
            t0 = time.perf_counter()
            model = LinearSVC(class_weight="balanced", random_state=SEED)
            model.fit(train_x, labels[train])
            elapsed = time.perf_counter() - t0
            bacc = 100 * balanced_accuracy_score(labels[test], model.predict(test_x))
            rows.append(dict(fold=fold, method="LinearSVC", bacc=bacc,
                             seconds=elapsed, terms=train_x.shape[1]))
            log(f"  LinearSVC:            BAcc {bacc:5.2f} | {elapsed:6.1f}s")

        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    frame_out = pd.DataFrame(rows)
    agg = frame_out.groupby("method").agg(bacc=("bacc", "mean"), sd=("bacc", "std"),
                                          segundos=("seconds", "mean"))
    ref = agg.loc["LinearSVC", "segundos"] if "LinearSVC" in agg.index else np.nan
    agg["x mais rápido"] = ref / agg["segundos"]
    print("\n" + agg.round(2).to_string())


if __name__ == "__main__":
    main()
