"""O WL paga NO PIPELINE? Marginal do bloco WL sobre o hand-8 inteiro.

O `evaluate_wl_subtree_multitask.py` mede o WL **sozinho** e acha sinal real
(fold 1: d2=19,33 contra 13,74 do controle de arestas embaralhadas). Mas o
§5.17 e o §5.9d medem que desempenho isolado NAO preve contribuicao marginal, e
o §5.9c mede que as quatro views de regiao ja se cobrem em 63-73% -- uma quinta
sobre as MESMAS particoes e a priori mais coberta que uma sobre substrato novo.
A percolacao sobre estes mesmos RAGs deu 0/5 folds (§10.3).

**OOF ANINHADO, e aqui ele e obrigatorio.** No teste standalone bastava treinar
no treino e prever no teste. Como bloco de ENTRADA do `BranchMLP`, as linhas de
treino tambem precisam de escore, e escore produzido por um modelo que viu o
rotulo daquela linha e auto-referencia -- o mecanismo que derrubou o SNF (§6) e
que o §5.18 trata com 5 folds internos por fold externo.

**Codificacao: TF-IDF -> LinearSVC -> 125 escores por classe**, a mesma do canal
de texto. Concatenar as ~600 mil colunas esparsas do bag of subtrees no
`BranchMLP` (que manda todo bloco para 128 dims) e a patologia do §5.17c.

**Ressalva de protocolo herdada do canal de texto (§5.18c):** `decision_function`
devolve margem de SVM, nao `p(c|x)`, entao este bloco NAO cumpre o §2 ("todo
bloco tipo-posterior nasce com log-lift"). Mantido assim de proposito, para ser
comparavel ao canal de texto que ja esta medido; a divida e a mesma dos dois.

Tres bracos: hand-8, + WL real, + WL com arestas embaralhadas. O placebo nao e
opcional -- o §5.6 mede que um bloco com informacao destruida ainda marca
+0,22pp so de arquitetura.

Uso:
    python experiments/evaluate_wl_pipeline_multitask.py --k 64 --folds 5
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

from corpora import HAND8                              # noqa: E402
from protocol import SEED, evaluate_mlp                # noqa: E402
from registry import load_blocks                       # noqa: E402
from evaluate_wl_subtree_multitask import (            # noqa: E402
    counts_matrix, global_edges, load_cache, wl_labels)

BANDA_HAND8 = (37.5, 39.2)   # §9.7 mede 38,53/38,71/38,74 entre execucoes;
                             # a margarita reproduz ~0,6pp abaixo (medido 30/set)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def wl_scores(counts, y: np.ndarray, outer_train: np.ndarray,
              outer_test: np.ndarray, n_classes: int, fold: int,
              inner_folds: int = 5) -> np.ndarray:
    """Escores por classe para TODAS as linhas, com OOF aninhado no treino."""
    scores = np.zeros((counts.shape[0], n_classes), dtype=np.float32)

    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=SEED + fold)
    for fit_idx, held_idx in inner.split(np.zeros(len(outer_train)), y[outer_train]):
        fit_rows, held_rows = outer_train[fit_idx], outer_train[held_idx]
        tfidf = TfidfTransformer(sublinear_tf=True)
        Xf = tfidf.fit_transform(counts[fit_rows])
        clf = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=3000)
        clf.fit(Xf, y[fit_rows])
        scores[held_rows] = clf.decision_function(tfidf.transform(counts[held_rows]))

    # teste externo: modelo ajustado em TODO o treino externo
    tfidf = TfidfTransformer(sublinear_tf=True)
    Xtr = tfidf.fit_transform(counts[outer_train])
    clf = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=3000)
    clf.fit(Xtr, y[outer_train])
    scores[outer_test] = clf.decision_function(tfidf.transform(counts[outer_test]))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/prism_data/multitask_slic_rag"))
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "multitask_wl_pipeline.csv")
    args = parser.parse_args()

    log(f"coorte de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset, list(HAND8))
    y = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(y))
    stems = [str(s) for s in cohort.stems]
    n_images = len(stems)
    hand8 = [cohort.blocks[name] for name in HAND8]
    log(f"  {n_images} imagens, {cohort.dimensions}d, {n_classes} classes")

    log(f"lendo cache de RAG em {args.cache_dir}")
    cache = load_cache(args.cache_dir, stems)
    src_real, dst_real = global_edges(cache["edges"], cache["node_offsets"])
    src_fake, dst_fake = global_edges(cache["edges"], cache["node_offsets"],
                                      rng=np.random.default_rng(SEED))
    log(f"  {cache['lab'].shape[0]:,} nos, {src_real.size:,} arestas")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(n_images), y))[:args.folds]

    rows: list[dict] = []
    for fold, (train, test) in enumerate(splits, start=1):
        # (1) hand-8 sozinho -- a referencia, e a guarda de que o caminho e o de producao
        t0 = time.perf_counter()
        m = evaluate_mlp(hand8, y, train, test, fold, epochs=150, patience=15)
        rows.append({"fold": fold, "arm": "hand-8", "dims": cohort.dimensions,
                     "seconds": time.perf_counter() - t0, "bacc": 100 * m["bacc"],
                     "accuracy": 100 * m["accuracy"], "macro_f1": 100 * m["macro_f1"]})
        log(f"  fold {fold}  hand-8                       BAcc {rows[-1]['bacc']:5.2f}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

        # (2) codebook de cor, ajustado SO no treino externo
        node_is_train = np.repeat(np.isin(np.arange(n_images), train),
                                  np.diff(cache["node_offsets"]))
        sample = cache["lab"][node_is_train]
        if sample.shape[0] > 400_000:
            idx = np.random.default_rng(SEED + fold).choice(sample.shape[0], 400_000,
                                                            replace=False)
            sample = sample[idx]
        km = MiniBatchKMeans(n_clusters=args.k, random_state=SEED + fold,
                             n_init=3, batch_size=4096).fit(sample)
        colour = km.predict(cache["lab"]).astype(np.int64)

        for label, (s, d) in [("WL", (src_real, dst_real)),
                              ("WL embaralhado", (src_fake, dst_fake))]:
            t0 = time.perf_counter()
            sets = wl_labels(colour, s, d, args.depth, SEED + fold)
            counts = counts_matrix(sets, cache["node_offsets"], n_images, args.min_df)
            block = wl_scores(counts, y, train, test, n_classes, fold)
            del counts, sets
            m = evaluate_mlp(hand8 + [block], y, train, test, fold,
                             epochs=150, patience=15)
            rows.append({"fold": fold, "arm": f"hand-8 + {label}",
                         "dims": cohort.dimensions + n_classes,
                         "seconds": time.perf_counter() - t0,
                         "bacc": 100 * m["bacc"], "accuracy": 100 * m["accuracy"],
                         "macro_f1": 100 * m["macro_f1"]})
            log(f"  fold {fold}  hand-8 + {label:18s} BAcc {rows[-1]['bacc']:5.2f}  "
                f"({rows[-1]['seconds']:.0f}s)")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    summary = table.groupby("arm", sort=False).agg(
        dims=("dims", "first"), bacc=("bacc", "mean"), sd=("bacc", "std"),
        acc=("accuracy", "mean"), f1=("macro_f1", "mean"), s=("seconds", "mean"))
    log("\n" + summary.round(2).to_string())

    base = table[table.arm.eq("hand-8")].set_index("fold")["bacc"]
    if not BANDA_HAND8[0] <= base.mean() <= BANDA_HAND8[1]:
        log(f"\n!! hand-8 em {base.mean():.2f}, fora da banda {BANDA_HAND8} -- "
            f"conferir o caminho antes de reportar os deltas")
    else:
        log(f"\nhand-8 em {base.mean():.2f}: dentro da banda, caminho validado")

    log("\ndelta contra hand-8, pareado por fold:")
    for name in ["hand-8 + WL", "hand-8 + WL embaralhado"]:
        other = table[table.arm.eq(name)].set_index("fold")["bacc"]
        delta = other - base
        p = stats.ttest_rel(other, base).pvalue if len(delta) > 1 else np.nan
        log(f"  {name:26s} {delta.mean():+6.2f} +- {delta.std():4.2f}  "
            f"p={p:.4f}  {int((delta > 0).sum())}/{len(delta)}")

    real = table[table.arm.eq("hand-8 + WL")].set_index("fold")["bacc"]
    fake = table[table.arm.eq("hand-8 + WL embaralhado")].set_index("fold")["bacc"]
    gap = real - fake
    p = stats.ttest_rel(real, fake).pvalue if len(gap) > 1 else np.nan
    log(f"\nWL real contra o PLACEBO (o teste que decide): "
        f"{gap.mean():+.2f} +- {gap.std():.2f}  p={p:.4f}  {int((gap > 0).sum())}/{len(gap)}")

    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
