"""Braço `logit-add` da cadeia cumulativa: o grafo entra SEM dimensao.

Complementa `evaluate_cumulative_twelve.py`. Naquele, o grafo de similaridade
entra CONCATENADO (282d em 125 classes) e custa -1,14pp no `BranchMLP`. Mas a
codificacao de producao para este corpus e o **`logit-add`** (§5.6): com 125
classes o bloco concatenado nasce quase todo constante -- k=30 vizinhos dao 0,24
por classe, entao >=76% da matriz de log-lift e o valor de "sem evidencia" -- e o
canal aditivo, que soma `lambda * log-lift` aos logits e custa ZERO dimensoes de
entrada, e o que paga. Publicar a cadeia so com o concatenado faria a figura
contradizer a configuracao declarada do artigo.

Este braco roda apenas de k=8 (onde o grafo entra) ate k=12; para k<=7 ele e
identico ao `BranchMLP` ja medido, porque o grafo ainda nao esta no conjunto --
a figura reusa aqueles valores em vez de gastar GPU repetindo-os.

**So o `BranchMLP` tem este braco.** O `LogitAdd` envolve o `BranchMLP` canonico,
e o `PlainMLP` denso nao tem caminho aditivo; o LDA nao tem logits treinaveis, e
portanto nao tem analogo nenhum. As curvas de LDA e MLP raso ficam com o grafo
concatenado, e esse contraste E o achado do §5.6 -- vale mostrar, nao esconder.

O log-lift e calculado como no `evaluate_logit_add_multitask.py`, com OOF
aninhado por fold sobre o espaco hand-8 padronizado (o MESMO espaco em todo k,
igual ao Shapley: reconstrui-lo por passo custaria 133s x 25 e mudaria a
pergunta de "quanto o canal aditivo vale" para "quanto vale o grafo daquele
subconjunto").
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_cumulative_twelve import ORDER  # noqa: E402
from evaluate_logit_add_multitask import build_oof, train_arm  # noqa: E402
from evaluate_shapley_twelve_views import DORMANT, HAND8, N_FOLDS, SEED  # noqa: E402
from registry import load_blocks, log  # noqa: E402

GRAPH_AT = ORDER.index("Similarity graph") + 1  # k em que o canal liga


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=30, help="vizinhos do log-lift")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_logit_add.csv")
    args = parser.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")

    blocks = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    for name in DORMANT:
        values_path, stems_path = EXTERNAL[name]
        values = np.load(ROOT / values_path)
        order = pd.Index(np.load(ROOT / stems_path, allow_pickle=True)
                         .astype(str)).get_indexer(stems)
        if (order < 0).any():
            raise SystemExit(f"{name}: {int((order < 0).sum())} stems ausentes")
        blocks[name] = values[order].astype(np.float32)

    # Texto vem do cache do Shapley (OOF aninhado, ~9 min por fold ja pagos).
    cache_dir = ROOT / "outputs/shapley_twelve_views"
    # Espaco do kNN: hand-8 concatenado, o mesmo do §5.6.
    space = np.hstack([blocks[n] for n in HAND8]).astype(np.float32)

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(stems, labels))

    done = set()
    if args.output.exists():
        done = {(int(r.k), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    for fold, (outer, test) in enumerate(splits, start=1):
        pending = [k for k in range(GRAPH_AT, len(ORDER) + 1) if (k, fold) not in done]
        if not pending:
            continue
        cache = cache_dir / f"fold_{fold}.npz"
        if not cache.exists():
            raise SystemExit(f"cache ausente: {cache}")
        per_fold = dict(blocks)
        per_fold["Text"] = np.load(cache)["text"].astype(np.float32)

        fit, validation = train_test_split(
            outer, test_size=0.10, stratify=labels[outer], random_state=SEED)
        started = time.perf_counter()
        scaler = StandardScaler().fit(space[outer])
        scaled = scaler.transform(space).astype(np.float32)
        oof, held = build_oof(scaled[outer], labels[outer], scaled[test],
                              n_classes, args.k)
        lift = np.zeros((len(labels), n_classes), dtype=np.float32)
        lift[outer], lift[test] = oof, held
        log(f"  fold {fold}: log-lift em {time.perf_counter() - started:.0f}s")

        for k in pending:
            # O grafo NAO entra na concatenacao: ele e o canal aditivo.
            names = [n for n in ORDER[:k] if n != "Similarity graph"]
            started = time.perf_counter()
            metrics, lam = train_arm([per_fold[n] for n in names], lift, labels,
                                     fit, validation, test, n_classes,
                                     np.zeros(n_classes, dtype=np.int64),
                                     SEED, args.epochs, args.patience)
            row = pd.DataFrame([dict(
                k=k, classifier="BranchMLP+logit-add", fold=fold,
                added=ORDER[k - 1], blocks="|".join(names),
                dimensions=sum(per_fold[n].shape[1] for n in names),
                bacc=metrics["bacc"], lam=float(np.ravel(lam)[0]),
                seconds=round(time.perf_counter() - started, 1))])
            row.to_csv(args.output, mode="a", index=False,
                       header=not args.output.exists(), float_format="%.4f")
            log(f"  fold {fold} k={k:2d} | {row.dimensions.iloc[0]:4d}d "
                f"| BAcc {metrics['bacc']:5.2f} | lambda {row.lam.iloc[0]:.4f} "
                f"| {row.seconds.iloc[0]:.0f}s")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
