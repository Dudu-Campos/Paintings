"""Escalonamento do grafo de similaridade contra o tamanho do corpus.

O grafo é a **única** peça do pipeline cujo custo não é por imagem: kNN e
difusão de rótulos acoplam todos os pontos. Este benchmark mede o expoente
empírico beta em t(n) ~ n^beta separando as três fases, porque elas têm
complexidades analíticas diferentes e só uma domina:

  - `LabelSpreading.fit`  : constrói o grafo kNN do treino e itera a difusão.
  - `kneighbors(query)`   : busca dos vizinhos das linhas de consulta.
  - agregação             : as 2*C+32 estatísticas, O(n*k*C), linear por
                            construção.

Em d=639 nenhuma estrutura de indexação ajuda (a maldição da dimensionalidade
derruba kd-tree/ball-tree para varredura), então a expectativa analítica é
beta -> 2 conforme n cresce. Medir importa porque o regime em que rodamos
(n ~ 5e4) pode estar longe da assíntota.
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.registry import load_blocks  # noqa: E402

VISUAL = ["Descriptors", "Felzenszwalb", "SLIC", "L(G_felz)", "Line graph",
          "Hilbert", "Fourier", "Gabor"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[2000, 4000, 8000, 16000, 32000, 53000])
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=1,
                        help="1 mede UMA construção de grafo; a produção paga "
                             "6 (5 folds internos de OOF + o ajuste final), e "
                             "medir 1 subestimou o custo absoluto em 12,6x")
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--keep-classes", type=int, default=25,
                        help="mantém as C classes mais frequentes; a produção "
                             "roda o grafo no recorte de 25")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/graph_scaling.csv")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, VISUAL, allow_external=False,
                         min_per_class=args.min_per_class)
    features = np.hstack([cohort.blocks[name] for name in VISUAL])
    labels = pd.Categorical(cohort.labels).codes.astype(np.int64)
    # O menor n da grade dividido pelo número de classes tem de deixar pelo
    # menos 2 linhas por classe nos dois lados do split, senão o stratify
    # recusa -- e o recorte de 25 é o que a produção usa de qualquer forma.
    frequent = pd.Series(labels).value_counts().index[:args.keep_classes]
    keep = np.isin(labels, np.asarray(frequent))
    features, labels = features[keep], labels[keep]
    labels = pd.Categorical(labels).codes.astype(np.int64)
    n_classes = int(labels.max()) + 1
    log(f"coorte {features.shape}, {n_classes} classes")

    rows = []
    for size in args.sizes:
        if size > len(features):
            log(f"n={size} maior que a coorte, pulado")
            continue
        for repeat in range(args.repeats):
            take, _ = train_test_split(
                np.arange(len(features)), train_size=size,
                random_state=1000 + repeat, stratify=labels)
            # Estratificar a divisão treino/consulta mantém todas as classes
            # presentes no treino: LabelSpreading recusa rótulo ausente.
            train_index, query_index = train_test_split(
                np.arange(size), test_size=args.query_fraction,
                random_state=repeat, stratify=labels[take])
            scaler = StandardScaler().fit(features[take][train_index])
            train_x = scaler.transform(features[take][train_index])
            query_x = scaler.transform(features[take][query_index])
            train_y = labels[take][train_index]

            started = time.perf_counter()
            model = LabelSpreading(kernel="knn", n_neighbors=args.n_neighbors,
                                   alpha=args.alpha, max_iter=args.max_iter)
            model.fit(train_x, train_y)
            fit_s = time.perf_counter() - started

            started = time.perf_counter()
            neighbors = NearestNeighbors(n_neighbors=args.n_neighbors).fit(train_x)
            distances, indices = neighbors.kneighbors(query_x)
            query_s = time.perf_counter() - started

            started = time.perf_counter()
            propagated = model.label_distributions_[indices].mean(axis=1).astype(np.float32)
            similarity = np.exp(-distances / max(float(np.median(distances)), 1e-6))
            one_hot = np.eye(n_classes, dtype=np.float32)[train_y[indices]]
            weighted = (one_hot * similarity[:, :, None].astype(np.float32)).sum(axis=1)
            weighted /= similarity.sum(axis=1)[:, None].clip(min=1e-8)
            aggregate_s = time.perf_counter() - started
            del propagated, weighted

            rows.append({"n": size, "repeat": repeat,
                         "n_train": len(train_index), "n_query": len(query_index),
                         "dimensions": features.shape[1], "n_classes": n_classes,
                         "fit_seconds": fit_s, "query_seconds": query_s,
                         "aggregate_seconds": aggregate_s,
                         "total_seconds": fit_s + query_s + aggregate_s})
            log(f"n={size:>6} rep={repeat}  fit={fit_s:7.2f}s  "
                f"query={query_s:6.2f}s  agg={aggregate_s:5.2f}s  "
                f"total={fit_s + query_s + aggregate_s:7.2f}s")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)

    frame = pd.DataFrame(rows).groupby("n")[
        ["fit_seconds", "query_seconds", "aggregate_seconds", "total_seconds"]].mean()
    log("\n" + frame.round(2).to_string())
    logn = np.log(frame.index.values.astype(float))
    log("\nexpoentes beta (ajuste log-log, t ~ n^beta):")
    for column in frame.columns:
        beta = np.polyfit(logn, np.log(frame[column].values), 1)[0]
        # A assíntota importa mais que o ajuste global: a maldição da
        # dimensionalidade só morde depois que a varredura domina o overhead.
        tail = np.polyfit(logn[-3:], np.log(frame[column].values[-3:]), 1)[0]
        log(f"  {column:<20} global {beta:5.2f}   últimos 3 pontos {tail:5.2f}")


if __name__ == "__main__":
    main()
