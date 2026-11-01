"""Pesos por bloco na métrica do grafo: o peso igual estava custando?

O kNN do grafo roda sobre blocos concatenados e padronizados. A distância
euclidiana ao quadrado se decompõe exatamente por bloco,

    d²(q,c) = sum_m d²_m(q,c),

então "peso igual por dimensão" significa que cada bloco entra com peso
proporcional ao **número de dimensões** dele -- as 130 de Descriptors valem 4x
as 32 de Gabor por contagem, não por mérito. A versão com pesos é

    d²_w(q,c) = sum_m w_m · d²_m(q,c),      w_m >= 0

e a pergunta é se algum `w` bate o implícito (w_m = 1 para todo m).

Medido antes de rodar isto (25/ago, Multitask 25cl, k=30): a concatenação com
peso igual dá pureza 22,71% contra 18,52% da melhor modalidade isolada
(Felzenszwalb) -- ou seja, mesmo crua ela já é melhor que qualquer distância
única. E as vizinhanças são quase disjuntas: Jaccard médio 0,0132.

**Truque que torna a busca barata.** Como d²_w é combinação linear das d²_m, dá
para pré-computar as distâncias por bloco uma vez sobre um **pool de candidatos**
(a união dos kNN de cada modalidade) e depois reordenar sob qualquer `w` com uma
soma ponderada -- milissegundos por avaliação, em vez de um kNN completo. O pool
generoso existe para que o ótimo não esteja fora dele; a verificação final é um
kNN **completo** sob os pesos aprendidos, sem pool.

Objetivo otimizado: pureza@k na validação. Não é a métrica final do pipeline,
mas é o que o grafo consome -- e é diferenciável do resultado, então serve de
teste da hipótese sem confundir com o classificador.
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
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def block_distances(spaces, queries, candidates, chunk=128):
    """(n_queries, n_candidates, n_blocks) de distâncias ao quadrado por bloco."""
    n_q, n_c = len(queries), candidates.shape[1]
    out = np.zeros((n_q, n_c, len(spaces)), dtype=np.float32)
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        rows = queries[start:stop]
        picked = candidates[start:stop]
        for b, space in enumerate(spaces):
            q = space[rows][:, None, :]
            c = space[picked]
            out[start:stop, :, b] = np.square(q - c).sum(-1)
    return out


def purity_under(weights, distances, candidate_labels, truth, k):
    combined = distances @ weights
    top = np.argpartition(combined, k, axis=1)[:, :k]
    hit = np.take_along_axis(candidate_labels, top, axis=1) == truth[:, None]
    return float(hit.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--extra-blocks", nargs="*", default=["Gabor"])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--pool-k", type=int, default=80,
                        help="vizinhos por modalidade que entram no pool de candidatos")
    parser.add_argument("--n-val", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/metric_block_weights.csv")
    args = parser.parse_args()

    names = VISUAL + list(args.extra_blocks)
    blocks, raw_labels, _ = load_corpus(args.dataset, names)
    if args.classes_file and args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))

    train, test = next(iter(StratifiedKFold(5, shuffle=True, random_state=SEED)
                            .split(np.zeros(len(labels)), labels)))
    fit, validation = train_test_split(train, test_size=0.15,
                                       stratify=labels[train], random_state=SEED)
    rng = np.random.default_rng(SEED)
    if len(validation) > args.n_val:
        validation = rng.choice(validation, args.n_val, replace=False)
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes | "
        f"fit={len(fit):,} val={len(validation):,} test={len(test):,}")

    spaces = [StandardScaler().fit(blocks[n][fit]).transform(blocks[n]).astype(np.float32)
              for n in names]

    started = time.perf_counter()
    pool = []
    for space in spaces:
        finder = NearestNeighbors(n_neighbors=args.pool_k, algorithm="brute").fit(space[fit])
        _, local = finder.kneighbors(space[validation])
        pool.append(fit[local])
    concat = np.hstack(spaces)
    finder = NearestNeighbors(n_neighbors=args.pool_k, algorithm="brute").fit(concat[fit])
    _, local = finder.kneighbors(concat[validation])
    pool.append(fit[local])
    candidates = np.concatenate(pool, axis=1)
    # dedup por linha mantendo largura fixa: repetidos viram o mesmo candidato,
    # o que não muda o top-k (a distância é idêntica).
    log(f"pool: {candidates.shape[1]} candidatos/consulta em "
        f"{time.perf_counter() - started:.0f}s")

    started = time.perf_counter()
    distances = block_distances(spaces, validation, candidates)
    candidate_labels = labels[candidates]
    truth = labels[validation]
    log(f"distâncias por bloco: {distances.shape} em {time.perf_counter() - started:.0f}s")

    equal = np.ones(len(names), dtype=np.float32)
    base = purity_under(equal, distances, candidate_labels, truth, args.k)
    log(f"peso igual (o atual): pureza@{args.k} na validação = {100 * base:.2f}")

    def objective(log_weights):
        return -purity_under(np.exp(log_weights).astype(np.float32),
                             distances, candidate_labels, truth, args.k)

    best_value, best_weights = base, equal
    for attempt in range(args.restarts):
        start = (np.zeros(len(names)) if attempt == 0
                 else rng.normal(0, 0.7, len(names)))
        result = minimize(objective, start, method="Nelder-Mead",
                          options={"maxiter": 1200, "xatol": 1e-3, "fatol": 1e-5})
        value = -result.fun
        if value > best_value:
            best_value, best_weights = value, np.exp(result.x).astype(np.float32)
        log(f"  restart {attempt + 1}: pureza {100 * value:.2f} "
            f"(melhor {100 * best_value:.2f})")

    best_weights = best_weights / best_weights.mean()
    log(f"pesos aprendidos (normalizados pela média): " +
        ", ".join(f"{n}={w:.2f}" for n, w in zip(names, best_weights)))

    # Verificação honesta: kNN COMPLETO no teste, sem pool, nos dois pesos.
    print()
    rows = []
    for label, weights in (("peso igual", equal), ("pesos aprendidos", best_weights)):
        scaled = np.hstack([s * np.sqrt(w) for s, w in zip(spaces, weights)])
        finder = NearestNeighbors(n_neighbors=args.k, algorithm="brute").fit(scaled[train])
        _, local = finder.kneighbors(scaled[test])
        hit = labels[train[local]] == labels[test][:, None]
        rows.append({"weights": label, "purity_k": 100 * hit.mean(),
                     "purity_1": 100 * hit[:, 0].mean(),
                     "oracle_5": 100 * hit[:, :5].any(1).mean(),
                     "oracle_k": 100 * hit.any(1).mean(),
                     **{f"w_{n}": float(w) for n, w in zip(names, weights)}})
        log(f"TESTE ({label}): pureza@1 {100 * hit[:, 0].mean():.2f}  "
            f"pureza@{args.k} {100 * hit.mean():.2f}  "
            f"oráculo@5 {100 * hit[:, :5].any(1).mean():.2f}")
        del scaled
    detail = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, float_format="%.4f")
    delta = detail.purity_k.iloc[1] - detail.purity_k.iloc[0]
    print()
    print(f"Δ pureza@{args.k} no teste: {delta:+.2f}pp")
    print("Leitura: se ~0, o peso igual já estava perto do ótimo e a métrica NÃO era")
    print("a perda -- o ativo do multiaresta seria a diversidade (Jaccard 0,013), não")
    print("a correção de pesos. Se >0, a métrica era a perda e o multiaresta herda o")
    print("mesmo ganho de forma mais cara.")


if __name__ == "__main__":
    main()
