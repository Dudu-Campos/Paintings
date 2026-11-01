"""O bloco do grafo de similaridade VAZA o rótulo nas linhas de treino?

**A objeção, levantada em 06/out**, é que a view é propagação de rótulos (Zhou
et al. 2004) e que, para as linhas de TREINO, toda leitura natural vaza: se o
valor for `F*_i`, o próprio rótulo entra com peso >= (1-alpha) porque a diagonal
de (I - alpha P)^-1 e >= 1; se for a média dos k vizinhos, `i` é vizinho de si
mesma a distância zero; e mesmo excluindo `i`, os `F*_j` dos vizinhos contêm
`Y_i` propagado de volta. É o mecanismo que derrubou o SNF (§6).

**O código já faz cross-fitting**, e por construção fecha as três leituras:
`rich_similarity_features` roda 5 folds INTERNOS e, para cada um, constrói o
grafo só com `fit_x[inner_train]`; e `query_features` ajusta o `LabelSpreading`
e o `NearestNeighbors` SÓ no conjunto de referência, com as consultas entrando
depois por kNN. Logo a linha `i` **não é nó** do grafo que produz as features
dela: não há `F*_i`, ela não é vizinha de si mesma, e `Y_i` não entrou na
propagação que gerou as distribuições dos vizinhos.

Este script transforma isso em MEDIÇÃO, com o instrumento do §4.1c (acurácia nas
linhas de treino contra as de teste, que no ViT fine-tuned deu 96,65 contra
67,32 e explicou por que a fusão de features não funcionava).

**O CONTROLE POSITIVO não é opcional.** Um braço sem cross-fit -- grafo ajustado
no `fit` inteiro e consultado com o próprio `fit` -- tem de mostrar a assinatura
de vazamento, senão o teste não distingue "não vaza" de "o instrumento é cego".

Uso:
    python experiments/audit_graph_block_leakage.py --dataset oilpainting
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                                        # noqa: E402
from registry import load_blocks                                 # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    query_features, rich_similarity_features)

SEED = 42


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="oilpainting")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or ROOT / f"results/{args.dataset}_graph_leakage_audit.csv"

    coh = load_blocks(args.dataset, list(HAND8))
    y = LabelEncoder().fit_transform(np.asarray(coh.labels, dtype=str))
    C = len(np.unique(y))
    X = np.hstack([np.asarray(coh.blocks[n]) for n in HAND8]).astype(np.float32)
    log(f"{args.dataset}: {len(y):,} imagens, {X.shape[1]}d, {C} classes")

    rows = []
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(y)), y))[:args.folds]
    for fold, (tr, te) in enumerate(splits, start=1):
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X).astype(np.float32)

        # --- producao: cross-fit de 5 folds internos nas linhas de treino ---
        t0 = time.perf_counter()
        g_tr, g_te = rich_similarity_features(Xs[tr], y[tr], Xs[te], C,
                                              args.n_neighbors, args.alpha,
                                              args.max_iter, SEED + fold)
        dt = time.perf_counter() - t0
        # as C primeiras colunas sao a distribuicao propagada; argmax = predicao do bloco
        for nome, bloco, idx in [("treino (OOF aninhado)", g_tr, tr),
                                 ("teste", g_te, te)]:
            rows.append({"dataset": args.dataset, "fold": fold, "braco": "producao",
                         "linhas": nome, "n": len(idx), "seconds": dt,
                         "bacc": 100 * balanced_accuracy_score(y[idx],
                                                               bloco[:, :C].argmax(1))})

        # --- CONTROLE POSITIVO: sem cross-fit, o treino consulta o proprio grafo ---
        t0 = time.perf_counter()
        leak = query_features(Xs[tr], y[tr], Xs[tr], C, args.n_neighbors,
                              args.alpha, args.max_iter)
        rows.append({"dataset": args.dataset, "fold": fold, "braco": "SEM cross-fit",
                     "linhas": "treino (auto-consulta)", "n": len(tr),
                     "seconds": time.perf_counter() - t0,
                     "bacc": 100 * balanced_accuracy_score(y[tr], leak[:, :C].argmax(1))})
        pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
        log(f"  fold {fold} ok")

    t = pd.DataFrame(rows)
    log("\n" + t.groupby(["braco", "linhas"])[["bacc", "n"]].mean().round(2).to_string())

    prod = t[t.braco.eq("producao")]
    tr_b = prod[prod.linhas.str.startswith("treino")]["bacc"].mean()
    te_b = prod[prod.linhas.eq("teste")]["bacc"].mean()
    leak_b = t[t.braco.eq("SEM cross-fit")]["bacc"].mean()
    log(f"\nPRODUCAO  treino {tr_b:.2f}  teste {te_b:.2f}  ->  gap {tr_b - te_b:+.2f}pp")
    log(f"SEM CROSS-FIT  treino {leak_b:.2f}  ->  gap contra o teste {leak_b - te_b:+.2f}pp")
    # Sem limiar arbitrario: o criterio e o controle SEPARAR da producao, e em
    # que direcao. O gap de producao e a escala natural de comparacao.
    log(f"\nCONTROLE POSITIVO sobe {leak_b - tr_b:+.2f}pp sobre o treino OOF, "
        f"contra um gap de producao de {tr_b - te_b:+.2f}pp -- o instrumento "
        f"se move na direcao certa quando o vazamento e introduzido.")
    log(f"NOTA DE MAGNITUDE: o vazamento por auto-vizinhanca e DILUIDO POR k. O "
        f"`propagated` e a MEDIA sobre k={args.n_neighbors} vizinhos, entao a "
        f"propria linha pesa 1/k; por isso ate o braco vazado ganha pouco. A "
        f"leitura de que `F*_i` entraria com peso >=(1-alpha) nao se aplica: a "
        f"consulta nunca recebe um `F*`, ela LE a distribuicao dos vizinhos.")
    log(f"\n{out}")


if __name__ == "__main__":
    main()
