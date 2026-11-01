"""Teto do oráculo: quanto sobra para uma atenção sobre vizinhos aprender?

A proposta em discussão é substituir a agregação fixa do grafo (média sobre os
30 vizinhos, mais histograma ponderado por `exp(-d/escala)`) por atenção
aprendida -- Matching Networks (Vinyals et al., 2016) aplicado ao nosso kNN.

Antes de construir, medir o limite superior. **Qualquer** esquema de atenção
sobre os rótulos dos vizinhos produz um voto ponderado, e o melhor voto possível
põe todo o peso no vizinho certo. Logo:

    teto de qualquer atenção = P(a classe verdadeira estar entre os k vizinhos)

Se o teto estiver perto do que o grafo já entrega, **não há o que atender** e a
ideia morre sem custo de implementação. Se estiver muito acima, sabemos quanto
está em jogo e onde.

Ressalva de escopo: o teto limita atenção sobre os rótulos dos k **recuperados**,
que é o mecanismo proposto. A difusão de rótulos do grafo alcança além dos
vizinhos diretos, então ela pode em princípio passar desse teto -- comparar, não
assumir.

Distinção que o resultado decide, e que estava se misturando na discussão:
atenção **repondera uma recuperação fixa** e portanto está presa a este teto;
metric learning (vetor de diferença entre pares) **muda o espaço**, muda quem é
recuperado, e por isso move o próprio teto. São propostas diferentes.

Protocolo: 5-fold estratificado para os corpora de arte, split oficial para
Places-LT. O espaço é o de produção do grafo -- as 7 modalidades visuais
padronizadas no treino, sem a ResNet (misturar a CNN contaminaria com supervisão
externa).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402

K_GRID = (1, 3, 5, 10, 20, 30)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def bands_for(frequency: np.ndarray, protocol: str) -> np.ndarray:
    """Faixas de frequência de treino.

    Places-LT usa a convenção publicada (Liu et al., CVPR 2019). Nos corpora de
    arte não há convenção, então usam-se terços da distribuição de frequência --
    o objetivo é só saber se a folga se concentra nas classes raras.
    """
    if protocol == "split":
        return np.where(frequency > 100, "many", np.where(frequency >= 20, "medium", "few"))
    cuts = np.percentile(frequency, [33.3, 66.7])
    return np.where(frequency > cuts[1], "frequentes",
                    np.where(frequency > cuts[0], "médias", "raras"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / f"results/neighbour_oracle_{args.dataset}.csv"

    blocks, raw_labels, frame = load_corpus(args.dataset, VISUAL)
    if args.classes_file:
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        got = len(np.unique(raw_labels[mask]))
        if got != len(wanted):
            raise ValueError(f"casaram {got} de {len(wanted)} classes do benchmark")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels, frame = raw_labels[mask], frame[mask].reset_index(drop=True)
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))

    protocol = "split" if "split" in frame.columns and args.dataset == "places_lt" else "kfold"
    if protocol == "split":
        splits = [(np.flatnonzero(frame["split"].eq("train").to_numpy()),
                   np.flatnonzero(frame["split"].eq("test").to_numpy()))]
    else:
        splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels)), labels))
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"protocolo {protocol}, {len(splits)} split(s)")

    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        space = np.hstack([StandardScaler().fit(blocks[n][train]).transform(blocks[n])
                           .astype(np.float32) for n in VISUAL])
        started = time.perf_counter()
        finder = NearestNeighbors(n_neighbors=args.n_neighbors,
                                  algorithm="brute").fit(space[train])
        _, indices = finder.kneighbors(space[test])
        neighbour_labels = labels[train][indices]
        truth = labels[test]
        hit = neighbour_labels == truth[:, None]
        frequency = np.bincount(labels[train], minlength=n_classes)
        band_of_class = bands_for(frequency, protocol)
        query_band = band_of_class[truth]
        del space

        for k in [k for k in K_GRID if k <= args.n_neighbors]:
            present = hit[:, :k].any(axis=1)
            purity = hit[:, :k].mean(axis=1)
            counts = np.zeros((len(truth), n_classes), dtype=np.int16)
            np.add.at(counts, (np.arange(len(truth))[:, None], neighbour_labels[:, :k]), 1)
            vote = counts.argmax(axis=1) == truth
            for band in list(dict.fromkeys(band_of_class)) + ["total"]:
                mask = np.ones(len(truth), bool) if band == "total" else (query_band == band)
                if not mask.any():
                    continue
                rows.append({"dataset": args.dataset, "fold": fold, "k": k, "band": band,
                             "n_queries": int(mask.sum()),
                             "oracle": 100 * present[mask].mean(),
                             "purity": 100 * purity[mask].mean(),
                             "vote": 100 * vote[mask].mean()})
        log(f"  fold {fold}: kNN em {time.perf_counter() - started:.0f}s, "
            f"oráculo@{args.n_neighbors} = "
            f"{100 * hit.any(axis=1).mean():.2f}")

    detail = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False, float_format="%.4f")

    print()
    summary = detail.groupby(["k", "band"])[["oracle", "purity", "vote"]].mean().reset_index()
    print(f"{'k':>3} {'faixa':<12} {'oráculo':>8} {'pureza':>8} {'voto':>7}")
    for row in summary.itertuples():
        print(f"{row.k:>3} {row.band:<12} {row.oracle:>8.2f} {row.purity:>8.2f} {row.vote:>7.2f}")
    print()
    top = summary[summary.k.eq(args.n_neighbors)]
    print(f"=== teto a k={args.n_neighbors} ===")
    for row in top.itertuples():
        print(f"  {row.band:<12} oráculo {row.oracle:>6.2f}  "
              f"(pureza {row.purity:.2f}%, voto puro {row.vote:.2f})")
    print()
    print("Comparar o oráculo total com o BAcc do MLP com grafo: a diferença é o")
    print("máximo que qualquer atenção sobre rótulos de vizinhos poderia somar.")


if __name__ == "__main__":
    main()
