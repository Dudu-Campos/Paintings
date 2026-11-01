"""§5.6 corrigido: o grafo perde com muitas classes por EVIDENCIA VAZIA, nao por
diluicao dimensional -- e o conserto e `k`, nao dimensao.

O texto antigo do §5.6 dizia que o bloco `2*n_classes+32` diluia o vetor (44% em
125 classes). Isso contradiz o codigo: o `BranchMLP` mapeia TODO bloco para 128
dims antes da cabeca (`nn.Linear(size, 128)` por bloco), entao o grafo ocupa 1/9
da cabeca com 82d ou com 282d. A fracao do vetor cru nao e o que a cabeca ve.

O que muda de verdade e vizinhos por classe:

    k = 30      25 classes -> 1,20 por classe
                125 classes -> 0,24
                365 classes -> 0,08

Com 125 classes a maioria recebe ZERO vizinhos e o log-lift vira o valor de "sem
evidencia" -- o bloco fica quase constante. A previsao e que k ~ 150 restaure em
125 classes o mesmo 1,2 vizinho por classe que funciona em 25.

E barato: o custo do grafo e a BUSCA, O(n^2 d) (§5.4b), nao o k. Aumentar k nao
muda a ordem de grandeza.

Numeros de referencia (BranchMLP estreito, 5 folds):
  125 classes: hand-8 38,74 | + grafo k=30  38,24  (-0,50)
   25 classes: hand-8 49,49 | + grafo k=30  50,28  (+0,79, p=0,049)

O braco de 25 classes e o CONTROLE: se a hipotese estiver certa, k deve pagar
muito em 125 e pouco ou nada em 25, que ja opera com evidencia suficiente.

--- RODAR NA MOJITO ---
    hostname                      # confirmar
    nvidia-smi -L
    .venv/bin/python3 experiments/sweep_similarity_graph_k.py \
        --output results/multitask_graph_k_sweep_125.csv

    .venv/bin/python3 experiments/sweep_similarity_graph_k.py \
        --restrict-stems data/multitask_25class_cohort_stems.csv \
        --output results/multitask_graph_k_sweep_25.csv

O §1 avisa: os venvs do NFS nao sao portaveis, use o `.venv` LOCAL da mojito e
NAO rode com o mesmo --output que a absinto. O script e retomavel por (condicao,
fold), entao pode reiniciar sem perder trabalho.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

import evaluate_preprocess_effect_oilpainting as base  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--k", type=int, nargs="+", default=[30, 60, 100, 150, 250])
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--restrict-stems", type=Path, default=None,
                    help="CSV com coluna `stem`; recorta a coorte (ex.: as 25 classes)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multitask_graph_k_sweep.csv")
    args = ap.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    blocks = [cohort.blocks[n] for n in HAND8]
    labels_raw = np.asarray(cohort.labels, dtype=str)

    if args.restrict_stems:
        wanted = set(pd.read_csv(args.restrict_stems, dtype={"stem": str})["stem"].astype(str))
        keep = np.array([s in wanted for s in stems])
        if not keep.any():
            raise ValueError(f"{args.restrict_stems}: nenhum stem casou")
        log(f"recorte: {keep.sum():,} de {len(stems):,} imagens")
        stems, labels_raw = stems[keep], labels_raw[keep]
        blocks = [b[keep] for b in blocks]

    labels = LabelEncoder().fit_transform(labels_raw)
    n_classes = len(np.unique(labels))
    graph_source = np.hstack(blocks)
    log(f"{args.dataset}: {len(stems):,} imagens, {n_classes} classes, "
        f"hand-8 {graph_source.shape[1]}d")
    for k in args.k:
        log(f"  k={k:4d} -> {k / n_classes:5.2f} vizinhos por classe em média")

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    plan = [("hand-8", None)] + [(f"hand-8 + grafo k={k}", k) for k in args.k]

    for condition, k in plan:
        for number, (train_idx, test_idx) in enumerate(folds, 1):
            if (condition, number) in done:
                continue
            if k is not None:
                # `fit` lê GRAPH do módulo em tempo de chamada; trocar aqui evita
                # duplicar 70 linhas e não edita arquivo que outro job importa.
                base.GRAPH = dict(n_neighbors=k, alpha=args.alpha, max_iter=args.max_iter)
            started = time.perf_counter()
            metrics = base.fit(blocks, labels, train_idx, test_idx, n_classes, SEED,
                               args.epochs, args.patience, "narrow",
                               graph_source if k is not None else None)
            row = {"condition": condition, "k": k or 0, "n_classes": n_classes,
                   "fold": number, "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                # §8: append com esquema mudado grava sob o cabeçalho antigo, em
                # silêncio. Reindexar amarra a ordem.
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{condition:24s} fold {number} | {row['dimensions']:5d}d | "
                f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    if "hand-8" in pivot.columns and pivot.notna().all().all():
        from scipy import stats
        b = pivot["hand-8"].to_numpy()
        print(f"\n{'condição':24s} {'BAcc':>6} {'Δ':>7} {'p':>8} {'folds':>7}")
        for c in pivot.columns:
            if c == "hand-8":
                print(f"{c:24s} {b.mean():6.2f}")
                continue
            x = pivot[c].to_numpy(); d = x - b
            _, p = stats.ttest_rel(x, b)
            print(f"{c:24s} {x.mean():6.2f} {d.mean():+7.2f} {p:8.4f} {(d > 0).sum():4d}/{len(d)}")


if __name__ == "__main__":
    main()
