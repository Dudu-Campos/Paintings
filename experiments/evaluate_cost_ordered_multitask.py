"""Acurácia contra custo, com as representações adicionadas do MAIS BARATO ao mais caro.

A figura anterior (`cost_vs_accuracy_multitask.png`) adiciona em ordem de
**Shapley**, o que faz a curva COMEÇAR pelo bloco mais caro (Felzenszwalb, 3,2 h
sozinho) e nunca explorar a ponta barata do eixo. O leitor não consegue
responder "e se eu tiver 1 CPU-hora?" -- que é a pergunta que ele tem.

Aqui a ordem é o custo. A narrativa que sai é a útil: **dá para ter um modelo
mais barato e um pouco pior**, e o leitor escolhe onde parar.

Não usa Shapley, nem fatorial, nem sonda LDA. É uma trajetória gulosa por custo,
e é isso que a legenda deve dizer -- não é a fronteira de Pareto (essa exigiria
os 255 subconjuntos sob o MLP, ~16 h).

Custos: relógio de parede com 14 workers, medidos nos logs da extração de
21-22/ago sobre as 99.816 imagens (`plot_cost_vs_accuracy_multitask.py`).
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
COST_MINUTES = {
    "Hilbert": 3.2, "Gabor": 3.6, "Fourier": 7.0, "Descriptors": 48.2,
    "SLIC": 81.6, "Line graph": 95.0, "Felzenszwalb": 193.25, "L(G_felz)": 194.5,
}
SHORT = {"Felzenszwalb": "Felz", "Descriptors": "Desc", "SLIC": "SLIC",
         "L(G_felz)": "L(G_felz)", "Line graph": "L(G_slic)", "Hilbert": "Hilbert",
         "Fourier": "Fourier", "Gabor": "Gabor"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path, default=ROOT / "results/cost_ordered_multitask.csv")
    args = ap.parse_args()

    order = sorted(HAND8, key=lambda n: COST_MINUTES[n])
    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")
    log("ordem por custo: " + " -> ".join(f"{SHORT[n]} ({COST_MINUTES[n]:.1f}min)" for n in order))

    done = set()
    if args.output.exists():
        done = {(int(r.step), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for step in range(1, len(order) + 1):
        names = order[:step]
        cost = sum(COST_MINUTES[n] for n in names) / 60
        blocks = [cohort.blocks[n] for n in names]
        for number, (train_idx, test_idx) in enumerate(folds, 1):
            if (step, number) in done:
                continue
            started = time.perf_counter()
            metrics = base.fit(blocks, labels, train_idx, test_idx, n_classes, SEED,
                               args.epochs, args.patience, "narrow", None)
            row = {"step": step, "added": SHORT[order[step - 1]],
                   "modalities": "+".join(names), "cost_hours": cost,
                   "fold": number, "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"passo {step} (+{SHORT[order[step-1]]:9s}) {cost:5.2f}h fold {number} | "
                f"{row['dimensions']:4d}d | BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    agg = frame.groupby(["step", "added", "cost_hours"]).agg(
        bacc=("bacc", "mean"), sd=("bacc", "std"), dims=("dimensions", "first"),
        n=("fold", "size")).reset_index().sort_values("step")
    if len(agg) and agg.n.min() == N_FOLDS:
        top = agg.bacc.max()
        topcost = agg.cost_hours.max()
        print(f"\n{'passo':>5} {'+bloco':>10} {'CPU-h':>7} {'dims':>5} {'BAcc':>6} "
              f"{'% teto':>7} {'% custo':>8}")
        for _, r in agg.iterrows():
            print(f"{int(r.step):5d} {r.added:>10} {r.cost_hours:7.2f} {int(r.dims):5d} "
                  f"{r.bacc:6.2f} {100*r.bacc/top:6.1f}% {100*r.cost_hours/topcost:7.1f}%")


if __name__ == "__main__":
    main()
