"""Fronteira de Pareto custo x acurácia, com o MLP de produção.

A figura `cost_vs_accuracy_multitask.png` adiciona representações em ordem de
**Shapley**, o que faz a curva COMEÇAR pelo bloco mais caro (Felzenszwalb, 3,2 h)
e nunca explorar a ponta barata do eixo. O leitor não consegue responder "e se eu
tiver 1 CPU-hora?".

Aqui o eixo é o orçamento: para cada custo, o melhor conjunto possível. As 511
máscaras do fatorial (`factorial_shapley_multitask_25class.csv`) dão a fronteira
sob **sonda LDA** -- 23 pontos, de 0,1 h a 10,4 h. Este script reavalia esses 23
com o **BranchMLP de produção** (§7.8).

Ressalva que a legenda tem de carregar: o conjunto candidato foi pré-selecionado
por LDA, e o §8 registra seis casos de "LDA extrai, MLP destrói" em vetores
multi-bloco. Logo isto é um **limite inferior** da fronteira sob MLP, não a
fronteira exata. Medir as 255 sob MLP custaria ~14 h.

Custos: `COST_MINUTES` de `plot_cost_vs_accuracy_multitask.py` -- relógio de
parede com 14 workers, medido nos logs da extração de 21-22/ago.
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
    "Felzenszwalb": 193.25, "Descriptors": 48.2, "SLIC": 81.6,
    "L(G_felz)": 194.5, "Line graph": 95.0,
    "Hilbert": 3.2, "Fourier": 7.0, "Gabor": 3.6,
}


def lda_frontier(path: Path) -> list[frozenset]:
    frame = pd.read_csv(path)
    frame["mods"] = frame.modalities.fillna("").apply(
        lambda s: frozenset(x for x in s.split("+") if x))
    grouped = frame.groupby("mods").bacc.mean()
    rows = [(sum(COST_MINUTES[x] for x in k) / 60, float(v), k)
            for k, v in grouped.items() if k and "Multiscale" not in k]
    rows.sort()
    best, frontier = -1.0, []
    for cost, bacc, mods in rows:
        if bacc > best:
            best = bacc
            frontier.append(mods)
    return frontier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factorial", type=Path,
                    default=ROOT / "results/factorial_shapley_multitask_25class.csv")
    ap.add_argument("--restrict-stems", type=Path,
                    default=ROOT / "data/multitask_25class_cohort_stems.csv",
                    help="o fatorial é de 25 classes; a coorte tem de casar")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path, default=ROOT / "results/cost_pareto_multitask.csv")
    args = ap.parse_args()

    frontier = lda_frontier(args.factorial)
    log(f"{len(frontier)} pontos na fronteira sob LDA")

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_raw = np.asarray(cohort.labels, dtype=str)
    blocks = {n: cohort.blocks[n] for n in HAND8}
    if args.restrict_stems and args.restrict_stems.exists():
        wanted = set(pd.read_csv(args.restrict_stems, dtype={"stem": str})["stem"].astype(str))
        keep = np.array([s in wanted for s in stems])
        log(f"recorte de 25 classes: {keep.sum():,} de {len(stems):,}")
        stems, labels_raw = stems[keep], labels_raw[keep]
        blocks = {n: b[keep] for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(labels_raw)
    n_classes = len(np.unique(labels))
    log(f"{len(stems):,} imagens, {n_classes} classes")

    done = set()
    if args.output.exists():
        done = {(str(r.mods), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for mods in frontier:
        name = "+".join(n for n in HAND8 if n in mods)
        cost = sum(COST_MINUTES[x] for x in mods) / 60
        chosen = [blocks[n] for n in HAND8 if n in mods]
        for number, (train_idx, test_idx) in enumerate(folds, 1):
            if (name, number) in done:
                continue
            started = time.perf_counter()
            metrics = base.fit(chosen, labels, train_idx, test_idx, n_classes, SEED,
                               args.epochs, args.patience, "narrow", None)
            row = {"mods": name, "n_blocks": len(mods), "cost_hours": cost,
                   "fold": number, "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{cost:5.1f}h {len(mods)}b fold {number} | BAcc {metrics['bacc']:5.2f} | "
                f"{row['seconds']:.0f}s | {name}")

    frame = pd.read_csv(args.output)
    agg = frame.groupby(["mods", "cost_hours", "n_blocks"]).agg(
        bacc=("bacc", "mean"), sd=("bacc", "std"), n=("fold", "size")).reset_index()
    agg = agg[agg.n == N_FOLDS].sort_values("cost_hours")
    if len(agg):
        top, topcost = agg.bacc.max(), agg.loc[agg.bacc.idxmax(), "cost_hours"]
        print(f"\nteto {top:.2f} BAcc a {topcost:.1f} h\n")
        print(f"{'h':>5} {'BAcc':>6} {'% teto':>7} {'% custo':>8} {'n':>2}  blocos")
        best = -1.0
        for _, r in agg.iterrows():
            mark = " *" if r.bacc > best else "  "
            best = max(best, r.bacc)
            print(f"{r.cost_hours:5.1f} {r.bacc:6.2f} {100*r.bacc/top:6.1f}% "
                  f"{100*r.cost_hours/topcost:7.1f}% {int(r.n_blocks):2d}{mark} {r.mods}")


if __name__ == "__main__":
    main()
