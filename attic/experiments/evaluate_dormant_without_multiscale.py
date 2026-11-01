"""Reavaliação das representações dormentes sobre uma base SEM quadrantes.

O achado 15 mediu as sete dormentes contra uma base que carregava os
quadrantes de Descriptors (520d de 1.154d, 45% do vetor). No MLP as quatro
derivadas de Hilbert custaram −2,39pp, enquanto no LDA as cinco somavam
+1,43pp -- o descolamento clássico de "consome capacidade".

Em 24/ago medimos o mesmo mecanismo de forma direta no OilPainting: o grafo de
similaridade dá exatamente 0,00pp com Multiscale na base e **+1,21pp sem**,
com o grafo idêntico nas duas. Se a explicação for capacidade e não conteúdo,
o sinal das dormentes deve mudar sobre uma base enxuta.

Grupos separados de propósito, porque a evidência prévia difere entre eles:

* **Hilbert-derivadas** (catch22, spectrum1d, visibility, wavelet): melhores
  candidatas. O descolamento LDA×MLP é a assinatura de capacidade.
* **Caminhadas** (random, hybrid): controle negativo esperado. O problema
  delas é redundância medida com Hilbert (r² mediano 0,918, p90 0,984), que
  independe da base.
* **Square graph**: sem evidência forte em nenhuma direção.

Se as caminhadas melhorarem tanto quanto as Hilbert-derivadas, o efeito é da
base e não das representações -- o que enfraqueceria a interpretação.
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
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from registry import load_blocks  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402

SEED, N_FOLDS = 42, 5
BASE = ["Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
        "Hilbert", "Fourier"]
GROUPS = {
    "Hilbert-derivadas": ["Hilbert catch22", "Hilbert spectrum1d",
                          "Hilbert visibility", "Hilbert wavelet"],
    "caminhadas": ["Random walk", "Hybrid walk"],
    "square graph": ["Square graph"],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--with-multiscale", action="store_true",
                        help="repete a medição com Multiscale na base, que é a "
                             "condição do achado 15 -- serve de controle")
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"])
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/dormant_without_multiscale.csv")
    args = parser.parse_args()

    base_names = BASE + (["Multiscale"] if args.with_multiscale else [])
    wanted = base_names + [n for group in GROUPS.values() for n in group]
    cohort = load_blocks(args.dataset, wanted, allow_external=False)

    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    base_blocks = [cohort.blocks[n] for n in base_names if n in cohort.blocks]
    base_dims = sum(b.shape[1] for b in base_blocks)
    log(f"base: {[n for n in base_names if n in cohort.blocks]} = {base_dims}d")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.classifier), str(r.condition), int(r.fold), bool(r.with_multiscale))
                for r in previous.itertuples()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    conditions = {"base": []}
    for tag, members in GROUPS.items():
        present = [n for n in members if n in cohort.blocks]
        if present:
            conditions[f"+ {tag}"] = present
    conditions["+ todas"] = [n for group in GROUPS.values() for n in group
                             if n in cohort.blocks]

    for classifier in args.classifiers:
        for tag, extra in conditions.items():
            for fold, (train, test) in enumerate(splits[:args.folds], start=1):
                key = (classifier, tag, fold, args.with_multiscale)
                if key in done:
                    continue
                blocks = base_blocks + [cohort.blocks[n] for n in extra]
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, train, test)
                else:
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                pd.DataFrame([{
                    "classifier": classifier, "condition": tag, "fold": fold,
                    "with_multiscale": args.with_multiscale,
                    "dimensions": sum(b.shape[1] for b in blocks),
                    **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}}]).to_csv(
                    args.output, mode="a", index=False,
                    header=not args.output.exists(), float_format="%.6f")
            log(f"{classifier} {tag}: pronto")

    detail = pd.read_csv(args.output)
    detail = detail[detail.with_multiscale.eq(args.with_multiscale)]
    print()
    for classifier in args.classifiers:
        block = detail[detail.classifier.eq(classifier)]
        pivot = block.pivot_table(index="fold", columns="condition", values="bacc")
        if "base" not in pivot:
            continue
        print(f"=== {classifier.upper()} "
              f"(base {'COM' if args.with_multiscale else 'SEM'} Multiscale) ===")
        for tag in conditions:
            if tag not in pivot:
                continue
            values = pivot[tag].to_numpy()
            line = f"  {tag:<22} {100 * values.mean():5.2f} ± {100 * values.std(ddof=1):4.2f}"
            if tag != "base":
                delta = values - pivot["base"].to_numpy()
                _, p = ttest_rel(values, pivot["base"].to_numpy())
                line += (f"   Δ {100 * delta.mean():+5.2f} ± {100 * delta.std(ddof=1):4.2f}pp"
                         f"  p={p:.4f}  {int((delta > 0).sum())}/{len(delta)}")
            print(line)
        print()


if __name__ == "__main__":
    main()
