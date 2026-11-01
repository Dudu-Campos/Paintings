"""O teste que decide: o ganho do banco é "mais pixels" ou são as representações?

O §9.5 mediu que o pipeline precisa de PELO MENOS UMA família de segmentação em
resolução alta -- as duas a 224 px custam 3,30pp, e preservar granularidade
recupera só 0,23. A objeção que sobra é a mais grave que o artigo enfrenta:
**e se toda a vantagem sobre as congeladas for apenas ver mais pixels?**

Os descritores globais rodam a 128 px (`TARGET`, sem justificativa registrada --
§7.1). Se eles também dispararem com resolução, a resolução é alavanca geral e a
narrativa vira "olhamos mais pixels que a CNN". Se não dispararem, a resolução
ajuda ESPECIFICAMENTE as views de grafo, o que é afirmação sobre as
representações.

Duas leituras, e as duas importam:
* **standalone**: Descriptors@128 vs @512 vs @1024 sozinho.
* **no pipeline**: hand-8 trocando só o bloco Descriptors. O §5.17 mediu que
  ordenamento isolado pode ficar ANTICORRELACIONADO com o do pipeline
  (`(256,32)` do Hilbert: +1,40pp isolado, -0,67pp concatenado), então o número
  que decide é o segundo.

CONFUNDIDOR a declarar junto: LBP (R=1 e R=3) e a distância do GLCM estão em
unidades de pixel, então mudar o alvo muda O QUE é medido, não só o custo. Mesma
pegadinha do `felz_scale` (§9.3); não há como separar sem reparametrizar.
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
ALT = {512: "outputs/multitask_descriptors_r512",
       1024: "outputs/multitask_descriptors_r1024"}


def load_desc(directory: Path, stems: np.ndarray, expected: int) -> np.ndarray:
    values = np.load(directory / "desc_full_features.npy")
    order = pd.Index(np.load(directory / "desc_full_stems.npy",
                             allow_pickle=True).astype(str)).get_indexer(stems)
    if (order < 0).any():
        raise ValueError(f"{directory}: {int((order < 0).sum())} stems ausentes")
    values = values[order].astype(np.float32)
    if values.shape[1] != expected:
        raise ValueError(f"{directory}: {values.shape[1]}d, esperava {expected}")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--targets", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multitask_descriptors_resolution.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    hand = [cohort.blocks[n] for n in HAND8]
    d128 = cohort.blocks["Descriptors"]

    variants = {128: d128}
    for target in args.targets:
        directory = ROOT / ALT[target]
        if not (directory / "desc_full_features.npy").exists():
            log(f"  target={target}: ainda não extraído, pulando")
            continue
        variants[target] = load_desc(directory, stems, d128.shape[1])
        log(f"  Descriptors@{target} carregado ({variants[target].shape[1]}d)")
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")

    conditions = {}
    for target, block in variants.items():
        conditions[f"Descriptors@{target} sozinho"] = [block]
    for target, block in variants.items():
        conditions[f"hand-8 (Descriptors@{target})"] = [
            block if n == "Descriptors" else b for n, b in zip(HAND8, hand)]

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for name, blocks in conditions.items():
        for number, (train_idx, test_idx) in enumerate(folds, 1):
            if (name, number) in done:
                continue
            started = time.perf_counter()
            metrics = base.fit(blocks, labels, train_idx, test_idx, n_classes, SEED,
                               args.epochs, args.patience, "narrow", None)
            row = {"condition": name, "fold": number,
                   "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{name:34s} fold {number} | BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    from scipy import stats
    print()
    for prefix, ref in (("Descriptors@", "Descriptors@128 sozinho"),
                        ("hand-8 (Descriptors@", "hand-8 (Descriptors@128)")):
        if ref not in pivot.columns:
            continue
        b = pivot[ref].to_numpy()
        print(f"{'condição':34s} {'BAcc':>6} {'Δ vs 128':>9} {'p':>8} {'folds':>8}")
        for c in sorted(x for x in pivot.columns if x.startswith(prefix)):
            x = pivot[c].to_numpy()
            if np.isnan(x).any():
                continue
            d = x - b
            if c == ref:
                print(f"{c:34s} {x.mean():6.2f}")
                continue
            _, p = stats.ttest_rel(x, b)
            print(f"{c:34s} {x.mean():6.2f} {d.mean():+9.2f} {p:8.4f} {(d > 0).sum():4d}/5 melhores")
        print()


if __name__ == "__main__":
    main()
