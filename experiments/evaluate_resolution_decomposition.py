"""Decompoe o -3,53pp do pareamento em 224 px: pixels ou granularidade de grafo?

`evaluate_resolution_pairing_multitask.py` mediu hand-8 caindo de 38,74 para
35,21 ao trocar os quatro blocos de segmentacao pela versao de 224 px. Mas os
dois segmentadores nao sofrem a mesma coisa (§9.3):

    SLIC  1.749 -> 2.005 segmentos   (granularidade PRESERVADA -- 2048 e alvo)
    Felz  1.398 ->   134 segmentos   (granularidade COLAPSADA -- scale/min_size
                                      estao em unidades de pixel)

Trocando um par por vez, sem extrair nada de novo:
* se o par do SLIC custar ~0 e o do Felz custar quase tudo -> o problema e
  GRANULARIDADE, e o conserto e reextrair a 224 px com `felz_scale` e
  `min_size` escalados por (224/781)^2 ~ 0,08, o que preserva a alegacao do
  artigo E deixa o preditor do §9.4 barato.
* se os dois custarem metade cada -> e PIXEL mesmo, e a assimetria de resolucao
  tem de ser reportada.
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
from evaluate_resolution_pairing_multitask import SWAP, load_224  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
GROUPS = {
    "slic": ("SLIC", "Line graph"),          # granularidade preservada
    "felz": ("Felzenszwalb", "L(G_felz)"),   # granularidade colapsada
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multitask_resolution_decomposition.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    hand = [cohort.blocks[n] for n in HAND8]
    small = {name: load_224(ROOT / SWAP[name], stems, cohort.blocks[name].shape[1])
             for name in SWAP}

    def mix(swap_names) -> list[np.ndarray]:
        return [small[n] if n in swap_names else b for n, b in zip(HAND8, hand)]

    conditions = {
        "tudo @1024": mix(()),
        "só SLIC+L(G) @224": mix(GROUPS["slic"]),
        "só Felz+L(G_felz) @224": mix(GROUPS["felz"]),
        "tudo @224": mix(tuple(SWAP)),
    }
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")

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
            log(f"{name:24s} fold {number} | BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    if pivot.notna().all().all():
        from scipy import stats
        b = pivot["tudo @1024"].to_numpy()
        print(f"\n{'condição':26s} {'BAcc':>6} {'Δ':>7} {'p':>8} {'folds':>7}")
        for c in ["tudo @1024", "só SLIC+L(G) @224", "só Felz+L(G_felz) @224", "tudo @224"]:
            x = pivot[c].to_numpy(); d = x - b
            if c == "tudo @1024":
                print(f"{c:26s} {x.mean():6.2f}")
                continue
            _, p = stats.ttest_rel(x, b)
            print(f"{c:26s} {x.mean():6.2f} {d.mean():+7.2f} {p:8.4f} {(d < 0).sum():4d}/5 piores")


if __name__ == "__main__":
    main()
