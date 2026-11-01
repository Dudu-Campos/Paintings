"""A previsao do §9.5: preservar a granularidade recupera os -3,53pp a 224 px?

O §9.5 mediu hand-8 caindo de 38,74 para 35,21 ao trocar os quatro blocos de
segmentacao pela extracao de 224 px, e decompos: o SLIC a 224 px EMPATA (+0,05,
n.s.) porque `--segments 2048` mantem a contagem, e o Felz perde porque
`felz_scale`/`min_size` estao em unidades de pixel e os segmentos colapsam de
1.398 para 134.

`run_multitask_224g_felz.sh` reextraiu o Felz a 224 px com `scale=4.0,
min_size=8`, calibrados contra o alvo de 1.398 segmentos. Realizado: **1.404**.
Granularidade pareada com 1/12 dos pixels.

A condicao que decide e `tudo @224g` = SLIC@224 + Felz@224g: TODOS os
segmentadores a 224 px, resolucao pareada com a da CNN, granularidade
preservada. Se ela reproduzir os 38,74:
* a comparacao do §9.2 com as congeladas passa a ser pareada em resolucao;
* o preditor de grafo do §9.4 pode operar em 224 px (0,30 GMACs).
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
SWAP_G = {
    "Felzenszwalb": "outputs/multitask224g_felz_features/combined_features.parquet",
    "L(G_felz)": "results/line_graph_felz_multitask224g_features.parquet",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multitask_resolution_granularity.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    hand = [cohort.blocks[n] for n in HAND8]
    dims = {n: cohort.blocks[n].shape[1] for n in HAND8}

    small = {n: load_224(ROOT / SWAP[n], stems, dims[n]) for n in SWAP}
    gran = {n: load_224(ROOT / SWAP_G[n], stems, dims[n]) for n in SWAP_G}
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")

    def mix(mapping) -> list[np.ndarray]:
        return [mapping.get(n, b) for n, b in zip(HAND8, hand)]

    slic_224 = {n: small[n] for n in ("SLIC", "Line graph")}
    conditions = {
        "tudo @1024": mix({}),
        "só Felz+L(G_felz) @224g": mix(gran),
        "tudo @224g (SLIC@224 + Felz@224g)": mix({**slic_224, **gran}),
    }

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
            log(f"{name:36s} fold {number} | BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    if pivot.notna().all().all():
        from scipy import stats
        b = pivot["tudo @1024"].to_numpy()
        print(f"\n{'condição':36s} {'BAcc':>6} {'Δ':>7} {'p':>8} {'folds':>8}")
        for c in conditions:
            x = pivot[c].to_numpy(); d = x - b
            if c == "tudo @1024":
                print(f"{c:36s} {x.mean():6.2f}")
                continue
            _, p = stats.ttest_rel(x, b)
            print(f"{c:36s} {x.mean():6.2f} {d.mean():+7.2f} {p:8.4f} {(d < 0).sum():4d}/5 piores")


if __name__ == "__main__":
    main()
