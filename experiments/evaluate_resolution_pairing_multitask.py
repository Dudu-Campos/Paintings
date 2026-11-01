"""§7.1: o nosso numero depende dos pixels extras?

A defesa do §7.1 contra "voces veem mais pixels que a ResNet" estava construida
sobre uma mediana de 343 px medida na copia local REDUZIDA. A coorte real tem
mediana 781 e 30% acima de 1024, entao o teto morde de verdade e a objecao e
legitima. Aqui os quatro blocos de segmentacao sao trocados pela versao extraida
com `--max-side 224` (`run_multitask_224_extraction.sh`), pareando a resolucao de
entrada com a da ResNet/ViT. Tudo o mais identico: mesmos folds, mesma cabeca,
mesmos outros quatro blocos.

O que o resultado decide:
* delta pequeno -> a objecao morre e vira frase de controle; e o preditor de
  grafo do §9.4 pode operar em 224 px (0,30 GMACs, 13,6x abaixo da ResNet50).
* delta grande  -> a assimetria tem de ser reportada, E o preditor precisa de
  mais pixels, o que encolhe a economia.

Confundidor que tem de ser dito junto (§9.3): baixar resolucao NAO isola
resolucao. `felz_scale` e `min_size` estao em unidades de pixel, entao o Felz
cai de 1.398 para 134 segmentos -- e coarsening de grafo, nao so menos pixels.
O SLIC nao sofre isso (2048 e alvo explicito): 1.749 -> 2.005.
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

from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_preprocess_effect_oilpainting import fit  # noqa: E402
from registry import feature_columns, load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
# Os quatro blocos que rodam acima de 224 px. Os outros quatro (Descriptors,
# Hilbert, Gabor a 128; Fourier a 256) ja estao abaixo e nao mudam.
SWAP = {
    "SLIC": "outputs/multitask224_slic_features/combined_features.parquet",
    "Felzenszwalb": "outputs/multitask224_felz_features/combined_features.parquet",
    "Line graph": "results/line_graph_multitask224_features.parquet",
    "L(G_felz)": "results/line_graph_felz_multitask224_features.parquet",
}


def load_224(path: Path, stems: np.ndarray, expected: int) -> np.ndarray:
    table = pd.read_parquet(path)
    table["stem"] = table["stem"].astype(str)
    columns = feature_columns(table)
    if len(columns) != expected:
        raise ValueError(f"{path.name}: {len(columns)} colunas, esperava {expected}")
    order = pd.Index(table["stem"]).get_indexer(stems)
    if (order < 0).any():
        raise ValueError(f"{path.name}: {int((order < 0).sum())} stems ausentes")
    return table[columns].to_numpy(dtype=np.float32)[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multitask_resolution_pairing.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))

    hand_1024 = [cohort.blocks[n] for n in HAND8]
    hand_224 = []
    for name, block in zip(HAND8, hand_1024):
        if name in SWAP:
            swapped = load_224(ROOT / SWAP[name], stems, block.shape[1])
            log(f"  {name:14s} trocado por 224px ({swapped.shape[1]}d)")
            hand_224.append(swapped)
        else:
            hand_224.append(block)
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes, "
        f"hand-8 {sum(b.shape[1] for b in hand_1024)}d")

    conditions = {
        "hand-8 @1024": (hand_1024, None),
        "hand-8 @224": (hand_224, None),
        "hand-8 + grafo @1024": (hand_1024, np.hstack(hand_1024)),
        "hand-8 + grafo @224": (hand_224, np.hstack(hand_224)),
    }

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for name, (blocks, source) in conditions.items():
        for number, (train_idx, test_idx) in enumerate(folds, 1):
            if (name, number) in done:
                continue
            started = time.perf_counter()
            metrics = fit(blocks, labels, train_idx, test_idx, n_classes, SEED,
                          args.epochs, args.patience, "narrow", source)
            row = {"condition": name, "fold": number,
                   "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{name:22s} fold {number} | {row['dimensions']:5d}d | "
                f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")


if __name__ == "__main__":
    main()
