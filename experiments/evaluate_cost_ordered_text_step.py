"""Passo 9 da curva de custo: o canal de texto, cujo custo é de GPU.

A curva `cost_ordered_multitask.png` tem o eixo em **relógio de CPU**. A geração
de legendas roda em **GPU** (47,6 img/s medidos, 35 min para as 99.816), então
pôr os dois no mesmo eixo é erro de categoria -- e o ponto cair fora da lei de
potência é a forma honesta de mostrar isso, em vez de escondê-lo.

Mede `hand-8 + texto`, SEM grafo, porque a curva é sem grafo. O §5.18 mediu o
texto sobre `hand-8 + grafo` (base diferente); reusar aquele número aqui
misturaria bases.

OOF aninhado obrigatório (§5.18, §6): o `build_text_scores.py` produz OOF simples
sobre a coorte inteira, então para o fold externo k os escores das linhas de
treino vieram de modelos que viram os rótulos do fold k.
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
from build_text_scores import make_stop_words  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_text_marginal_multitask import CAPTIONS, load_captions, nested_text_block  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/cost_ordered_text_step.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    labels = LabelEncoder().fit_transform(labels_str)
    n_classes = len(np.unique(labels))
    hand = [cohort.blocks[n] for n in HAND8]
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    stop_words = make_stop_words(classes)
    texts = load_captions(CAPTIONS["estudante"], stems)
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes, "
        f"{np.mean([len(t.split()) for t in texts]):.1f} tokens/legenda")

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for number, (train_idx, test_idx) in enumerate(folds, 1):
        name = "hand-8 + texto"
        if (name, number) in done:
            continue
        started = time.perf_counter()
        log(f"  fold {number}: OOF aninhado do texto")
        block = nested_text_block(texts, labels_str, classes, index_of,
                                  stop_words, train_idx, test_idx, number)
        metrics = base.fit(hand + [block], labels, train_idx, test_idx, n_classes,
                           SEED, args.epochs, args.patience, "narrow", None)
        row = {"condition": name, "fold": number,
               "seconds": time.perf_counter() - started, **metrics}
        frame = pd.DataFrame([row])
        if args.output.exists():
            frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
        frame.to_csv(args.output, mode="a", index=False,
                     header=not args.output.exists(), float_format="%.4f")
        log(f"{name} fold {number} | {row['dimensions']:4d}d | "
            f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    if len(frame) == N_FOLDS:
        print(f"\nhand-8 + texto: {frame.bacc.mean():.2f} ± {frame.bacc.std():.2f}")
        print("hand-8 (passo 8 da curva): 38.71")


if __name__ == "__main__":
    main()
