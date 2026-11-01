"""Marginal leave-one-out por bloco, com MLP — para cruzar com a redundância.

A figura `block_redundancy_vs_shapley.png` usa o Shapley exato das 511 máscaras,
mas aquele factorial foi avaliado com **LDA** (`evaluate_lda`), que está preso a
`n_classes - 1` = 24 componentes. O §8 registra seis casos de "LDA extrai, MLP
destrói", então a contribuição medida pelos dois pode não coincidir.

Shapley exato com MLP seria 2^8 x 5 folds = 1.280 treinos. O marginal
leave-one-out dá a mesma leitura por 45: `hand-8` menos `hand-8 sem o bloco`,
pareado por fold. Não é o mesmo estimador que Shapley (ignora interações de
ordem alta), mas é o que o projeto já usa nos testes passo a passo do §5.1.
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
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean  # noqa: E402

HAND = VISUAL + ["Gabor"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/loo_marginal_multitask.csv")
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in HAND:
        frame = frame.merge(load_block("multitask", name), on="stem", how="inner")
    wanted = {key(l) for l in (ROOT / "data/multitask_25class_benchmark.txt")
              .read_text().splitlines() if l.strip()}
    frame = frame[[key(v) in wanted for v in frame["style"].to_numpy(str)]].reset_index(drop=True)
    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(str))
    blocks = {n: clean(frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
                       .to_numpy(np.float32)) for n in HAND}
    log(f"{len(labels):,} imagens, {len(np.unique(labels))} classes")

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold))
                for r in pd.read_csv(args.output).itertuples()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    conditions = {"hand-8": HAND}
    for name in HAND:
        conditions[f"sem {name}"] = [n for n in HAND if n != name]

    for fold, (train, test) in enumerate(StratifiedKFold(
            5, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels), start=1):
        if fold > args.folds:
            break
        for condition, names in conditions.items():
            if (condition, fold) in done:
                continue
            chosen = [blocks[n] for n in names]
            started = time.perf_counter()
            metrics = evaluate_mlp(chosen, labels, train, test, fold, 150, 15)
            pd.DataFrame([{"condition": condition, "fold": fold,
                           "dimensions": sum(b.shape[1] for b in chosen),
                           "seconds": time.perf_counter() - started,
                           **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}}]).to_csv(
                args.output, mode="a", index=False,
                header=not args.output.exists(), float_format="%.6f")
            log(f"  f{fold} {condition:<20} BAcc={100 * metrics['bacc']:5.2f} "
                f"({sum(b.shape[1] for b in chosen)}d, {time.perf_counter() - started:.0f}s)")

    detail = pd.read_csv(args.output)
    pivot = detail.pivot_table(index="fold", columns="condition", values="bacc") * 100
    print()
    print(f"{'bloco':<14} {'marginal LOO (pp)':>18} {'p':>8} {'folds+':>7}")
    rows = []
    for name in HAND:
        column = f"sem {name}"
        if column not in pivot or "hand-8" not in pivot:
            continue
        delta = (pivot["hand-8"] - pivot[column]).to_numpy()
        _, p = ttest_rel(pivot["hand-8"], pivot[column])
        rows.append({"block": name, "loo_marginal": delta.mean(),
                     "loo_std": delta.std(ddof=1), "p": p,
                     "positive_folds": int((delta > 0).sum())})
        print(f"{name:<14} {delta.mean():>+13.2f} ± {delta.std(ddof=1):4.2f} {p:>8.4f} "
              f"{int((delta > 0).sum())}/{len(delta):>3}")
    pd.DataFrame(rows).to_csv(str(args.output).replace(".csv", "_summary.csv"),
                              index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
