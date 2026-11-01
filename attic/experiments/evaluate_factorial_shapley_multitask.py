"""Ablação fatorial 2^9 com sonda LDA e Shapley exato, no MultitaskPainting100k.

Nove fatores (as 8 modalidades canônicas mais Gabor, que entrou no pipeline em
21/ago e nunca passou por medida de importância junto das outras). 512
máscaras x 5 folds. Incremental e retomável: cada linha vai pro CSV assim que
sai, e uma re-execução pula o que já está lá -- o padrão do projeto para jobs
longos (ver `evaluate_full_factorial_ablation_text.py`).

Sonda LDA, não MLP, por custo: o Shapley exato exige avaliar todos os
subconjuntos, e o LDA reajusta em segundos. **Ressalva registrada**: o projeto
já mediu seis casos de "LDA extrai, MLP destrói" (achados 11, 12, 15, 16), e a
discordância é sistemática -- o LDA reponderá features redundantes de graça,
enquanto no MLP um branch fraco consome capacidade. Ler esta tabela como
ordenação de importância, não como previsão do que acontece no MLP.

Saída: results/factorial_shapley_multitask.csv (por máscara x fold) e
results/factorial_shapley_multitask_summary.csv (Shapley por modalidade).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402

SEED, N_FOLDS = 42, 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate_lda(blocks, labels, train, test, n_classes) -> dict:
    """LDA direto na dimensão nativa, priors uniformes -- a sonda canônica."""
    parts_train, parts_test = [], []
    for block in blocks:
        scaler = StandardScaler().fit(block[train])
        parts_train.append(scaler.transform(block[train]))
        parts_test.append(scaler.transform(block[test]))
    train_x, test_x = np.hstack(parts_train), np.hstack(parts_test)
    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
    model.fit(train_x, labels[train])
    predicted = model.predict(test_x)
    return {"bacc": balanced_accuracy_score(labels[test], predicted),
            "accuracy": accuracy_score(labels[test], predicted),
            "macro_f1": f1_score(labels[test], predicted, average="macro"),
            "dimensions": train_x.shape[1]}


def shapley(scores: dict[int, float], n_factors: int) -> np.ndarray:
    """Shapley exato a partir do valor de todos os 2^n subconjuntos."""
    values = np.zeros(n_factors)
    for index in range(n_factors):
        bit = 1 << index
        total = 0.0
        for mask in range(1 << n_factors):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            weight = (math.factorial(size) * math.factorial(n_factors - size - 1)
                      / math.factorial(n_factors))
            total += weight * (scores[mask | bit] - scores[mask])
        values[index] = total
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50",))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/factorial_shapley_multitask.csv")
    args = parser.parse_args()

    meta_path, label_column = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / meta_path).dropna(subset=[label_column])
    meta["stem"] = meta["stem"].astype(str)
    meta["label"] = meta[label_column].astype(str)

    available = {}
    for name in BLOCKS:
        if name in args.exclude:
            continue
        block = load_block(args.dataset, name)
        if block is None:
            log(f"{name}: não extraído, pulando")
        else:
            available[name] = block
    factors = list(available)

    frame = meta[["stem", "label"]]
    for name, block in available.items():
        frame = frame.merge(block, on="stem", how="inner")

    counts = frame["label"].value_counts()
    if args.classes_file:
        def normalize(value: str) -> str:
            return value.replace("_", " ").replace("-", " ").replace("ï", "i").strip().lower()
        wanted = {normalize(line) for line in args.classes_file.read_text().splitlines() if line.strip()}
        by_norm = {normalize(label): label for label in counts.index}
        missing = wanted - set(by_norm)
        if missing:
            raise ValueError(f"classes não encontradas: {missing}")
        frequent = [by_norm[name] for name in wanted]
    else:
        frequent = counts[counts >= args.min_per_class].index
    frame = frame[frame["label"].isin(frequent)]

    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    n_classes = len(np.unique(labels))
    arrays = {name: frame[[c for c in frame.columns if c.startswith(f"{name}_")]]
              .to_numpy(dtype=np.float32) for name in factors}
    for name, values in arrays.items():
        bad = ~np.isfinite(values)
        if bad.any():
            medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
            log(f"  ! {name}: {int(bad.sum())} não-finitos imputados pela mediana")

    n_factors = len(factors)
    log(f"{len(frame):,} imagens, {n_classes} classes, {n_factors} fatores: {factors}")
    log(f"{1 << n_factors} máscaras x {args.folds} folds")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(int(r.mask), int(r.fold)) for r in previous.itertuples()}
        log(f"retomando: {len(done)} combinações já feitas")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = 0
    for mask in range(1, 1 << n_factors):
        names = [factors[i] for i in range(n_factors) if mask >> i & 1]
        for fold, (train, test) in enumerate(splits[:args.folds], start=1):
            if (mask, fold) in done:
                continue
            metrics = evaluate_lda([arrays[n] for n in names], labels, train, test, n_classes)
            row = pd.DataFrame([{"mask": mask, "fold": fold,
                                 "n_modalities": len(names),
                                 "modalities": "+".join(names), **metrics}])
            row.to_csv(args.output, mode="a", index=False,
                       header=not args.output.exists(), float_format="%.6f")
            completed += 1
        if mask % 16 == 0:
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed else 0
            remaining = ((1 << n_factors) - mask) * args.folds
            log(f"máscara {mask}/{(1 << n_factors) - 1}; {completed} avaliações; "
                f"{rate:.2f}/s; ETA {remaining / rate / 60:.0f} min" if rate else
                f"máscara {mask}")

    result = pd.read_csv(args.output)
    per_fold = []
    for fold in sorted(result.fold.unique()):
        block = result[result.fold.eq(fold)]
        scores = dict(zip(block["mask"].astype(int), block["bacc"]))
        scores[0] = 1.0 / n_classes  # representação vazia = acaso balanceado
        if len(scores) != 1 << n_factors:
            log(f"fold {fold} incompleto ({len(scores)}/{1 << n_factors}), pulando Shapley")
            continue
        per_fold.append(shapley(scores, n_factors))
    if per_fold:
        values = np.vstack(per_fold)
        summary = pd.DataFrame({"modality": factors,
                                "shapley": values.mean(axis=0) * 100,
                                "std": values.std(axis=0, ddof=1) * 100,
                                "dimensions": [arrays[n].shape[1] for n in factors]})
        summary = summary.sort_values("shapley", ascending=False)
        summary_path = args.output.with_name(args.output.stem + "_summary.csv")
        summary.to_csv(summary_path, index=False, float_format="%.4f")
        print()
        print("Shapley exato (pp de BAcc), sonda LDA:")
        print(summary.round(2).to_string(index=False))
        log(f"salvo: {summary_path}")


if __name__ == "__main__":
    main()
