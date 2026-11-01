"""MLP de produção em subconjuntos aninhados, para a curva custo x acurácia.

A progressão aninhada só tinha sido medida com sonda LDA. Aqui é o MLP de
produção (um branch por modalidade), que é o classificador cujo número vale
para o paper -- e o projeto já mediu seis casos em que LDA e MLP discordam de
sinal, então a curva de um não substitui a do outro.

O último passo é o grafo de similaridade, com nested OOF + log-lift e cache.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402
from evaluate_production_visual_rich_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features,
)
from similarity_graph_cache import cached_similarity_features  # noqa: E402

SEED, N_FOLDS = 42, 5
GRAPH = "Similarity graph"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50", "Multiscale"))
    parser.add_argument("--with-graph", action="store_true", default=True)
    parser.add_argument("--no-graph", dest="with_graph", action="store_false")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--shapley", type=Path,
                        default=ROOT / "results/factorial_shapley_multitask_25class_summary.csv")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/nested_mlp_multitask.csv")
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
        if block is not None:
            available[name] = block

    frame = meta[["stem", "label"]]
    for name, block in available.items():
        frame = frame.merge(block, on="stem", how="inner")

    counts = frame["label"].value_counts()
    if args.classes_file:
        def normalize(value: str) -> str:
            return value.replace("_", " ").replace("-", " ").replace("ï", "i").strip().lower()
        wanted = {normalize(line) for line in args.classes_file.read_text().splitlines() if line.strip()}
        by_norm = {normalize(label): label for label in counts.index}
        frequent = [by_norm[n] for n in wanted]
    else:
        frequent = counts[counts >= args.min_per_class].index
    frame = frame[frame["label"].isin(frequent)]

    order = [n for n in pd.read_csv(args.shapley)["modality"] if n in available]
    order += [n for n in available if n not in order]

    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    n_classes = len(np.unique(labels))
    arrays = {name: frame[[c for c in frame.columns if c.startswith(f"{name}_")]]
              .to_numpy(dtype=np.float32) for name in order}
    for name, values in arrays.items():
        bad = ~np.isfinite(values)
        if bad.any():
            medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])

    log(f"{len(frame):,} imagens, {n_classes} classes, ordem: {order}")
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(int(r.n_modalities), int(r.fold)) for r in previous.itertuples()}
        log(f"retomando: {len(done)} combinações já feitas")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_steps = len(order) + (1 if args.with_graph else 0)
    for m in range(1, total_steps + 1):
        graph_step = args.with_graph and m == total_steps
        names = order if graph_step else order[:m]
        label = "+".join(names) + (f"+{GRAPH}" if graph_step else "")
        for fold, (train, test) in enumerate(splits[:args.folds], start=1):
            if (m, fold) in done:
                continue
            started = time.perf_counter()
            blocks = [arrays[n] for n in names]
            if graph_step:
                scaled = []
                for name in names:
                    scaler = StandardScaler().fit(arrays[name][train])
                    scaled.append(scaler.transform(arrays[name]).astype(np.float32))
                space = np.hstack(scaled)
                del scaled
                oof, target = cached_similarity_features(
                    lambda: rich_similarity_features(
                        space[train], labels[train], space[test], n_classes,
                        args.n_neighbors, args.alpha, args.max_iter, SEED + fold),
                    names, train, test, n_classes,
                    args.n_neighbors, args.alpha, args.max_iter, SEED + fold, log=log)
                graph_block = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
                graph_block[train] = prior_correct_rich(oof, labels[train], n_classes)
                graph_block[test] = prior_correct_rich(target, labels[train], n_classes)
                blocks = blocks + [graph_block]
                del space, oof, target

            metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                   args.epochs, args.patience)
            pd.DataFrame([{
                "n_modalities": m, "modalities": label, "fold": fold,
                "dimensions": sum(b.shape[1] for b in blocks),
                **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}}]).to_csv(
                args.output, mode="a", index=False,
                header=not args.output.exists(), float_format="%.6f")
            log(f"m={m} fold={fold}: BAcc={100 * metrics['bacc']:.2f} "
                f"em {time.perf_counter() - started:.0f}s")

    result = pd.read_csv(args.output)
    print()
    summary = result.groupby(["n_modalities", "modalities"])["bacc"].agg(["mean", "std"]) * 100
    print(summary.round(2).to_string())
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
