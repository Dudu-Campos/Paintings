"""Curvas BAcc(k) por número de modalidades, no MultitaskPainting100k.

Uma curva por subconjunto **aninhado**, na ordem de Shapley medida pelo
fatorial 2^9 (`results/factorial_shapley_multitask_25class_summary.csv`).
Eixo x = k componentes de PCA ajustada só no treino do fold; eixo y = BAcc do
LDA sobre essas k componentes.

Por que aninhado em ordem de Shapley, e não o máximo sobre subconjuntos: com 9
fatores há 126 combinações no meio da faixa, e pegar o máximo infla por seleção
(nota já registrada em `plot_ablation_overview.py`). A ordem vem de uma medição
independente da curva, então a progressão não é escolhida olhando o resultado.

PCA e não top-k supervisionado porque é a curva que identifica teto: o achado 7
mostrou que a seleção supervisionada individual não decorrelaciona, e a curva
fica reta em log-log por tempo demais para o ajuste `a − b·k^(−c)` convergir.

Configuração de 24/ago: **Multiscale sai** (1.032d por +0,36 ± 0,27pp, p=0,053
sobre a base com grafo -- não paga 2,6x as dimensões) e o **grafo de
similaridade entra** como última posição da progressão. O grafo é construído
sobre as 8 modalidades que o precedem e recalculado por fold com nested OOF +
log-lift, então não pode ser pré-computado como bloco estático.
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402
from evaluate_production_visual_rich_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features,
)

SEED, N_FOLDS = 42, 5
GRAPH = "Similarity graph"
K_GRID = (2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50", "Multiscale"))
    parser.add_argument("--with-graph", action="store_true", default=True,
                        help="acrescenta o grafo de similaridade como última "
                             "posição da progressão")
    parser.add_argument("--no-graph", dest="with_graph", action="store_false")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--shapley", type=Path,
                        default=ROOT / "results/factorial_shapley_multitask_25class_summary.csv",
                        help="define a ordem dos subconjuntos aninhados")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/pca_curve_by_subset_multitask.csv")
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
        frequent = [by_norm[n] for n in wanted]
    else:
        frequent = counts[counts >= args.min_per_class].index
    frame = frame[frame["label"].isin(frequent)]

    order = [n for n in pd.read_csv(args.shapley)["modality"] if n in available]
    order += [n for n in available if n not in order]
    log(f"ordem de Shapley: {order}")

    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    n_classes = len(np.unique(labels))
    arrays = {name: frame[[c for c in frame.columns if c.startswith(f"{name}_")]]
              .to_numpy(dtype=np.float32) for name in order}
    for name, values in arrays.items():
        bad = ~np.isfinite(values)
        if bad.any():
            medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
            log(f"  ! {name}: {int(bad.sum())} não-finitos imputados pela mediana")

    log(f"{len(frame):,} imagens, {n_classes} classes, {len(order)} modalidades")
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(int(r.n_modalities), int(r.fold), int(r.k)) for r in previous.itertuples()}
        log(f"retomando: {len(done)} pontos já feitos")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # O grafo entra como uma posição extra ao fim da progressão. Ele é derivado
    # das modalidades que o precedem e recalculado por fold (nested OOF), então
    # não existe como bloco estático em `arrays`.
    total_steps = len(order) + (1 if args.with_graph else 0)

    for m in range(1, total_steps + 1):
        graph_step = args.with_graph and m == total_steps
        names = order if graph_step else order[:m]
        label = "+".join(names) + (f"+{GRAPH}" if graph_step else "")
        for fold, (train, test) in enumerate(splits[:args.folds], start=1):
            base_dims = sum(arrays[n].shape[1] for n in names)
            grid = [k for k in K_GRID
                    if k <= min(base_dims + (42 if graph_step else 0),
                                len(train) - n_classes)]
            if all((m, fold, k) in done for k in grid):
                continue
            started = time.perf_counter()
            parts_train, parts_test = [], []
            for name in names:
                scaler = StandardScaler().fit(arrays[name][train])
                parts_train.append(scaler.transform(arrays[name][train]))
                parts_test.append(scaler.transform(arrays[name][test]))
            train_x, test_x = np.hstack(parts_train), np.hstack(parts_test)
            del parts_train, parts_test

            if graph_step:
                # Espaço do grafo = as modalidades já padronizadas neste fold.
                oof_train, target = rich_similarity_features(
                    train_x.astype(np.float32), labels[train],
                    test_x.astype(np.float32), n_classes,
                    args.n_neighbors, args.alpha, args.max_iter, SEED + fold)
                train_x = np.hstack([train_x, prior_correct_rich(
                    oof_train, labels[train], n_classes)])
                test_x = np.hstack([test_x, prior_correct_rich(
                    target, labels[train], n_classes)])
                del oof_train, target

            total_dims = train_x.shape[1]
            reducer = PCA(n_components=max(grid), svd_solver="randomized",
                          random_state=SEED).fit(train_x)
            train_p, test_p = reducer.transform(train_x), reducer.transform(test_x)
            del train_x, test_x, reducer

            for k in grid:
                if (m, fold, k) in done:
                    continue
                model = LinearDiscriminantAnalysis(
                    n_components=min(n_classes - 1, k),
                    priors=np.ones(n_classes) / n_classes)
                model.fit(train_p[:, :k], labels[train])
                predicted = model.predict(test_p[:, :k])
                row = pd.DataFrame([{
                    "n_modalities": m, "modalities": label, "fold": fold, "k": k,
                    "dimensions": total_dims,
                    "bacc": balanced_accuracy_score(labels[test], predicted),
                    "accuracy": accuracy_score(labels[test], predicted),
                    "macro_f1": f1_score(labels[test], predicted, average="macro")}])
                row.to_csv(args.output, mode="a", index=False,
                           header=not args.output.exists(), float_format="%.6f")
            del train_p, test_p
            log(f"m={m} fold={fold}: {len(grid)} valores de k em "
                f"{time.perf_counter() - started:.0f}s")

    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
