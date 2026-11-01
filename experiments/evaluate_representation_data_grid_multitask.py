"""Grade representação x dados no MultitaskPainting100k.

Uma curva de erro contra tamanho de treino **por número de representações**,
aninhadas na ordem de Shapley medida pelo fatorial 2^9. Replica o achado 20
(medido só no WikiArt até agora: o expoente do erro cresce monotonicamente com
o número de modalidades, 0,032 -> 0,075, ou seja representações adicionais
devolvem o modelo ao regime em que mais dados ainda pagam).

Pergunta nova que só este dataset permite: o **grafo de similaridade** é um
operador sobre as representações existentes, não uma modalidade nova. Ele
também move o expoente, ou só conteúdo novo faz isso? Por isso ele entra como
o último passo da progressão, recalculado a cada (tamanho, fold) -- o grafo
depende do treino, então subamostrar o treino muda o grafo.

O fold de teste **nunca** é reduzido: só o treino encolhe, senão a variância do
alvo muda junto com o tamanho e a curva deixa de medir escala de dados.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402
from evaluate_production_visual_rich_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features,
)
from similarity_graph_cache import cached_similarity_features  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402

SEED, N_FOLDS = 42, 5
GRAPH = "Similarity graph"
TRAIN_SIZES = (2500, 5000, 10000, 20000, 40000, 0)   # 0 = treino completo
K_GRID = (8, 16, 32, 64, 128, 192, 256, 384, 512, 768)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def perdas(proba: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, float]:
    """NLL simples e NLL MACRO (media das NLL por classe).

    As duas existem porque a curva do §5.1 e lida em BAcc, que e recall MACRO --
    o analogo em perda e a NLL macro, nao a simples. Reportar so a simples
    repetiria o erro de metrica do §4.1f (misturar uma quantidade macro com uma
    quantidade pooled na mesma leitura).

    POR QUE A PERDA, e nao `100 - BAcc`: medido em 13/out, 7 das 9 curvas de
    `log(100-BAcc)` tem termo quadratico significativo (p<=0,025, coeficiente
    POSITIVO nas nove), entao a reta ajusta uma curva que nao e reta -- e e isso
    que faz o expoente mover ~0,005 ao tirar o menor ponto, sempre para baixo.
    A perda usa o posterior inteiro em vez do argmax, e nao tem o teto que
    comprime `100 - BAcc` na faixa de 59% a 77% em que esta grade vive.
    """
    q = np.clip(proba, 1e-12, 1.0)
    ll = -np.log(q[np.arange(len(y)), y])
    macro = np.mean([ll[y == c].mean() for c in range(n_classes) if (y == c).any()])
    return {"nll": float(ll.mean()), "nll_macro": float(macro)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--classifier", default="lda", choices=["lda", "mlp"],
                        help="a grade original usa sonda LDA, que fica presa a "
                             "n_classes-1 = 24 componentes em 25 classes (ressalva "
                             "do §5.1). `mlp` usa o BranchMLP de produção, um branch "
                             "por bloco, sem PCA e sem grade de k.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50", "Multiscale"))
    parser.add_argument("--with-graph", action="store_true", default=True)
    parser.add_argument("--no-graph", dest="with_graph", action="store_false")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    parser.add_argument("--shapley", type=Path,
                        default=ROOT / "results/factorial_shapley_multitask_25class_summary.csv")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/representation_data_grid_multitask.csv")
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
        done = {(int(r.n_modalities), int(r.train_size), int(r.fold))
                for r in previous.itertuples()}
        log(f"retomando: {len(done)} células já feitas")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_steps = len(order) + (1 if args.with_graph else 0)
    for m in range(1, total_steps + 1):
        graph_step = args.with_graph and m == total_steps
        names = order if graph_step else order[:m]
        label = "+".join(names) + (f"+{GRAPH}" if graph_step else "")

        for fold, (train_all, test) in enumerate(splits[:args.folds], start=1):
            for requested in args.train_sizes:
                size = len(train_all) if requested == 0 else requested
                if size > len(train_all) or (m, size, fold) in done:
                    continue
                started = time.perf_counter()
                if size == len(train_all):
                    train = train_all
                else:
                    train, _ = train_test_split(
                        train_all, train_size=size, stratify=labels[train_all],
                        random_state=SEED + fold)

                parts_train, parts_test = [], []
                for name in names:
                    scaler = StandardScaler().fit(arrays[name][train])
                    parts_train.append(scaler.transform(arrays[name][train]))
                    parts_test.append(scaler.transform(arrays[name][test]))
                block_sizes = [p.shape[1] for p in parts_train]
                train_x, test_x = np.hstack(parts_train), np.hstack(parts_test)
                del parts_train, parts_test

                if graph_step:
                    # O grafo depende do treino: subamostrar o treino muda o
                    # grafo de verdade, então cada célula tem o seu. O cache é
                    # chaveado pelos índices exatos, então só reaproveita entre
                    # execuções da mesma célula -- nunca mistura partições.
                    oof, target = cached_similarity_features(
                        lambda: rich_similarity_features(
                            train_x.astype(np.float32), labels[train],
                            test_x.astype(np.float32), n_classes,
                            args.n_neighbors, args.alpha, args.max_iter, SEED + fold),
                        names, train, test, n_classes,
                        args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
                        log=log)
                    graph_train = prior_correct_rich(oof, labels[train], n_classes)
                    graph_test = prior_correct_rich(target, labels[train], n_classes)
                    block_sizes.append(graph_train.shape[1])
                    train_x = np.hstack([train_x, graph_train])
                    test_x = np.hstack([test_x, graph_test])
                    del oof, target, graph_train, graph_test

                if args.classifier == "mlp":
                    # Um branch por bloco, como na produção -- achatar tudo num
                    # vetor só mediria outra arquitetura. Sem PCA e sem grade de
                    # k: o MLP não tem o teto de `n_classes - 1` que motiva a
                    # busca por componentes na sonda LDA.
                    blocks = np.split(train_x, np.cumsum(block_sizes)[:-1], axis=1)
                    index = np.arange(len(train) + len(test))
                    stacked = [np.vstack([a, b]) for a, b in
                               zip(blocks, np.split(test_x, np.cumsum(block_sizes)[:-1], axis=1))]
                    joined = np.concatenate([labels[train], labels[test]])
                    dims = train_x.shape[1]
                    del train_x, test_x, blocks
                    metrics = evaluate_mlp(stacked, joined, index[:len(train)],
                                           index[len(train):], fold,
                                           args.epochs, args.patience)
                    best = {"bacc": metrics["bacc"], "k": 0,
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"]}
                    del stacked
                    pd.DataFrame([{"n_modalities": m, "modalities": label,
                                   "train_size": size, "fold": fold,
                                   "dimensions": dims, **best}]).to_csv(
                        args.output, mode="a", index=False,
                        header=not args.output.exists(), float_format="%.6f")
                    log(f"m={m} n={size:,} fold={fold}: BAcc={100 * best['bacc']:.2f} "
                        f"(MLP) em {time.perf_counter() - started:.0f}s")
                    continue

                grid = [k for k in K_GRID if k <= min(train_x.shape[1], size - n_classes)]
                if not grid:
                    continue
                reducer = PCA(n_components=max(grid), svd_solver="randomized",
                              random_state=SEED).fit(train_x)
                train_p, test_p = reducer.transform(train_x), reducer.transform(test_x)
                dims = train_x.shape[1]
                del train_x, test_x, reducer

                best = {"bacc": -1.0}
                for k in grid:
                    model = LinearDiscriminantAnalysis(
                        n_components=min(n_classes - 1, k),
                        priors=np.ones(n_classes) / n_classes)
                    model.fit(train_p[:, :k], labels[train])
                    predicted = model.predict(test_p[:, :k])
                    bacc = balanced_accuracy_score(labels[test], predicted)
                    if bacc > best["bacc"]:
                        # a perda sai do MESMO k que a BAcc escolheu, senao as
                        # duas colunas descreveriam modelos diferentes na mesma
                        # linha. O criterio de selecao NAO muda -- a BAcc tem de
                        # reproduzir o CSV publicado celula a celula.
                        best = {"bacc": bacc, "k": k,
                                "accuracy": accuracy_score(labels[test], predicted),
                                "macro_f1": f1_score(labels[test], predicted, average="macro"),
                                **perdas(model.predict_proba(test_p[:, :k]),
                                         labels[test], n_classes)}
                del train_p, test_p

                pd.DataFrame([{"n_modalities": m, "modalities": label,
                               "train_size": size, "fold": fold, "dimensions": dims,
                               **best}]).to_csv(
                    args.output, mode="a", index=False,
                    header=not args.output.exists(), float_format="%.6f")
                log(f"m={m} n={size:,} fold={fold}: BAcc={100 * best['bacc']:.2f} "
                    f"(k={best['k']}) em {time.perf_counter() - started:.0f}s")

    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
