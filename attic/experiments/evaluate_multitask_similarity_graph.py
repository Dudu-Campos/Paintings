"""Grafo de similaridade indutivo como branch extra, no MultitaskPainting100k.

Reusa **sem alteração** as funções validadas de
`evaluate_production_visual_rich_similarity_graph.py`:

  - `rich_similarity_features`: nested OOF de verdade (5 folds internos no
    treino) para gerar a feature do lado do treino sem auto-referência. É
    exatamente o que faltava no SNF quebrado de 21/ago, onde o
    `train_posterior` tinha BAcc=100% contra o próprio `y_train` e o
    classificador final aprendia a decodificar um vazamento.
  - `query_features`: um `LabelSpreading` ajustado **só no treino**, e cada
    ponto de consulta escorado pela média das distribuições dos seus k
    vizinhos de treino -- indutivo, nunca vê outros pontos de teste.
  - `prior_correct_rich`: log-lift `log p(c|x) − log p(c)` com o prior do
    treino do fold. Regra geral do projeto (achado 12): todo bloco
    tipo-posterior nasce com log-lift; sem ela o posterior propagado herda o
    prior e a acurácia balanceada despenca sob desbalanceamento.

O espaço do grafo é o mesmo espaço visual do classificador (as N modalidades
hand-designed padronizadas por bloco), como no braço de produção do WikiArt.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS = "16"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_production_visual_rich_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features,
)

torch.set_num_threads(int(THREADS))
SEED, N_FOLDS = 42, 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50",))
    parser.add_argument("--graph-exclude", nargs="+", default=("Multiscale",),
                        help="modalidades fora do ESPAÇO DO GRAFO, mas ainda "
                             "na base do classificador. Multiscale sai por "
                             "padrão: com 1.032 de 1.671 dims ele domina a "
                             "distância euclidiana do kNN e o grafo vira o "
                             "grafo do Multiscale (viés registrado no "
                             "CLAUDE.md de 21/ago, motivo original do SNF)")
    parser.add_argument("--text-scores", type=Path, default=None,
                        help="prefixo dos escores de texto (build_text_scores.py). "
                             "Entra como bloco extra APOS o grafo. Depende de "
                             "supervisao externa (captioner treinado no COCO), "
                             "entao fica fora do numero principal do paper.")
    parser.add_argument("--prior-correct", action="store_true", default=True)
    parser.add_argument("--no-prior-correct", dest="prior_correct",
                        action="store_false",
                        help="desliga o log-lift. A regra do projeto (achado 12) "
                             "foi estabelecida no WikiArt, com desbalanceamento "
                             "de 139,9x; em corpus mais balanceado a correcao "
                             "tem menos vies para corrigir e pode virar so "
                             "amplificacao de ruido.")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"],
                        choices=["lda", "mlp"])
    parser.add_argument("--official-split", action="store_true",
                        help="usa a coluna `split` do metadado (train/test) em vez "
                             "de StratifiedKFold. No Places-LT os 996x de "
                             "desbalanceamento vivem só no split de treino; o pool "
                             "das 106.300 tem 40,8x, mais brando que o WikiArt.")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/multitask_similarity_graph")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "fold_results.csv"

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
    names = list(available)

    carry = ["stem", "label"] + (["split"] if "split" in meta.columns else [])
    frame = meta[carry]
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
    blocks = [frame[[c for c in frame.columns if c.startswith(f"{name}_")]].to_numpy(dtype=np.float32)
              for name in names]
    for name, values in zip(names, blocks):
        bad = ~np.isfinite(values)
        if bad.any():
            medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
            log(f"  ! {name}: {int(bad.sum())} não-finitos imputados pela mediana")

    dims = sum(b.shape[1] for b in blocks)
    graph_names = [n for n in names if n not in args.graph_exclude]
    graph_index = [names.index(n) for n in graph_names]
    graph_dims = sum(blocks[i].shape[1] for i in graph_index)
    log(f"{len(frame):,} imagens x {dims}d em {len(blocks)} blocos, {n_classes} classes")
    log(f"base do classificador: {names}")
    log(f"espaço do grafo: {graph_names} ({graph_dims}d)")
    log(f"grafo: k={args.n_neighbors}, alpha={args.alpha}, nested OOF + log-lift")

    if args.official_split:
        if "split" not in frame.columns:
            raise ValueError(f"{args.dataset}: --official-split pedido, mas o "
                             f"metadado não tem coluna `split`")
        where = frame["split"].to_numpy(str)
        train_ix = np.flatnonzero(where == "train")
        test_ix = np.flatnonzero(where == "test")
        splits = [(train_ix, test_ix)]
        log(f"split oficial: train={len(train_ix):,}, test={len(test_ix):,}, "
            f"val descartado={int((where == 'val').sum()):,}")
    else:
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels)), labels))

    done = set()
    if detail_path.exists():
        previous = pd.read_csv(detail_path)
        done = {(str(r.classifier), str(r.condition), int(r.fold)) for r in previous.itertuples()}
        log(f"retomando: {len(done)} combinações já feitas")

    text_block = None
    if args.text_scores:
        decisions = np.load(f"{args.text_scores}_decisions.npy")
        text_stems = np.load(f"{args.text_scores}_stems.npy", allow_pickle=True).astype(str)
        position = {stem: index for index, stem in enumerate(text_stems)}
        rows = [position.get(stem) for stem in frame["stem"].astype(str)]
        if any(r is None for r in rows):
            raise ValueError("escores de texto nao cobrem a coorte")
        text_block = decisions[np.asarray(rows)].astype(np.float32)
        log(f"Text: {text_block.shape[1]}d para {len(text_block):,} imagens")

    conditions = ["base", "base+graph"] + (["base+graph+text"] if text_block is not None else [])
    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        pending = [(c, cond) for c in args.classifiers
                   for cond in conditions
                   if (c, cond, fold) not in done]
        if not pending:
            log(f"fold {fold}: já completo, pulando")
            continue

        # Espaço do grafo: modalidades escolhidas, padronizadas por bloco no
        # treino. Não é a base inteira -- ver `--graph-exclude`.
        started = time.perf_counter()
        scaled = []
        for index in graph_index:
            scaler = StandardScaler().fit(blocks[index][train])
            scaled.append(scaler.transform(blocks[index]).astype(np.float32))
        graph_space = np.hstack(scaled)
        del scaled

        oof_train, test_features = rich_similarity_features(
            graph_space[train], labels[train], graph_space[test], n_classes,
            args.n_neighbors, args.alpha, args.max_iter, SEED + fold)
        graph_block = np.zeros((len(labels), oof_train.shape[1]), dtype=np.float32)
        adjust = (prior_correct_rich if args.prior_correct
                  else (lambda values, *_a: values))
        graph_block[train] = adjust(oof_train, labels[train], n_classes)
        graph_block[test] = adjust(test_features, labels[train], n_classes)
        del graph_space, oof_train, test_features
        log(f"fold {fold}: grafo pronto ({graph_block.shape[1]}d) em "
            f"{time.perf_counter() - started:.0f}s")

        for classifier, condition in pending:
            if condition == "base":
                use = blocks
            elif condition == "base+graph":
                use = blocks + [graph_block]
            else:
                use = blocks + [graph_block, text_block]
            if classifier == "lda":
                # `evaluate_lda` deriva n_classes internamente.
                metrics = evaluate_lda(use, labels, train, test)
            else:
                metrics = evaluate_mlp(use, labels, train, test, fold,
                                       args.epochs, args.patience)
            row = pd.DataFrame([{"classifier": classifier, "condition": condition,
                                 "fold": fold,
                                 "dimensions": sum(b.shape[1] for b in use),
                                 **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}}])
            row.to_csv(detail_path, mode="a", index=False,
                       header=not detail_path.exists(), float_format="%.6f")
            log(f"  {classifier:<4} {condition:<11} BAcc={100 * metrics['bacc']:.2f}")

    detail = pd.read_csv(detail_path)
    print()
    for classifier in sorted(detail.classifier.unique()):
        block = detail[detail.classifier.eq(classifier)]
        base = block[block.condition.eq("base")].sort_values("fold")
        graph = block[block.condition.eq("base+graph")].sort_values("fold")
        if len(base) != len(graph) or base.empty:
            continue
        delta = graph.bacc.to_numpy() - base.bacc.to_numpy()
        statistic, p = ttest_rel(graph.bacc.to_numpy(), base.bacc.to_numpy())
        print(f"{classifier.upper():<4} base={100 * base.bacc.mean():.2f} ± {100 * base.bacc.std():.2f}"
              f"  +grafo={100 * graph.bacc.mean():.2f} ± {100 * graph.bacc.std():.2f}"
              f"  delta={100 * delta.mean():+.2f} ± {100 * delta.std():.2f}pp"
              f"  p={p:.4f}  {int((delta > 0).sum())}/{len(delta)} folds")
    log(f"salvo: {detail_path}")


if __name__ == "__main__":
    main()
