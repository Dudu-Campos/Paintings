"""Cadeia cumulativa das DOZE views, sob LDA e MLP raso, Multitask 125 classes.

Substitui `evaluate_cumulative_by_shapley.py`, que rodava so as dez views
estaticas. Motivo (decisao do usuario em 03/set): o artigo mostra primeiro a
fronteira de Pareto, que enumera as doze -- entao a figura seguinte nao pode
fazer o grafo de similaridade e o texto desaparecerem sem explicacao.

**ORDEM.** As sete primeiras vem do Shapley, como antes. O grafo e o texto
entram em 8 e 9, logo ANTES das tres que subtraem. Nao e a posicao de Shapley
deles (seriam 1 e 2): o grafo e DERIVADO do espaco visual concatenado e nao
existe em k=1, e poe-los no inicio faria a cadeia comecar por um bloco que
pressupoe o resto. Encostados no fim, eles ficam no regime em que o leitor os
encontra na pratica -- somados a um banco visual ja completo -- e a figura
mantem as tres prejudiciais no extremo direito, que e a afirmacao dela.

**MLP raso de proposito** (decisao do usuario em 03/set): nenhuma cabeca e proxy
de todas as configuracoes, e o `PlainMLP` denso e o instrumento generico; o
`BranchMLP` de producao responderia sobre si mesmo. Consequencia obrigatoria na
legenda: o NIVEL nao e comparavel com as tabelas do artigo (o `hand-8` da 36,32
aqui contra 38,74 no §9.2, e a diferenca inteira e o cabecote).

O grafo e o texto saem do cache por fold do Shapley (`outputs/shapley_twelve_views`),
que ja os tem com OOF aninhado -- sao ~9 min por fold que nao se repetem. Os
splits sao construidos exatamente como la, senao o cache desalinha em silencio.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
# `branch_fit` e o MESMO `fit` das tabelas de producao, importado e nao
# reimplementado: o §5.6 registra um caso em que reescrever o laco custou 0,47pp
# num braco de controle (label_smoothing omitido e semente diferente), e um
# controle que nao reproduz o valor conhecido nao e controle.
from evaluate_preprocess_effect_oilpainting import fit as branch_fit  # noqa: E402
from evaluate_redundancy_selection_curve import fit as plain_fit  # noqa: E402
from evaluate_shapley_twelve_views import DORMANT, HAND8, N_FOLDS, SEED  # noqa: E402
from registry import load_blocks, log  # noqa: E402

# Sete de Shapley, depois grafo e texto, depois as tres que subtraem.
ORDER = ["Felzenszwalb", "SLIC", "Descriptors", "L(G_felz)", "Line graph",
         "Gabor", "Hilbert", "Similarity graph", "Text",
         "Fourier", "Random walk", "HVG"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "outputs/shapley_twelve_views")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_twelve.csv")
    args = parser.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")

    blocks = {name: cohort.blocks[name].astype(np.float32) for name in HAND8}
    for name in DORMANT:
        values_path, stems_path = EXTERNAL[name]
        values = np.load(ROOT / values_path)
        order = pd.Index(np.load(ROOT / stems_path, allow_pickle=True)
                         .astype(str)).get_indexer(stems)
        if (order < 0).any():
            raise SystemExit(f"{name}: {int((order < 0).sum())} stems ausentes")
        blocks[name] = values[order].astype(np.float32)

    # Splits IDENTICOS aos do Shapley -- o cache por fold depende disso.
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(stems, labels))[:args.folds]

    done = set()
    if args.output.exists():
        done = {(int(r.k), str(r.classifier), int(r.fold))
                for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        cache = args.cache_dir / f"fold_{fold}.npz"
        if not cache.exists():
            raise SystemExit(f"cache ausente: {cache} -- rodar o Shapley antes")
        saved = np.load(cache)
        per_fold = dict(blocks)
        per_fold["Similarity graph"] = saved["graph"].astype(np.float32)
        per_fold["Text"] = saved["text"].astype(np.float32)
        if any(per_fold[n].shape[0] != len(labels) for n in ORDER):
            raise SystemExit(f"fold {fold}: bloco com numero de linhas errado")

        scaled = {n: StandardScaler().fit(per_fold[n][train_idx])
                  .transform(per_fold[n]).astype(np.float32) for n in ORDER}

        for k in range(1, len(ORDER) + 1):
            names = ORDER[:k]
            stacked = None
            for classifier in ("LDA", "MLP", "BranchMLP"):
                if (k, classifier, fold) in done:
                    continue
                start = time.time()
                if classifier == "BranchMLP":
                    # Recebe os blocos SEPARADOS (um encoder por view), e faz a
                    # propria padronizacao no sub-treino -- por isso passa os
                    # blocos crus, nao os ja escalados.
                    metrics = branch_fit([per_fold[n] for n in names], labels,
                                         train_idx, test_idx, n_classes,
                                         seed=SEED, epochs=150, patience=15,
                                         head="narrow")
                    bacc = metrics["bacc"]
                    dims = sum(per_fold[n].shape[1] for n in names)
                    row = pd.DataFrame([dict(
                        k=k, classifier=classifier, fold=fold, added=ORDER[k - 1],
                        blocks="|".join(names), dimensions=dims, bacc=bacc,
                        seconds=round(time.time() - start, 1))])
                    row.to_csv(args.output, mode="a", index=False,
                               header=not args.output.exists(), float_format="%.4f")
                    log(f"  fold {fold} k={k:2d} {classifier:9s} | {dims:4d}d "
                        f"| BAcc {bacc:5.2f} | {row.seconds.iloc[0]:.0f}s")
                    continue
                if stacked is None:
                    stacked = np.hstack([scaled[n] for n in names])
                if classifier == "LDA":
                    model = LinearDiscriminantAnalysis(
                        priors=np.ones(n_classes) / n_classes)
                    model.fit(stacked[train_idx], labels[train_idx])
                    bacc = 100 * balanced_accuracy_score(
                        labels[test_idx], model.predict(stacked[test_idx]))
                else:
                    fit_idx, val_idx = train_test_split(
                        train_idx, test_size=0.1, stratify=labels[train_idx],
                        random_state=SEED + fold)
                    bacc = plain_fit(stacked[fit_idx], labels[fit_idx],
                                     stacked[val_idx], labels[val_idx],
                                     stacked[test_idx], labels[test_idx],
                                     n_classes, seed=SEED + fold)["bacc"]
                row = pd.DataFrame([dict(
                    k=k, classifier=classifier, fold=fold, added=ORDER[k - 1],
                    blocks="|".join(names), dimensions=stacked.shape[1],
                    bacc=bacc, seconds=round(time.time() - start, 1))])
                row.to_csv(args.output, mode="a", index=False,
                           header=not args.output.exists(), float_format="%.4f")
                log(f"  fold {fold} k={k:2d} {classifier:3s} | {stacked.shape[1]:4d}d "
                    f"| BAcc {bacc:5.2f} | {row.seconds.iloc[0]:.0f}s")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
