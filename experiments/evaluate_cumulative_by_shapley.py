"""Curva cumulativa em ordem de SHAPLEY, sob LDA e MLP, no Multitask 125 classes.

Difere de `evaluate_cumulative_by_loo.py` em tres pontos, e os tres importam:

1. **A ordem vem do Shapley, nao do LOO.** O §5.9d mediu que o LOO nao serve
   para ordenar views -- o desvio entre folds chega a +-2 a 4pp no MLP (SLIC deu
   +2,31 num fold e -2,19 noutro), porque LOO e o termo de coalizao MAXIMA e nas
   quatro views de grafo de regiao o residuo e dominado por ruido. O Shapley dos
   mesmos dados tem desvio de 0,09 a 0,27 sobre 4.096 mascaras por fold.
2. **125 classes**, nao o recorte de 25. O §5.1 registra que a sonda LDA fica
   presa em `n_classes - 1` componentes; em 25 classes isso e 24 para uma entrada
   que cresce ate 617d, e o teto penaliza o vetor multi-bloco mais que o denso.
3. **5 folds**, nao 3.

**Grafo de similaridade e Text ficam FORA da cadeia, de proposito**, apesar de
terem os dois maiores Shapley (6,11 e 5,22). O grafo e DERIVADO do espaco visual
concatenado -- ele nao existe antes das outras views, entao "acrescentar o grafo
em k=1" nao e uma configuracao executavel, e o Shapley dele e o do bloco de
features pre-computado (ressalva 3 do §5.9c). O Text e supervisao externa e muda
o degrau da escada do §4.1. Os dois entram como RAMO a partir do conjunto
completo, igual ao que o §9.7 fez na figura de custo.

Os conjuntos sao IDENTICOS nos dois classificadores a cada k, entao a distancia
vertical entre as curvas e puramente capacidade, sem confundir com escolha de
views.
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
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_redundancy_selection_curve import fit as plain_fit  # noqa: E402

# Fora da cadeia: derivado (grafo) e supervisao externa (texto). Ver docstring.
BRANCH = {"Similarity graph", "Text"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def shapley_order(path: Path) -> list[str]:
    table = pd.read_csv(path, index_col=0)
    table = table[~table.index.isin(BRANCH)]
    return table.sort_values("shapley", ascending=False).index.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapley", type=Path,
                        default=ROOT / "results/shapley_twelve_summary.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_by_shapley.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    # Coorte inteira: 125 classes. O recorte de 25 e o benchmark do §7.11 e
    # prenderia o LDA em 24 componentes (§5.1).
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    log(f"coorte {len(labels):,} imagens, {n_classes} classes")

    meta = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})
    reference = meta.stem.to_numpy()
    if len(reference) != len(labels):
        raise SystemExit(f"metadado {len(reference)} != coorte {len(labels)}")
    for name, (values_path, stems_path) in EXTERNAL.items():
        values = np.load(ROOT / values_path)
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise SystemExit(f"{name}: {int(index.isna().sum())} stems ausentes")
        blocks[name] = clean(values[index.to_numpy(dtype=int)])

    order = [n for n in shapley_order(args.shapley) if n in blocks]
    missing = set(shapley_order(args.shapley)) - set(blocks)
    if missing:
        raise SystemExit(f"views do Shapley ausentes nos blocos: {sorted(missing)}")
    log(f"ordem de Shapley: {' > '.join(order)}")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    scaled = {fold: {n: StandardScaler().fit(b[train]).transform(b).astype(np.float32)
                     for n, b in blocks.items()}
              for fold, (train, _) in enumerate(splits, start=1)}

    def run(names, fold, train, test, classifier):
        stacked = np.hstack([scaled[fold][n] for n in names])
        if classifier == "LDA":
            model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            model.fit(stacked[train], labels[train])
            return 100 * balanced_accuracy_score(labels[test], model.predict(stacked[test]))
        fit_idx, val_idx = train_test_split(
            train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
        return plain_fit(stacked[fit_idx], labels[fit_idx], stacked[val_idx], labels[val_idx],
                         stacked[test], labels[test], n_classes, seed=SEED + fold)["bacc"]

    # Retomavel: cada linha e (k, classificador, fold) e o arquivo e appendado.
    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(int(r.k), str(r.classifier), int(r.fold)) for r in previous.itertuples()}
        log(f"retomando {len(done)} linhas")

    rows = []
    for k in range(1, len(order) + 1):
        names = order[:k]
        for classifier in ("LDA", "MLP"):
            for fold, (train, test) in enumerate(splits, start=1):
                if (k, classifier, fold) in done:
                    continue
                start = time.time()
                bacc = run(names, fold, train, test, classifier)
                rows.append(dict(k=k, classifier=classifier, fold=fold,
                                 added=order[k - 1], blocks="|".join(names),
                                 dimensions=sum(blocks[n].shape[1] for n in names),
                                 bacc=bacc, seconds=round(time.time() - start, 1)))
                log(f"  k={k:2d} {classifier:3s} fold {fold} | {rows[-1]['dimensions']:4d}d "
                    f"| BAcc {bacc:5.2f} | {rows[-1]['seconds']:.0f}s")
                frame = pd.DataFrame(rows)
                if args.output.exists():
                    frame = pd.concat([pd.read_csv(args.output), frame], ignore_index=True)
                frame.to_csv(args.output, index=False)
                rows = []
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
