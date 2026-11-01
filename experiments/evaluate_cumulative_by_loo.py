"""Curva cumulativa: blocos entrando em ordem de LOO, sob LDA e MLP.

Ordenar por LOO e acumular responde algo que a eliminação para trás não
responde: **quanto do banco basta**, e se a resposta muda com o classificador.
A eliminação dá um caminho de remoção; esta dá a curva de saturação.

Duas ordenações, porque os dois classificadores discordam sobre um bloco em
particular (Fourier: LOO +0,74 no LDA e −0,04 no MLP, o único que inverte de
sinal):

  `mlp`  ordem pelo LOO do MLP -- os conjuntos são IDÊNTICOS nos dois braços em
         cada k, então a distância vertical entre as curvas é puramente o
         classificador, sem confundir com escolha de blocos.
  `lda`  ordem pelo LOO do LDA -- mostra se o LDA se sai melhor usando o próprio
         critério, isto é, se a discordância de ordenação custa alguma coisa.

O LOO usado é o do conjunto COMPLETO de 10 blocos. Ele só é bem-posto para
remover um bloco de cada vez, e é por isso que aqui ele serve de **ordenação**,
não de regra de remoção: a soma de LOOs não prevê remoção conjunta (medido:
Fourier −0,04 e HVG −0,71 somam −0,75, mas removidos juntos deram +1,53).
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


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulative_by_loo.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    if args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
    else:
        # `--classes-file none` roda a coorte inteira (multitask125). O recorte
        # de 25 é o benchmark do §7.11; sem ele o LDA deixa de ficar preso em
        # 24 componentes, que é a ressalva do §5.1. `wanted` vira o conjunto
        # completo porque ele é reusado adiante para recortar o metadado.
        wanted = {key(v) for v in raw_labels}
        mask = np.ones(len(raw_labels), dtype=bool)
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    n_classes = len(np.unique(labels))

    meta = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})
    reference = meta[[key(str(v)) in wanted for v in meta["style"]]].stem.to_numpy()
    for name, (values_path, stems_path) in EXTERNAL.items():
        values = np.load(ROOT / values_path)
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        blocks[name] = clean(values[index.to_numpy(dtype=int)])

    order = [n for n in VISUAL + ["Gabor"] if n in blocks] + list(EXTERNAL)
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

    # LOO do LDA no conjunto de 10 (o do fatorial é sobre 8 e não serve aqui).
    lda_loo = {}
    base_lda = np.mean([run(order, f, tr, te, "LDA")
                        for f, (tr, te) in enumerate(splits, start=1)])
    for name in order:
        without = [n for n in order if n != name]
        value = np.mean([run(without, f, tr, te, "LDA")
                         for f, (tr, te) in enumerate(splits, start=1)])
        lda_loo[name] = base_lda - value
        log(f"  LOO LDA  {name:<14}{lda_loo[name]:+6.2f}")

    elimination = pd.read_csv(ROOT / "results/backward_elimination.csv")
    step0 = elimination[elimination.step.eq(0)]
    base_mlp = step0[step0.removed.eq("-")].bacc.mean()
    mlp_loo = (base_mlp - step0[step0.removed.ne("-")].groupby("removed").bacc.mean()).to_dict()

    orderings = {
        "by MLP LOO": [n for n, _ in sorted(mlp_loo.items(), key=lambda x: -x[1])],
        "by LDA LOO": [n for n, _ in sorted(lda_loo.items(), key=lambda x: -x[1])],
    }
    for tag, names in orderings.items():
        log(f"{tag}: {names}")

    rows = []
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
    done = {(str(r["ordering"]), int(r["k"]), str(r["classifier"]), int(r["fold"]))
            for r in rows}

    for tag, names in orderings.items():
        for k in range(1, len(names) + 1):
            subset = names[:k]
            for classifier in ("LDA", "MLP"):
                for fold, (train, test) in enumerate(splits, start=1):
                    if (tag, k, classifier, fold) in done:
                        continue
                    value = run(subset, fold, train, test, classifier)
                    rows.append({"ordering": tag, "k": k, "classifier": classifier,
                                 "fold": fold, "added": subset[-1],
                                 "blocks": "+".join(subset),
                                 "dimensions": sum(scaled[fold][n].shape[1] for n in subset),
                                 "bacc": value})
                    pd.DataFrame(rows).to_csv(args.output, index=False)
            frame = pd.DataFrame(rows)
            cut = frame[frame.ordering.eq(tag) & frame.k.eq(k)]
            log(f"  {tag}  k={k:>2} +{subset[-1]:<14} "
                f"LDA {cut[cut.classifier.eq('LDA')].bacc.mean():5.2f}  "
                f"MLP {cut[cut.classifier.eq('MLP')].bacc.mean():5.2f}")

    frame = pd.DataFrame(rows)
    log("\n" + frame.pivot_table(index=["ordering", "k"], columns="classifier",
                                 values="bacc").round(2).to_string())


if __name__ == "__main__":
    main()
