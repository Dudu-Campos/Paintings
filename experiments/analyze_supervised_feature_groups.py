"""Agrupar features pelo que elas PREDIZEM, não pelo que elas correlacionam.

Companheiro supervisionado de `analyze_unsupervised_feature_groups.py`. Aquele
agrupa por r², que vive no espaço de **variância**; este agrupa por concordância
entre as predições de cada feature, que vive no espaço **discriminante**. O
projeto mediu seis vezes que os dois não são a mesma coisa, então a partição
também não tem por que ser.

Protocolo: uma sonda LDA de **uma feature só** por coluna, predições fora do
fold, e a afinidade entre duas colunas é o quanto elas rotulam as mesmas imagens
do mesmo jeito.

**A correção de acaso não é opcional aqui, é o experimento.** A taxonomia por
co-ocorrência de erro foi descartada em 16/ago justamente por ordenar por
acurácia (Spearman +0,747 com a BAcc isolada): duas features ruins concordam
muito, porque ambas caem no prior. Concordância crua agruparia por qualidade e
o resultado seria "features fracas de um lado, fortes do outro". Por isso a
afinidade é o **kappa de Cohen** -- concordância descontada da esperada sob
independência das marginais de cada uma:

    kappa(i,j) = (p_observada − p_esperada) / (1 − p_esperada)

Duas features que sempre predizem Impressionism concordam ~100% e têm kappa ~0,
que é a resposta certa: elas não compartilham canal, compartilham prior.

Subamostragem: a matriz de concordância custa 27 produtos (d × n)·(n × d), o que
com n = 79.207 e d = 1.609 são ~10^13 operações. Com 15.000 imagens o erro
padrão de uma taxa de concordância é ~0,4pp, muito abaixo do que separa os
grupos, e o custo cai 5×. O treino das sondas usa a coorte inteira.

Saída: results/supervised_feature_groups.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
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
sys.path.insert(0, str(ROOT / "src/pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from registry import load_blocks, log  # noqa: E402
from analyze_unsupervised_feature_groups import (  # noqa: E402
    FAMILIES, QUADRANT_SOURCE, choose_k, standardise)

SEED, N_FOLDS = 42, 5


def oof_predictions(values: np.ndarray, labels: np.ndarray,
                    classes: np.ndarray) -> np.ndarray:
    """Predição fora do fold de uma sonda LDA de UMA feature, para todas elas.

    Com d=1 o discriminante é fechado: a regra é maximizar `μ_c·x/σ² − μ_c²/2σ²`
    com priors uniformes. Isso resolve as 1.609 sondas de uma vez, em vez de
    1.609 chamadas ao sklearn.
    """
    predictions = np.empty(values.shape, dtype=np.int16)
    splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for train, test in splits.split(np.zeros(len(labels), dtype=np.uint8), labels):
        block = values[train]
        means = np.stack([block[labels[train] == c].mean(axis=0) for c in classes])
        # Variância intra-classe agrupada, por coluna.
        centred = block - means[np.searchsorted(classes, labels[train])]
        variance = np.maximum((centred ** 2).sum(axis=0) / (len(block) - len(classes)),
                              1e-8)
        weight = means / variance                            # (C, d)
        bias = -0.5 * means * weight
        scores = values[test][:, None, :] * weight[None] + bias[None]
        predictions[test] = np.argmax(scores, axis=1).astype(np.int16)
    return predictions


def kappa_affinity(predictions: np.ndarray, n_classes: int,
                   sample: int, rng: np.random.Generator) -> np.ndarray:
    rows = (rng.choice(len(predictions), sample, replace=False)
            if sample and sample < len(predictions) else np.arange(len(predictions)))
    block = predictions[rows]
    n = len(block)
    observed = np.zeros((block.shape[1], block.shape[1]), dtype=np.float64)
    marginal = np.zeros((n_classes, block.shape[1]), dtype=np.float64)
    for label in range(n_classes):
        indicator = (block == label).astype(np.float32)
        observed += indicator.T @ indicator
        marginal[label] = indicator.mean(axis=0)
    observed /= n
    expected = marginal.T @ marginal
    kappa = (observed - expected) / np.maximum(1.0 - expected, 1e-9)
    return np.clip(kappa, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=14)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--compare", type=Path,
                        default=ROOT / "results/unsupervised_feature_groups.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/supervised_feature_groups.csv")
    args = parser.parse_args()

    blocks = sorted({b for names in FAMILIES.values() for b in names})
    cohort = load_blocks(args.dataset, blocks + ["Multiscale"], allow_external=False)
    pieces, origin, names, source = [], [], [], []
    for family, members in FAMILIES.items():
        for block in members:
            pieces.append(standardise(cohort.blocks[block]))
            origin += [family] * cohort.blocks[block].shape[1]
            names += list(cohort.columns[block])
            source += [block] * cohort.blocks[block].shape[1]
    quadrant_columns = np.array(cohort.columns["Multiscale"])
    quadrants = standardise(cohort.blocks["Multiscale"])
    for block, family in QUADRANT_SOURCE.items():
        mask = np.array([f"_{block}_" in c for c in quadrant_columns])
        if mask.any():
            pieces.append(quadrants[:, mask])
            origin += [family] * int(mask.sum())
            names += list(quadrant_columns[mask])
            source += [f"Multiscale/{block}"] * int(mask.sum())

    values = np.hstack(pieces).astype(np.float64)
    origin, source = np.array(origin), np.array(source)
    names = np.array([n.split("_", 1)[-1] if "_" in n else n for n in names])
    keep = values.std(axis=0) > 1e-8
    values, origin, names, source = (values[:, keep], origin[keep], names[keep],
                                     source[keep])
    labels = LabelEncoder().fit_transform(cohort.labels)
    classes = np.unique(labels)
    log(f"{values.shape[1]} features · {len(classes)} classes")

    started = time.perf_counter()
    predictions = oof_predictions(values, labels, classes)
    log(f"sondas de uma feature: {time.perf_counter() - started:.0f}s")
    accuracy = np.array([(predictions[:, i] == labels).mean()
                         for i in range(predictions.shape[1])])

    rng = np.random.default_rng(SEED)
    affinity = kappa_affinity(predictions, len(classes), args.sample, rng)
    np.fill_diagonal(affinity, 1.0)

    k, spectrum = choose_k(affinity, args.min_k, args.max_k)
    k = args.k or k
    log("espectro: " + " ".join(f"{v:.3f}" for v in spectrum))
    log(f"k escolhido = {k}")

    from sklearn.cluster import SpectralClustering
    groups = SpectralClustering(n_clusters=k, affinity="precomputed",
                                assign_labels="kmeans", random_state=SEED,
                                n_init=20).fit_predict(affinity)

    table = pd.DataFrame({"block": source, "feature": names, "origin": origin,
                          "group": groups, "solo_accuracy": accuracy})
    table.to_csv(args.output, index=False, float_format="%.6f")

    log("\n=== grupo × procedência ===")
    log(pd.crosstab(table.group, table.origin).to_string())
    log("\n=== conteúdo ===")
    for group in range(k):
        members = np.flatnonzero(groups == group)
        if not len(members):
            continue
        share = pd.Series(origin[members]).value_counts(normalize=True)
        inner = affinity[np.ix_(members, members)].mean(axis=1)
        central = members[np.argsort(inner)[::-1][:5]]
        log(f"\n  grupo {group} ({len(members)} features) "
            f"{'MISTO' if share.iloc[0] < 0.8 else share.index[0]} · "
            f"acurácia solo média {100 * accuracy[members].mean():.2f}%")
        log("    procedência: "
            + ", ".join(f"{n} {100 * v:.0f}%" for n, v in share.items()))
        log("    centrais: " + " · ".join(names[central]))

    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_mutual_info_score as ami
    # A checagem que matou a taxonomia anterior: se a afinidade ainda ordena por
    # qualidade, os grupos separam features boas de ruins e não canal de canal.
    inner_mean = np.array([affinity[i][groups == groups[i]].mean()
                           for i in range(len(groups))])
    rho, p_value = spearmanr(inner_mean, accuracy)
    log(f"\nafinidade média no grupo × acurácia solo: Spearman {rho:+.3f} "
        f"(p={p_value:.4f})  <- perto de zero = correção de acaso funcionou")
    log(f"AMI com a procedência: {ami(origin, groups):.3f}")
    if args.compare.exists():
        other = pd.read_csv(args.compare)
        merged = table.merge(other[["block", "feature", "group"]],
                             on=["block", "feature"], suffixes=("", "_r2"))
        log(f"AMI com o agrupamento por r²: "
            f"{ami(merged.group_r2, merged.group):.3f}  (n={len(merged)})")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
