"""A lei de escala fica melhor sobre canais medidos ou sobre blocos de engenharia?

O achado 20 mostrou que o expoente do erro cresce com o número de modalidades
(0,032 → 0,075 de 1 para 6): representações adicionais devolvem o modelo ao
regime em que mais dados ainda pagam. Aquele eixo x eram os **blocos** do
pipeline, que rotulam a procedência da coluna e não o que ela mede -- nada
impede uma "feature de grafo" que constrói o RAG, extrai a cor média e descarta
o grafo.

`analyze_unsupervised_feature_groups.py` mediu os grupos reais por agrupamento
espectral sobre a afinidade r², sem usar rótulo de origem, e eles se organizam
por **grandeza medida**: luminância, canal a, canal b, dispersão de cor,
distribuição de área, bins de histograma. A informação mútua ajustada com a
procedência é 0,330 -- relacionados, não iguais.

Este script repete a grade representação × dados nos dois eixos, com o mesmo
código, para que qualquer diferença seja do eixo e não do protocolo.

Três decisões que o resultado depende, todas explícitas:

**Grupos minúsculos saem.** O eigengap escolheu k=13 e quatro grupos ficaram com
2 a 9 colunas. Somar um grupo de 2 features não é comparável a somar um de 606,
e deixá-los inflaria a contagem de "modalidades" sem conteúdo. `--min-group`
descarta ~1% das colunas.

**A ordem empírica é gulosa, não Shapley.** Com 9 grupos, Shapley exato são 512
máscaras por tamanho e por fold. A ordem gulosa (a cada passo, o grupo de maior
marginal) é o análogo honesto de "aninhado em ordem de Shapley" e custa 45
ajustes. É calculada **uma vez**, no maior tamanho e no fold 1, e depois fixada
-- exatamente como a ordem de Shapley foi fixada no achado 20.

**O teste nunca encolhe.** Só o treino é subamostrado; o fold de teste é sempre
o inteiro, senão a variância cresceria junto com o efeito que queremos medir.

Saída incremental: outputs/unsupervised_channel_grid/fold_results.csv
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
SIZES = (2500, 5000, 10000, 20000, 40000, 0)      # 0 = todo o treino do fold
# Ordem de Shapley do achado 1, que é a do achado 20. Multiscale fica fora pelo
# mesmo motivo de lá: com 1.032d ele é 7× a maior das outras e misturaria salto
# de tamanho com adição de conteúdo.
# Os dois braços particionam **as mesmas colunas**; só o critério muda. Sem
# isso a comparação mediria o tamanho do pool em vez do eixo. Multiscale entra
# como um bloco, que é como a engenharia de fato a trata.
ENGINEERED = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
              "Hilbert", "Fourier", "Multiscale")
QUADRANT_SOURCE = ("Descriptors", "SLIC", "Hilbert")


def k_grid(limit: int) -> list[int]:
    grid = [k for k in (16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536)
            if k <= limit]
    if not grid or grid[-1] != limit:
        grid.append(limit)
    return grid


def probe(values, labels, train, test, n_classes) -> float:
    scaler = StandardScaler().fit(values[train])
    train_x = scaler.transform(values[train])
    test_x = scaler.transform(values[test])
    limit = min(train_x.shape[1], len(train) - n_classes)
    if limit < 2:
        return 1.0 / n_classes
    reducer = PCA(n_components=limit, svd_solver="randomized",
                  random_state=SEED).fit(train_x)
    train_p, test_p = reducer.transform(train_x), reducer.transform(test_x)
    best = 0.0
    for k in k_grid(limit):
        model = LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, k),
            priors=np.ones(n_classes) / n_classes).fit(train_p[:, :k], labels[train])
        best = max(best, balanced_accuracy_score(
            labels[test], model.predict(test_p[:, :k])))
    return best


def subsample(train: np.ndarray, labels: np.ndarray, size: int,
              rng: np.random.Generator) -> np.ndarray:
    if not size or size >= len(train):
        return train
    # Estratificado: com 139× de desbalanceamento uma amostra uniforme perderia
    # classes inteiras em 2.500 e a acurácia balanceada viraria ruído.
    chosen = []
    share = size / len(train)
    for label in np.unique(labels[train]):
        rows = train[labels[train] == label]
        take = max(1, int(round(len(rows) * share)))
        chosen.append(rng.choice(rows, min(take, len(rows)), replace=False))
    return np.concatenate(chosen)


def greedy_order(blocks: dict[str, np.ndarray], labels, train, test,
                 n_classes) -> list[str]:
    """A cada passo, o grupo que mais soma ao que já foi escolhido."""
    remaining, order, current = list(blocks), [], []
    while remaining:
        scores = {}
        for name in remaining:
            values = np.hstack([blocks[n] for n in current + [name]])
            scores[name] = probe(values, labels, train, test, n_classes)
        best = max(scores, key=scores.get)
        order.append(best)
        current.append(best)
        remaining.remove(best)
        log(f"    passo {len(order)}: +{best} -> {100 * scores[best]:.2f}%")
    return order


def build_unsupervised(cohort, assignment: Path, min_group: int):
    table = pd.read_csv(assignment)
    # Casar por CHAVE (bloco, feature) e nunca por posição: os dois scripts
    # iteram os blocos em ordens diferentes, e um desalinhamento posicional
    # atribuiria grupos errados sem erro nenhum.
    columns, lookup = {}, {}
    for block in cohort.blocks:
        if block == "Multiscale":
            continue
        for index, name in enumerate(cohort.columns[block]):
            lookup[(block, name.split("_", 1)[-1])] = (block, index)
    quadrant_columns = list(cohort.columns["Multiscale"])
    for source in QUADRANT_SOURCE:
        for index, name in enumerate(quadrant_columns):
            if f"_{source}_" in name:
                lookup[(f"Multiscale/{source}", name.split("_", 1)[-1])] = (
                    "Multiscale", index)

    missing, selection = 0, {}
    for row in table.itertuples(index=False):
        key = (str(row.block), str(row.feature))
        if key not in lookup:
            missing += 1
            continue
        selection.setdefault(f"g{row.group}", []).append(lookup[key])
    if missing:
        raise ValueError(f"{missing} linhas do assignment não casaram com o registro")

    groups = {}
    for name, entries in sorted(selection.items()):
        if len(entries) < min_group:
            continue
        parts = [cohort.blocks[block][:, index] for block, index in entries]
        groups[name] = np.column_stack(parts)
    log(f"  {len(groups)} grupos empíricos (≥{min_group} colunas): "
        + ", ".join(f"{k}({v.shape[1]})" for k, v in groups.items()))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--assignment", type=Path,
                        default=ROOT / "results/unsupervised_feature_groups.csv")
    parser.add_argument("--min-group", type=int, default=20)
    parser.add_argument("--arms", nargs="+", default=["unsupervised", "engineered"])
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/unsupervised_channel_grid")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, list(ENGINEERED), allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    arms = {}
    if "engineered" in args.arms:
        # Ordem gulosa também aqui: um braço ordenado por Shapley e o outro por
        # ganância mediria o critério de ordenação junto com a partição.
        arms["engineered"] = ({n: cohort.blocks[n] for n in ENGINEERED}, None)
    if "unsupervised" in args.arms:
        groups = build_unsupervised(cohort, args.assignment, args.min_group)
        arms["unsupervised"] = (groups, None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows, done = [], set()
    if result_path.exists():
        previous = pd.read_csv(result_path)
        rows = previous.to_dict("records")
        done = set(zip(previous.arm, previous.n_groups, previous.size, previous.fold))
        log(f"retomando {len(rows)} linhas")

    for arm, (blocks, order) in arms.items():
        cached = args.output_dir / f"{arm}_order.json"
        if order is None and cached.exists():
            # A ordenação gulosa custa ~2h no braço de engenharia e não era
            # persistida antes do laço, então qualquer reinício a refazia do
            # zero. Reaproveitar o arquivo torna o script de fato retomável.
            order = list(pd.read_json(cached, typ="series"))
            log(f"{arm}: ordem lida de {cached.name}")
        if order is None:
            log(f"{arm}: ordem gulosa (maior treino, fold 1)")
            train, test = splits[0]
            order = greedy_order(blocks, labels, train, test, n_classes)
            (args.output_dir / f"{arm}_order.json").write_text(
                pd.Series(order).to_json(orient="values"), encoding="utf-8")
        log(f"{arm}: ordem = {' -> '.join(order)}")

        for fold, (train, test) in enumerate(splits, start=1):
            rng = np.random.default_rng(SEED + fold)
            for size in SIZES:
                subset = subsample(train, labels, size, rng)
                label = size or len(train)
                for count in range(1, len(order) + 1):
                    key = (arm, count, label, fold)
                    if key in done:
                        continue
                    values = np.hstack([blocks[n] for n in order[:count]])
                    started = time.perf_counter()
                    score = probe(values, labels, subset, test, n_classes)
                    rows.append({"arm": arm, "n_groups": count, "size": label,
                                 "fold": fold, "dims": values.shape[1],
                                 "n_train": len(subset), "bacc": 100 * score,
                                 "error": 100 * (1 - score),
                                 "elapsed_seconds": time.perf_counter() - started})
                    done.add(key)
                    pd.DataFrame(rows).to_csv(result_path, index=False,
                                              float_format="%.6f")
                    log(f"  {arm} | fold {fold} | n={label:>6,} | {count:2d} grupos "
                        f"| {values.shape[1]:5d}d | BAcc {100 * score:5.2f} "
                        f"| {rows[-1]['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    log("\n=== expoente do erro (erro ∝ n^−c), por braço e nº de grupos ===")
    for arm, group in table.groupby("arm"):
        for count, piece in group.groupby("n_groups"):
            curve = piece.groupby("size")["error"].mean()
            if len(curve) < 3:
                continue
            slope = np.polyfit(np.log(curve.index.to_numpy(dtype=float)),
                               np.log(curve.to_numpy()), 1)[0]
            log(f"  {arm:<13} {count:2d} grupos: c = {-slope:.4f}  "
                f"(erro final {curve.iloc[-1]:.2f}%)")
    log(f"\nSalvo em: {result_path}")


if __name__ == "__main__":
    main()
