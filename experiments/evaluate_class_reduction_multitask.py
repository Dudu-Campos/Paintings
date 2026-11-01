"""Redução POR CLASSE em vez de vizinhança global: o grafo sem inanição.

O §5.6 mediu que o grafo de similaridade perde em 125 classes por **evidência
vazia**: com k=30 vizinhos globais e 125 classes são 0,24 vizinhos por classe,
então a maioria das classes recebe zero e o log-lift delas vira o valor de "sem
evidência". A cura óbvia -- aumentar k -- foi testada em 31/ago e **refutada**
(k=30/60/100 dão −0,70/−0,33/−0,68, sem gradiente).

Este script testa a outra cura. O problema nunca foi *quantidade* de vizinhos,
foi o **critério de seleção**: o top-k global descarta as classes raras antes de
olhar para elas. Reduzindo por classe, cada consulta recebe a distância ao
membro mais próximo de **cada** classe -- C coordenadas sempre preenchidas,
imunes a inanição por construção, para qualquer C.

**O custo é o mesmo, e essa é a razão de a rota ser viável.** Em d=617 nenhuma
árvore ajuda e o sklearn cai para força bruta, materializando a matriz
consulta × treino em lotes; só o top-k sobrevive à redução. Fazendo a redução
nós mesmos, `min por classe` sai da mesma matriz. Muda o que se guarda, não a
complexidade -- continua O(n²d).

Diagnóstico que autorizou a implementação (fold 1, 2.000 consultas): amplitude
relativa `(max−min)/média` dos d_min por consulta = **2,16** (não há
concentração em d=617, que era o risco), e a classe do d_min é a verdadeira em
**32,60%** contra 0,80% de acaso -- 41x. Uma coordenada crua, sem difusão nem
ponderação, já carrega mais sinal que o kNN global extrai em 125 classes.

Três blocos, medidos separadamente de propósito:

    dmin      C dimensões, a distância ao membro mais próximo de cada classe.
              **Zero parâmetros** -- preserva a alegação do §5.2, que é o que dá
              força ao achado do operador.
    dmean-m   C dimensões, média das `m` mais próximas de cada classe. O `m` É
              um parâmetro, da mesma família do k; por isso não entra junto com
              o dmin, senão a alegação de "sem parâmetros" morre por contágio.
    ambos     para medir se o `m` acrescenta ao dmin.

**OOF aninhado é obrigatório**, não opcional: sem ele as linhas de treino da
fusão enxergam o próprio rótulo pelo vizinho mais próximo -- é o mecanismo que
derrubou o SNF para 3,4% (§6) e que zerou a fusão de features contra backbone
finetunada (§4.1c, treino 96,65% contra teste 67,32%). Cada fold externo refaz a
redução em INNER_FOLDS folds internos.

Saída: results/multitask_class_reduction.csv e o resumo pareado no log.
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
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
import evaluate_preprocess_effect_oilpainting as base  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS, INNER_FOLDS = 42, 5, 5
CHUNK = 256


def class_distances(train_x: np.ndarray, train_y: np.ndarray,
                    query_x: np.ndarray, n_classes: int,
                    m: int) -> tuple[np.ndarray, np.ndarray]:
    """`(d_min, d_mean_m)` por classe, ambos (n_query, n_classes).

    Ordenar o treino por rótulo uma vez transforma "as linhas da classe c" num
    intervalo contíguo, então a redução por classe é uma fatia -- sem máscara
    booleana por classe, que custaria C passagens sobre a matriz inteira.
    """
    order = np.argsort(train_y, kind="stable")
    sorted_x, sorted_y = train_x[order], train_y[order]
    bounds = np.searchsorted(sorted_y, np.arange(n_classes + 1))
    squared = (sorted_x ** 2).sum(axis=1)

    minimum = np.empty((len(query_x), n_classes), dtype=np.float32)
    average = np.empty((len(query_x), n_classes), dtype=np.float32)
    for start in range(0, len(query_x), CHUNK):
        block = query_x[start:start + CHUNK]
        # ||a-b||² expandido: a matriz completa do bloco, que é exatamente o que
        # a força bruta do sklearn já calcularia para o top-k.
        d2 = squared[None, :] - 2.0 * (block @ sorted_x.T) + (block ** 2).sum(axis=1)[:, None]
        np.maximum(d2, 0.0, out=d2)
        for label in range(n_classes):
            low, high = bounds[label], bounds[label + 1]
            if high <= low:
                # Classe ausente do treino interno: marcada e imputada abaixo
                # pelo pior caso observado, nunca por zero -- zero seria "vizinho
                # colado", o oposto da verdade.
                minimum[start:start + len(block), label] = np.nan
                average[start:start + len(block), label] = np.nan
                continue
            piece = d2[:, low:high]
            take = min(m, piece.shape[1])
            if take == 1:
                nearest = piece.min(axis=1)
                minimum[start:start + len(block), label] = nearest
                average[start:start + len(block), label] = nearest
            else:
                partial = np.partition(piece, take - 1, axis=1)[:, :take]
                minimum[start:start + len(block), label] = partial.min(axis=1)
                average[start:start + len(block), label] = partial.mean(axis=1)
    minimum, average = np.sqrt(minimum), np.sqrt(average)
    for values in (minimum, average):
        holes = np.isnan(values)
        if holes.any():
            values[holes] = np.nanmax(values)
    return minimum, average


def standardise_rows(values: np.ndarray) -> np.ndarray:
    """Normaliza cada consulta pela própria escala.

    Consultas em regiões densas têm todas as distâncias menores; sem isto o
    bloco codificaria densidade local em vez de identidade de classe. Medido no
    diagnóstico: a dispersão ENTRE consultas (9,27) é maior que a dispersão
    ENTRE classes dentro de uma consulta (5,52), então a escala por linha é o
    termo dominante e precisa sair.
    """
    centre = values.mean(axis=1, keepdims=True)
    spread = values.std(axis=1, keepdims=True)
    return ((values - centre) / np.maximum(spread, 1e-6)).astype(np.float32)


def build_block(fit_x, fit_y, target_x, n_classes, m, variant):
    """OOF aninhado: cada fold interno prevê o seu próprio recorte de treino."""
    def encode(train_x, train_y, query_x):
        minimum, average = class_distances(train_x, train_y, query_x, n_classes, m)
        parts = []
        if variant in ("dmin", "both"):
            parts.append(standardise_rows(minimum))
        if variant in ("dmean", "both"):
            parts.append(standardise_rows(average))
        return np.hstack(parts)

    oof = None
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        values = encode(fit_x[inner_train], fit_y[inner_train], fit_x[inner_val])
        if oof is None:
            oof = np.zeros((len(fit_x), values.shape[1]), dtype=np.float32)
        oof[inner_val] = values
    assert oof is not None
    return oof, encode(fit_x, fit_y, target_x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multitask_class_reduction.csv")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    blocks = [cohort.blocks[name] for name in cohort.blocks]
    space = np.hstack(blocks)
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"hand-8 {space.shape[1]}d, m={args.m}")

    conditions = [("hand-8", None), ("+ dmin por classe", "dmin"),
                  (f"+ dmean-{args.m} por classe", "dmean"),
                  (f"+ dmin & dmean-{args.m}", "both")]
    rows = []
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        log(f"retomando {len(rows)} linhas")
    done = {(r["condition"], r["fold"]) for r in rows}

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    for fold, (train, test) in enumerate(splits, start=1):
        scaler = StandardScaler().fit(space[train])
        scaled = scaler.transform(space).astype(np.float32)
        for condition, variant in conditions:
            if (condition, fold) in done:
                continue
            started = time.perf_counter()
            extra = None
            if variant is not None:
                oof, held = build_block(scaled[train], labels[train], scaled[test],
                                        n_classes, args.m, variant)
                extra = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
                extra[train], extra[test] = oof, held
            chosen = blocks + ([extra] if extra is not None else [])
            metrics = base.fit(chosen, labels, train, test, n_classes, SEED,
                               args.epochs, args.patience)
            rows.append({"condition": condition, "fold": fold,
                         "dimensions": metrics["dimensions"],
                         "elapsed_seconds": time.perf_counter() - started,
                         **{k: v for k, v in metrics.items() if k != "dimensions"}})
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
            log(f"  {condition:<26} fold {fold} | {metrics['dimensions']:5d}d "
                f"| BAcc {metrics['bacc']:.2f} | {rows[-1]['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    reference = table[table.condition == "hand-8"].set_index("fold")["bacc"]
    log("\n=== marginal sobre hand-8 (5 folds, pareado) ===")
    for condition, _ in conditions:
        current = table[table.condition == condition].set_index("fold")["bacc"]
        line = f"  {condition:<26} {current.mean():6.2f} ± {current.std(ddof=1):4.2f}"
        if condition != "hand-8" and len(current) == len(reference) == N_FOLDS:
            delta = (current - reference).dropna()
            line += (f"   Δ {delta.mean():+5.2f} ± {delta.std(ddof=1):4.2f}pp"
                     f"  p={stats.ttest_rel(current, reference).pvalue:.4f}"
                     f"  {int((delta > 0).sum())}/{len(delta)}")
        log(line)
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
