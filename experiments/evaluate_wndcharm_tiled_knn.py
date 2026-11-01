"""kNN (recipe do §5.9d) sobre WND-CHARM tiled: cota inferior mais fiel a WND5
do que o MLP, sem reimplementar o classificador exato do artigo.

Complementa `evaluate_wndcharm_tiled_vs_hand8.py`. A objecao que motivou este
script: o MLP e o classificador de TODO o resto do projeto, mas o metodo do
artigo (Orlov et al. 2008 / Shamir et al. 2010, WND5) e feature+classificador
JUNTOS -- kNN ponderado por escore de Fisher, sem combinacao nao-linear
aprendida. Avaliar so pelo MLP arrisca subestimar o banco deles se o MLP for
pior em explorar aquele espaco de 2895d redundante do que o classificador que
eles desenharam pra ele.

**Nao reimplementa o WND5 exato** (a formula de agregacao por classe nao esta
determinada com certeza suficiente -- risco do §7.11b, "nao chamar de
reproducao o que nao e"). Em vez disso, DUAS variantes de kNN, ambas com a
receita ja validada do `evaluate_standalone_by_classifier.py` (k=30, voto
ponderado por 1/n_c contra desbalanceamento -- e o analogo do class_weight
balanceado que os outros classificadores do projeto usam):

  1. **kNN plano** (Euclidiana padronizada) -- MESMO tratamento que TODAS as
     views do banco ja recebem na figura do §5.9d. Da o ponto de comparacao
     interna: "kNN tambem e ruim pra hand-8?"
  2. **kNN ponderado por Fisher** -- a distancia usa
     `w_f * (x_f - y_f)^2`, com `w_f` = escore de Fisher (ANOVA F) computado
     SO no treino de cada fold. E EXATAMENTE a frase do artigo, "feature
     weights are the Fisher scores", aplicada a uma distancia bem definida
     (equivale a escalar cada coluna por sqrt(w_f) antes da NearestNeighbors
     euclidiana padrao -- sem metrica customizada, sem risco de formula
     errada).

Sem fase de treino em nenhuma das duas -- diferente do MLP, isso amplia o
intervalo de vies indutivo coberto e nao pode "aprender a ignorar" a
redundancia do jeito que um encoder denso poderia.

Uso:
    python experiments/evaluate_wndcharm_tiled_knn.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import HAND8, clean                     # noqa: E402
from protocol import SEED                            # noqa: E402
from registry import load_blocks                      # noqa: E402

K = 30


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                n_classes: int, k: int = K) -> np.ndarray:
    """Mesma receita do `evaluate_standalone_by_classifier.run_knn` (§5.9d):
    voto ponderado por 1/n_c, em lotes (a matriz densa consulta x treino nao
    cabe de uma vez em 125 classes)."""
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    weight = 1.0 / np.maximum(counts, 1.0)
    index = NearestNeighbors(n_neighbors=min(k, len(train_y)), n_jobs=16).fit(train_x)
    prediction = np.empty(len(test_x), dtype=train_y.dtype)
    for start in range(0, len(test_x), 4096):
        chunk = slice(start, start + 4096)
        neighbours = index.kneighbors(test_x[chunk], return_distance=False)
        votes = weight[train_y[neighbours]]
        tally = np.zeros((neighbours.shape[0], n_classes), dtype=np.float64)
        np.add.at(tally, (np.arange(neighbours.shape[0])[:, None], train_y[neighbours]), votes)
        prediction[chunk] = tally.argmax(axis=1)
    return prediction


def metrics_of(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {
        "bacc": 100 * balanced_accuracy_score(truth, prediction),
        "accuracy": 100 * accuracy_score(truth, prediction),
        "macro_f1": 100 * f1_score(truth, prediction, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--tiled-dir", type=Path,
                        default=ROOT / "outputs" / "multitask_wndcharm_tiled")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "wndcharm_tiled_knn.csv")
    args = parser.parse_args()

    log(f"coorte de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset, list(HAND8))
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    stems = [str(s) for s in cohort.stems]
    log(f"  {len(stems)} imagens, {n_classes} classes")

    tiled_stems = [str(s) for s in np.load(args.tiled_dir / "stems.npy")]
    if tiled_stems != stems:
        raise SystemExit("stems.npy nao reproduz cohort.stems -- abortando (§5.18b)")

    log("carregando features.npy e fazendo pooling por media dos tiles")
    started = time.time()
    tiles = np.load(args.tiled_dir / "features.npy")
    pooled = clean(tiles.mean(axis=1).astype(np.float32, copy=False))
    del tiles
    alive = pooled.std(axis=0) > 0
    pooled = pooled[:, alive]
    log(f"  pooled {pooled.shape} ({time.time()-started:.0f}s, "
        f"{int((~alive).sum())} colunas constantes removidas)")

    hand8 = np.hstack([cohort.blocks[name] for name in HAND8]).astype(np.float32)
    log(f"  hand-8 concatenado: {hand8.shape}")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]

    rows: list[dict] = []
    for fold, (train, test) in enumerate(splits, start=1):
        # hand-8, kNN plano -- referencia interna: kNN tambem sofre aqui?
        scaler = StandardScaler().fit(hand8[train])
        train_h, test_h = scaler.transform(hand8[train]), scaler.transform(hand8[test])
        started = time.perf_counter()
        pred = knn_predict(train_h, labels[train], test_h, n_classes)
        m = metrics_of(labels[test], pred)
        rows.append({"condition": "hand-8 (kNN plano)", "fold": fold,
                     "dimensions": hand8.shape[1], "seconds": time.perf_counter() - started,
                     **m})
        log(f"  hand-8 (kNN plano)                    f{fold}  BAcc {m['bacc']:5.2f}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

        # WND-CHARM tiled pooled, kNN plano
        scaler = StandardScaler().fit(pooled[train])
        train_c, test_c = scaler.transform(pooled[train]), scaler.transform(pooled[test])
        started = time.perf_counter()
        pred = knn_predict(train_c, labels[train], test_c, n_classes)
        m = metrics_of(labels[test], pred)
        rows.append({"condition": "WND-CHARM tiled pooled (kNN plano)", "fold": fold,
                     "dimensions": pooled.shape[1], "seconds": time.perf_counter() - started,
                     **m})
        log(f"  WND-CHARM tiled pooled (kNN plano)    f{fold}  BAcc {m['bacc']:5.2f}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

        # WND-CHARM tiled pooled, kNN ponderado por Fisher (so no treino)
        f_values, _ = f_classif(train_c, labels[train])
        f_values = np.nan_to_num(f_values, nan=0.0)
        f_values = np.clip(f_values, 0.0, None)
        sqrt_w = np.sqrt(f_values / max(f_values.mean(), 1e-12))  # normalizado, media 1
        started = time.perf_counter()
        pred = knn_predict(train_c * sqrt_w, labels[train], test_c * sqrt_w, n_classes)
        m = metrics_of(labels[test], pred)
        rows.append({"condition": "WND-CHARM tiled pooled (kNN Fisher-weighted)",
                     "fold": fold, "dimensions": pooled.shape[1],
                     "seconds": time.perf_counter() - started, **m})
        log(f"  WND-CHARM tiled pooled (kNN Fisher-w)  f{fold}  BAcc {m['bacc']:5.2f}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    summary = table.groupby("condition", sort=False).agg(
        dims=("dimensions", "first"), bacc=("bacc", "mean"), sd=("bacc", "std"),
        acc=("accuracy", "mean"), f1=("macro_f1", "mean"), s=("seconds", "mean"))
    log("\n" + summary.round(2).to_string())
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
