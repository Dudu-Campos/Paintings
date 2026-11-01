"""WND5 EXATO, do codigo-fonte oficial (github.com/wnd-charm/wnd-charm), nao
aproximacao. Motivado pela pergunta "voce fez a avaliacao oficial deles?" --
os avaliadores anteriores (MLP, kNN) deliberadamente evitavam reimplementar o
WND5 de memoria (risco de formula errada); este script usa a formula lida
direto de `wndcharm/FeatureSpacePrediction.py::NewWND5` e
`wndcharm/FeatureWeights.py::FisherFeatureWeights.NewFromFeatureSpace`.

**DUAS DIFERENCAS do que os scripts anteriores assumiram, achadas no codigo:**

1. **Fisher weight NAO e `sklearn.f_classif`.** E uma formula propria, SEM
   ponderar por tamanho de classe:

       W_f = [ sum_c (mu_f - mu_{c,f})^2 / (n_classes - 1) ] / mean_c(var_{c,f})

   `mu_{c,f}` = media da feature f na classe c; `mu_f` = media geral (sobre
   TODAS as linhas, nao media das medias de classe); `var_{c,f}` = variancia
   amostral (ddof=1) da classe c. O numerador e a soma (nao a media ponderada
   por n_c) das distancias quadraticas ao redor da media geral; o denominador
   e a media SIMPLES (nao ponderada) das variancias por classe. Com
   desbalanceamento de ate 1052x no nosso corpus (§3), isso pesa uma classe de
   10 imagens IGUAL a uma de 10.000 -- oposto do que a ANOVA F padrao faria.

2. **Normalizacao e min-max pra [0,100], NAO z-score.** `mins`/`maxs`
   calculados SO no treino; o teste e CLIPADO a esse intervalo antes de
   escalar. Colunas de amplitude zero no treino viram 0.

**Classificacao (WND5):**

    w = x * W_f                                    (elemento a elemento)
    D(test, train_i) = sum_f (w_test_f - w_train_i_f)^2      (euclidiana ao quadrado)
    sim(test, train_i) = D(test, train_i) ^ -5               (p=-5, empirico do artigo)
    d_{test,c} = media de sim(test, train_i) sobre os train_i da classe c
    marginal_probabilities = d_{test,.} / soma_c d_{test,c}
    predicted = argmax_c marginal_probabilities

**Simplificacao declarada**: o codigo oficial MASCARA pares com distancia
<= epsilon de maquina (protege contra auto-match, ex. avaliar no proprio
treino). Entre imagens DIFERENTES em 2894 features continuas a chance de
distancia exatamente zero e nula por construcao -- a mascara nunca dispararia
aqui, entao ela foi omitida. Ainda assim ha um piso numerico (`clamp`) pra
evitar divisao por zero literal em ponto flutuante.

**Ainda opera sobre features POOLED** (media dos 16 tiles), no MESMO espaco
do `evaluate_wndcharm_tiled_knn.py` -- nao a versao por-tile fiel
(`evaluate_wndcharm_tiled_pertile.py`, que usa MLP). Fazer WND5 exato por-tile
exigiria uma matriz de distancia (n_test*16) x (n_train*16) -- ~1,5 TB em
float32 mesmo em lotes por causa do volume de saida por lote acumulado;
inviavel no orcamento desta sessao. O pooled ja fecha a pergunta da FORMULA;
o per-tile fiel ja fechou a pergunta da AGREGACAO (script separado).

Uso:
    python experiments/evaluate_wndcharm_wnd5.py [--folds 5]
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
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import HAND8, clean                     # noqa: E402
from protocol import SEED, prediction_metrics          # noqa: E402
from registry import load_blocks                       # noqa: E402
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_handcrafted_banks_lda import OFFICIAL, TEST_COHORT  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fisher_weights_official(train_x: np.ndarray, train_y: np.ndarray, n_classes: int) -> np.ndarray:
    """Formula exata de `FisherFeatureWeights.NewFromFeatureSpace` -- NAO
    ponderada por tamanho de classe (ver docstring do modulo)."""
    pop_mean = train_x.mean(axis=0)
    d = train_x.shape[1]
    class_means = np.empty((n_classes, d), dtype=np.float64)
    class_vars = np.empty((n_classes, d), dtype=np.float64)
    for c in range(n_classes):
        rows = train_x[train_y == c]
        class_means[c] = rows.mean(axis=0)
        class_vars[c] = rows.var(axis=0, ddof=1) if len(rows) > 1 else 0.0
    numerator = np.square(pop_mean - class_means).sum(axis=0) / (n_classes - 1)
    denominator = class_vars.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = numerator / denominator
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w.astype(np.float32)


def minmax_fit_apply(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max pra [0,100], min/max SO do treino; teste clipado antes de escalar."""
    mins = train_x.min(axis=0)
    maxs = train_x.max(axis=0)
    span = maxs - mins
    zero = span == 0
    span_safe = np.where(zero, 1.0, span)

    train_scaled = (np.clip(train_x, mins, maxs) - mins) / span_safe * 100.0
    test_scaled = (np.clip(test_x, mins, maxs) - mins) / span_safe * 100.0
    train_scaled[:, zero] = 0.0
    test_scaled[:, zero] = 0.0
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def wnd5_classify(train_w: np.ndarray, train_y: np.ndarray, test_w: np.ndarray,
                  n_classes: int, batch: int = 2000) -> np.ndarray:
    """WND5: distancia euclidiana ao quadrado -> ^-5 -> media por classe ->
    normaliza. Em lotes de TESTE via matmul na GPU (evita materializar a
    matriz densa n_test x n_train inteira)."""
    train_t = torch.from_numpy(train_w).to(DEVICE)
    train_y_t = torch.from_numpy(train_y).long().to(DEVICE)
    train_sq = (train_t ** 2).sum(1)
    onehot = torch.zeros(len(train_y), n_classes, device=DEVICE)
    onehot[torch.arange(len(train_y), device=DEVICE), train_y_t] = 1.0
    class_counts = onehot.sum(0).clamp_min(1.0)

    n_test = test_w.shape[0]
    probs = np.empty((n_test, n_classes), dtype=np.float64)
    for start in range(0, n_test, batch):
        chunk = torch.from_numpy(test_w[start:start + batch]).to(DEVICE)
        test_sq = (chunk ** 2).sum(1, keepdim=True)
        cross = chunk @ train_t.T
        dist = (test_sq + train_sq.unsqueeze(0) - 2 * cross).clamp_min(1e-12)
        sim = dist.pow(-5.0).double()
        class_sum = sim @ onehot.double()
        class_mean = class_sum / class_counts.double().unsqueeze(0)
        total = class_mean.sum(1, keepdim=True).clamp_min(1e-300)
        marginal = class_mean / total
        probs[start:start + chunk.shape[0]] = marginal.cpu().numpy()
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    # o default SEGUE o dataset: fixo em multitask ele silenciaria a troca de
    # corpus (a guarda de stems pegaria, mas depois de 5 min de pooling)
    parser.add_argument("--tiled-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "wndcharm_wnd5_official.csv")
    args = parser.parse_args()

    if args.tiled_dir is None:
        args.tiled_dir = ROOT / "outputs" / f"{args.dataset}_wndcharm_tiled"
    log(f"device: {DEVICE}")
    log(f"coorte de {args.dataset} pelo registry")
    ds = args.dataset
    cohort = load_blocks(ds, list(HAND8))
    labels_raw = np.asarray(cohort.labels, dtype=str)
    stems = [str(s) for s in cohort.stems]
    n_treino = None
    if ds in TEST_COHORT:
        held = load_blocks(TEST_COHORT[ds], list(HAND8), allow_external=True)
        n_treino = len(stems)
        held_stems = [str(s) for s in held.stems]
        labels_raw = np.concatenate([labels_raw, np.asarray(held.labels, dtype=str)])
        log(f"{ds}: coorte EXTERNA {TEST_COHORT[ds]} como teste "
            f"({n_treino:,} + {len(held_stems):,})")
    labels = LabelEncoder().fit_transform(labels_raw)
    n_classes = len(np.unique(labels))

    def poola(d: Path, esperados: list[str]) -> np.ndarray:
        """Media dos 16 tiles, REINDEXADA POR STEM.

        A versao anterior exigia ordem identica e abortava no places_lt, que tem
        os MESMOS 106.300 stems em ordem diferente -- o caso que o §4.1f registra
        no MINC ("sempre alinhar por stem, nunca por posicao"). Exigir igualdade
        de lista e forte demais: o que importa e que todo stem da coorte exista.
        """
        onde = {str(x): i for i, x in enumerate(np.load(d / "stems.npy"))}
        falta = [x for x in esperados if x not in onde]
        if falta:
            raise SystemExit(f"{d.name}: {len(falta)} stems da coorte ausentes "
                             f"(ex. {falta[:3]}) -- abortando (§5.18b)")
        idx = np.array([onde[x] for x in esperados])
        t = np.load(d / "features.npy", mmap_mode="r")
        out = np.empty((len(idx), t.shape[2]), dtype=np.float64)
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]
            out[i:i + len(j)] = np.asarray(t[np.sort(j)], dtype=np.float64).mean(axis=1)[
                np.argsort(np.argsort(j))]
        return out

    log("carregando features.npy e fazendo pooling por media dos tiles")
    started = time.time()
    pooled = poola(args.tiled_dir, stems)
    if ds in TEST_COHORT:
        alvo_dir = args.tiled_dir.parent / f"{TEST_COHORT[ds]}_wndcharm_tiled"
        pooled = np.vstack([pooled, poola(alvo_dir, held_stems)])
    pooled = clean(pooled)
    alive = pooled.std(axis=0) > 0
    pooled = pooled[:, alive]
    log(f"  pooled {pooled.shape} ({time.time()-started:.0f}s)")

    # O RAMO ANTIGO FAZIA StratifiedKFold INCONDICIONALMENTE. Isso e o bug do §8
    # ("constante de protocolo definida e nunca usada"): nos quatro de split
    # publicado ele ignorava a particao oficial, e no places_lt isso troca 996x
    # de desbalanceamento por 41x (§7.5). Os tamanhos vao IMPRESSOS, que e o
    # antidoto prescrito -- conferir contra o numero publicado antes de reportar.
    if ds in TEST_COHORT:
        splits = [(np.arange(n_treino), np.arange(n_treino, len(labels)))]
        tipo = f"coorte externa {TEST_COHORT[ds]}"
    elif ds in OFFICIAL:
        meta = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})
        onde = meta.set_index("stem")["split"].reindex(stems).to_numpy(str)
        if (onde == "nan").any():
            raise SystemExit(f"{ds}: {(onde == 'nan').sum()} stems fora do metadado")
        splits = [(np.flatnonzero(onde == "train"), np.flatnonzero(onde == "test"))]
        tipo = "split OFICIAL"
    else:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels)), labels))[:args.folds]
        tipo = f"CV {len(splits)} folds"
    log(f"  [{tipo}] treino={len(splits[0][0]):,} teste={len(splits[0][1]):,} "
        f"classes={n_classes}")

    rows: list[dict] = []
    for fold, (train, test) in enumerate(splits, start=1):
        started = time.perf_counter()
        train_x, test_x = minmax_fit_apply(pooled[train], pooled[test])
        train_y = labels[train]

        wts = fisher_weights_official(train_x.astype(np.float64), train_y, n_classes)
        log(f"  fold {fold}: pesos de Fisher OK (min {wts.min():.3g} max {wts.max():.3g} "
            f"zeros {int((wts == 0).sum())} de {len(wts)})")

        w_train = (train_x * wts).astype(np.float32)
        w_test = (test_x * wts).astype(np.float32)

        probs = wnd5_classify(w_train, train_y, w_test, n_classes)
        pred = probs.argmax(axis=1)
        metrics = prediction_metrics(labels[test], pred)

        elapsed = time.perf_counter() - started
        rows.append({
            "fold": fold, "dimensions": pooled.shape[1],
            "seconds": elapsed,
            "bacc": 100 * metrics["bacc"], "accuracy": 100 * metrics["accuracy"],
            "macro_f1": 100 * metrics["macro_f1"],
        })
        log(f"  fold {fold}  WND5 oficial  BAcc {rows[-1]['bacc']:5.2f}  ({elapsed:.0f}s)")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    log("\n" + table.round(2).to_string(index=False))
    log(f"\nBAcc media: {table['bacc'].mean():.2f} +- {table['bacc'].std():.2f}")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
