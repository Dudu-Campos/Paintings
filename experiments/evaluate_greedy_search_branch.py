"""Busca gulosa para frente sobre as doze views, com o `BranchMLP` de producao.

A enumeracao completa (2^12 x 5 folds) custaria 256 h com esta cabeca -- medido,
45 s por ajuste. O guloso custa 390 ajustes (~4,9 h) e responde a pergunta que
interessa: **uma busca com a cabeca de producao passa da configuracao de
producao?** Difere da cadeia de `evaluate_cumulative_twelve.py` porque la a
ordem e FIXA (Shapley) e aqui ela e escolhida a cada passo.

DUAS GUARDAS CONTRA SUPERAJUSTE DE SELECAO, e sem elas o resultado nao presta:

1. **Escolhe pelo `val`, reporta o `test`.** Com 5 folds e sigma ~1,2pp, pegar o
   maximo entre ~11 candidatos captura ruido: o vencedor tende a ser o que teve
   sorte, e a curva sai otimista por construcao. O `fit` ja calcula a BAcc de
   validacao e a descartava; aqui ela vira o criterio, e o teste so e lido depois
   que a escolha esta feita.
2. **Registra a MARGEM de cada escolha.** Se o 1o e o 2o candidato diferem por
   0,2pp com sigma de 1,2, a escolha e arbitraria e isso tem de aparecer -- o
   passo vira "qualquer um destes", nao "este".

O grafo de similaridade entra como CANAL, nao como bloco: um dos candidatos de
cada passo e "ligar o `logit-add`", que custa zero dimensoes. Assim a busca
escolhe sozinha entre concatenar e somar aos logits, que e a decisao que o §5.6
formalizou como regra -- e o resultado dela e um teste independente da regra.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_logit_add_multitask import build_oof, train_arm  # noqa: E402
from evaluate_shapley_twelve_views import DORMANT, HAND8, N_FOLDS, SEED  # noqa: E402
from registry import load_blocks, log  # noqa: E402

CONCAT = list(HAND8) + DORMANT + ["Text"]
CHANNEL = "graph (logit-add)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/greedy_search_branch.csv")
    args = parser.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes")

    blocks = {n: cohort.blocks[n].astype(np.float32) for n in HAND8}
    for name in DORMANT:
        values_path, stems_path = EXTERNAL[name]
        values = np.load(ROOT / values_path)
        order = pd.Index(np.load(ROOT / stems_path, allow_pickle=True)
                         .astype(str)).get_indexer(stems)
        if (order < 0).any():
            raise SystemExit(f"{name}: {int((order < 0).sum())} stems ausentes")
        blocks[name] = values[order].astype(np.float32)

    cache_dir = ROOT / "outputs/shapley_twelve_views"
    space = np.hstack([blocks[n] for n in HAND8]).astype(np.float32)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(stems, labels))[:args.folds]

    # Pre-computa, por fold, o texto (cache do Shapley) e o log-lift do grafo.
    per_fold, lifts, inner = {}, {}, {}
    for fold, (outer, test) in enumerate(splits, start=1):
        cache = cache_dir / f"fold_{fold}.npz"
        if not cache.exists():
            raise SystemExit(f"cache ausente: {cache}")
        blob = dict(blocks)
        blob["Text"] = np.load(cache)["text"].astype(np.float32)
        per_fold[fold] = blob
        fit, validation = train_test_split(
            outer, test_size=0.10, stratify=labels[outer], random_state=SEED)
        inner[fold] = (fit, validation, test)
        started = time.perf_counter()
        scaled = StandardScaler().fit(space[outer]).transform(space).astype(np.float32)
        oof, held = build_oof(scaled[outer], labels[outer], scaled[test],
                              n_classes, args.k)
        lift = np.zeros((len(labels), n_classes), dtype=np.float32)
        lift[outer], lift[test] = oof, held
        lifts[fold] = lift
        log(f"  fold {fold}: log-lift em {time.perf_counter() - started:.0f}s")

    def evaluate(names, use_channel):
        """Media sobre folds da BAcc de VALIDACAO e da de TESTE, nessa ordem."""
        val, test = [], []
        for fold in per_fold:
            fit, validation, test_idx = inner[fold]
            metrics, _ = train_arm([per_fold[fold][n] for n in names],
                                   lifts[fold] if use_channel else None, labels,
                                   fit, validation, test_idx, n_classes,
                                   np.zeros(n_classes, dtype=np.int64) if use_channel else None,
                                   SEED, args.epochs, args.patience)
            val.append(metrics["val_bacc"] if "val_bacc" in metrics
                       else metrics.get("best_val_bacc", np.nan))
            test.append(metrics["bacc"])
        return float(np.mean(val)), float(np.mean(test)), float(np.std(test))

    chosen, channel_on, rows = [], False, []
    if args.output.exists():
        log(f"saida ja existe: {args.output} -- apague para refazer")
    step = 0
    while True:
        candidates = [(n, False) for n in CONCAT if n not in chosen]
        if not channel_on:
            candidates.append((CHANNEL, True))
        if not candidates:
            break
        step += 1
        scored = []
        for name, is_channel in candidates:
            names = chosen if is_channel else chosen + [name]
            if not names:          # o canal sozinho nao e configuracao valida
                continue
            started = time.perf_counter()
            val, test, sd = evaluate(names, channel_on or is_channel)
            scored.append(dict(step=step, candidate=name, val=val, test=test,
                               sd=sd, seconds=round(time.perf_counter() - started, 1)))
            log(f"  passo {step:2d} candidato {name:<18} val {val:5.2f} "
                f"test {test:5.2f} ± {sd:4.2f} | {scored[-1]['seconds']:.0f}s")
        scored.sort(key=lambda r: -r["val"])
        best, second = scored[0], (scored[1] if len(scored) > 1 else None)
        best["selected"] = True
        best["margin"] = best["val"] - second["val"] if second else np.nan
        best["runner_up"] = second["candidate"] if second else ""
        rows.extend(scored)
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
        if best["candidate"] == CHANNEL:
            channel_on = True
        else:
            chosen.append(best["candidate"])
        log(f"  >> passo {step}: {best['candidate']} (margem {best['margin']:+.2f} "
            f"sobre {best['runner_up']}) | test {best['test']:.2f}")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
