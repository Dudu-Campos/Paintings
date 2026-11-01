"""Leave-one-out das DEZ views: 8 visuais + grafo de similaridade + texto novo.

O §5.10 tirou LOO de graça das 511 máscaras do fatorial, mas aquele fatorial tem
Multiscale (aposentado no §3) e **não tem nem o grafo nem o texto** -- os dois
blocos que mais mudaram desde então. Aqui o LOO é medido direto: `v(N) - v(N\\i)`
para cada um dos dez.

Decisões de protocolo, todas declaradas porque cada uma muda o que o número
significa:

* **O grafo entra CONCATENADO**, que é a codificação do §9.2 (38,24 em 125
  classes), não o `logit-add` do §5.6. Motivo: o LOO compara remoção de blocos de
  ENTRADA, e o logit-add não é bloco de entrada -- ele soma aos logits. Um LOO
  com logit-add mediria outra coisa.
* **O grafo é reconstruído dentro do `base.fit` em toda condição**, sobre o
  espaço que restou. Então remover uma view visual remove-a também do espaço do
  kNN -- é "remover a view do pipeline", não "remover da entrada do MLP". Custa
  ~133 s por fold por condição, e é o que faz o número ser LOO de verdade.
* **O texto usa OOF aninhado**, calculado UMA vez por fold e reusado nas
  condições -- ele não depende dos outros blocos. Sem o aninhamento os escores das
  linhas de treino vêm de modelos que viram o fold de teste (§5.18, §6).
* **Nada de reimplementar o laço de treino.** O §5.6 registra que uma versão
  própria do `BranchMLP` custou 0,47pp de discrepância; aqui usa-se `base.fit`.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

import evaluate_preprocess_effect_oilpainting as base  # noqa: E402
from build_text_scores import make_stop_words  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_text_marginal_multitask import CAPTIONS, load_captions, nested_text_block  # noqa: E402
from registry import load_blocks, log  # noqa: E402
from view_names import display  # noqa: E402

SEED, N_FOLDS = 42, 5
GRAPH, TEXT = "Similarity graph", "Text"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output", type=Path, default=ROOT / "results/multitask_loo_all_views.csv")
    args = ap.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    labels = LabelEncoder().fit_transform(labels_str)
    n_classes = len(np.unique(labels))
    hand = {n: cohort.blocks[n] for n in HAND8}
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    stop_words = make_stop_words(classes)
    texts = load_captions(CAPTIONS["estudante"], stems)
    log(f"{args.dataset}: {len(stems):,} imagens, {n_classes} classes, "
        f"hand-8 {sum(b.shape[1] for b in hand.values())}d")

    views = list(HAND8) + [GRAPH, TEXT]
    conditions = ["full"] + [f"drop {v}" for v in views]

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    for number, (train_idx, test_idx) in enumerate(folds, 1):
        pending = [c for c in conditions if (c, number) not in done]
        if not pending:
            continue
        text_block = None
        if any(c != f"drop {TEXT}" for c in pending):
            log(f"  fold {number}: OOF aninhado do texto")
            text_block = nested_text_block(texts, labels_str, classes, index_of,
                                           stop_words, train_idx, test_idx, number)
        for condition in pending:
            drop = condition[5:] if condition.startswith("drop ") else None
            names = [n for n in HAND8 if n != drop]
            blocks = [hand[n] for n in names]
            if drop != TEXT:
                blocks = blocks + [text_block]
            source = np.hstack([hand[n] for n in names]) if drop != GRAPH else None
            started = time.perf_counter()
            metrics = base.fit(blocks, labels, train_idx, test_idx, n_classes, SEED,
                               args.epochs, args.patience, "narrow", source)
            row = {"condition": condition, "dropped": drop or "", "fold": number,
                   "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{condition:28s} fold {number} | {row['dimensions']:5d}d | "
                f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    if "full" in pivot.columns and pivot.notna().all().all():
        from scipy import stats
        full = pivot["full"].to_numpy()
        rows = []
        for v in views:
            x = pivot[f"drop {v}"].to_numpy()
            loo = full - x                      # quanto se perde ao remover
            _, p = stats.ttest_rel(full, x)
            rows.append((display(v), float(loo.mean()), float(loo.std(ddof=1)),
                         p, int((loo > 0).sum())))
        rows.sort(key=lambda r: -r[1])
        print(f"\npipeline completo: {full.mean():.2f} BAcc\n")
        print(f"{'view':22}{'LOO':>7}{'±':>7}{'p':>8}{'folds':>8}")
        for name, m, sd, p, k in rows:
            print(f"{name:22}{m:+7.2f}{sd:7.2f}{p:8.4f}{k:5d}/5")


if __name__ == "__main__":
    main()
