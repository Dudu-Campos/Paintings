"""A escolha da SONDA LINEAR é neutra entre banco hand-designed e embedding de CNN?

NAO E, e esta medicao e o que delimita onde o LDA pode ser usado.

O contexto: a objecao de que a arquitetura do `BranchMLP` pode ser confundidor
nas comparacoes entre views levou a proposta de avaliar o banco hand-designed
com LDA. O §4.1e ja mede que a estrutura de branches NAO interage (tirar as
branches inteiras muda -0,11pp, p=0,60; dar largura proporcional PIORA 0,87pp,
p=0,011), e o §8 registra que LDA e MLP divergem ~9pp em vetor multi-bloco. Mas
a pergunta que faltava era outra: trocar de sonda muda a DISTANCIA entre as duas
familias de representacao?

Se muda, nenhuma sonda pode atravessar familias sem que a escolha dela vire
parte do resultado -- e a saida e confinar o LDA a comparacoes DENTRO da familia
hand-designed, onde o §5.9d ja o validou (ordenacao de views conservada nos cinco
classificadores, Spearman +0,939 a +0,988; `sozinha (LDA) x Shapley (LDA)` = +1,000).

Tres sondas lineares, duas representacoes, mesmos folds:
  * LDA svd          -- a de producao (`protocol.evaluate_lda`), sem `n_components`,
                        entao usada como CLASSIFICADOR sobre todas as d features
  * LDA lsqr+shrinkage -- Ledoit-Wolf; conserta covariancia mal condicionada, e NAO
                        levanta o teto de C-1, que e estrutural ao criterio de Fisher
  * logistica multinomial -- mesma familia linear, sem a hipotese gaussiana de
                        covariancia compartilhada; e a sonda que o §5.18c queria
                        para o canal de texto, por emitir probabilidade calibrada

Uso:
    python experiments/evaluate_probe_family_bias.py --datasets oilpainting rijksdraw
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from corpora import HAND8                     # noqa: E402
from registry import load_blocks              # noqa: E402

SEED = 42


def log(m: str) -> None:
    print(m, flush=True)


def sondas(n_classes: int) -> dict:
    return {
        "LDA svd (producao)": lambda: LDA(priors=np.ones(n_classes) / n_classes),
        "LDA lsqr + shrinkage": lambda: LDA(solver="lsqr", shrinkage="auto",
                                            priors=np.ones(n_classes) / n_classes),
        "logistica multinomial": lambda: LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["oilpainting", "rijksdraw"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "probe_family_bias.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    for ds in args.datasets:
        coh = load_blocks(ds, list(HAND8) + ["ResNet50"], allow_external=True)
        y = LabelEncoder().fit_transform(np.asarray(coh.labels, dtype=str))
        C = len(np.unique(y))
        reps = {"hand-8": np.hstack([np.asarray(coh.blocks[n]) for n in HAND8]),
                "ResNet50 congelada": np.asarray(coh.blocks["ResNet50"])}
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(y)), y))[:args.folds]
        log(f"\n{ds}: {len(y):,} imagens, {C} classes, {len(splits)} folds")
        for fold, (tr, te) in enumerate(splits, start=1):
            for rn, X in reps.items():
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                for pn, mk in sondas(C).items():
                    t0 = time.perf_counter()
                    pred = mk().fit(Xtr, y[tr]).predict(Xte)
                    rows.append({"dataset": ds, "fold": fold, "rep": rn,
                                 "dims": X.shape[1], "sonda": pn,
                                 "bacc": 100 * balanced_accuracy_score(y[te], pred),
                                 "seconds": time.perf_counter() - t0})
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
            log(f"  fold {fold} ok")

    t = pd.DataFrame(rows)
    log("\n=== BAcc media por sonda e representacao ===")
    piv = t.pivot_table(index=["dataset", "sonda"], columns="rep", values="bacc")
    piv["hand-8 menos ResNet"] = piv["hand-8"] - piv["ResNet50 congelada"]
    log(piv.round(2).to_string())

    log("\n=== A SONDA MUDA A DISTANCIA ENTRE AS FAMILIAS? ===")
    for ds, sub in piv.groupby(level=0):
        d = sub["hand-8 menos ResNet"]
        log(f"  {ds:12s} faixa {d.min():+6.2f} a {d.max():+6.2f}  "
            f"amplitude {d.max() - d.min():5.2f}pp")
    log("\n=== e DENTRO de cada representacao, quanto a sonda move? ===")
    for (ds, rep), sub in t.groupby(["dataset", "rep"]):
        g = sub.groupby("sonda")["bacc"].mean()
        log(f"  {ds:12s} {rep:20s} faixa {g.min():6.2f} a {g.max():6.2f}  "
            f"amplitude {g.max() - g.min():5.2f}pp")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
