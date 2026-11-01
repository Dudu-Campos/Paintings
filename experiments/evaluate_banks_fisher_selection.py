"""Selecao por Fisher ANINHADA NO FOLD, nos dois bancos. Simetrica por construcao.

Duas razoes independentes, e nenhuma delas e "para ganharmos":

1. **FIDELIDADE AO METODO DELES.** Shamir et al. (2010) rejeitam os 85% mais
   fracos por escore de Fisher -- *"resulting in a feature space of 548 image
   features"*. O `extract_wndcharm_tiled_features.py` declara que *"a selecao de
   Fisher e o WND ficam no avaliador, porque tem de ser ajustados DENTRO de cada
   fold"*, e o avaliador do §5.19 **nao faz**. O WND-CHARM da tabela roda em modo
   nao-canonico, com 2.895 colunas.
2. **O §5.15 ja mediu que poda ajuda o NOSSO banco**: remover as 256 piores de
   639 por permutation importance da +0,60pp (3/3) sob MLP, contra -0,85 de
   remover 256 aleatorias.

E a direcao do efeito NAO e obvia a nosso favor -- pelo contrario. Com n=38.636
de treino, a razao de estimacao do Σ e 13:1 no WND-CHARM (d=2.895) contra 63:1
no hand-8 (d=617). Cortar o lado deles para 548 leva a 70:1, ou seja **condiciona
muito melhor o banco deles do que o nosso**. Este experimento pode piorar a nossa
posicao, e e por isso que ele passa o teste do §4.1f (adotariamos se nao
ajudasse).

**ANINHADO E OBRIGATORIO.** Escolher feature na coorte inteira e o mecanismo do
§6 que derrubou o SNF: o escore de Fisher usa o rotulo, entao calcular fora do
fold vaza o teste. Aqui o escore sai so das linhas de TREINO de cada fold.

Orcamentos varridos em fracao, nao em contagem absoluta, porque os bancos tem
dimensionalidade 4,7x diferente -- fixar 548 nos dois seria 19% do WND-CHARM e
89% do hand-8, que nao e a mesma intervencao. O 15% do artigo entra como ponto.

Uso:
    python experiments/evaluate_banks_fisher_selection.py --dataset rijksdraw
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                                      # noqa: E402
from registry import load_blocks                                # noqa: E402
from evaluate_handcrafted_banks_lda import (                    # noqa: E402
    fit_probe, pooled_wndcharm)

SEED, N_FOLDS = 42, 5
FRACOES = [0.15, 0.30, 0.50, 1.00]      # 0,15 = os "548 de 2.895" do artigo


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fisher(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Escore de Fisher por feature: variancia ENTRE classes / variancia DENTRO.

    So das linhas passadas (o treino do fold). Classe com 1 linha nao contribui
    para a variancia dentro; colunas sem variancia nenhuma recebem escore 0 e
    caem no fim da ordenacao.
    """
    classes = np.unique(y)
    mu = X.mean(axis=0)
    entre = np.zeros(X.shape[1])
    dentro = np.zeros(X.shape[1])
    for c in classes:
        m = y == c
        n_c = int(m.sum())
        xc = X[m]
        entre += n_c * (xc.mean(axis=0) - mu) ** 2
        if n_c > 1:
            dentro += n_c * xc.var(axis=0)
    return np.divide(entre, dentro, out=np.zeros_like(entre), where=dentro > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="rijksdraw")
    ap.add_argument("--variante", default="shrinkage", choices=["svd", "shrinkage"])
    ap.add_argument("--fracoes", nargs="+", type=float, default=FRACOES)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/banks_fisher_selection.csv")
    args = ap.parse_args()
    ds = args.dataset

    c = load_blocks(ds, list(HAND8))
    stems = list(c.stems)
    y = LabelEncoder().fit_transform(np.asarray(c.labels, dtype=str))
    C = len(np.unique(y))
    bancos = {"hand-8": np.hstack([c.blocks[b] for b in c.columns])}
    wnd = pooled_wndcharm(ds, stems)
    if wnd is None:
        raise SystemExit(f"{ds}: sem WND-CHARM")
    bancos["WND-CHARM"] = wnd
    log(f"{ds}: {len(stems):,} imagens | {C} classes | "
        + ", ".join(f"{k} {v.shape[1]}d" for k, v in bancos.items()))

    rows = pd.read_csv(args.output).to_dict("records") if args.output.exists() else []
    done = {(r["dataset"], r["banco"], float(r["fracao"]), int(r["fold"]))
            for r in rows}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        for nome, X in bancos.items():
            # o escore de Fisher sai do mesmo pre-processamento que a sonda ve,
            # senao ele ordena colunas numa escala e o LDA recebe outra
            Z = np.where(np.isfinite(X), X, np.nan)
            imp = SimpleImputer(strategy="median",
                                keep_empty_features=True).fit(Z[tr])
            Z = imp.transform(Z)
            ordem = np.argsort(-fisher(Z[tr], y[tr]))
            for fr in args.fracoes:
                if (ds, nome, float(fr), fold) in done:
                    continue
                k = max(C, int(round(fr * X.shape[1])))
                sel = np.sort(ordem[:k]) if fr < 1.0 else np.arange(X.shape[1])
                t0 = time.perf_counter()
                m, _, _ = fit_probe(X[:, sel], y, tr, te, args.variante, C)
                rows.append({"dataset": ds, "banco": nome, "fracao": float(fr),
                             "fold": fold, "variante": args.variante,
                             "dims": int(len(sel)), "dims_totais": X.shape[1],
                             "seconds": time.perf_counter() - t0, **m})
                pd.DataFrame(rows).to_csv(args.output, index=False,
                                          float_format="%.4f")
                log(f"  f{fold} {nome:<10} fr={fr:.2f} {len(sel):>5}d "
                    f"({m['dims_vivas']:>5} vivas)  BAcc {m['bacc']:5.2f}  "
                    f"Acc {m['accuracy']:5.2f}  ({rows[-1]['seconds']:.0f}s)")

    t = pd.DataFrame(rows)
    t = t[t.dataset.eq(ds) & t.variante.eq(args.variante)]
    log(f"\n=== {ds}, {args.variante}: Fisher aninhado, os dois bancos ===")
    log(f"{'fracao':>8}{'dims h8':>9}{'hand-8':>18}{'dims wnd':>10}"
        f"{'WND-CHARM':>18}{'delta BAcc':>12}")
    for fr in sorted(t.fracao.unique()):
        g = t[t.fracao.eq(fr)]
        a = g[g.banco.eq("hand-8")].set_index("fold")
        b = g[g.banco.eq("WND-CHARM")].set_index("fold")
        if not len(a) or not len(b):
            continue
        x, yy = a.bacc.align(b.bacc, join="inner")
        d = x - yy
        log(f"{fr:>8.2f}{int(a.dims.iloc[0]):>9}"
            f"{a.bacc.mean():>11.2f} +-{a.bacc.std():.2f}"
            f"{int(b.dims.iloc[0]):>10}"
            f"{b.bacc.mean():>11.2f} +-{b.bacc.std():.2f}"
            f"{d.mean():>+9.2f} {(d>0).sum()}/{len(d)}")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
