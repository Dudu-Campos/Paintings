#!/usr/bin/env python3
"""Shapley e LOO das nove views (hand-8 + grafo) a partir das mascaras em disco.

POR QUE ESTE INSTRUMENTO. Julgar a inclusao do grafo por `hand-8 + grafo` e usar
o termo LOO -- o marginal na coalizao CHEIA, o mais adverso do somatorio. O
§5.10 mede o vies: nenhum bloco do banco passa de 0,7pp de LOO enquanto os
Shapley vao de 1,2 a 6,1, e no §5.9c o grafo tem o MAIOR Shapley das doze views
(6,11) com LOO de 0,90 -- 85% do valor dele esta em coalizoes menores. Quanto
mais saturado o banco, menos o LOO enxerga.

`substituicao = Shapley - LOO` sai de graca e e a quantidade que responde
"quanto deste bloco os outros ja cobrem".
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                      # noqa: E402
from view_names import display                 # noqa: E402

VIEWS = list(HAND8) + ["Similarity graph"]
ORD = ["oilpainting", "rijksdraw", "multitask", "minc2500", "food101", "places_lt"]


def shapley(valores: dict[int, float], n: int) -> np.ndarray:
    phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        for m, v in valores.items():
            if m & bit:
                continue
            k = bin(m).count("1")
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            phi[i] += w * (valores[m | bit] - v)
    return phi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/shapley_hand8_grafo.csv")
    args = ap.parse_args()
    n = len(VIEWS)
    cheia = 2 ** n - 1
    linhas = []
    for ds in ORD:
        f = ROOT / f"results/{ds}_shapley_hand8_grafo.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if d["mask"].nunique() < 2 ** n:
            print(f"{ds}: {d['mask'].nunique()} de {2**n} mascaras -- PARCIAL, pulando")
            continue
        phis, loos = [], []
        for fold, g in d.groupby("fold"):
            v = dict(zip(g["mask"].astype(int), g.bacc.astype(float)))
            if len(v) < 2 ** n:
                continue
            phis.append(shapley(v, n))
            loos.append([v[cheia] - v[cheia ^ (1 << i)] for i in range(n)])
        if not phis:
            continue
        phi, loo = np.array(phis), np.array(loos)
        print(f"\n=== {ds} ===  {len(phis)} particao(oes), completo {d[d['mask'].eq(cheia)].bacc.mean():.2f}")
        print(f"{'view':22s} {'Shapley':>9s} {'+-':>6s} {'LOO':>8s} {'substituicao':>13s}")
        ordem = np.argsort(-phi.mean(axis=0))
        for i in ordem:
            s_, l_ = phi[:, i].mean(), loo[:, i].mean()
            marca = "  <--" if VIEWS[i] == "Similarity graph" else ""
            print(f"{display(VIEWS[i]):22s} {s_:9.3f} {phi[:, i].std(ddof=1) if len(phis)>1 else 0:6.3f} "
                  f"{l_:8.3f} {s_-l_:13.3f}{marca}")
            linhas.append({"dataset": ds, "view": VIEWS[i], "shapley": s_,
                           "shapley_sd": phi[:, i].std(ddof=1) if len(phis) > 1 else np.nan,
                           "loo": l_, "substituicao": s_ - l_, "particoes": len(phis)})
        soma, dif = phi.mean(axis=0).sum(), d[d["mask"].eq(cheia)].bacc.mean() - d[d["mask"].eq(0)].bacc.mean()
        print(f"  eficiencia: soma dos Shapley {soma:.4f} contra completo-vazio {dif:.4f} "
              f"(erro {abs(soma-dif):.6f})")
    if linhas:
        pd.DataFrame(linhas).to_csv(args.output, index=False, float_format="%.4f")
        print(f"\n[gravado em {args.output.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
