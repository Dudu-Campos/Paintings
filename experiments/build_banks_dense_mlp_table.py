"""Tabela dos quatro bancos sob MLP SEM BRANCH, nas TRES metricas.

Agrega `results/<ds>_banks_dense_mlp.csv` e nao recalcula nada.

**Por que sem branch.** O `BranchMLP` de producao da a cada bloco 128 dims e a
cabeca fica com `128 * n_blocos`, entao o `hand-8` recebe **8x a largura de
cabeca** (1024x512 = 524 mil pesos na primeira camada contra 66 mil dos outros
tres) so por ser declarado como 8 blocos. Isso nao e propriedade do banco, e de
como escrevemos o `HAND8`. Com um bloco unico a rede e IDENTICA nos quatro --
`Linear(d,128) -> 512-256-128 -> C` -- e o que resta diferente e `d`, que E os
bancos diferindo. O §4.1e, citado no §7.19 para dizer que o confundidor e
inerte, mede outra coisa: largura extra em UM bloco, nunca 8 branches contra 1.

**A barra e o desvio ENTRE SEMENTES** (convencao do §9.0), com a media por
semente calculada sobre os folds ANTES de agregar -- a semente e a replicata
mais independente, e com MLP ela e informativa (ruido de treino existe: o §9.7
registra o mesmo `hand-8` em 38,53 / 38,71 / 38,74). Isso e o oposto do LDA,
onde a semente e inerte (§5.19).

**AS TRES METRICAS JUNTAS, nao a BAcc sozinha.** O §5.19 mediu que sob LDA a
BAcc INVERTE de sinal contra o WND-CHARM em 2 de 6 corpora, entao reportar so
ela seria usar a metrica como escudo (§7.20).

Uso:
    python experiments/build_banks_dense_mlp_table.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BANCOS = ["Haralick", "HOG", "WND-CHARM", "hand-8"]
METRICAS = ["bacc", "accuracy", "macro_f1"]
ROTULO = {"bacc": "BAcc", "accuracy": "Acc", "macro_f1": "macro-F1"}
ORDEM = ["oilpainting", "rijksdraw", "multitask",
         "nct_crc", "minc2500", "food101", "places_lt"]
NOME = {"oilpainting": "OilPainting 17cl", "rijksdraw": "Rijksdraw 11cl",
        "multitask": "Multitask 125cl", "nct_crc": "NCT-CRC ext 9cl",
        "minc2500": "MINC-2500 23cl", "food101": "Food-101 101cl",
        "places_lt": "Places-LT 365cl"}
REF = "hand-8"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arquitetura", default="denso-128")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/banks_dense_mlp_table.csv")
    args = ap.parse_args()

    d = pd.concat([pd.read_csv(f) for f in
                   glob.glob(str(ROOT / "results/*_banks_dense_mlp.csv"))],
                  ignore_index=True)
    d = d[d.arquitetura.eq(args.arquitetura)]

    linhas = []
    for ds in ORDEM:
        g = d[d.dataset.eq(ds)]
        if not len(g):
            continue
        for b in BANCOS:
            h = g[g.banco.eq(b)]
            if not len(h):
                continue
            reg = {"dataset": ds, "arquitetura": args.arquitetura, "banco": b,
                   "dims": int(h.dims.iloc[0]),
                   "dims_vivas": int(h.dims_vivas.iloc[0]),
                   "sementes": int(h.seed.nunique()),
                   "folds": int(h.fold.nunique()), "unidades": len(h),
                   "barra": "sd_sementes"}
            for met in METRICAS:
                # media por semente PRIMEIRO, depois media e desvio entre elas
                por_seed = h.groupby("seed")[met].mean()
                reg |= {met: por_seed.mean(), f"{met}_erro": por_seed.std()}
            linhas.append(reg)
    t = pd.DataFrame(linhas)
    t.to_csv(args.output, index=False, float_format="%.4f")

    bar = "=" * 92
    for met in METRICAS:
        print("\n" + bar)
        print(f"QUATRO BANCOS, MLP SEM BRANCH ({args.arquitetura}) -- {ROTULO[met]}")
        print(bar + "\n")
        print("corpus".ljust(19) + "".join(b.rjust(18) for b in BANCOS)
              + "unid.".rjust(8))
        for ds in ORDEM:
            g = t[t.dataset == ds]
            if not len(g):
                continue
            linha = NOME[ds].ljust(19)
            for b in BANCOS:
                r = g[g.banco == b]
                if not len(r):
                    linha += "--".rjust(18)
                    continue
                r = r.iloc[0]
                marca = "*" if b == REF else " "
                linha += ("%.2f +-%.2f%s" % (r[met], r[met + "_erro"],
                                             marca)).rjust(18)
            print(linha + str(int(g.unidades.max())).rjust(8))

    def faixa(b):
        v = sorted(set(t[t.banco == b].dims_vivas.astype(int)))
        if not v:
            return "--"
        dd = int(t[t.banco == b].dims.iloc[0])
        corpo = str(v[0]) if len(v) == 1 else "%d-%d" % (v[0], v[-1])
        return corpo if corpo == str(dd) else corpo + " de " + str(dd)
    print("\n" + "dims vivas".ljust(19) + "".join(faixa(b).rjust(18) for b in BANCOS))
    print("* = nosso banco   ·   barra = desvio ENTRE SEMENTES")

    print("\n" + bar)
    print("DELTA DO HAND-8 CONTRA O WND-CHARM -- pareado por unidade (semente, fold)")
    print(bar + "\n")
    print("corpus".ljust(19) + "".join(ROTULO[m].rjust(27) for m in METRICAS))
    for ds in ORDEM:
        g = d[d.dataset.eq(ds)]
        linha = NOME[ds].ljust(19)
        for met in METRICAS:
            a = g[g.banco.eq(REF)].set_index(["seed", "fold"])[met]
            c = g[g.banco.eq("WND-CHARM")].set_index(["seed", "fold"])[met]
            x, y = a.align(c, join="inner")
            if not len(x):
                linha += "--".rjust(27)
                continue
            dd = x - y
            _, p = stats.ttest_rel(x, y)
            linha += ("%+.2f  %d/%d  p=%.0e" % (dd.mean(), (dd > 0).sum(),
                                                len(dd), p)).rjust(27)
        print(linha)
    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
