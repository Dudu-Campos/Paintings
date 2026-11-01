"""Tabela final dos quatro bancos hand-designed, com incerteza nas sete linhas.

Agrega os dois produtores e NAO recalcula nada:
  results/<ds>_handcrafted_banks_lda.csv       ponto e dispersao entre folds
  results/handcrafted_banks_bootstrap*.csv     erro padrao da amostra de teste

**AS DUAS BARRAS NAO SAO A MESMA QUANTIDADE, e a coluna `barra` declara qual
e.** Nos tres de arte e o DESVIO entre as 5 particoes de uma coorte (convencao
do §2); nos quatro de split fixo e o ERRO PADRAO da BAcc sob a particao
publicada, por reamostragem estratificada do teste -- ali a particao nao varia e
o LDA e fechado, entao nao existe dispersao entre execucoes para medir.
Imprimir as duas sob um `+-` generico seria o erro de metrica do §4.1f.

Guarda: aborta se o ponto do bootstrap divergir do CSV do produtor acima de
0,01pp em qualquer celula -- as duas fontes tem de concordar por construcao.

Uso:
    python experiments/build_handcrafted_banks_table.py
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

ROOT = Path(__file__).resolve().parents[1]
BANCOS = ["Haralick", "HOG", "WND-CHARM", "hand-8"]
ARTE = ["oilpainting", "rijksdraw", "multitask"]
OFIC = ["nct_crc", "minc2500", "food101", "places_lt"]
REF = "hand-8"
METRICAS = ["bacc", "accuracy", "macro_f1"]
ROTULO_MET = {"bacc": "BAcc", "accuracy": "Acc", "macro_f1": "macro-F1"}
ROTULO = {"oilpainting": "OilPainting 17cl", "rijksdraw": "Rijksdraw 11cl",
          "multitask": "Multitask 125cl", "nct_crc": "NCT-CRC ext 9cl",
          "minc2500": "MINC-2500 23cl", "food101": "Food-101 101cl",
          "places_lt": "Places-LT 365cl"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variantes", nargs="+", default=["shrinkage", "svd"])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/handcrafted_banks_table.csv")
    args = ap.parse_args()

    cv = pd.concat([pd.read_csv(p) for p in
                    glob.glob(str(ROOT / "results/*_handcrafted_banks_lda.csv"))],
                   ignore_index=True)
    bo = pd.read_csv(ROOT / "results/handcrafted_banks_bootstrap.csv")
    pa = pd.read_csv(ROOT / "results/handcrafted_banks_bootstrap_pares.csv")

    linhas = []
    for var in args.variantes:
        for ds in ARTE + OFIC:
            for b in BANCOS:
                g = cv[(cv.dataset == ds) & (cv.banco == b) & (cv.variante == var)]
                if not len(g):
                    continue
                reg = {"dataset": ds, "variante": var, "banco": b,
                       "dims": int(g.dims.iloc[0]),
                       "dims_vivas": int(g.dims_vivas.iloc[0]),
                       "unidades": len(g)}
                # as TRES metricas com barra, porque o §5.19 mede que a BAcc
                # inverte de sinal contra o WND-CHARM em 2 de 6 corpora
                if ds in ARTE:
                    reg["barra"] = "sd_folds"
                    for met in METRICAS:
                        reg |= {met: g[met].mean(), f"{met}_erro": g[met].std(),
                                f"{met}_lo": np.nan, f"{met}_hi": np.nan}
                else:
                    r = bo[(bo.dataset == ds) & (bo.banco == b)
                           & (bo.variante == var)]
                    if not len(r):
                        raise SystemExit(f"{ds}/{var}/{b}: sem bootstrap")
                    r = r.iloc[0]
                    reg["barra"] = "se_bootstrap"
                    reg["n_teste"] = int(r.n_teste)
                    for met in METRICAS:
                        if abs(r[met] - g[met].mean()) > 0.01:
                            raise SystemExit(
                                f"{ds}/{var}/{b}/{met}: bootstrap {r[met]:.4f} "
                                f"contra produtor {g[met].mean():.4f}")
                        reg |= {met: r[met], f"{met}_erro": r[f"{met}_se"],
                                f"{met}_lo": r[f"{met}_lo"],
                                f"{met}_hi": r[f"{met}_hi"]}
                linhas.append(reg)

    t = pd.DataFrame(linhas)
    t.to_csv(args.output, index=False, float_format="%.4f")

    bar = "=" * 104
    for var in args.variantes:
        s = t[t.variante == var]
        for met in METRICAS:
            print("\n" + bar)
            print("QUATRO BANCOS HAND-DESIGNED SOB SONDA LINEAR -- "
                  + var + " -- " + ROTULO_MET[met])
            print(bar + "\n")
            cab = "corpus".ljust(19) + "".join(b.rjust(19) for b in BANCOS)
            print(cab + "barra".rjust(16) + "unid.".rjust(7))
            for ds in ARTE + OFIC:
                g = s[s.dataset == ds]
                linha = ROTULO[ds].ljust(19)
                for b in BANCOS:
                    r = g[g.banco == b]
                    if not len(r):
                        linha += "--".rjust(19)
                        continue
                    r = r.iloc[0]
                    marca = "*" if b == REF else " "
                    cel = "%.2f +-%.2f%s" % (r[met], r[met + "_erro"], marca)
                    linha += cel.rjust(19)
                br = "sd folds" if g.barra.iloc[0] == "sd_folds" else "se teste"
                print(linha + br.rjust(16) + str(int(g.unidades.max())).rjust(7))

        def faixa(b):
            v = sorted(set(s[s.banco == b].dims_vivas.astype(int)))
            if not v:
                return "--"
            d = int(s[s.banco == b].dims.iloc[0])
            corpo = str(v[0]) if len(v) == 1 else "%d-%d" % (v[0], v[-1])
            return corpo if corpo == str(d) else corpo + " de " + str(d)
        print("\n" + "dims vivas".ljust(19) + "".join(faixa(b).rjust(19) for b in BANCOS))
        print("* = nosso banco")

        print("\n--- delta do " + REF + " contra o WND-CHARM, nas tres metricas ---")
        print("corpus".ljust(19) + "".join(ROTULO_MET[m].rjust(28) for m in METRICAS))
        for ds in ARTE + OFIC:
            linha = ROTULO[ds].ljust(19)
            for met in METRICAS:
                if ds in ARTE:
                    sel = (cv.dataset == ds) & (cv.variante == var)
                    a = cv[sel & (cv.banco == REF)].set_index(["seed", "fold"])[met]
                    c = cv[sel & (cv.banco == "WND-CHARM")].set_index(["seed", "fold"])[met]
                    x, y = a.align(c, join="inner")
                    if not len(x):
                        linha += "--".rjust(28)
                        continue
                    dd = x - y
                    cel = "%+.2f  %d/%d folds" % (dd.mean(), (dd > 0).sum(), len(dd))
                else:
                    r = pa[(pa.dataset == ds) & (pa.contra == "WND-CHARM")
                           & (pa.variante == var)]
                    if not len(r):
                        linha += "--".rjust(28)
                        continue
                    r = r.iloc[0]
                    cel = "%+.2f [%+.2f,%+.2f]" % (r["d_" + met], r["d_" + met + "_lo"],
                                                   r["d_" + met + "_hi"])
                linha += cel.rjust(28)
            print(linha)

    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
