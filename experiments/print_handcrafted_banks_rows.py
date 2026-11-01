#!/usr/bin/env python3
"""Imprime a comparacao dos quatro bancos hand-designed AGRUPADA POR DATASET.

Uma linha por metodo, com BAcc / Acc / macro-F1 e `+-`, no formato de colar numa
tabela. Le os dois agregados que se auto-conferem (§5.19, §5.19b) e nao calcula
nada: se um numero aqui divergir do CSV, o bug esta neste script.

A ORIGEM DA BARRA MUDA POR CORPUS e vai impressa, porque as duas nao sao a mesma
quantidade (§5.19): nos de CV e dispersao entre particoes; nos de split publicado
e o erro padrao da BAcc sob aquela particao, por bootstrap estratificado do
teste. Imprimir as duas sob um `+-` generico e o erro de metrica do §4.1f.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BANCOS = ["Haralick", "HOG", "WND-CHARM", "hand-8"]
DATASETS = ["oilpainting", "rijksdraw", "multitask",
            "nct_crc", "minc2500", "food101", "places_lt"]
CLASSES = {"oilpainting": 17, "rijksdraw": 11, "multitask": 125,
           "nct_crc": 9, "minc2500": 23, "food101": 101, "places_lt": 365}
BARRA = {"sd_sementes": "sd entre sementes", "sd_folds": "sd entre folds",
         "se_bootstrap": "se bootstrap estratificado do teste"}
METRICAS = (("bacc", "BAcc"), ("accuracy", "Acc"), ("macro_f1", "macro-F1"))


def bloco(csv: Path, titulo: str, linhas: list[str]) -> None:
    d = pd.read_csv(csv)
    linhas += ["=" * 78, titulo, f"fonte: {csv.relative_to(ROOT)}", "=" * 78]
    for ds in DATASETS:
        sub = d[d["dataset"] == ds]
        if sub.empty:
            continue
        barra = BARRA.get(sub["barra"].iloc[0], str(sub["barra"].iloc[0]))
        u = int(sub["unidades"].iloc[0])
        linhas += ["", f"{ds}  ({CLASSES[ds]} classes)   +- = {barra}, {u} unidades"]
        cab = "    %-12s %5s  " % ("metodo", "dims")
        cab += "".join("%17s" % nome for _, nome in METRICAS)
        linhas.append(cab)
        for b in BANCOS:
            r = sub[sub["banco"] == b]
            if r.empty or pd.isna(r.iloc[0]["bacc"]):
                linhas.append("    %-12s %5s  %17s" % (b, "--", "ausente"))
                continue
            r = r.iloc[0]
            cel = "".join("%17s" % ("%6.2f +- %4.2f" % (r[m], r[m + "_erro"]))
                          for m, _ in METRICAS)
            linhas.append("    %-12s %5d  %s" % (b, int(r["dims_vivas"]), cel))
    linhas.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "results/handcrafted_banks_rows.txt")
    args = ap.parse_args()

    linhas: list[str] = []
    bloco(ROOT / "results/banks_dense_mlp_table.csv",
          "MLP SEM BRANCH (denso-128) -- a sonda principal da comparacao", linhas)
    bloco(ROOT / "results/handcrafted_banks_table.csv",
          "LDA lsqr + shrinkage -- a sonda linear do §7.19", linhas)
    texto = "\n".join(linhas)
    args.output.write_text(texto + "\n")
    print(texto)
    print(f"gravado em {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
