#!/usr/bin/env python3
"""A tabela principal com TODOS os bancos na mesma cabeca (enc=128).

REGRA, e ela e simetrica: todo banco entra com encoder de 128 por bloco e
cabeca 512-256-128. A unica diferenca e o NUMERO de blocos -- 8 no hand-8, 1
nos outros -- e isso e a arquitetura do metodo, nao vantagem concedida: nao
existe versao de 8 ramos do WND-CHARM sem fatiar as 2.895 dimensoes dele
arbitrariamente.

O Fisher vector NAO entra nessa regra, e a excecao e declarada: ele perde ~9pp
sob cabeca estreita e a sonda do protocolo publicado dele e o classificador
LINEAR. Forca-lo a 128 mediria a sonda, nao o banco. Coerente com o §7.19 -- o
que decide a sonda e a representacao ter sido otimizada para separabilidade
linear, e o FV e o caso extremo (+10,4pp a favor do linear contra -2,3 do
hand-8).

BRACO DE SENSIBILIDADE OBRIGATORIO: o `denso-512` (cabeca 2048-1024-1024) foi
medido e AJUDA os baselines (+0,87 a +2,75, todos significativos no rijksdraw).
Ele vai impresso junto. Omitir um braco medido que favorece o adversario e o que
o §5.19 veta ao declarar que a variante adotada e a MENOS favoravel.

Uso:  python experiments/print_tabela_128.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["oilpainting", "rijksdraw", "multitask",
            "nct_crc", "minc2500", "food101", "places_lt"]
CLASSES = {"oilpainting": 17, "rijksdraw": 11, "multitask": 125,
           "nct_crc": 9, "minc2500": 23, "food101": 101, "places_lt": 365}
CV = {"oilpainting", "rijksdraw", "multitask"}
BASE = ["Haralick", "HOG", "WND-CHARM"]
MET = (("bacc", "BAcc"), ("accuracy", "Acc"), ("macro_f1", "macro-F1"))


def celula(sub: pd.DataFrame) -> str:
    return "  ".join(f"{sub[m].mean():6.2f} +-{sub[m].std(ddof=1):4.2f}" for m, _ in MET)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "results/tabela_128.txt")
    args = ap.parse_args()
    L: list[str] = ["=" * 92,
                    "TABELA PRINCIPAL -- todos os bancos com encoder 128 e cabeca 512-256-128",
                    "hand-8: BranchMLP (8 ramos x 128)   baselines: 1 bloco x 128",
                    "Fisher vector: EXCECAO declarada, sonda linear (o protocolo publicado dele)",
                    "ordem das metricas: BAcc  ·  Acc  ·  macro-F1",
                    "=" * 92]
    for ds in DATASETS:
        fb = ROOT / f"results/{ds}_banks_dense_mlp.csv"
        ff = ROOT / f"results/{ds}_final_table.csv"
        if not (fb.exists() and ff.exists()):
            L += ["", f"{ds}: faltando CSV"]
            continue
        d = pd.read_csv(fb)
        f = pd.read_csv(ff)
        barra = ("sd entre 25 unidades (5 folds x 5 sementes)" if ds in CV
                 else "sd entre 5 sementes (particao publicada)")
        L += ["", f"{ds}  ({CLASSES[ds]} classes)   +- = {barra}"]
        for b in BASE:
            s = d[(d.banco == b) & (d.arquitetura == "denso-128")]
            L.append(f"{'':4s}{b:28s} {int(s.dims.iloc[0]):6d}d  {celula(s)}"
                     if not s.empty else f"{'':4s}{b:28s}  (ausente)")
        h = f[f["condition"] == "hand-8"]
        L.append(f"{'':4s}{'hand-8 (PRISM)':28s} {int(h.dimensions.iloc[0]):6d}d  {celula(h)}")
        # Fisher vector, onde existe, na sonda dele
        for nome, rot in (("Fisher vector pub", "Fisher vector [linear]"),
                          ("Fisher vector ind", "Fisher vector IND [linear]")):
            s = d[(d.banco == nome) & (d.arquitetura == "linear")]
            if not s.empty:
                nf = s.fold.nunique()
                L.append(f"{'':4s}{rot:28s} {int(s.dims.iloc[0]):6d}d  {celula(s)}"
                         f"   ({len(s)}u, {nf} fold{'s' if nf > 1 else ''})")
        # sensibilidade
        s512 = d[d.arquitetura == "denso-512"]
        if not s512.empty:
            g = s512[s512.banco.isin(BASE)].groupby("banco").bacc.mean()
            if len(g):
                d128 = d[(d.arquitetura == "denso-128") &
                         (d.banco.isin(g.index))].groupby("banco").bacc.mean()
                dif = (g - d128).reindex(g.index)
                L.append(f"{'':4s}  sensibilidade, cabeca 2048-1024-1024: " +
                         "  ".join(f"{k} {v:+.2f}" for k, v in dif.items()))
    L += ["", "=" * 92,
          "A cabeca larga AJUDA os baselines e o vao encolhe ~20% onde foi medida,",
          "sem trocar a ordem entre o melhor baseline e o PRISM. A tabela acima e",
          "portanto a configuracao SIMETRICA, nao a mais favoravel a nos.",
          "=" * 92]
    txt = "\n".join(L)
    print(txt)
    args.output.write_text(txt + "\n")
    print(f"\n[gravado em {args.output.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
