#!/usr/bin/env python3
"""O bloco do MLP SEM BRANCH agrupado por dataset, no formato de colar na tabela.

Companheiro do `print_handcrafted_banks_rows.py`, que faz o mesmo para o LDA
(§5.19). Aqui a sonda e o MLP sem branch do §5.19b -- `denso-128` por padrao --,
e a diferenca que importa NAO e so a sonda:

**A SEMENTE DEIXA DE SER INERTE.** No LDA ela e literalmente inocua em split
fixo (o estimador e fechado, reexecutar da bit-a-bit identico), e por isso
aquela tabela usa bootstrap do teste como unica fonte de incerteza. O MLP tem
ruido de TREINO, entao em split fixo as 5 sementes sao dispersao de verdade e a
barra e `sd entre sementes`. Nos tres de CV ha DUAS fontes (particao e treino)
e as 25 unidades sao 5 folds x 5 sementes.

A decomposicao das duas fontes vai impressa junto, porque reportar so o total
esconderia qual delas manda -- e o §5.6 registra que neste projeto o que decide
o MLP e semente, nao fold.

Nao recalcula metrica nenhuma: le os CSVs por corpus e agrega. Se um numero
aqui divergir do CSV, o bug esta neste script.

Uso:
    python experiments/print_banks_dense_mlp_rows.py
    python experiments/print_banks_dense_mlp_rows.py --arquitetura linear
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BANCOS = ["Haralick", "HOG", "WND-CHARM", "hand-8"]
DATASETS = ["oilpainting", "rijksdraw", "multitask",
            "nct_crc", "minc2500", "food101", "places_lt"]
CLASSES = {"oilpainting": 17, "rijksdraw": 11, "multitask": 125,
           "nct_crc": 9, "minc2500": 23, "food101": 101, "places_lt": 365}
CV = {"oilpainting", "rijksdraw", "multitask"}
METRICAS = (("bacc", "BAcc"), ("accuracy", "Acc"), ("macro_f1", "macro-F1"))


def decompoe(sub: pd.DataFrame, metrica: str) -> tuple[float, float]:
    """sd entre folds (media por fold sobre sementes) e sd entre sementes."""
    por_fold = sub.groupby("fold")[metrica].mean()
    por_seed = sub.groupby("seed")[metrica].mean()
    f = float(por_fold.std(ddof=1)) if len(por_fold) > 1 else float("nan")
    s = float(por_seed.std(ddof=1)) if len(por_seed) > 1 else float("nan")
    return f, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arquitetura", default="denso-128")
    ap.add_argument("--bancos", nargs="+", default=BANCOS)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/banks_dense_mlp_rows.txt")
    args = ap.parse_args()

    L: list[str] = []
    L += ["=" * 78,
          f"MLP SEM BRANCH ({args.arquitetura}) -- a sonda do §5.19b",
          "fonte: results/{dataset}_banks_dense_mlp.csv",
          "=" * 78]
    for ds in DATASETS:
        f = ROOT / f"results/{ds}_banks_dense_mlp.csv"
        if not f.exists():
            L += ["", f"{ds}  -- sem CSV"]
            continue
        d = pd.read_csv(f)
        d = d[d["arquitetura"] == args.arquitetura]
        if d.empty:
            L += ["", f"{ds}  -- sem o braco '{args.arquitetura}'"]
            continue
        n = int(d.groupby("banco").size().max())
        barra = ("sd entre as 25 unidades (5 folds x 5 sementes)" if ds in CV
                 else "sd entre as 5 sementes (particao publicada, fixa)")
        L += ["", f"{ds}  ({CLASSES[ds]} classes)   +- = {barra}"]
        for b in args.bancos:
            sub = d[d["banco"] == b]
            if sub.empty:
                L.append(f"{'':16s}{b:12s}  (ausente)")
                continue
            cel = [f"{sub[m].mean():6.2f} +- {sub[m].std(ddof=1):4.2f}"
                   for m, _ in METRICAS]
            L.append(f"{'':16s}{b:12s} {cel[0]}  {cel[1]}  {cel[2]}"
                     f"   {int(sub['dims'].iloc[0]):5d}d")
        # decomposicao da barra: qual fonte manda
        if ds in CV:
            part = []
            for b in args.bancos:
                sub = d[d["banco"] == b]
                if sub.empty:
                    continue
                ff, ss = decompoe(sub, "bacc")
                part.append(f"{b} {ff:.2f}/{ss:.2f}")
            L.append(f"{'':16s}  sd_fold/sd_semente em BAcc: " + "  ".join(part))

    L += ["", "=" * 78,
          "ordem das metricas: BAcc  ·  Acc  ·  macro-F1",
          "AS BARRAS NAO SAO A MESMA QUANTIDADE entre os dois grupos de corpora",
          "(§4.1f): nos de CV ela mistura particao e ruido de treino; nos de split",
          "publicado a particao e fixa e sobra so o ruido de treino. No LDA (§5.19)",
          "a semente e INERTE em split fixo, entao la a barra e bootstrap do teste",
          "-- as duas tabelas nao tem barras comparaveis, e isso tem de ir na legenda.",
          "=" * 78]
    txt = "\n".join(L)
    print(txt)
    args.output.write_text(txt + "\n")
    print(f"\n[gravado em {args.output.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
