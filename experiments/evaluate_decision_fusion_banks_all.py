#!/usr/bin/env python3
"""Fusao de DECISAO entre os dois bancos hand-designed, nos SETE corpora.

`PRISM` (o lado de PRODUCAO -- `hand-8 + grafo` sob BranchMLP nos de CV,
`hand-8` sob WideBranchMLP nos de split oficial) contra o Fisher vector sob a
sonda `linear`, media de log-probs com peso **FIXO 0,5**, zero parametro.

**POR QUE ISTO EXISTE** (§5.23): concatenar os dois bancos nao convertia a
complementaridade medida -- `hand-8 + FV` dava +0,81pp sobre o FV sozinho no
rijksdraw, p=0,23 -- porque 41.984 dimensoes contra 617 na mesma cabeca fazem o
ramo maior dominar (§4.1c). A fusao de decisao contorna o mecanismo.

**O BRACO COM TEXTO existe porque o peso fixo pode QUEBRAR** -- pergunta do
usuario em 13/out: *"temos que medir se a inclusao do texto nao destroi a fusao,
ficamos melhor talvez o peso fixo nao seja o suficiente"*. O §5.18e mede que o
texto move o NOSSO lado de +0,01 a +20,18pp e nao toca no FV, e o §4.1f mede que
o peso fixo quebra quando um lado fica muito pior. **Os dois bracos sao
reportados**; a linha SEM texto e a que sustenta "zero supervisao externa" (§7.2).

Tudo e aritmetica sobre posteriors em disco: nenhum treino. Alinhamento por
STEM e nunca por posicao (§4.1f item 7), pareado por unidade.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
SEEDS = [42, 43, 44, 45, 46]
MET = (("BAcc", balanced_accuracy_score), ("Acc", accuracy_score),
       ("F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)))

# corpus -> (cache OOF sem texto, com texto, padrao do pred do FV, folds)
#
# O FV dos tres de arte e INDUTIVO por fold (§5.20): cada fold tem o SEU
# codebook e a sonda so e honesta nele, entao o padrao leva `{k}` nos dois
# lugares. Rodar um codebook nos outros quatro folds seria transducao em
# silencio -- e o rijksdraw tem a variante `pub` (transdutiva) so no fold 1.
CFG = {
    "oilpainting": ("oilpainting_our_oof_cv", "oilpainting_our_oof_cv_text",
                    "oilpainting__Fisher_vector_ind_f{k}_linear__s{sd}_f{k}", [1, 2, 3, 4, 5]),
    "multitask":   ("multitask_our_oof_cv", "multitask_our_oof_cv_text",
                    "multitask__Fisher_vector_ind_f{k}_linear__s{sd}_f{k}", [1, 2, 3, 4, 5]),
    "rijksdraw":   ("rijksdraw_our_oof_cv", "rijksdraw_our_oof_cv_text",
                    "rijksdraw__Fisher_vector_pub_linear__s{sd}_f{k}", [1]),
    "nct_crc":     ("nct_crc_our_posteriors", "nct_crc_our_posteriors_text",
                    "nct_crc__Fisher_vector_pub_linear__s{sd}_f{k}", [1]),
    "minc2500":    ("minc2500_our_posteriors", "minc2500_our_posteriors_text",
                    "minc2500__Fisher_vector_pub_linear__s{sd}_f{k}", [1]),
    "food101":     ("food101_our_posteriors", "food101_our_posteriors_text",
                    "food101__Fisher_vector_pub_linear__s{sd}_f{k}", [1]),
    "places_lt":   ("places_lt_our_posteriors", "places_lt_our_posteriors_text",
                    "places_lt__Fisher_vector_pub_linear__s{sd}_f{k}", [1]),
}


def unidades(ds: str, cache: str, pad: str, folds: list[int], peso: float):
    """Uma linha por (fold, semente): os tres lados e a decomposicao por imagem."""
    z = np.load(ROOT / f"outputs/{cache}.npz", allow_pickle=True)
    pos = {s: i for i, s in enumerate(np.asarray(z["stems"], dtype=str))}
    yall = np.asarray(z["y_true"])
    linhas, dec = [], []
    for k in folds:
        for sd in SEEDS:
            p = ROOT / f"outputs/banks_mlp_pred/{pad.format(k=k, sd=sd)}.npz"
            if not p.exists():
                continue
            f = np.load(p, allow_pickle=True)
            if "probs" not in f.files:
                raise SystemExit(f"{p.name}: gravado sem posteriors -- rode com --pred-dir")
            st = np.asarray(f["stems"], dtype=str)
            y = np.asarray(f["truth"])
            falta = [s for s in st if s not in pos]
            if falta:
                raise SystemExit(f"{ds}: {len(falta)} stems do FV ausentes no cache do PRISM")
            idx = np.array([pos[s] for s in st])
            if not np.array_equal(yall[idx], y):
                raise SystemExit(f"{ds} f{k} s{sd}: rotulos divergem depois de alinhar")
            pb = np.asarray(f["probs"], np.float64)
            pa = np.asarray(z[f"seed_{sd}"], np.float64)[idx]
            lf = peso * np.log(pa + EPS) + (1 - peso) * np.log(pb + EPS)
            pr = {"PRISM": pa.argmax(1), "FV": pb.argmax(1), "fusao": lf.argmax(1)}
            linha = {"fold": k, "seed": sd}
            for lado, q in pr.items():
                for nome, fn in MET:
                    linha[f"{lado}_{nome}"] = 100 * fn(y, q)
            linhas.append(linha)
            ca, cb = pr["PRISM"] == y, pr["FV"] == y
            dec.append([100 * v.mean() for v in (ca & cb, ca & ~cb, ~ca & cb, ca | cb)])
    return pd.DataFrame(linhas), (np.array(dec).mean(0) if dec else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CFG))
    ap.add_argument("--bracos", nargs="+", default=["sem", "com"],
                    choices=["sem", "com"])
    ap.add_argument("--peso", type=float, default=0.5)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/decision_fusion_banks_all.csv")
    args = ap.parse_args()

    saida = []
    for braco in args.bracos:
        rot = "SEM texto (zero supervisao externa)" if braco == "sem" else "COM texto"
        print("\n" + "=" * 104)
        print(f"PRISM x Fisher vector -- peso fixo {args.peso} -- {rot}")
        print("=" * 104)
        print(f"{'corpus':12s} {'PRISM':>7s} {'FV':>7s} {'FUSAO':>7s} | "
              f"{'sobre o PRISM (BAcc/Acc/F1)':>40s} | {'sobre o FV':>14s} {'unid':>5s}")
        for ds in args.datasets:
            semf, comf, pad, folds = CFG[ds]
            d, dec = unidades(ds, semf if braco == "sem" else comf, pad, folds, args.peso)
            if not len(d):
                print(f"{ds:12s} sem posteriors do FV em disco -- pulando")
                continue
            cel = []
            for nome, _ in MET:
                dd = d[f"fusao_{nome}"] - d[f"PRISM_{nome}"]
                cel.append(f"{dd.mean():+6.2f}({int((dd > 0).sum())}/{len(dd)})")
            dfv = d["fusao_BAcc"] - d["FV_BAcc"]
            print(f"{ds:12s} {d.PRISM_BAcc.mean():7.2f} {d.FV_BAcc.mean():7.2f} "
                  f"{d.fusao_BAcc.mean():7.2f} | {' '.join(cel):>40s} | "
                  f"{dfv.mean():+6.2f}({int((dfv > 0).sum())}/{len(dfv)}) {len(d):5d}")
            for nome, _ in MET:
                dd = d[f"fusao_{nome}"] - d[f"PRISM_{nome}"]
                p = stats.ttest_rel(d[f"fusao_{nome}"], d[f"PRISM_{nome}"])[1] if len(d) > 1 else np.nan
                saida.append({"dataset": ds, "braco": braco, "metrica": nome,
                              "peso": args.peso, "unidades": len(d),
                              "prism": d[f"PRISM_{nome}"].mean(),
                              "fv": d[f"FV_{nome}"].mean(),
                              "fusao": d[f"fusao_{nome}"].mean(),
                              # sd ENTRE UNIDADES, ddof=1 -- a mesma convencao
                              # do §2 e do `cel()` da tabela. Duas barras lidas
                              # lado a lado tem de sair do mesmo ddof (§5.18g).
                              "sd_prism": d[f"PRISM_{nome}"].std(),
                              "sd_fv": d[f"FV_{nome}"].std(),
                              "sd_fusao": d[f"fusao_{nome}"].std(),
                              "delta_prism": dd.mean(), "p": p,
                              "positivas": int((dd > 0).sum()),
                              "ambos": dec[0], "so_prism": dec[1],
                              "so_fv": dec[2], "oraculo": dec[3]})

    out = pd.DataFrame(saida)
    if args.output.exists():
        velho = pd.read_csv(args.output)
        out = pd.concat([velho, out]).drop_duplicates(
            subset=["dataset", "braco", "metrica", "peso"], keep="last")
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n[{args.output}]")


if __name__ == "__main__":
    main()
