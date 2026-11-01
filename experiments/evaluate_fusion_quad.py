#!/usr/bin/env python3
"""O 2x2 do NOSSO LADO, sempre fundido com o backbone fine-tuned.

Pedido do usuario em 13/out, depois de a figura de dispersao do mecanismo ficar
ilegivel: *"faca 4x4, com e sem texto, com e sem fisher e a fusao final com os
backbones"*. Sao quatro configuracoes do lado nosso --

    {sem texto, com texto} x {sem Fisher vector, com Fisher vector}

-- e em todas as quatro o desfecho medido e o MESMO: `BAcc da fusao com o
backbone` menos `BAcc do backbone sozinho`, em pp. Isso torna os quatro paineis
comparaveis num eixo so, que e o que a figura de dispersao nao conseguia.

**O PESO E ANINHADO, e a escolha importa (§5.23).** `((PRISM + FV) + backbone)`
com 0,5 em cada nivel da **0,25 / 0,25 / 0,50**, nao um terco para cada lado.
Isso e deliberado e e o que torna os paineis comparaveis: **o backbone recebe
0,50 nas quatro condicoes**, e o que muda e so como o nosso lado se divide
internamente. A regra plana (1/3 cada) foi medida e DISCORDA EM SINAL entre
corpora -- melhor no rijksdraw, pior no food101 --, entao escolhe-la por corpus
seria selecao por desfecho.

**AS QUATRO CONDICOES SAO AVALIADAS NAS MESMAS LINHAS, e isso muda numero
publicado.** O Fisher vector dos tres corpora de arte so e honesto nos folds
cujo codebook foi ajustado sem aquelas linhas (§5.20), entao ele cobre um
subconjunto: no `rijksdraw`, o fold 1. Avaliar `sem FV` na coorte inteira e
`com FV` so naquele fold compararia dois conjuntos de linhas diferentes. Aqui
TODAS as condicoes rodam nas linhas em que o FV existe, e por isso as celulas
`sem FV` dos tres de arte **nao reproduzem os valores publicados do §4.1f**, que
sao da coorte inteira. Nos quatro de split oficial o FV cobre o teste inteiro e
nao ha diferenca -- e e la que a guarda de regressao do §5.23 confere.

Nenhum treino: aritmetica sobre posteriors em disco, alinhamento por STEM e
nunca por posicao (§4.1f item 7), pareado por (fold, semente, modelo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

# FONTE UNICA dos caminhos do par PRISM x FV (§2: nao criar tabela nova).
from evaluate_decision_fusion_banks_all import CFG, SEEDS, EPS  # noqa: E402

MET = (("BAcc", balanced_accuracy_score), ("Acc", accuracy_score),
       ("F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)))
MODELOS = ["resnet50", "vit_b_16"]
# corpus -> subdiretorio da grade de fine-tuning. O multitask guarda features
# sob `multitask` e posteriors sob `multitask125`; o `evaluate_decision_fusion_
# cv.py` carrega a mesma excecao.
GRADE = {"rijksdraw": "rijksdraw", "multitask": "multitask125",
         "oilpainting": "oilpainting", "nct_crc": "nct_crc",
         "minc2500": "minc2500", "food101": "food101", "places_lt": "places_lt"}
CV = {"rijksdraw", "multitask", "oilpainting"}
# §5.23: `PRISM + backbone` com w=0,5 no food101, para a guarda de regressao.
PUBLICADO = {("food101", "resnet50", "sem"): -0.479,
             ("food101", "vit_b_16", "sem"): -1.286}


def log(m: str) -> None:
    print(m, flush=True)


def backbone(ds: str, modelo: str, sd: int, stems: np.ndarray,
             y: np.ndarray, n_classes: int) -> np.ndarray | None:
    """Posterior do backbone alinhado a `stems`, por STEM e com conferencia.

    Nos corpora de CV ele e montado juntando os 5 folds da grade num OOF de
    coorte inteira -- as linhas vem de CINCO modelos diferentes, que e a
    ressalva 2 do §4.1f e vale aqui igual.
    """
    base = ROOT / "outputs/final_finetune_grid" / GRADE[ds] / modelo / f"seed_{sd}"
    arquivos = sorted(base.glob("fold_*/fold_*_posteriors.npz"))
    if not arquivos:
        return None
    pos = pd.Index(list(stems))
    saida = np.zeros((len(stems), n_classes), np.float64)
    visto = np.zeros(len(stems), bool)
    for p in arquivos:
        z = np.load(p, allow_pickle=True)
        idx = pos.get_indexer(np.asarray(z["stems"], dtype=str))
        ok = idx >= 0
        if not ok.any():
            continue
        if not np.array_equal(np.asarray(z["y_true"])[ok], y[idx[ok]]):
            raise SystemExit(f"{p}: rotulos divergem depois de alinhar por stem")
        saida[idx[ok]] = np.asarray(z["probabilities"], np.float64)[ok]
        visto[idx[ok]] = True
    if not visto.all():
        raise SystemExit(f"{ds}/{modelo}/s{sd}: {int((~visto).sum())} linhas sem backbone")
    return saida


def unidades(ds: str, braco: str) -> list[dict]:
    """Uma linha por (fold, semente, modelo), com as quatro condicoes."""
    semf, comf, pad, folds = CFG[ds]
    z = np.load(ROOT / f"outputs/{semf if braco == 'sem' else comf}.npz",
                allow_pickle=True)
    pos = {s: i for i, s in enumerate(np.asarray(z["stems"], dtype=str))}
    yall = np.asarray(z["y_true"])
    n_classes = int(yall.max()) + 1
    linhas = []
    for k in folds:
        for sd in SEEDS:
            p = ROOT / f"outputs/banks_mlp_pred/{pad.format(k=k, sd=sd)}.npz"
            if not p.exists():
                continue
            f = np.load(p, allow_pickle=True)
            st = np.asarray(f["stems"], dtype=str)
            y = np.asarray(f["truth"])
            if any(s not in pos for s in st):
                raise SystemExit(f"{ds}: stems do FV fora do cache do PRISM")
            idx = np.array([pos[s] for s in st])
            if not np.array_equal(yall[idx], y):
                raise SystemExit(f"{ds} f{k} s{sd}: rotulos divergem")
            lp = np.log(np.asarray(z[f"seed_{sd}"], np.float64)[idx] + EPS)
            lf = np.log(np.asarray(f["probs"], np.float64) + EPS)
            for modelo in MODELOS:
                B = backbone(ds, modelo, sd, st, y, n_classes)
                if B is None:
                    continue
                lb = np.log(B + EPS)
                # AS QUATRO CONDICOES. O backbone pesa 0,50 nas quatro; o que
                # muda e so como o nosso lado se divide por dentro.
                q = {"backbone": lb,
                     "sem_fv": 0.5 * lp + 0.5 * lb,
                     "com_fv": 0.5 * (0.5 * lp + 0.5 * lf) + 0.5 * lb,
                     # contexto, nao vai na figura
                     "prism": lp, "fv": lf, "prism_fv": 0.5 * lp + 0.5 * lf}
                linha = {"dataset": ds, "braco": braco, "modelo": modelo,
                         "fold": k, "seed": sd, "n": len(y)}
                for cond, v in q.items():
                    pred = v.argmax(1)
                    for nome, fn in MET:
                        linha[f"{cond}_{nome}"] = 100 * fn(y, pred)
                linhas.append(linha)
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CFG))
    ap.add_argument("--bracos", nargs="+", default=["sem", "com"],
                    choices=["sem", "com"])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_quad.csv")
    args = ap.parse_args()

    linhas: list[dict] = []
    feito: set[tuple] = set()
    if args.output.exists() and args.output.stat().st_size:
        try:
            linhas = pd.read_csv(args.output).to_dict("records")
        except pd.errors.EmptyDataError:
            log("! CSV vazio, comecando do zero")
        feito = {(r["dataset"], r["braco"]) for r in linhas}
        log(f"retomando: {len(linhas)} unidades")

    for braco in args.bracos:
        for ds in args.datasets:
            if (ds, braco) in feito:
                continue
            novas = unidades(ds, braco)
            linhas += novas
            pd.DataFrame(linhas).to_csv(args.output, index=False, float_format="%.4f")
            log(f"  {ds}/{braco}: {len(novas)} unidades")

    t = pd.DataFrame(linhas)

    # GUARDA DE REGRESSAO contra o §5.23. So vale nos de split oficial, onde o
    # FV cobre o teste inteiro e as linhas sao as mesmas da medicao publicada.
    pior, onde = 0.0, "-"
    for (ds, mo, br), alvo in PUBLICADO.items():
        g = t[t.dataset.eq(ds) & t.modelo.eq(mo) & t.braco.eq(br)]
        if not len(g):
            continue
        d = abs((g.sem_fv_BAcc - g.backbone_BAcc).mean() - alvo)
        if d > pior:
            pior, onde = d, f"{ds}/{mo}/{br}"
    log(f"\nguarda contra o §5.23: maior divergencia {pior:.4f}pp ({onde})")
    if pior > args.tol:
        raise SystemExit(f"ABORTADO: {onde} divergiu {pior:.4f}pp > {args.tol}")

    log("\n" + "=" * 104)
    log("DELTA DA FUSAO SOBRE O BACKBONE SOZINHO (BAcc, pp) -- o 2x2 do nosso lado")
    log("=" * 104)
    log(f"{'corpus':12s} {'modelo':9s} | {'backbone':>8s} | "
        f"{'sem txt/sem FV':>14s} {'com txt/sem FV':>14s} "
        f"{'sem txt/com FV':>14s} {'com txt/com FV':>14s}")
    for ds in args.datasets:
        for mo in MODELOS:
            cel = []
            bb = np.nan
            for br in ("sem", "com"):
                for cond in ("sem_fv", "com_fv"):
                    g = t[t.dataset.eq(ds) & t.modelo.eq(mo) & t.braco.eq(br)]
                    if not len(g):
                        cel.append("--")
                        continue
                    bb = g.backbone_BAcc.mean()
                    d = g[f"{cond}_BAcc"] - g.backbone_BAcc
                    cel.append(f"{d.mean():+6.2f}({int((d > 0).sum())}/{len(d)})")
            ordem = [cel[0], cel[2], cel[1], cel[3]]
            log(f"{ds:12s} {mo:9s} | {bb:8.2f} | " + " ".join(f"{c:>14s}" for c in ordem))
    log(f"\n[{args.output}]")


if __name__ == "__main__":
    main()
