#!/usr/bin/env python3
"""A tabela de resultados na regra de 12/out: cada banco no SEU melhor estimador.

Difere da `print_tabela_128.py` (regra "todos na mesma cabeca") em uma coisa, e
ela corre CONTRA nos: os baselines hand-designed entram no **denso-512**
(encoder 512, cabeca 2048-1024-1024), que e o melhor deles por rank medio sobre
os sete corpora -- Haralick 1,29 · HOG 1,14 · WND-CHARM 1,00 entre quatro
sondas, contra 2,29 / 2,57 / 3,00 do denso-128. **E a sonda unica por banco, nao
escolha por corpus**, que seria selecao por desfecho (§5.19).

Consequencia que a legenda TEM de carregar: nessa regra os baselines ficam com
classificador **estritamente maior que o nosso** -- o WND-CHARM com 5,70M de
parametros contra os 774k do `hand-8`, 7,4x, e o Haralick com 4,2M a partir de
13 dimensoes de entrada. A frase que isso compra: *para cada baseline
reportamos o melhor entre o protocolo do artigo original e quatro sondas
alternativas, e para os tres isso e uma cabeca mais larga do que a que o nosso
proprio metodo usa*.

O `hand-8` entra com `BranchMLP` (8 ramos x 128, cabeca 512-256-128), que e a
arquitetura do metodo e a decisao de reporte do §5.19b -- o ablation das
branches ja esta no artigo e reportar o braco sem elas seria subreportar.

O Fisher vector entra sob a sonda **linear**, o protocolo publicado dele
(§5.20), e so nos corpora onde existe.

A **fusao de decisao** dos dois bancos (§5.23) fecha cada bloco, no braco SEM
texto -- e ele, sozinho, que sustenta "zero supervisao externa" (§7.2); o braco
com texto fica no CSV. Ela nao tem coluna de dimensao porque combina os
posteriors de dois classificadores, nao um espaco de features conjunto, e e
exatamente por isso que ela converte a complementaridade que a concatenacao nao
convertia (+0,81pp n.s. contra +6,14 no rijksdraw).

**AS PARCELAS DELA NAO SAO AS DUAS LINHAS ACIMA, e isso vai impresso** -- e o
defeito que o §4.1f achou na tabela do manuscrito ("o PRISM da fusao nao e o
PRISM da tabela"). O lado PRISM e o de PRODUCAO (`hand-8 + grafo` sob BranchMLP
nos tres de CV, `hand-8` sob WideBranchMLP nos quatro de split oficial) e o lado
FV roda so nos folds cujo codebook e honesto. Sem as parcelas impressas, um
revisor tenta reconciliar a fusao com as linhas de cima e nao consegue.

AS TRES METRICAS, por decisao de 12/out (§7.20).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["oilpainting", "rijksdraw", "multitask",
            "nct_crc", "minc2500", "food101", "places_lt"]
CV = {"oilpainting", "rijksdraw", "multitask"}
CLASSES = {"oilpainting": 17, "rijksdraw": 11, "multitask": 125,
           "nct_crc": 9, "minc2500": 23, "food101": 101, "places_lt": 365}
BASE = ["Haralick", "HOG", "WND-CHARM"]
MET = ["bacc", "accuracy", "macro_f1"]
# a mesma ordem de metrica do `MET`, com os nomes que o produtor da fusao usa
FMET = ["BAcc", "Acc", "F1"]
FUSAO = ROOT / "results/decision_fusion_banks_all.csv"


def cel(g: pd.DataFrame) -> str:
    return "  ".join(f"{g[m].mean():6.2f} +-{g[m].std():4.2f}" for m in MET)


def fusao_linhas(fus: pd.DataFrame | None, ds: str) -> list[str]:
    """A fusao de decisao do corpus, no braco SEM texto, com as PARCELAS dela.

    As parcelas vao impressas porque nao sao as linhas de cima: o lado PRISM e o
    de producao e o lado FV roda so nos folds de codebook honesto. Imprimir a
    fusao sozinha deixaria a tabela irreconciliavel (§4.1f).
    """
    if fus is None:
        return [f"{'':4s}{'PRISM + Fisher vector':28s} {'--':>7s}  (CSV da fusao ausente)"]
    g = fus[fus.dataset.eq(ds) & fus.braco.eq("sem")].set_index("metrica")
    if len(g) != len(FMET):
        return [f"{'':4s}{'PRISM + Fisher vector':28s} {'--':>7s}  (nao medido ainda)"]
    n = int(g.unidades.iloc[0])
    alerta = "" if n >= 5 else "   !! PARCIAL, nao reportar"

    def linha(rot: str, celulas: list[str], cauda: str = "") -> str:
        """Mesmo prefixo de 42 colunas das linhas de banco, para as metricas
        cairem SOB as colunas delas -- a coluna de dimensao fica vazia nas
        linhas de parcela porque fusao de decisao nao tem dimensao."""
        return f"{'':4s}{rot:28s} {'':>7s}  " + "  ".join(celulas) + cauda

    corpo = [f"{g.fusao[m]:6.2f} +-{g.sd_fusao[m]:4.2f}" for m in FMET]
    lados = [f"{g.prism[m]:6.2f}|{g.fv[m]:6.2f}" for m in FMET]
    delta = [f"{g.delta_prism[m]:+6.2f}{'':7s}" for m in FMET]
    return [f"{'':4s}{'PRISM + Fisher vector':28s} {'--':>7s}  "
            + "  ".join(corpo) + f"   ({n}u){alerta}",
            linha("  parcelas  PRISM | FV", lados),
            linha("  delta sobre o PRISM", delta,
                  f" {g.positivas['BAcc']}/{n} em BAcc, p={g.p['BAcc']:.4f}")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "results/tabela_final.txt")
    args = ap.parse_args()

    fus = pd.read_csv(FUSAO) if FUSAO.exists() else None

    L = ["=" * 100,
         "TABELA DE RESULTADOS -- cada banco no SEU melhor estimador",
         "  baselines hand-designed: denso-512 (enc 512, cabeca 2048-1024-1024)",
         "  hand-8 (PRISM):          BranchMLP, 8 ramos x 128, cabeca 512-256-128",
         "  Fisher vector:           sonda linear, o protocolo publicado dele",
         "  PRISM + FV:              fusao de DECISAO, peso fixo 0,5, zero parametro",
         "ordem das metricas: BAcc  ·  Acc  ·  macro-F1",
         "=" * 100]

    for ds in DATASETS:
        fb, ff = (ROOT / f"results/{ds}_banks_dense_mlp.csv",
                  ROOT / f"results/{ds}_final_table.csv")
        if not (fb.exists() and ff.exists()):
            L += ["", f"{ds}: faltando CSV"]
            continue
        d, f = pd.read_csv(fb), pd.read_csv(ff)
        barra = ("sd entre 25 unidades (5 folds x 5 sementes)" if ds in CV
                 else "sd entre 5 sementes (particao publicada)")
        L += ["", f"{ds}  ({CLASSES[ds]} classes)   +- = {barra}"]

        for b in BASE:
            g = d[d.banco.eq(b) & d.arquitetura.eq("denso-512")]
            if not len(g):
                L.append(f"{'':4s}{b:28s} {'--':>6s}   (sem denso-512)")
                continue
            L.append(f"{'':4s}{b:28s} {int(g.dims.iloc[0]):6d}d  {cel(g)}")

        h = f[f.condition.eq("hand-8") & f["head"].eq("narrow")]
        L.append(f"{'':4s}{'hand-8 (PRISM)':28s} {int(h.dimensions.iloc[0]):6d}d  {cel(h)}"
                 f"   <- com branches")

        # SO a config de PUBLICACAO (§5.20 config E). O `food101` tem CINCO
        # configs em disco e juntar todas daria a media da busca de
        # configuracao, nao o banco -- foi o que a primeira versao fez (32,29 a
        # 32.768d, misturando A/B/C/D/E, contra 34,85 a 41.984d do E).
        # No `oilpainting` a config E so existe na forma INDUTIVA, um codebook
        # por fold, e cada sonda roda so no fold cujo codebook e honesto: as 25
        # unidades sao a CV inteira, nao cinco medicoes do mesmo banco.
        fvl = d[d.arquitetura.eq("linear")]
        pub = fvl[fvl.banco.astype(str).eq("Fisher vector pub")]
        ind = fvl[fvl.banco.astype(str).str.fullmatch(r"Fisher vector ind(_f\d)?")]
        if not (len(pub) or len(ind)):
            L.append(f"{'':4s}{'Fisher vector':28s} {'--':>6s}   (nao medido ainda)")
        for rot, g in (("Fisher vector", pub), ("Fisher vector [INDUTIVO]", ind)):
            if not len(g):
                continue
            alerta = "" if len(g) >= 5 else "   !! PARCIAL, nao reportar"
            L.append(f"{'':4s}{rot:28s} {int(g.dims.iloc[0]):6d}d  {cel(g)}"
                     f"   ({len(g)}u){alerta}")

        L += fusao_linhas(fus, ds)

    L += ["", "-" * 100,
          "LEGENDA OBRIGATORIA",
          "  · os baselines estao no estimador MAIS FAVORAVEL a eles entre quatro",
          "    (LDA svd, LDA shrinkage, denso-128, denso-512) mais o protocolo do",
          "    artigo original do WND-CHARM (WND5, que perde em 7 de 7, §5.19c).",
          "    A escolha e UMA por banco, por rank medio sobre os sete corpora.",
          "  · nessa regra o classificador deles e MAIOR que o nosso: WND-CHARM com",
          "    5,70M de parametros contra 774k do hand-8 (7,4x).",
          "  · treino com class_weight: a celula reportada e a PONDERADA do §7.12,",
          "    que troca ~7,3pp de Acc por ~2,8pp de BAcc. Nos tres de teste",
          "    balanceado (minc2500, food101, places_lt) Acc = BAcc exatamente.",
          "  · o Fisher vector e GPU; o §5.20 proibe comparar relogio com o hand-8.",
          "",
          "SOBRE A LINHA DE FUSAO (§5.23)",
          "  · media de log-probs com peso FIXO 0,5, zero parametro ajustado,",
          "    alinhamento por STEM. Nao tem coluna de dimensao porque combina os",
          "    posteriors de dois classificadores, nao um espaco de features",
          "    conjunto -- e e por isso que ela converte a complementaridade que a",
          "    CONCATENACAO dos mesmos dois bancos nao convertia (+0,81pp n.s.",
          "    contra +6,14 sobre o FV no rijksdraw).",
          "  · AS PARCELAS DELA NAO SAO AS DUAS LINHAS ACIMA, e e por isso que vao",
          "    impressas. O lado PRISM e o de PRODUCAO -- `hand-8 + grafo` sob",
          "    BranchMLP nos tres de CV, `hand-8` sob WideBranchMLP nos quatro de",
          "    split oficial -- contra o `hand-8` de cabeca estreita e sem grafo da",
          "    linha `hand-8 (PRISM)`. E o lado FV roda so nos folds cujo codebook",
          "    e honesto, entao no rijksdraw ele e 5u (fold 1) contra os 25u da",
          "    linha de cima. Sem as parcelas a tabela e irreconciliavel (§4.1f).",
          "  · braco SEM texto, que e o unico que sustenta zero supervisao externa",
          "    (§7.2). O braco COM texto esta em results/decision_fusion_banks_all",
          "    .csv e o ganho encolhe na proporcao do quanto o texto melhora o",
          "    nosso lado -- no places_lt vai a +0,25 com 3/5, indistinguivel de",
          "    zero. Promove-lo para esta tabela trocaria o TIPO da alegacao.",
          "  · 21 de 21 celulas positivas nos sete corpora, 100% das unidades.",
          "-" * 100]

    texto = "\n".join(L)
    args.output.write_text(texto + "\n")
    print(texto)
    print(f"\n[{args.output}]")


if __name__ == "__main__":
    main()
