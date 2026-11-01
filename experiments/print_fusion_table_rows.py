#!/usr/bin/env python3
"""As linhas de fusao COM TEXTO para as tabelas de resultados.

Pedido do usuario em 13/out: *"preciso das linhas prism com texto + fisher e
prism com texto + fisher + backbones para colocar nas tabelas de resultados"*.

Emite tres blocos por corpus, nas TRES metricas (§7.20), com `+-` e contagem de
unidades:

    PRISM+texto + FV                 fusao de dois bancos, peso 0,5/0,5
    PRISM+texto + FV + ResNet50 ft   fusao de tres,  0,25/0,25/0,50
    PRISM+texto + FV + ViT-B/16 ft   idem

**O PESO DA LINHA DE TRES E ANINHADO, e tem de ir na legenda.** `((PRISM+FV) +
backbone)` com 0,5 em cada nivel da **0,25 / 0,25 / 0,50**, nao um terco para
cada lado. A regra plana foi medida (§5.23) e DISCORDA EM SINAL entre corpora,
entao escolher por corpus seria selecao por desfecho. A aninhada se justifica
antes de ver o resultado -- e o enquadramento do §4.1f, *o nosso lado contra o
backbone*, com o nosso lado sendo agora dois bancos.

**§7.2, e aqui ela vale DUAS vezes.** Com texto o nosso lado carrega supervisao
externa; com backbone, os dois lados carregam. Entao estas linhas NAO sustentam
"zero supervisao externa" -- essa e a linha `PRISM + FV` sem texto e sem
backbone, que o `print_tabela_final.py` ja imprime. A alegacao aqui muda de
TIPO, nao de forca.

**AS LINHAS RODAM NAS LINHAS EM QUE O FISHER VECTOR E HONESTO** (§5.20), que nos
tres corpora de arte e um subconjunto -- no `rijksdraw`, o fold 1. Medido: a
celula `PRISM + backbone` recomputada nesse subconjunto reproduz o valor
publicado do §4.1f dentro de **0,06pp**, entao o recorte nao desloca o numero;
mas a contagem de unidades muda e vai impressa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDEM = ["oilpainting", "rijksdraw", "multitask",
         "nct_crc", "minc2500", "food101", "places_lt"]
CLASSES = {"oilpainting": 17, "rijksdraw": 11, "multitask": 125,
           "nct_crc": 9, "minc2500": 23, "food101": 101, "places_lt": 365}
MET = ["BAcc", "Acc", "F1"]
BACK = {"resnet50": "ResNet50 ft", "vit_b_16": "ViT-B/16 ft"}


def cel(g: pd.DataFrame, pref: str) -> str:
    return "  ".join(f"{g[f'{pref}_{m}'].mean():6.2f} +-{g[f'{pref}_{m}'].std():4.2f}"
                     for m in MET)


def delta(g: pd.DataFrame, a: str, b: str) -> str:
    out = []
    for m in MET:
        d = g[f"{a}_{m}"] - g[f"{b}_{m}"]
        out.append(f"{d.mean():+6.2f}({int((d > 0).sum())}/{len(d)})")
    return "  ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_table_rows.txt")
    args = ap.parse_args()

    t = pd.read_csv(ROOT / "results/fusion_quad.csv")
    t = t[t.braco.eq("com")]
    banks = pd.read_csv(ROOT / "results/decision_fusion_banks_all.csv")
    banks = banks[banks.braco.eq("com")].set_index(["dataset", "metrica"])

    L = ["=" * 102,
         "LINHAS DE FUSAO COM TEXTO, para as tabelas de resultados",
         "  PRISM+texto + FV            media de log-probs, 0,5 / 0,5",
         "  ... + backbone fine-tuned   aninhado: 0,25 PRISM / 0,25 FV / 0,50 backbone",
         "ordem das metricas: BAcc  ·  Acc  ·  macro-F1",
         "=" * 102]

    pior = 0.0
    for ds in ORDEM:
        g = t[t.dataset.eq(ds)]
        if not len(g):
            L += ["", f"{ds}: sem dado"]
            continue
        um = g[g.modelo.eq("resnet50")]          # o par de bancos nao depende do backbone
        n_u = len(um)
        L += ["", f"{ds}  ({CLASSES[ds]} classes)   n={n_u} unid. no par de bancos, "
                  f"{len(g[g.modelo.eq('resnet50')])} por backbone"]
        L.append(f"{'':4s}{'PRISM+texto':34s} {cel(um, 'prism')}")
        L.append(f"{'':4s}{'Fisher vector':34s} {cel(um, 'fv')}")
        L.append(f"{'':4s}{'PRISM+texto + FV':34s} {cel(um, 'prism_fv')}")
        L.append(f"{'':8s}{'sobre o PRISM+texto':30s} {delta(um, 'prism_fv', 'prism')}")

        # GUARDA: o par de bancos tem de bater com o produtor dedicado, que e
        # outro caminho de codigo sobre os mesmos posteriors.
        for m in MET:
            if (ds, m) in banks.index:
                pior = max(pior, abs(um[f"prism_fv_{m}"].mean() - banks.fusao[(ds, m)]))

        for mo, rot in BACK.items():
            h = g[g.modelo.eq(mo)]
            if not len(h):
                continue
            L.append(f"{'':4s}{rot + ' sozinho':34s} {cel(h, 'backbone')}")
            L.append(f"{'':4s}{'PRISM+texto + FV + ' + rot:34s} {cel(h, 'com_fv')}"
                     f"   ({len(h)}u)")
            L.append(f"{'':8s}{'sobre o backbone sozinho':30s} "
                     f"{delta(h, 'com_fv', 'backbone')}")
            L.append(f"{'':8s}{'sobre PRISM+texto + backbone':30s} "
                     f"{delta(h, 'com_fv', 'sem_fv')}")

    L += ["", "-" * 102,
          "LEGENDA OBRIGATORIA",
          "  · §7.2: com texto o nosso lado carrega supervisao externa, e com o",
          "    backbone os DOIS lados carregam. Estas linhas nao sustentam 'zero",
          "    supervisao externa' -- essa e a linha PRISM + FV sem texto e sem",
          "    backbone, na tabela_final.txt. A alegacao muda de TIPO, nao de forca.",
          "  · o peso da linha de tres e ANINHADO (0,25/0,25/0,50), nao 1/3 cada. A",
          "    regra plana discorda em sinal entre corpora (§5.23); a aninhada se",
          "    justifica antes de ver o resultado, e nao por desfecho.",
          "  · contra o BACKBONE SOZINHO a linha de tres e positiva em 14/14 celulas",
          "    (pior +0,39). Mas contra a linha de DOIS (PRISM+texto + backbone) o",
          "    Fisher vector PIORA 3 de 14, duas delas significativas:",
          "      nct_crc/ResNet50   -0,38 (p=0,007, 0/5)",
          "      places_lt/ResNet50 -0,39 (p=0,024, 1/5)",
          "      places_lt/ViT      -0,11 (n.s., 2/5)",
          "    O mecanismo e o §4.1f um nivel abaixo: nos dois o FV fica ~11,5pp",
          "    ABAIXO do nosso lado (76,8 contra 88,3 no nct_crc; 11,2 contra 22,8 no",
          "    places_lt) e o peso aninhado lhe da 0,25 do voto assim mesmo. Se a",
          "    tabela trouxer as duas linhas, isto vai declarado -- um revisor que",
          "    compare as duas acha as celulas.",
          "  · peso FIXO em todos os niveis, zero parametro ajustado. E a escolha",
          "    CONSERVADORA: o §4.1f mede que escolher fora da dobra recupera mais.",
          "  · as linhas rodam onde o Fisher vector e honesto (§5.20): nos tres de",
          "    arte e um subconjunto de folds, e no rijksdraw e o fold 1. A celula",
          "    PRISM + backbone recomputada ali reproduz o §4.1f dentro de 0,06pp.",
          "  · treino com class_weight: celula PONDERADA do §7.12. Nos tres de teste",
          "    balanceado (minc2500, food101, places_lt) Acc = BAcc exatamente.",
          "-" * 102]

    texto = "\n".join(L)
    args.output.write_text(texto + "\n")
    print(texto)
    print(f"\nguarda contra o produtor do par de bancos: maior divergencia {pior:.4f}pp")
    if pior > 0.01:
        raise SystemExit("ABORTADO: o par de bancos nao reproduz o CSV dedicado")
    print(f"[{args.output}]")


if __name__ == "__main__":
    main()
