#!/usr/bin/env python3
"""O CLIP congelado nos sete corpora, em arquivo PROPRIO.

Pedido do usuario em 13/out: *"acha que devemos rodar um clip congelado em todos
os datasets? provavelmente o ganho sobre ele e negativo, mas acha que um revisor
vai estranhar a falta de um backbone moderno?"* -- e depois *"crie um arquivo a
parte com os resultados do clip, o zero shot nao precisa so o finetunned"*.

**POR QUE ELE EXISTE, e a razao e a regra que o projeto aplica a si mesmo.** O
§7.11 proibe citar so o adversario que nos favorece -- foi o que obrigou a citar
o RFDC 50,76 junto com a CNN no Food-101, e o que obrigou a implementar o Fisher
vector (§5.20, "por honestidade de baseline"). O CLIP e o mesmo caso, com um
agravante nosso: **o artigo JA USA CLIP** -- o encoder do estudante que gera as
legendas e CLIP ViT-B/32 congelado (§5.18). Usa-lo onde ajuda e omiti-lo onde
ganha e exatamente o movimento que o §7.11 veta.

**O ZERO-SHOT FICA DE FORA, por decisao do usuario.** Ele e o instrumento que o
§5.18f usa para provar que o vazamento e LEXICAL (16,3x o nulo embaralhado), e
continua valendo la; aqui a tabela e so de encoder congelado com cabeca
treinada.

**A SONDA E MLP, e isso nao e escolha livre.** O §7.19, refinado pelo §5.20, fixa
que o eixo e *se a representacao foi otimizada para separabilidade linear* -- e o
CLIP foi, por alinhamento contrastivo, como o penultimo de uma CNN. Entao LDA
mediria o alinhamento e nao o banco.

**A SENSIBILIDADE DE CABECA VAI JUNTO, porque ela INVERTE o sinal.** O §5.18f
mede a complementaridade do `hand-8` sobre o CLIP em **+0,91 (p=0,107, 4/5) na
cabeca estreita e -0,94 (p=0,079, 1/5) na larga** -- nenhuma das duas se
estabelece, e reportar so uma seria escolher por desfecho. As duas arquiteturas
vao impressas.

**O QUE ESTA TABELA NAO AUTORIZA: uma alegacao de complementaridade contra o
CLIP.** Ela nao se estabelece (acima), e o §5.18f ja da o enquadramento certo --
*contra supervisao externa a alegacao muda de TIPO, nao de forca*, e o eixo
contra o CLIP e auditabilidade: a decisao dele e um vetor de 512d sem
equivalente inspecionavel, a nossa passa por legenda + n-gramas que um revisor
confere. Isso e MEDIDO e nao retorico, pelo par do §5.9b (mesma legenda, duas
codificacoes: embedding -0,09pp, TF-IDF +2,31/+4,25pp).

**GOTCHA QUE ESTE TRABALHO PRODUZIU, e custou o primeiro resultado inteiro.** O
cache `outputs/<ds>_clip_vitb32.npy` e um .npy SEM stems, gravado na ordem do
`results/<ds>_student_captions.csv` e NAO na da coorte do registry. O consumidor
publicado le os rotulos do mesmo CSV e por isso e auto-consistente; casar por
POSICAO aqui deu **BAcc 0,0017 contra os 54,39 do §5.18f** -- ruido puro, com o
tamanho batendo exatamente nos sete corpora. E o §5.18b outra vez. O
`evaluate_banks_dense_mlp.py` agora reindexa por stem e aborta se faltar algum.
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
CV = {"oilpainting", "rijksdraw", "multitask"}
MET = ["bacc", "accuracy", "macro_f1"]
ARQ = [("denso-128", "cabeca estreita (512-256-128)"),
       ("denso-512", "cabeca larga (2048-1024-1024)")]
# as outras congeladas e o nosso lado, do produtor que ja existe
# As congeladas do ImageNet nas tabelas se chamam pelo PRE-PROCESSAMENTO, nao
# por "congelada" -- e o §7.17 embutido. O §8 mede que `letterbox` custa 4,91pp
# contra `resize+crop` nos MESMOS pesos, entao preferir a melhor variante e
# obrigatorio: reportar a pior handicaparia o adversario em silencio. A variante
# usada vai impressa na propria linha.
REF = [(["ResNet50 resize+crop", "ResNet50 letterbox"], "ResNet50 congelada"),
       (["ViT resize+crop", "ViT letterbox"], "ViT-B/16 congelado"),
       (["hand-8"], "hand-8 (PRISM)")]


def cel(g: pd.DataFrame) -> str:
    return "  ".join(f"{g[m].mean():6.2f} +-{g[m].std():4.2f}" for m in MET)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "results/clip_frozen.csv")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/clip_frozen_rows.txt")
    args = ap.parse_args()

    t = pd.read_csv(args.csv)
    # o places_lt roda num RECORTE proprio e por isso sai noutro arquivo: o
    # cache CLIP nao cobre 528 linhas de treino (as 5 classes permutadas do
    # §5.18d, cujas imagens nunca foram materializadas) nem 5.520 de `val`, que
    # o protocolo publicado descarta. O TESTE esta 100% coberto. Com o corte e a
    # guarda de classe rara o split fica em 61.971 / 36.200 / 362 classes, que
    # reproduz o §5.18e ao numero -- por dois caminhos de codigo independentes.
    pl = ROOT / "results/clip_frozen_places_lt.csv"
    if pl.exists():
        t = pd.concat([t, pd.read_csv(pl)], ignore_index=True)
    L = ["=" * 100,
         "CLIP ViT-B/32 CONGELADO + cabeca treinada, nos sete corpora",
         "  encoder congelado; so o MLP e treinado, igual as outras congeladas",
         "  as DUAS cabecas vao impressas: o §5.18f mede que a largura INVERTE o",
         "  sinal da nossa complementaridade sobre o CLIP (+0,91 x -0,94, as duas n.s.)",
         "ordem das metricas: BAcc  ·  Acc  ·  macro-F1",
         "=" * 100]

    for ds in ORDEM:
        g = t[t.dataset.eq(ds)]
        barra = ("sd entre 25 unid. (5 folds x 5 sementes)" if ds in CV
                 else "sd entre 5 sementes (particao publicada)")
        ncl = 362 if ds == "places_lt" else CLASSES[ds]
        L += ["", f"{ds}  ({ncl} classes)   +- = {barra}"]
        if not len(g):
            L.append(f"{'':4s}CLIP: sem dado -- ver o log")
        gc = g[g.banco.eq("CLIP ViT-B/32 congelado")] if "banco" in g.columns else g
        for arq, rot in ARQ:
            h = gc[gc.arquitetura.eq(arq)]
            if not len(h):
                L.append(f"{'':4s}{'CLIP, ' + rot:40s} {'--':>6s}")
                continue
            L.append(f"{'':4s}{'CLIP ViT-B/32, ' + rot:40s} {cel(h)}   ({len(h)}u)")

        # as referencias do MESMO corpus, do produtor ja existente. A cabeca
        # delas e `narrow`, que e a estreita -- comparar com a larga do CLIP
        # misturaria duas arquiteturas e e o erro que o §7.8 registra.
        if ds == "places_lt":
            # a referencia vem do MESMO recorte, nao da tabela de 365 classes
            h = g[g.banco.eq("hand-8") & g.arquitetura.eq("denso-128")]
            if len(h):
                L.append(f"{'':4s}{'hand-8 (PRISM), MESMO recorte':40s} {cel(h)}")
            L.append(f"{'':4s}!! 362 classes, 61.971/36.200 -- NAO comparar com as")
            L.append(f"{'':4s}   linhas de places_lt das outras tabelas (365 cl)")
            continue
        f = ROOT / f"results/{ds}_final_table.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d["head"].eq("narrow")] if "head" in d.columns else d
            for conds, rot in REF:
                for cond in conds:
                    h = d[d.condition.eq(cond)]
                    if not len(h):
                        continue
                    marca = "" if cond in ("hand-8", conds[0]) else "  (letterbox!)"
                    L.append(f"{'':4s}{rot + ' (estreita)':40s} "
                             + "  ".join(f"{h[m].mean():6.2f} +-{h[m].std():4.2f}"
                                         for m in MET) + marca)
                    break

    L += ["", "-" * 100,
          "LEGENDA OBRIGATORIA",
          "  · MESMO DEGRAU DE SUPERVISAO que as congeladas do ImageNet: o encoder",
          "    nunca viu rotulo deste corpus. Mas o pre-treino e de 400M de pares",
          "    imagem-texto contra 1,28M de rotulos do ImageNet, e o §5.18f mede",
          "    que o CLIP reconhece os NOMES dos estilos -- zero-shot a 16,3x o",
          "    nulo embaralhado no multitask, escalando com o tamanho do encoder.",
          "  · ARQUITETURA CONTROLADA e corre CONTRA o CLIP: ViT-B/32 e o mesmo",
          "    ViT-Base do ViT-B/16, com patch 32 e portanto 4x MENOS tokens",
          "    visuais. E o encoder visualmente mais pobre dos dois e ganha mesmo",
          "    assim -- nao e arquitetura, e o que ele viu (§5.18f).",
          "  · A COMPLEMENTARIDADE SOBRE O CLIP NAO SE ESTABELECE, e esta tabela",
          "    nao a alega: +0,91 (p=0,107, 4/5) na cabeca estreita e -0,94",
          "    (p=0,079, 1/5) na larga (§5.18f). O enquadramento certo e o do",
          "    §7.2 -- contra supervisao externa a alegacao muda de TIPO --, e o",
          "    eixo contra o CLIP e AUDITABILIDADE, medida pelo par do §5.9b.",
          "  · as congeladas de referencia saem na MELHOR variante de",
          "    pre-processamento disponivel (resize+crop; `letterbox!` marca onde",
          "    so havia a pior). O §8 mede 4,91pp entre as duas nos mesmos pesos,",
          "    entao reportar a pior handicaparia o adversario (§7.17).",
          "  · o B/32 e o encoder mais FRACO: o §5.18g mede o L/14 em +4,68pp no",
          "    zero-shot. O cache do L/14 so existe para o multitask; reportar o",
          "    B/32 nos sete e a opcao disponivel, nao a mais adversa.",
          "  · TAMANHO MEDIDO (hook em Conv2d/Linear, o instrumento do §5.4b), e",
          "    ele corre CONTRA a leitura de que o CLIP ganha por ser maior:",
          "        ResNet50 (ImageNet-1k)   25.557.032 par.   4,09 GMACs   2048d",
          "        ViT-B/16 (ImageNet-1k)   86.567.656 par.  11,27 GMACs    768d",
          "        CLIP ViT-B/32 (WIT-400M) 87.456.000 par.   4,36 GMACs    512d",
          "    O CLIP e o MESMO ViT-Base do ViT-B/16 (12 camadas, 768 de largura)",
          "    com +1,0% de parametros -- so a projecao de patch muda, 32x32 em",
          "    vez de 16x16 -- e custa 2,6x MENOS FLOPs, porque a sequencia cai de",
          "    196 para 49 tokens. Contra a ResNet50 o custo e praticamente o",
          "    mesmo (+6,6%). E ele entrega o MENOR vetor dos tres. O que separa",
          "    os +17,3pp nao e capacidade: sao 400M de pares imagem-texto contra",
          "    1,28M de rotulos.",
          "  · treino com class_weight: celula PONDERADA do §7.12. Nos tres de",
          "    teste balanceado Acc = BAcc exatamente.",
          "-" * 100]

    texto = "\n".join(L)
    args.output.write_text(texto + "\n")
    print(texto)
    print(f"\n[{args.output}]")


if __name__ == "__main__":
    main()
