"""Monta a matriz do artigo: escada de supervisao x (BAcc, Acc, F1).

Linhas, na ordem de supervisao externa crescente:
  Haralick (13d) · HOG (1764d)          baseline classica, zero supervisao
  nos = hand-8 + grafo (§3)             zero supervisao, com agregacao
  ResNet50 / ViT congelados             supervisao externa, sem adaptacao ao alvo
  nos + congelado                       fusao de FEATURES (valida: congelado
                                        nunca viu rotulo do corpus)
  ResNet50 / ViT fine-tuned             supervisao externa + adaptacao
  nos + fine-tuned                      fusao de DECISAO, zero parametros -- o
                                        unico instrumento valido contra
                                        fine-tuned (§4.1c)

`nos` NUNCA inclui Text (supervisao externa, §3) e SEMPRE inclui o grafo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [("bacc", "BAcc"), ("accuracy", "Acc"), ("macro_f1", "F1")]


def from_final_table(dataset: str) -> dict[str, dict]:
    path = ROOT / f"results/{dataset}_final_table.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame["head"] == "narrow"]
    out = {}
    for name, block in frame.groupby("condition"):
        if len(block) < 5:
            continue
        out[str(name)] = {"dims": int(block.dimensions.iloc[0]),
                          "bacc": block.bacc.mean(), "sd": block.bacc.std(),
                          "accuracy": block.accuracy.mean(), "sd_acc": block.accuracy.std(),
                          "macro_f1": block.macro_f1.mean(), "sd_f1": block.macro_f1.std()}
    return out


def from_logit_add(dataset: str) -> dict[str, dict]:
    """O grafo na codificacao ADITIVA (§5.6), que e a canonica desde 02/set.

    Escolhida pelo usuario sobre a concatenada apesar de CUSTAR no oilpainting
    (43,01 contra 44,01) -- e a escolha conservadora, e evita reportar uma
    codificacao diferente por corpus, que seria selecao pos-hoc.

    `dims` e o do hand-8 puro de proposito: o logit-add soma lambda*log-lift aos
    LOGITS, entao acrescenta **zero dimensoes de entrada**. E o argumento de
    operador quase sem parametros do §5.2 -- um escalar contra 2C+32 colunas.
    """
    path = ROOT / f"results/{dataset}_logit_add_metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for name, key in (("hand-8", "hand-8 puro"),
                      ("+ logit-add \u03bb global", "hand-8 + logit-add")):
        block = frame[frame.condition == name]
        if len(block) < 5:
            continue
        out[key] = {"dims": np.nan, "bacc": block.bacc.mean(), "sd": block.bacc.std(),
                    "accuracy": block.accuracy.mean(), "sd_acc": block.accuracy.std(),
                    "macro_f1": block.macro_f1.mean(), "sd_f1": block.macro_f1.std()}
    return out


def from_text(dataset: str) -> dict[str, dict]:
    """O canal de texto sobre o grafo aditivo (§5.18b).

    ATENCAO DE DEGRAU: o encoder do captioner e o CLIP, entao esta linha carrega
    **supervisao externa** e NAO pertence ao degrau de zero supervisao. Ela entra
    no mesmo degrau das congeladas -- supervisao externa sem adaptacao ao alvo --
    e e ali que deve ser lida (§7.2). A linha que sustenta a alegacao de zero
    supervisao externa continua sendo `NOS` sem texto.

    O que a linha acrescenta nesse degrau nao e desempate de acuracia, e
    **natureza**: a decisao passa por um gargalo legivel (legenda + n-gramas de
    maior peso), enquanto 2.048 dims de GAP nao tem equivalente. O controle que
    torna isso medicao e nao retorica esta no §5.9b -- a MESMA legenda como
    embedding de 384d vale -0,09pp.
    """
    path = ROOT / f"results/{dataset}_logit_add_text.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for name, key in (("+ logit-add + texto", "NOS + texto"),
                      ("hand-8 + texto", "hand-8 + texto (sem grafo)")):
        block = frame[frame.condition == name]
        if len(block) < 5:
            continue
        out[key] = {"dims": int(block.dimensions.iloc[0]),
                    "bacc": block.bacc.mean(), "sd": block.bacc.std(),
                    "accuracy": block.accuracy.mean(), "sd_acc": block.accuracy.std(),
                    "macro_f1": block.macro_f1.mean(), "sd_f1": block.macro_f1.std()}
    return out


def from_grid(dataset: str) -> dict[str, dict]:
    path = ROOT / "results/final_finetune_grid_units.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame.dataset == dataset]
    out = {}
    for model, block in frame.groupby("model"):
        by_seed = block.groupby("seed")[["bacc", "acc", "f1"]].mean()
        label = {"resnet50": "ResNet50 fine-tuned", "vit_b_16": "ViT fine-tuned"}[str(model)]
        out[label] = {"dims": np.nan, "bacc": by_seed.bacc.mean(), "sd": by_seed.bacc.std(),
                      "accuracy": by_seed.acc.mean(), "sd_acc": by_seed.acc.std(),
                      "macro_f1": by_seed.f1.mean(), "sd_f1": by_seed.f1.std()}
    return out


def from_fusion(dataset: str) -> dict[str, dict]:
    for suffix in ("_decision_fusion_cv.csv", "_decision_fusion.csv"):
        path = ROOT / f"results/{dataset}{suffix}"
        if path.exists():
            break
    else:
        return {}
    frame = pd.read_csv(path)
    out = {}
    for model, block in frame.groupby("model"):
        label = {"resnet50": "nos + ResNet50 ft", "vit_b_16": "nos + ViT ft"}[str(model)]
        out[label] = {"dims": np.nan, "bacc": block.fusion_bacc.mean(), "sd": block.fusion_bacc.std(),
                      "accuracy": block.fusion_accuracy.mean(), "sd_acc": block.fusion_accuracy.std(),
                      "macro_f1": block.fusion_macro_f1.mean(), "sd_f1": block.fusion_macro_f1.std()}
    return out


# nome interno -> rotulo do artigo.
#
# PRE-PROCESSAMENTO DAS CONGELADAS: FIXO em `Resize(256)+CenterCrop(224)` em
# todos os corpora (decisao do usuario em 02/set -- reportar duas variantes
# obrigaria a explicar as duas no texto e chamaria atencao para uma escolha que
# nao e o assunto do artigo).
#
# Por que resize+crop e nao Letterbox, apesar de o Letterbox ser melhor no
# rijksdraw: (a) e a avaliacao CANONICA do ImageNet, e os pesos foram treinados
# com RandomResizedCrop -- qualquer outra coisa exigiria justificativa nossa;
# (b) e a mesma transformacao que os backbones FINE-TUNED usam na grade do §9,
# entao congelado e fine-tuned ficam comparaveis; (c) e a escolha que corre
# CONTRA nos em dois dos tres corpora.
#
# O efeito medido em 02/set, que fica como nota de robustez e nao como duas
# linhas na tabela -- o §7.17 supunha Letterbox handicapando sempre, e nao e:
#
#     letterbox -> resize+crop     ResNet50      ViT
#     oilpainting                    +0,49     +0,95
#     multitask                      +0,42     -0,12
#     rijksdraw                      -0,81     -1,68
#
# No rijksdraw o Letterbox e MELHOR, provavelmente porque sao desenhos com
# margem larga e proporcao irregular: o CenterCrop corta conteudo, e perder
# borda custa mais que o desvio de distribuicao das barras cinzas. Fixar
# resize+crop nos favorece ali em <=1,7pp, numa margem de ~7,8 -- imaterial para
# a conclusao, e vale uma frase de nota em vez de uma coluna a mais.
ROWS = [
    ("Haralick", ["Haralick"]),
    ("HOG", ["HOG"]),
    ("hand-8 (sem grafo)", ["hand-8"]),
    # A codificacao do grafo e escolhida por REGRA, nao fixada nem por corpus
    # (§5.6): o bloco concatenado degrada com o numero de classes (gira -1,29pp
    # de 25 para 125 no experimento controlado) enquanto o aditivo fica plano, e
    # o lambda que o modelo aprende cresce monotonicamente com C (0,032 em 17
    # classes -> 0,080 em 25 -> 0,138 em 125). Onde C e pequeno o bloco e barato
    # (2C+32 = 66d em 17 classes) e a concatenacao vence; onde C e grande ele
    # custa 282d e esparsos, e o canal aditivo vence. A ordem dos candidatos
    # abaixo implementa isso, e a variante nao usada aparece na linha seguinte.
    ("NOS = hand-8 + grafo", ["hand-8 + grafo", "hand-8 + logit-add"]),
    ("  variante: outra codificacao", ["hand-8 + logit-add", "hand-8 + grafo"]),
    ("ResNet50 congelada", ["ResNet50 resize+crop", "ResNet50 letterbox"]),
    ("ViT congelado", ["ViT resize+crop", "ViT letterbox"]),
    ("NOS + texto (CLIP)", ["NOS + texto"]),
    ("nos + ResNet50 congelada", ["hand-8 + grafo + ResNet50 resize+crop",
                                  "hand-8 + grafo + ResNet50 letterbox"]),
    ("nos + ViT congelado", ["hand-8 + grafo + ViT resize+crop",
                             "hand-8 + grafo + ViT letterbox"]),
    ("ResNet50 fine-tuned", ["ResNet50 fine-tuned"]),
    ("ViT fine-tuned", ["ViT fine-tuned"]),
    ("nos + ResNet50 ft (decisao)", ["nos + ResNet50 ft"]),
    ("nos + ViT ft (decisao)", ["nos + ViT ft"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["oilpainting", "rijksdraw", "multitask"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/artistic_matrix.csv")
    args = parser.parse_args()

    # `multitask` = 125 classes, onde a regra pede o canal ADITIVO. Os demais
    # corpora de arte tem C pequeno e ficam com o concatenado.
    additive_first = {"multitask"}
    tables = {}
    for dataset in args.datasets:
        merged = from_final_table(dataset)
        merged |= from_logit_add(dataset)
        merged |= from_text(dataset)
        merged |= from_grid("multitask125" if dataset == "multitask" else dataset)
        merged |= from_fusion(dataset)
        tables[dataset] = merged

    rows = []
    for label, candidates in ROWS:
        row = {"condicao": label}
        for dataset in args.datasets:
            # A inversao vale SO para a linha do grafo. Condicionar em
            # `len(candidates) == 2` pegava tambem as congeladas, que tem dois
            # candidatos de pre-processamento -- e o multitask ficava com
            # Letterbox em vez de resize+crop, em silencio.
            order = candidates
            if dataset in additive_first and any("logit-add" in c for c in candidates):
                order = candidates[::-1]
            found = next((c for c in order if c in tables[dataset]), None)
            values = tables[dataset].get(found) if found else None
            for key, tag in METRICS:
                row[f"{dataset}_{tag}"] = values[key] if values else np.nan
            for key, tag in (("sd", "BAcc"), ("sd_acc", "Acc"), ("sd_f1", "F1")):
                row[f"{dataset}_sd_{tag}"] = (values or {}).get(key, np.nan)
            row[f"{dataset}_dims"] = values["dims"] if values else np.nan
            row[f"{dataset}_sd"] = values["sd"] if values else np.nan
        rows.append(row)
    matrix = pd.DataFrame(rows)
    matrix.to_csv(args.output, index=False, float_format="%.2f")

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        print(f"{'condicao':30s} {'dims':>6s} {'BAcc':>14s} {'Acc':>14s} {'F1':>14s}")
        print("-" * 68)
        for _, r in matrix.iterrows():
            b = r[f"{dataset}_BAcc"]
            if pd.isna(b):
                print(f"{r.condicao:30s} {'':>6s} {'— pendente':>14s}")
                continue
            dims = "" if pd.isna(r[f"{dataset}_dims"]) else f"{int(r[f'{dataset}_dims']):d}"
            # As tres metricas saem com desvio: sem ele a tabela do artigo nao
            # fecha, e os sd de Acc e F1 ja estavam no CSV, so nao eram impressos.
            def cell(tag):
                mean, sd = r[f"{dataset}_{tag}"], r[f"{dataset}_sd_{tag}"]
                if pd.isna(mean):
                    return f"{'—':>14s}"
                return f"{mean:8.2f} {'' if pd.isna(sd) else f'± {sd:.2f}':>5s}"
            print(f"{r.condicao:30s} {dims:>6s} {cell('BAcc')} {cell('Acc')} {cell('F1')}")
    print(f"\nsalvo: {args.output}")


if __name__ == "__main__":
    main()
