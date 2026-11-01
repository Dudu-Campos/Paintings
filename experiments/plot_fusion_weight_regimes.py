"""A quebra da fusao acima de ~20pp e do ESTIMADOR, nao da complementaridade.

Tres series sobre o MESMO eixo, uma coluna por corpus:

  oraculo          o teto: toda imagem que um dos dois lados acerta
  peso 0,5         o publicado -- cruza zero perto de 20pp de gap e fica negativo
  peso escolhido   cross-fit -- positivo nos SETE, inclusive a 27 e 49pp de gap
  fora do dado

O peso do terceiro e escolhido por cross-fitting DENTRO do teste -- o teste e
particionado em 5 dobras, o peso sai das outras quatro e e aplicado so na dobra
retida, entao nenhuma imagem e avaliada com um peso que a viu. Nao equivale a
escolher no `val` oficial (continua usando a distribuicao do teste), mas remove
a circularidade que torna o numero do §4.1d nao reportavel.

**PAINEL UNICO desde 19/set, por decisao do usuario** ("a imagem ficou um pouco
cansativa de ver"). A versao anterior tinha dois paineis empilhados com eixos x
compartilhados, e o leitor precisava saltar entre eles para comparar teto e
realizado do mesmo corpus. As tres series estao na MESMA unidade -- pontos
percentuais sobre o backbone --, entao dividi-las em dois eixos nunca teve
justificativa de escala; era so heranca de como a medicao foi feita. Num eixo so
a distancia vertical entre o verde e o azul vira diretamente "o que a fusao
deixa na mesa", que nenhum dos dois paineis mostrava.

**A coluna do oraculo e a `oracle_bacc`, nao a `oracle_gain`, e o painel unico e
o que obrigou a troca.** A `oracle_gain` herdada do `*_decision_fusion*.csv` e
decomposicao por imagem, portanto ACURACIA SIMPLES (`oracle - backbone_accuracy`
== `only_ours`), enquanto `d_half`/`d_cross` sao BAcc. Com dois paineis e dois
rotulos de eixo a mistura passava; num eixo so ela produziu **cross-fit ACIMA do
teto** no multitask (4,31 contra 4,14). Com o oraculo recalculado em BAcc o teto
domina o cross-fit nas 70 unidades.

O conector cinza liga as tres condicoes do MESMO corpus. Ele e o que torna 21
pontos legiveis: agrupa por coluna antes de o leitor ler qualquer cor. Ligar
corpora diferentes seria a interpolacao que o `plot_fusion_gap.py` ja recusa.

O rotulo vai acima do oraculo, que e o ponto mais alto de toda coluna nos sete
corpora -- um rotulo por corpus, sem dicionario de deslocamento manual.

ATUALIZADO em 19/set: o cross-fit cobre os SETE corpora. A versao anterior tinha
ponto azul so nos quatro de split oficial, com a ressalva de que "os tres de
arte nao tem posteriors nesse formato" -- isso estava errado. Os posteriors OOF
dos tres estao em `outputs/{corpus}_our_oof_cv.npz` desde que o
`evaluate_decision_fusion_cv.py` os cacheou, e o lado do backbone sai dos 5
folds da grade. Nenhum treino novo.

Diferenca de protocolo que a legenda tem de carregar: nos quatro oficiais o
cross-fit particiona um TESTE separado; nos tres de arte particiona a coorte em
OOF, cujas linhas vem de cinco modelos diferentes (um por fold da CV).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HALF, CROSS, ORACLE = "#e34948", "#2a78d6", "#1baf7a"
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
LABEL = {"multitask": "Multitask", "nct_crc": "NCT-CRC", "minc2500": "MINC-2500",
         "oilpainting": "OilPainting", "rijksdraw": "Rijksmuseum",
         "food101": "Food-101", "places_lt": "Places-LT"}
# Deslocamento horizontal do rotulo, em pontos. So os dois corpora que ficam a
# 1pp de gap um do outro precisam: OilPainting (18,7) e Places-LT (19,8).
LABEL_DX = {"oilpainting": -6, "places_lt": 6}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=ROOT / "results/decision_fusion_gap_points_extended.csv")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_regimes.png")
    ap.add_argument("--title",
                    default="Complementarity never vanishes; "
                            "the failure belongs to the fixed weight")
    # Os dois paineis do par (sem texto / com texto) TEM de compartilhar os dois
    # eixos. Com limites automaticos o Food-101 fica na borda direita nos dois,
    # em gap 48 e em gap 29, e o deslocamento que o segundo painel existe para
    # mostrar fica invisivel -- mesma posicao visual, valor diferente, e o tipo
    # de composicao que engana sem errar nenhum numero.
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    args = ap.parse_args()
    t = pd.read_csv(args.input)
    g = t.groupby("corpus").agg(gap=("gap", "mean"), oracle=("oracle_bacc", "mean"),
                                half=("d_half", "mean"), cross=("d_cross", "mean")
                                ).sort_values("gap")

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    fig.patch.set_facecolor("white")

    # Conector por CORPUS, nao entre corpora: agrupa a coluna antes da cor.
    for _, row in g.iterrows():
        lo = min(row.half, row.cross, row.oracle)
        hi = max(row.half, row.cross, row.oracle)
        ax.plot([row.gap, row.gap], [lo, hi], color="#c8c8c8", linewidth=1.3, zorder=1)

    ax.axhline(0, color="#5a5a5a", linewidth=1.1, zorder=2)
    ax.plot(g.gap, g.oracle, "o", color=ORACLE, markersize=11, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.6,
            label="oracle ceiling — either side gets the image right")
    ax.plot(g.gap, g.half, "s", color=HALF, markersize=11, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.6,
            label="decision fusion, fixed weight 0.5")
    ax.plot(g.gap, g.cross, "o", color=CROSS, markersize=11, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=1.6,
            label="decision fusion, weight chosen off the evaluated data")

    for name, row in g.iterrows():
        ax.annotate(LABEL[name], (row.gap, row.oracle), textcoords="offset points",
                    xytext=(LABEL_DX.get(name, 0), 13), ha="center", va="bottom",
                    fontsize=9.5, color="#3a3a3a")

    ax.set_ylabel("gain over the backbone (pp)", fontsize=12, color=MUTED)
    ax.set_xlabel("backbone's advantage over ours (pp)", fontsize=12, color=MUTED)
    ax.set_title(args.title, fontsize=12.5, fontweight="bold", color=INK,
                 loc="left", pad=10)
    ax.set_xlim(-4, args.xmax if args.xmax is not None else g.gap.max() + 6)
    ax.set_ylim(args.ymin if args.ymin is not None else min(g.half.min(), g.cross.min()) - 1.0,
                args.ymax if args.ymax is not None else g.oracle.max() + 2.6)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right", labelcolor=INK,
              handletextpad=0.6, borderaxespad=0.8)
    style(ax)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"escrito: {args.output}")


if __name__ == "__main__":
    main()
