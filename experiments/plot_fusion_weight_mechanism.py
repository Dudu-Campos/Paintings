#!/usr/bin/env python3
"""O regime do peso e do ESTIMADOR, nao de qual modelo esta do outro lado.

A figura responde a pergunta do usuario em 13/out: *"temos (prism com e sem
texto) fisher vec e (resnet ou vit) e 3 jeitos de estabelecer o peso, como
visualizariamos isso?"*

**O EIXO E ESCOLHIDO POR COMENSURABILIDADE, nao por estetica.** Nos eixos de
desempenho as quatro dimensoes NAO sao comparaveis -- `Delta sobre o backbone` e
`Delta sobre o Fisher vector` nao sao a mesma quantidade, e junta-las num eixo e
o erro de metrica que o §4.1f item 1 registra (ele produziu cross-fit ACIMA do
proprio teto de oraculo). Aqui:

    x = |w* - 0,5|   deslocamento do peso, adimensional
    y = recuperacao do cross-fit sobre o peso fixo, em pp

As duas sao a mesma quantidade para qualquer segundo modelo, entao os 42 pontos
sao um unico conjunto e nao tres conjuntos empilhados.

**UM PAINEL, e isso e medido.** A inclinacao e a mesma nos dois bracos de texto
(+6,67 sem contra +6,55 com), entao um segundo painel duplicaria a relacao e
cortaria pela metade a densidade do unico achado. O braco entra como
PREENCHIMENTO, e a figura passa a afirmar a invariancia aos tres eixos de uma
vez. E o oposto do `fusion_weight_pair.png`, onde o painel duplo se justifica
porque lá o x e o gap e o texto o desloca.

**OS TRES JEITOS DE ESTABELECER O PESO FICAM FORA, de proposito.** Medido em
13/out: `a/(a+b)` e o cross-fit diferem por menos de 0,1pp em 6 dos 7 corpora do
par com Fisher vector -- eles nao se separam visualmente, e tres series
sobrepostas triplicariam os pontos sem acrescentar leitura. A unica celula que
os distingue (`places_lt` com texto) tem numero, e numero vai em tabela. A serie
plotada e o **cross-fit**, que e o estimador que o §4.1f ja reporta na figura.

**SEM RETA AJUSTADA**, por dois motivos somados. O do §4.1f item 7 -- nenhuma
forma funcional e identificavel -- e um que e proprio desta figura: **a origem e
IDENTIDADE, nao dado.** Se o peso honesto for exatamente 0,5 a recuperacao e
exatamente zero por construcao, entao qualquer ajuste tem parte do R² garantida.
A alegacao ordinal sobrevive sem essa vizinhanca (x > 0,05: rho=+0,909, n=31),
e e isso que a legenda reporta.

RESSALVAS que tem de ir na legenda:
 1. O `x` e o peso que **ESPIA o teste** (`best_weight`), que e diagnostico e
    nao e reportavel como resultado. Ele e o x certo porque e o ALVO que os
    estimadores honestos tentam acertar -- mas tem de ser declarado.
 2. A serie `Fisher vector / sem texto` **sozinha nao resolve** (rho=+0,393,
    p=0,38). Nao e falha: a faixa de x dela e 0,010-0,120, curta demais. E
    justamente a serie que gerou a pergunta do usuario.
 3. O `|w*-0,5|` agregado por CORPUS reproduz o +0,955 do §4.1f ao milesimo;
    por CELULA (corpus x modelo) da +0,906. Sao estatisticas diferentes e nao
    devem ser citadas lado a lado.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

# FONTE UNICA da paleta por corpus: o §7.18 registra que ja houve QUATRO mapas
# view->cor incompativeis em experiments/, e os sete slots do `plot_fusion_gap`
# ja passaram a busca sobre os tres dicromatismos de Machado.
from plot_fusion_gap import COLOUR, LABEL  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
# forma = SEGUNDO MODELO. Identidade nunca e so cor (§7.18): a forma separa os
# tres segundos modelos e o preenchimento separa os dois bracos de texto.
FORMA = {"Fisher vector": ("s", "Fisher vector"),
         "ResNet50 ft": ("o", "ResNet50 fine-tuned"),
         "ViT-B/16 ft": ("^", "ViT-B/16 fine-tuned")}
FONTES_BACKBONE = {"sem": "results/fusion_weight_crossfit.csv",
                   "com": "results/fusion_weight_crossfit_text.csv"}
FONTE_FV = "results/fusion_weight_banks.csv"
MODELO = {"resnet50": "ResNet50 ft", "vit_b_16": "ViT-B/16 ft"}


def pontos() -> pd.DataFrame:
    """Os 42 pontos: 7 corpora x 3 segundos modelos x 2 bracos de texto.

    O `y` sai da MESMA definicao nos dois pares -- BAcc do cross-fit menos BAcc
    do peso fixo, pareado por unidade antes de agregar --, e o `x` do peso que
    espia o teste, que e o que os dois CSVs guardam em comum.
    """
    L = []
    for braco, arq in FONTES_BACKBONE.items():
        p = ROOT / arq
        if not p.exists():
            raise SystemExit(f"ausente: {arq}")
        d = pd.read_csv(p)
        for (corpus, modelo), g in d.groupby(["corpus", "model"]):
            L.append({"corpus": corpus, "par": MODELO[modelo], "braco": braco,
                      "x": abs(g.best_weight.mean() - 0.5),
                      "y": (g.w_cross - g.w_half).mean(), "n": len(g)})
    d = pd.read_csv(ROOT / FONTE_FV)
    for (corpus, braco), g in d.groupby(["dataset", "braco"]):
        L.append({"corpus": corpus, "par": "Fisher vector", "braco": braco,
                  "x": abs(g.w_espia.mean() - 0.5),
                  "y": (g.cross_BAcc - g.meio_BAcc).mean(), "n": len(g)})
    return pd.DataFrame(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_mechanism.png")
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "results/fusion_weight_mechanism.csv")
    ap.add_argument("--mono", action="store_true",
                    help="sem cor por corpus: a figura afirma SO a relacao, e "
                         "ai nada e identificado por cor (§7.18)")
    ap.add_argument("--rotulos", action="store_true",
                    help="rotula o corpus em cada ponto; com 42 pontos colide")
    args = ap.parse_args()

    t = pontos()
    t.to_csv(args.csv, index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10.5)

    # y=0 e a fronteira: abaixo dela escolher o peso PIORA o desfecho.
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=1)

    # CONECTOR entre os dois bracos de texto da MESMA celula (corpus x modelo).
    # E a mesma funcao que o §4.1f da ao conector cinza: agrupar antes que o
    # leitor leia qualquer cor. E ele mostra de graca o que o texto FAZ --
    # desloca a celula AO LONGO da relacao, nao para fora dela.
    cor = (lambda c: "#3f6fa8") if args.mono else (lambda c: COLOUR[c])
    for (corpus, par), g in t.groupby(["corpus", "par"]):
        if len(g) == 2:
            ax.plot(g.x, g.y, color=GRID, linewidth=1.4, zorder=2)

    for r in t.itertuples():
        m, _ = FORMA[r.par]
        cheio = r.braco == "com"
        ax.plot(r.x, r.y, marker=m, markersize=9.5, linestyle="none",
                markerfacecolor=cor(r.corpus) if cheio else "white",
                markeredgecolor=cor(r.corpus), markeredgewidth=1.8,
                zorder=3)
        if args.rotulos:
            ax.annotate(LABEL[r.corpus], (r.x, r.y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color=MUTED)

    rho, p = stats.spearmanr(t.x, t.y)
    m = t[t.x > 0.05]
    rho2, p2 = stats.spearmanr(m.x, m.y)

    ax.set_xlabel("weight displacement  |w* − 0.5|", fontsize=11.5, color=MUTED)
    ax.set_ylabel("gain of the out-of-fold weight over w = 0.5  (pp)",
                  fontsize=11.5, color=MUTED)
    ax.set_title("The weight regime belongs to the estimator,\n"
                 "not to the model on the other side",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    # A chave de FORMA fica dentro (o canto superior esquerdo e vazio por
    # construcao: deslocamento pequeno nao pode dar ganho grande). A de COR vai
    # para FORA do eixo -- dentro ela caia sobre os pontos e a linha y=0 cortava
    # o rotulo do Places-LT.
    #
    # O preenchimento NAO ganha glifo de legenda: um circulo vazado ali
    # significaria "ResNet50" na primeira chave e "sem texto" na segunda, duas
    # coisas com o mesmo desenho. Vai como NOTA, que e inequivoco.
    chave = [Line2D([], [], marker=mk, color=MUTED, linestyle="none",
                    markersize=9, markerfacecolor="white", markeredgewidth=1.8,
                    label=rot)
             for mk, rot in FORMA.values()]
    primeira = ax.legend(handles=chave, frameon=False, fontsize=10,
                         loc="upper left", labelcolor=INK, handletextpad=0.6,
                         borderaxespad=1.4, title="second model",
                         title_fontproperties={"weight": "bold", "size": 9.5})
    primeira._legend_box.align = "left"
    ax.add_artist(primeira)
    ax.text(0.018, 0.655, "hollow = PRISM without text\nfilled  = PRISM with text",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")

    if not args.mono:
        cores = [Line2D([], [], marker="s", color=COLOUR[c], linestyle="none",
                        markersize=8, label=LABEL[c]) for c in COLOUR]
        fig.legend(handles=cores, frameon=False, fontsize=9.5, loc="lower center",
                   labelcolor=INK, handletextpad=0.5, ncol=4,
                   bbox_to_anchor=(0.5, -0.005))
        fig.subplots_adjust(bottom=0.23)

    fig.tight_layout(rect=(0, 0 if args.mono else 0.11, 1, 1))
    fig.savefig(args.output, dpi=200, facecolor="white")

    print(f"{len(t)} pontos -> {args.output}")
    print(f"  Spearman todos        rho={rho:+.3f}  p={p:.2e}")
    print(f"  sem a origem (x>0,05) rho={rho2:+.3f}  p={p2:.2e}  n={len(m)}")
    print("\npor serie:")
    for (br, par), g in t.groupby(["braco", "par"]):
        r, pp = stats.spearmanr(g.x, g.y)
        print(f"  {br} texto / {par:14s} n={len(g)}  rho={r:+.3f} p={pp:.4f}"
              f"  inclinacao {np.polyfit(g.x, g.y, 1)[0]:+5.2f}")
    print("\npor painel hipotetico (a razao de NAO dividir em dois):")
    for br, g in t.groupby("braco"):
        print(f"  {br} texto  n={len(g)}  inclinacao {np.polyfit(g.x, g.y, 1)[0]:+5.2f}")
    print(f"\n[{args.csv}]")


if __name__ == "__main__":
    main()
