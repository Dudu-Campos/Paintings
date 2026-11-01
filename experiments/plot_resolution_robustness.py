#!/usr/bin/env python3
"""Robustez a RESOLUCAO: o banco precisa de UMA familia de segmentacao em alta.

Dois paineis, os dois em pontos percentuais de BAcc contra a configuracao de
producao -- e e essa unidade comum que faz a figura dizer algo: **a resolucao
paga ~5x mais para as views de estrutura de segmentacao do que para estatistica
global**.

PAINEL A -- o 2x2 de Felzenszwalb x SLIC (§9.5). Nao e um estudo de ablacao
disfarcado: as quatro celulas sao um fatorial completo, e o achado e a
INTERACAO. Estragar uma familia custa ~0; estragar as duas custa 3,5pp.

    Felz \\ SLIC        @1024      @224
    @1024              38,74      38,79      <- tirar o SLIC de alta: +0,05 (n.s.)
    @224               37,34      35,21
                         ^ tirar o Felz de alta: -1,39

    aditivo previa -1,34   ·   medido -3,53   ·   INTERACAO -2,19

**O eixo x separa PIXELS de SEGMENTOS, e e por isso que a coluna do meio
existe.** O `felz_scale` esta em unidades de pixel (§9.3), entao baixar a
resolucao colapsa a contagem de nos de 1.398 para 134 -- duas mudancas ao mesmo
tempo. O braco `@224g` recalibra (`scale=4,0, min_size=8`) e realiza **1.404
segmentos com 1/12 dos pixels**, isolando a resolucao. Ele recupera 0,60pp no
braco isolado e **so 0,23 dos 3,53** no conjunto: granularidade existe e nao e o
que explica a queda.

PAINEL B -- Descriptors (§9.6), a estatistica global, no mesmo eixo. Subir de
128 para 1024 px paga **+0,66pp (5/5 folds)** no banco, contra os 3,30 que as
duas familias de segmentacao custam. Sozinho ele e PLANO (23,70 / 24,42 / 24,00,
nao-monotonico, nada significativo), que e o §5.17 por outro instrumento.

**RESSALVAS que vao na legenda:**
 1. O §9.6 e NAO-MONOTONICO (128 -> 512 desce, 512 -> 1024 sobe) e o efeito vem
    dos folds 1 e 2 (+1,26 e +1,50 contra ~0,2 nos outros tres), com p=0,089.
    O teste que decide e SEMENTE, nao fold (§5.6), e nao foi feito. A figura
    desenha a barra de erro justamente para o leitor ver isso.
 2. Mudar a resolucao muda O QUE e medido, nao so o custo: LBP (R=1 e R=3) e a
    distancia do GLCM estao em unidades de pixel, igual ao `felz_scale`. Nao ha
    separacao limpa sem reparametrizar em unidades relativas.
 3. Tudo e multitask 125 classes, BranchMLP estreito, 5 folds, uma semente.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
# cor por FAMILIA, do §7.18: grafos de regiao em azul, estatistica de imagem em
# aqua. O SLIC usa o tom claro da rampa e o Felz o cheio.
AZUL, AZUL_CLARO, AQUA = "#2a78d6", "#88b3e8", "#1baf7a"
BASE = "tudo @1024"


def serie(d: pd.DataFrame, cond: str):
    g = d[d.condition.eq(cond)].bacc
    return g.mean(), g.std(), len(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/resolution_robustness.png")
    args = ap.parse_args()

    dec = pd.read_csv(ROOT / "results/multitask_resolution_decomposition.csv")
    gra = pd.read_csv(ROOT / "results/multitask_resolution_granularity.csv")
    des = pd.read_csv(ROOT / "results/multitask_descriptors_resolution.csv")
    ref = serie(dec, BASE)[0]

    # PAINEL A: 3 posicoes de Felz x 2 series de SLIC, em delta contra producao
    x = [0, 1, 2]
    xlab = ["Felz @1024\n1.398 seg.", "Felz @224g\n1.404 seg.",
            "Felz @224\n134 seg."]
    slic_alto = [serie(dec, BASE), serie(gra, "só Felz+L(G_felz) @224g"),
                 serie(dec, "só Felz+L(G_felz) @224")]
    slic_baixo = [serie(dec, "só SLIC+L(G) @224"),
                  serie(gra, "tudo @224g (SLIC@224 + Felz@224g)"),
                  serie(dec, "tudo @224")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 5.4),
                               gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.axhline(0, color=MUTED, linewidth=1.1)

    for pts, cor, rot in ((slic_alto, AZUL, "SLIC @1024"),
                          (slic_baixo, AZUL_CLARO, "SLIC @224")):
        m = [p[0] - ref for p in pts]
        s = [p[1] for p in pts]
        a.errorbar(x, m, yerr=s, color=cor, marker="o", markersize=9,
                   linewidth=2.4, capsize=4, markeredgecolor="white",
                   markeredgewidth=1.0, label=rot, zorder=3)
    a.set_xticks(x)
    a.set_xticklabels(xlab, fontsize=9.5)
    a.set_xlim(-0.3, 3.05)
    a.set_ylabel("Δ BAcc against the production setup  (pp)",
                 fontsize=11, color=MUTED)
    a.set_title("A · two segmentation families: the loss is super-additive",
                fontsize=11.5, loc="left", color=INK, fontweight="bold", pad=8)
    a.legend(frameon=False, fontsize=10, loc="lower left", labelcolor=INK)
    # A INTERACAO ANOTADA: e a afirmacao do painel, e sem ela o leitor tem de
    # fazer a subtracao de cabeca.
    ad = (slic_baixo[0][0] - ref) + (slic_alto[2][0] - ref)
    med = slic_baixo[2][0] - ref
    # As DUAS anotacoes viram UMA. Separadas elas colidiam, e nao por acaso: a
    # previsao aditiva (-1,34) cai praticamente EM CIMA do ponto de Felz sozinho
    # (-1,39), porque o efeito do SLIC e ~0. Um rotulo "additive prediction" ali
    # fica sobre a linha azul; o bloco unico diz a mesma coisa sem competir com
    # o traco (§9.7).
    a.annotate("", xy=(2.08, med), xytext=(2.08, ad),
               arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.4))
    a.plot([1.97, 2.19], [ad, ad], color=MUTED, linewidth=1.2,
           linestyle=(0, (4, 3)), zorder=2)
    a.text(2.26, (med + ad) / 2,
           f"interaction\n{med - ad:+.2f} pp\n\nadditive would\npredict {ad:+.2f}",
           fontsize=9, color=MUTED, va="center", linespacing=1.35)

    # PAINEL B: Descriptors, isolado x no banco, delta contra 128 px
    px = [128, 512, 1024]
    iso = [serie(des, f"Descriptors@{p} sozinho") for p in px]
    # `hand-8` aqui e CHAVE DE CSV, nao rotulo: o §7.18 proibe renomear as
    # chaves internas (quebrariam todo o results/) e manda traduzir so na
    # borda de apresentacao, que e a legenda logo abaixo.
    pipe = [serie(des, f"hand-8 (Descriptors@{p})") for p in px]
    for pts, cor, rot, ls in ((pipe, AQUA, "inside PRISM", "-"),
                              (iso, AQUA, "the view on its own", "--")):
        base = pts[0][0]
        b.errorbar(range(3), [p[0] - base for p in pts], yerr=[p[1] for p in pts],
                   color=cor, marker="s" if ls == "-" else "D", markersize=8,
                   linewidth=2.4, linestyle=ls, capsize=4,
                   markeredgecolor="white", markeredgewidth=1.0,
                   label=rot, zorder=3, alpha=1.0 if ls == "-" else 0.55)
    b.set_xticks(range(3))
    b.set_xticklabels([f"{p} px" for p in px], fontsize=10)
    b.set_xlabel("Image descriptors: operating resolution",
                 fontsize=10.5, color=MUTED)
    b.set_title("B · global statistics: resolution barely pays",
                fontsize=11.5, loc="left", color=INK, fontweight="bold", pad=8)
    b.legend(frameon=False, fontsize=10, loc="lower right", labelcolor=INK)
    # o ylim comum e o que torna os dois paineis comparaveis -- e a unidade
    # comum e a afirmacao da figura. Mas ele tem de COBRIR as barras do B, que
    # sao mais altas que as do A no topo.
    lo = min(a.get_ylim()[0], min(p[0] - q[0][0] - p[1]
                                  for q in (pipe, iso) for p in q) - 0.25)
    hi = max(a.get_ylim()[1], max(p[0] - q[0][0] + p[1]
                                  for q in (pipe, iso) for p in q) + 0.25)
    a.set_ylim(lo, hi)
    b.set_ylim(lo, hi)

    fig.suptitle("The bank needs ONE segmentation family at high resolution — "
                 "the second is substitutable",
                 fontsize=13, fontweight="bold", color=INK, x=0.008, ha="left",
                 y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=200, facecolor="white")

    print(f"{args.output}\n")
    print("PAINEL A -- o 2x2 (delta contra producao, pp):")
    for rot, pts in (("SLIC @1024", slic_alto), ("SLIC @224", slic_baixo)):
        print(f"  {rot:11s} " + "  ".join(
            f"{p[0] - ref:+6.2f} +-{p[1]:4.2f}" for p in pts))
    print(f"\n  aditivo previa {ad:+.2f}   medido {med:+.2f}   "
          f"INTERACAO {med - ad:+.2f}pp")
    print("\nPAINEL B -- Descriptors (delta contra 128 px, pp):")
    for rot, pts in (("no banco", pipe), ("sozinho", iso)):
        print(f"  {rot:9s} " + "  ".join(
            f"{p[0] - pts[0][0]:+6.2f} +-{p[1]:4.2f}" for p in pts))
    print(f"\nrazao: as duas familias de segmentacao custam {-med:.2f}pp; "
          f"subir Descriptors a 1024 paga {pipe[2][0] - pipe[0][0]:+.2f}pp "
          f"-- {-med / (pipe[2][0] - pipe[0][0]):.1f}x")


if __name__ == "__main__":
    main()
