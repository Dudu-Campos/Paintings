#!/usr/bin/env python3
"""O 2x2 do nosso lado, sempre contra o backbone fine-tuned.

Pedido do usuario em 13/out: *"faca 4x4, com e sem texto, com e sem fisher e a
fusao final com os backbones"*. Sao quatro configuracoes do lado nosso --
{sem texto, com texto} x {sem Fisher vector, com Fisher vector} -- e em todas o
desfecho medido e o mesmo: BAcc da fusao com o backbone menos BAcc do backbone
sozinho.

**O EIXO Y E A DIFERENCA ENTRE OS MODELOS, e a primeira versao errava nisso.**
Ela punha o corpus no eixo y, em escada. O usuario rejeitou com o argumento
certo: *"com essa escadinha eu queria mostrar que mesmo uma fusao ingenua e boa
quando os modelos sao comparaveis, mas sem eixo y de diferenca dos modelos nao
da pra saber isso"*. Com o corpus no eixo, a variavel da tese fica FORA da
figura e a escada vira ordenacao de corpora, que nao e o achado. (A versao
antiga continua acessivel em `--por-corpus`, porque ela mostra o detalhe por
corpus que esta nao mostra.)

Agora cada painel e uma dispersao:

    x = diferenca de LOG-ODDS entre o nosso lado e o backbone (§4.1g)
    y = ganho da fusao de decisao sobre o backbone SOZINHO, em pp

e a INCLINACAO DENTRO DE CADA PAINEL e a afirmacao. Medido: rho=+0,901
(p=3e-21) nos 56 pontos, e +0,864 a +0,903 por painel.

**O `x` e o lado QUE REALMENTE ENTRA na fusao.** Na linha de baixo o nosso lado
e `PRISM + FV`, nao o `PRISM`, entao o x usa a BAcc da fusao dos dois bancos.
E isso que torna a figura causal em vez de descritiva: cada adicao ao nosso lado
move o ponto para a DIREITA (encurta a diferenca) e para CIMA (aumenta o ganho),
e sao os dois movimentos juntos que sustentam a tese.

**O que a figura mede, e e ordinal, sem curva ajustada:**

    diferenca de log-odds     n    ganho medio    celulas positivas
    [-2,5 , -1,5)             6      -0,16pp           3/6
    [-1,5 , -0,8)            15      -0,12pp           7/15
    [-0,8 , -0,3)            23      +2,10pp          23/23
    [-0,3 , +0,1)            10      +4,80pp          10/10

Acima de -0,90 de diferenca, **33 de 33 celulas sao positivas**; abaixo, 10 de
21. O limiar e LIDO do dado, nao ajustado, entao vai no texto e **nao e
desenhado** -- uma vertical ali seria lida como lei, e o §4.1f item 7 mede que
nenhuma forma funcional e identificavel com estes corpora.

**O 2x2 continua sendo o layout**: ler da esquerda para a direita e o que o
TEXTO compra, de cima para baixo o que o FISHER VECTOR compra, e os quatro
paineis compartilham os dois eixos (§5.9d -- a comparacao e de silhuetas e so
existe se a escala for a mesma).

**Cor por corpus e segura AQUI.** A paleta de sete tem uma colisao forcada de
matiz (violeta a 22,9 graus do azul, §7.18) e a mitigacao declarada no
`plot_fusion_gap.py` e por nos os dois corpora mais distantes no eixo x. Naquela
figura o x e o gap, e nesta tambem: medido, `Multitask` fica em -0,43 e
`MINC-2500` em -1,31. A forma continua separando os dois backbones.

RESSALVAS que vao na legenda:
 1. **As quatro condicoes rodam nas MESMAS linhas** -- as em que o Fisher vector
    e honesto --, entao nos tres corpora de arte as celulas `sem FV` nao sao, em
    principio, os valores publicados do §4.1f, que usam a coorte inteira.
    Medido: elas reproduzem dentro de **0,06pp**, inclusive no `rijksdraw`, que
    aqui usa 9.659 das 48.295 linhas.
 2. **O peso e aninhado**: `((PRISM + FV) + backbone)` com 0,5 em cada nivel da
    0,25/0,25/0,50. O backbone recebe 0,50 nas QUATRO condicoes, e e isso que
    torna os paineis comparaveis.
 3. O `n` por celula e 10 nos quatro de split oficial e no `rijksdraw` (onde o
    FV so e honesto no fold 1) e 50 no `oilpainting` e no `multitask`.
 4. Com o backbone dentro, NENHUM painel sustenta "zero supervisao externa"
    (§7.2) -- a figura e sobre complementaridade. A coluna com texto muda o TIPO
    da alegacao mais uma vez, nao a forca.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
PERDA = "#f4e8e8"
COLOUR = {"multitask": "#2a78d6", "minc2500": "#4a3aa7", "nct_crc": "#1baf7a",
          "oilpainting": "#e34948", "rijksdraw": "#eda100",
          "food101": "#e87ba4", "places_lt": "#008300"}
LABEL = {"multitask": "Multitask 125cl", "nct_crc": "NCT-CRC",
         "minc2500": "MINC-2500", "oilpainting": "OilPainting",
         "rijksdraw": "Rijksmuseum drawings", "food101": "Food-101",
         "places_lt": "Places-LT"}
MARCA = {"resnet50": ("o", "ResNet50 fine-tuned"),
         "vit_b_16": ("^", "ViT-B/16 fine-tuned")}
# (coluna, linha) -> titulo.  colunas = texto, linhas = Fisher vector
TITULO = {("sem", "sem_fv"): "PRISM",
          ("com", "sem_fv"): "PRISM + text",
          ("sem", "com_fv"): "PRISM + Fisher vector",
          ("com", "com_fv"): "PRISM + text + Fisher vector"}
# qual coluna do CSV e o NOSSO LADO em cada condicao
NOSSO = {"sem_fv": "prism", "com_fv": "prism_fv"}


def logit(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def celulas(t: pd.DataFrame, M: str) -> pd.DataFrame:
    """Uma linha por (braco, condicao, corpus, backbone): x, y e o lado nosso."""
    L = []
    for braco in ("sem", "com"):
        for cond in ("sem_fv", "com_fv"):
            g = t[t.braco.eq(braco)]
            for (ds, mo), h in g.groupby(["dataset", "modelo"]):
                nosso = h[f"{NOSSO[cond]}_{M}"].mean()
                back = h[f"backbone_{M}"].mean()
                L.append({"braco": braco, "cond": cond, "dataset": ds,
                          "modelo": mo, "nosso": nosso, "backbone": back,
                          "x": logit(nosso / 100) - logit(back / 100),
                          "y": h[f"{cond}_{M}"].mean() - back,
                          "pos": int((h[f"{cond}_{M}"] > h[f"backbone_{M}"]).sum()),
                          "n": len(h)})
    return pd.DataFrame(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "results/fusion_quad.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "results/fusion_quad.png")
    ap.add_argument("--metrica", default="BAcc", choices=["BAcc", "Acc", "F1"])
    ap.add_argument("--por-corpus", action="store_true",
                    help="a versao antiga: corpus no eixo y, em escada")
    args = ap.parse_args()

    t = pd.read_csv(args.csv)
    M = args.metrica
    c = celulas(t, M)
    c.to_csv(args.output.with_suffix(".cells.csv"), index=False, float_format="%.4f")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True, sharey=True)
    xlo, xhi = c.x.min(), c.x.max()
    ylo, yhi = c.y.min(), c.y.max()
    px, py = 0.09 * (xhi - xlo), 0.10 * (yhi - ylo)

    for i, cond in enumerate(["sem_fv", "com_fv"]):
        for j, braco in enumerate(["sem", "com"]):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            ax.axhspan(ylo - py, 0, color=PERDA, zorder=0)
            ax.axhline(0, color=MUTED, linewidth=1.1, zorder=2)
            ax.grid(color=GRID, linewidth=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=10)

            g = c[c.braco.eq(braco) & c["cond"].eq(cond)]
            # conector cinza entre os dois backbones do MESMO corpus: agrupa os
            # 14 pontos antes de o leitor ler qualquer cor (§4.1f)
            for ds, h in g.groupby("dataset"):
                if len(h) == 2:
                    ax.plot(h.x, h.y, color=GRID, linewidth=1.3, zorder=2)
            for r in g.itertuples():
                mk, _ = MARCA[r.modelo]
                ax.plot(r.x, r.y, marker=mk, markersize=9, linestyle="none",
                        markerfacecolor=COLOUR[r.dataset], markeredgecolor="white",
                        markeredgewidth=0.9, zorder=4)

            ax.set_title(TITULO[(braco, cond)], fontsize=11.5, loc="left",
                         color=INK, fontweight="bold", pad=7)
            ax.set_xlim(xlo - px, xhi + px)
            ax.set_ylim(ylo - py, yhi + py)
            if i == 1:
                ax.set_xlabel("difference in log-odds   (our side − backbone)",
                              fontsize=10.5, color=MUTED)
            # o rotulo de y vai UMA vez, no `supylabel`: posto nos dois
            # paineis da coluna esquerda os dois se sobrepoem no meio da figura

    chave = [Line2D([], [], marker=mk, color=MUTED, linestyle="none",
                    markersize=9, markeredgecolor="white", markeredgewidth=0.9,
                    label=rot) for mk, rot in MARCA.values()]
    axes[0][0].legend(handles=chave, frameon=False, fontsize=9.5,
                      loc="upper left", labelcolor=INK, handletextpad=0.5)
    cores = [Line2D([], [], marker="s", color=COLOUR[k], linestyle="none",
                    markersize=8, label=LABEL[k]) for k in COLOUR]
    fig.legend(handles=cores, frameon=False, fontsize=9.5, loc="lower center",
               labelcolor=INK, handletextpad=0.5, ncol=4,
               bbox_to_anchor=(0.5, -0.004))
    fig.supylabel(f"fusion gain over the backbone alone  ({M}, pp)",
                  fontsize=10.5, color=MUTED, x=0.008)
    # TITULO: "pays EXACTLY when comparable" seria um SE E SOMENTE SE, e o lado
    # esquerdo nao sustenta -- abaixo de -0,90 o desfecho e 8/18, cara ou coroa,
    # nao perda confiavel. O que e verdade e a direcao: acima do limiar, 37/38.
    fig.suptitle("Equal-weight fusion pays once the two sides are comparable",
                 fontsize=13, fontweight="bold", color=INK,
                 x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    fig.savefig(args.output, dpi=200, facecolor="white")

    rho, p = stats.spearmanr(c.x, c.y)
    print(f"{args.output}\n  {len(c)} celulas   rho={rho:+.3f}  p={p:.2e}")
    print(f"\n{'condicao':30s} {'rho':>7s} {'celulas > 0':>12s} {'media':>8s} {'pior':>8s}")
    for cond in ("sem_fv", "com_fv"):
        for braco in ("sem", "com"):
            g = c[c.braco.eq(braco) & c["cond"].eq(cond)]
            print(f"{TITULO[(braco, cond)]:30s} {stats.spearmanr(g.x, g.y)[0]:+7.3f} "
                  f"{int((g.y > 0).sum()):6d}/{len(g):<5d} {g.y.mean():+8.2f} {g.y.min():+8.2f}")
    print("\na tese, em faixas da diferenca (sem curva ajustada):")
    for lo, hi in [(-2.5, -1.5), (-1.5, -0.8), (-0.8, -0.3), (-0.3, 0.2)]:
        g = c[(c.x >= lo) & (c.x < hi)]
        print(f"  x em [{lo:+.1f},{hi:+.1f})  n={len(g):2d}  ganho medio {g.y.mean():+5.2f}pp"
              f"   positivos {int((g.y > 0).sum())}/{len(g)}")
    alto = c[c.x > -0.90]
    print(f"\nacima de -0,90 de diferenca: {int((alto.y > 0).sum())}/{len(alto)} positivas"
          f"   abaixo: {int((c[c.x <= -0.90].y > 0).sum())}/{len(c[c.x <= -0.90])}")


if __name__ == "__main__":
    main()
