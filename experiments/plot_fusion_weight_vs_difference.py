#!/usr/bin/env python3
"""O peso otimo contra a DIFERENCA ASSINADA de desempenho entre os dois lados.

Proposta do usuario em 13/out: *"o que acha de um scatter peso ideal e quao
longe estao os 2 ou 3 modelos [...] se a diferenca de performance e grande ou
pequena"*. Ela estende a `fusion_weight_vs_gap.png` do §4.1f item 7, que tem os
14 pontos de backbone, com o par PRISM x Fisher vector e com o braco de texto.

**O QUE O FISHER VECTOR ACRESCENTA, e e a razao da figura existir: a METADE
POSITIVA DO EIXO.** Contra os backbones fine-tuned nos somos o lado pior em
**28 de 28** pontos, entao a figura antiga so amostra `diferenca < 0` e a
alegacao do §4.1f item 7 de que *"o mecanismo e simetrico, nao uma correcao
unilateral para quando somos piores"* se apoia num unico ponto (o rijksdraw
acima de 0,5). Contra o Fisher vector nos somos o lado MELHOR em **12 de 14**,
e o eixo passa a ir de -48,7 a +23,5pp, cruzando a paridade.

**O eixo x e a diferenca de LOG-ODDS**, nao de pontos percentuais (§4.1g):
rho=+0,755 contra +0,688, e a razao e a mesma de la -- a mesma diferenca em pp
vale mais perto do teto e do piso.

**O eixo y comeca em ZERO e vai a UM, e isso e a figura inteira.** `w = 0` e
literalmente a hipotese "o combinador aprende a ignorar o nosso lado" e `w = 1`
e o espelho dela. Com o eixo apertado na faixa dos dados sobraria so a
inclinacao, que e a metade banal da pergunta.

**SEM RETA AJUSTADA** (§4.1f item 7), e aqui ha um motivo a mais, medido em
13/out: um ajuste linear sobre este x cruza `w=0,5` em **-4,45pp**, o que
sugeriria que o nosso lado recebe mais de metade do voto mesmo sendo pior. Isso
**NAO sobrevive ao teste nao-parametrico** -- na faixa `|dif| < 5pp` a mediana
de `w` e 0,494 com 6/13 acima de 0,5 (binomial p=1,0). O cruzamento deslocado
era artefato de ajustar reta num x assimetrico (30 pontos negativos, indo ate
-48, contra 12 positivos). Nao escrever aquela leitura.

**A RESSALVA CENTRAL, e ela tem de ir na legenda: os dois pares quase nao se
sobrepoem em x.** O par com Fisher vector ocupa -3,2 a +23,5 e o par com
backbone -48,7 a -0,6. Entao a figura NAO testa "a mesma relacao vale para
qualquer segundo modelo" -- ela mostra dois regimes adjacentes. E na faixa fina
onde eles se encontram (-3,3 a -0,5pp, 6 pontos) o `w` vai de **0,464 a 0,790**,
amplitude de 0,33, que e **53% da amplitude total de `w` na serie inteira**. Ou
seja: a mesma diferenca de desempenho admite pesos muito diferentes.

Isso NAO e defeito da figura, e o achado que ela torna visivel. O §4.1f item 7
ja mediu a residual como correlacao parcial -- `folga do oraculo, controlando o
gap`, +0,482 (p=0,081) -- e concluiu que o peso pondera por **contribuicao** e
nao so por **confiabilidade**. Ali aquilo e um numero marginal numa tabela; aqui
e o espalhamento vertical a x fixo, que o leitor ve sozinho. A alegacao que a
figura sustenta e *a diferenca de desempenho e necessaria e nao suficiente*.
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
TINTA = "#3f6fa8"
FORMA = {"Fisher vector": ("s", "Fisher vector"),
         "ResNet50 ft": ("o", "ResNet50 fine-tuned"),
         "ViT-B/16 ft": ("^", "ViT-B/16 fine-tuned")}
MODELO = {"resnet50": "ResNet50 ft", "vit_b_16": "ViT-B/16 ft"}
FONTES = {"sem": "results/fusion_weight_crossfit.csv",
          "com": "results/fusion_weight_crossfit_text.csv"}


def logit(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def pontos() -> pd.DataFrame:
    """42 pontos: 7 corpora x 3 segundos modelos x 2 bracos.

    O `nosso` e o `outro` saem do `fusion_quad.csv` para o par com backbone --
    e o unico produtor que guarda os dois lados nas MESMAS linhas -- e do
    `fusion_weight_banks.csv` para o par com Fisher vector. O `w*` e o peso que
    ESPIA o teste nos dois, que e o que os dois CSVs tem em comum e e o alvo
    que os estimadores honestos tentam acertar.
    """
    q = pd.read_csv(ROOT / "results/fusion_quad.csv")
    L = []
    for braco, arq in FONTES.items():
        d = pd.read_csv(ROOT / arq)
        for (corpus, modelo), g in d.groupby(["corpus", "model"]):
            h = q[q.dataset.eq(corpus) & q.braco.eq(braco) & q.modelo.eq(modelo)]
            if not len(h):
                continue
            L.append({"corpus": corpus, "par": MODELO[modelo], "braco": braco,
                      "nosso": h.prism_BAcc.mean(), "outro": h.backbone_BAcc.mean(),
                      "w": g.best_weight.mean()})
    d = pd.read_csv(ROOT / "results/fusion_weight_banks.csv")
    for (corpus, braco), g in d.groupby(["dataset", "braco"]):
        L.append({"corpus": corpus, "par": "Fisher vector", "braco": braco,
                  "nosso": g.prism.mean(), "outro": g.fv.mean(),
                  "w": g.w_espia.mean()})
    t = pd.DataFrame(L)
    t["dif"] = t.nosso - t.outro
    t["dlog"] = logit(t.nosso / 100) - logit(t.outro / 100)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eixo", default="dlog", choices=["dlog", "dif"])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_vs_difference.png")
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "results/fusion_weight_vs_difference.csv")
    args = ap.parse_args()

    t = pontos()
    t.to_csv(args.csv, index=False, float_format="%.4f")
    X = args.eixo

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10.5)

    # As DUAS hipoteses que o eixo y nomeia, e a paridade de voto no meio.
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=(0, (5, 4)), zorder=1)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=1)

    for (corpus, par), g in t.groupby(["corpus", "par"]):
        if len(g) == 2:
            ax.plot(g[X], g.w, color=GRID, linewidth=1.4, zorder=2)
    for r in t.itertuples():
        m, _ = FORMA[r.par]
        cheio = r.braco == "com"
        ax.plot(getattr(r, X), r.w, marker=m, markersize=9.5, linestyle="none",
                markerfacecolor=TINTA if cheio else "white",
                markeredgecolor=TINTA, markeredgewidth=1.8, zorder=3)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    pad = 0.07 * (t[X].max() - t[X].min())
    ax.set_xlim(t[X].min() - pad, t[X].max() + pad)

    # As duas pontas do eixo y sao as duas hipoteses nomeadas; sem isso o leitor
    # le so a inclinacao, que e a metade banal da pergunta.
    ax.text(0.985, 0.02, "w = 0 : the combiner ignores our side",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    ax.text(0.985, 0.955, "w = 1 : it ignores the other model",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    ax.text(0.012, 0.52, "equal vote", transform=ax.transAxes,
            fontsize=9, color=MUTED)

    rot = ("difference in log-odds   (ours − other model)" if X == "dlog"
           else "difference in balanced accuracy   (ours − other model, pp)")
    ax.set_xlabel(rot, fontsize=11.5, color=MUTED)
    ax.set_ylabel("optimal weight on our side,  w*", fontsize=11.5, color=MUTED)
    ax.set_title("The weight tracks the performance difference —\n"
                 "but never reaches zero, and the difference is not enough",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    chave = [Line2D([], [], marker=mk, color=TINTA, linestyle="none",
                    markersize=9, markerfacecolor="white", markeredgewidth=1.8,
                    label=r) for mk, r in FORMA.values()]
    chave += [Line2D([], [], marker="o", color=MUTED, linestyle="none",
                     markersize=9, markerfacecolor="white", markeredgewidth=1.8,
                     label="hollow: PRISM without text"),
              Line2D([], [], marker="o", color=MUTED, linestyle="none",
                     markersize=9, markerfacecolor=MUTED, label="filled: with text")]
    ax.legend(handles=chave, frameon=False, fontsize=9.5, loc="upper left",
              labelcolor=INK, handletextpad=0.6, borderaxespad=1.1)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, facecolor="white")

    rho, p = stats.spearmanr(t[X], t.w)
    print(f"{len(t)} pontos -> {args.output}")
    print(f"  w* x {X}:  Spearman rho={rho:+.3f}  p={p:.2e}")
    print(f"  w* de {t.w.min():.3f} a {t.w.max():.3f}  (nunca 0, nunca 1)")
    print(f"  x de {t.dif.min():+.1f} a {t.dif.max():+.1f}pp")
    print("\nquem ocupa cada metade do eixo:")
    for par, g in t.groupby("par"):
        print(f"  {par:14s} nosso lado melhor em {int((g.dif > 0).sum())}/{len(g)}"
              f"   x de {g.dif.min():+7.2f} a {g.dif.max():+6.2f}pp")
    ov = t[t.dif.between(-3.3, -0.5)]
    print(f"\nfaixa de sobreposicao entre pares ({len(ov)} pontos): w de "
          f"{ov.w.min():.3f} a {ov.w.max():.3f}, amplitude "
          f"{100 * (ov.w.max() - ov.w.min()) / (t.w.max() - t.w.min()):.0f}% da total")
    print(f"[{args.csv}]")


if __name__ == "__main__":
    main()
