"""O peso da fusao contra a vantagem do backbone: ele encolhe, e nao vai a zero.

Responde a pergunta do usuario (22/set): *"sabemos a relacao entre peso escolhido
e diferenca de performance dos backbones? quando somos pior ele somente aprende a
ignorar o nosso?"* A resposta e o §4.1f item 7, e esta figura carrega as duas
metades dela num scatter so.

**O EIXO Y COMECA EM ZERO, e isto nao e negociavel.** A afirmacao da figura e a
DISTANCIA ate zero -- `w = 0` e literalmente a hipotese "ele ignora o nosso lado
e vira o backbone puro". Com o eixo comecando em 0,15 o achado fica invisivel e a
figura passa a mostrar so a inclinacao, que e a metade banal.

**Duas referencias horizontais, e cada uma nomeia uma hipotese:**
  `w = 0,5`  o peso fixo, a constante arbitraria que o item 2 contrasta
  `w = 0`    "ignorar o nosso lado" -- o piso que nenhum ponto encosta

**SEM RETA AJUSTADA**, pela mesma razao que `decision_fusion_gap.png` nao tem: o
item 5 mede que nenhuma forma funcional e identificavel com 7 corpora, e a
logistica vencedora ajusta uma transicao de 1pp de largura num vao sem dado. A
alegacao e ordinal e a figura tem de parecer ordinal.

**O Δ da fusao NAO entra aqui**, nem como tamanho de marcador. "O ganho nunca
some" ja e a afirmacao de `fusion_weight_pair.png`; repetir sobrecarrega esta
figura sem acrescentar, e tamanho de marcador seria lido como importancia.

Um conector cinza liga os DOIS backbones do mesmo corpus, que e o que torna 14
pontos legiveis -- o leitor agrupa por par antes de ler qualquer marcador, mesmo
truque de `plot_fusion_weight_pair.py`. Identidade do modelo vai por FORMA, nao
por cor (§7.18: nenhuma figura pode identificar serie so por cor).
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
GRID, INK, MUTED = "#dedede", "#0b0b0b", "#52514e"
OURS, BACKBONE = "#2a78d6", "#e34948"
LABEL = {"multitask": "Multitask", "nct_crc": "NCT-CRC", "minc2500": "MINC-2500",
         "oilpainting": "OilPainting", "rijksdraw": "Rijksmuseum",
         "food101": "Food-101", "places_lt": "Places-LT"}
MARK = {"resnet50": ("o", "ResNet50"), "vit_b_16": ("^", "ViT-B/16")}
# (dx, dy, ha) em pontos, ancorado no marcador de CIMA quando dy>0 e no de BAIXO
# quando dy<0. O par critico e OilPainting (gap 18-19) contra Places-LT (19,7):
# quase a mesma coluna, separados so em y -- um vai para cima, o outro para baixo.
NUDGE = {"rijksdraw": (8, 9, "left"), "nct_crc": (0, -17, "center"),
         "multitask": (0, 11, "center"), "oilpainting": (6, 10, "left"),
         "places_lt": (2, -17, "center"), "minc2500": (-13, -19, "right"),
         "food101": (-8, 10, "right")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path,
                    default=ROOT / "results/fusion_weight_vs_gap.csv")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_vs_gap.png")
    args = ap.parse_args()
    t = pd.read_csv(args.points)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    # As duas hipoteses, como linhas nomeadas.
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=(0, (5, 4)), zorder=1)
    ax.text(49.6, 0.515, "equal weight (w = 0.5)", fontsize=9, color=MUTED,
            ha="right", va="bottom")
    ax.axhline(0.0, color=BACKBONE, linewidth=1.4, zorder=1)
    ax.text(49.6, 0.018, "backbone alone (w = 0) — never approached",
            fontsize=9, color=BACKBONE, ha="right", va="bottom")

    for corpus, part in t.groupby("corpus"):
        part = part.sort_values("gap")
        ax.plot(part.gap, part.weight, color=GRID, linewidth=1.4, zorder=2)
        # Desvio ENTRE SEMENTES. Sem ele o piso parece artefato de media, e a
        # pergunta certa de um cetico e se a barra encosta em zero -- nao encosta
        # em nenhuma celula (a mais larga e MINC/ResNet, 0,11).
        ax.errorbar(part.gap, part.weight, yerr=part.weight_sd, fmt="none",
                    ecolor=OURS, elinewidth=1.1, capsize=2.5, alpha=0.55, zorder=2)
        for _, r in part.iterrows():
            ax.scatter(r.gap, r.weight, marker=MARK[r.model][0], s=74, color=OURS,
                       edgecolor="white", linewidth=1.0, zorder=3)
        dx, dy, ha = NUDGE[corpus]
        anchor = part.loc[part.weight.idxmax() if dy > 0 else part.weight.idxmin()]
        ax.annotate(LABEL[corpus], (anchor.gap, anchor.weight),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9.5, color=INK)

    ax.set_ylim(-0.02, 0.88)
    ax.set_xlim(-1.5, 50.5)
    ax.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax.set_xlabel("backbone's advantage over ours (pp)", fontsize=11, color=MUTED)
    ax.set_ylabel("weight the fusion gives our side", fontsize=11, color=MUTED)
    ax.set_title("The weight shrinks with the gap, but never reaches zero",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    handles = [plt.Line2D([], [], marker=m, linestyle="", color=OURS,
                          markeredgecolor="white", markersize=8, label=name)
               for m, name in MARK.values()]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper right",
              labelcolor=INK, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
