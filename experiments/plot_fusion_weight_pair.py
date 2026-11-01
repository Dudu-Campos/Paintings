"""O peso fixo contra o peso escolhido fora do dado, com e sem o canal de texto.

Dois paineis lado a lado, uma condicao de peso por cor. Substitui
`plot_fusion_weight_regimes.py` como figura do artigo.

**O ORACULO SAIU, por decisao do usuario em 21/set, e a razao e de mecanismo.**
Ele estava no painel como "teto", e a distancia ate ele era lida como o que a
fusao deixa na mesa. Duas medicoes derrubam essa leitura:

  1. O teto e INALCANCAVEL por construcao. Chegar nele exige saber, imagem a
     imagem, qual dos dois lados esta certo; a unica fonte disso nos posteriors
     e a confianca, e um seletor por `max p` perde da media simples em 6 dos 7
     corpora (`results/oracle_reachability.csv`). Nao e folga desperdicada.
  2. Contra um nulo de independencia o teto medido e DEFICITARIO nos sete
     (`results/oracle_independence_null.csv`), entao a magnitude dele tambem nao
     e evidencia de complementaridade.

Nas palavras do usuario: *"o teto nao importa, e dai que um modelo aleatorio
teria teto maior? por definicao nao ha mecanismo que recupere esse ganho."*
Tirar e mais honesto que anotar com a linha de acaso -- uma serie que ninguem
pode perseguir nao pertence a uma figura sobre estimadores.

O ganho de tira-lo nao e so de escopo: o eixo y ia de ~20pp de amplitude para
~7pp, entao a diferenca entre vermelho e azul -- que E a afirmacao -- passa a
ocupar a altura do painel em vez de um terco dela.

**Os dois paineis COMPARTILHAM os dois eixos, e isso nao e negociavel.** Com
limites automaticos o Food-101 fica na borda direita nos dois, em gap 48 e em
gap 29, e o deslocamento que o painel da direita existe para mostrar some --
mesma posicao visual, valor diferente, sem errar numero nenhum.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HALF, CROSS = "#e34948", "#2a78d6"
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
LABEL = {"multitask": "Multitask", "nct_crc": "NCT-CRC", "minc2500": "MINC-2500",
         "oilpainting": "OilPainting", "rijksdraw": "Rijksmuseum",
         "food101": "Food-101", "places_lt": "Places-LT"}
# (dx, dy, ha) em pontos. O rotulo e ancorado no marcador de CIMA quando dy>0 e
# no de BAIXO quando dy<0, entao ele nunca cai sobre o outro ponto do par.
NUDGE = {"rijksdraw": (0, -15, "center"), "nct_crc": (0, -15, "center"),
         "multitask": (0, 12, "center"), "oilpainting": (0, 12, "center"),
         "places_lt": (-9, -15, "right"), "minc2500": (0, -15, "center"),
         "food101": (-9, -15, "right")}
# Os gaps mudam entre os paineis, entao a posicao que resolve um nao resolve o
# outro -- no painel com texto o NCT-CRC e o Places-LT ficam a 5pp um do outro.
# Mesmo motivo pelo qual a versao empilhada precisava de dicionarios separados.
NUDGE_TEXT = dict(NUDGE, nct_crc=(-10, -15, "right"),
                  places_lt=(0, -15, "center"))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10.5)


def load(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    return (t.groupby("corpus").agg(gap=("gap", "mean"), half=("d_half", "mean"),
                                    cross=("d_cross", "mean")).sort_values("gap"))


def panel(ax, g, title, nudge):
    for _, r in g.iterrows():
        ax.plot([r.gap, r.gap], [min(r.half, r.cross), max(r.half, r.cross)],
                color="#c8c8c8", linewidth=1.3, zorder=1)
    ax.axhline(0, color="#5a5a5a", linewidth=1.1, zorder=2)
    ax.plot(g.gap, g.half, "s", color=HALF, markersize=10, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="fixed weight 0.5")
    ax.plot(g.gap, g.cross, "o", color=CROSS, markersize=10, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="weight chosen off the evaluated data")
    for name, r in g.iterrows():
        dx, dy, ha = nudge[name]
        anchor = max(r.half, r.cross) if dy > 0 else min(r.half, r.cross)
        ax.annotate(LABEL[name], (r.gap, anchor), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="bottom" if dy > 0 else "top",
                    fontsize=9, color="#3a3a3a")
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax.set_xlabel("backbone's advantage over ours (pp)", fontsize=11, color=MUTED)
    style(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visual", type=Path,
                    default=ROOT / "results/decision_fusion_gap_points_extended.csv")
    ap.add_argument("--text", type=Path,
                    default=ROOT / "results/decision_fusion_gap_points_text.csv")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_pair.png")
    args = ap.parse_args()
    left, right = load(args.visual), load(args.text)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    panel(a, left, "Visual bank only — no external supervision on our side", NUDGE)
    panel(b, right, "With the text channel — every corpus moves left", NUDGE_TEXT)

    lo = min(left[["half", "cross"]].min().min(), right[["half", "cross"]].min().min())
    hi = max(left[["half", "cross"]].max().max(), right[["half", "cross"]].max().max())
    a.set_xlim(-4, max(left.gap.max(), right.gap.max()) + 6)
    a.set_ylim(lo - 0.9, hi + 0.9)
    a.set_ylabel("decision-fusion gain over the backbone (pp)", fontsize=11, color=MUTED)
    a.legend(frameon=False, fontsize=10, loc="upper right", labelcolor=INK,
             handletextpad=0.6, borderaxespad=0.8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"escrito: {args.output}")


if __name__ == "__main__":
    main()
