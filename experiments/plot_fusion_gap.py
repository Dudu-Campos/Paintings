"""Delta da fusao de decisao contra a vantagem do backbone sobre nos.

A pergunta que a figura responde: QUANDO a fusao de decisao paga? O §4.1d
respondeu com n=5 pontos de protocolos misturados (fold unico do §4.1c ao lado
de 5 sementes). Aqui sao **14 pontos homogeneos** -- 7 corpora x 2 backbones,
todos com 5 sementes, media de log-probs sem nenhum parametro ajustado.

Eixo x = gap = BAcc do backbone fine-tuned menos a nossa.
Eixo y = delta = BAcc da fusao menos a do backbone sozinho.
A linha y=0 e a fronteira de decisao: acima dela a fusao vale a pena.

Cor = corpus, marcador = backbone. Os quatro slots (1,3,4,8) sairam de busca
exaustiva sobre as permutacoes da paleta do projeto maximizando o pior par sob
os tres dicromatismos de Machado: pior dE 21,2 e matiz minima 46,1 graus,
contra pisos de 12,0 e 60 do §7.18. Identidade NUNCA e so cor -- o marcador
separa os backbones e os corpora vao rotulados no proprio ponto.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "minc2500":    "results/minc2500_decision_fusion.csv",
    "nct_crc":     "results/nct_crc_decision_fusion.csv",
    "multitask":   "results/multitask_decision_fusion_cv.csv",
    "oilpainting": "results/oilpainting_decision_fusion_cv.csv",
    "rijksdraw":   "results/rijksdraw_decision_fusion_cv.csv",
    "food101":     "results/food101_decision_fusion.csv",
    "places_lt":   "results/places_lt_decision_fusion.csv",
}
# Sete corpora exigem sete cores, e dos 8 slots do §7.18 só UM subconjunto de
# sete passa o piso de dE 12 sob os tres dicromatismos: (1,3,4,5,6,7,8), com o
# laranja fora. Ele carrega uma colisao FORCADA de matiz -- violeta a 22,9 graus
# do azul, exatamente o par que o §7.18 rejeita --, e nao existe alternativa com
# sete categorias. Mitigado de duas formas: os dois vao para os corpora mais
# distantes no eixo x (multitask com gap ~10, minc2500 com ~27), e todo ponto
# leva rotulo de texto, entao identidade nunca e so cor.
COLOUR = {"multitask": "#2a78d6", "minc2500": "#4a3aa7", "nct_crc": "#1baf7a",
          "oilpainting": "#e34948", "rijksdraw": "#eda100",
          "food101": "#e87ba4", "places_lt": "#008300"}
LABEL = {"multitask": "Multitask 125cl", "nct_crc": "NCT-CRC",
         "minc2500": "MINC-2500", "oilpainting": "OilPainting",
         "rijksdraw": "Rijksmuseum drawings", "food101": "Food-101",
         "places_lt": "Places-LT"}
MARKER = {"resnet50": ("o", "ResNet50"), "vit_b_16": ("^", "ViT-B/16")}
# (dx, dy em PONTOS, alinhamento). Ajustado contra as colisoes observadas: o
# rotulo do rijksdraw encostava no eixo y e no topo, o do places_lt caia sobre
# os pontos do oilpainting, e o do nct_crc ficava sob o proprio ponto.
NUDGE = {"rijksdraw":   (  0,  11, "center", "bottom"),
         "nct_crc":     (  0, -16, "center", "top"),
         "multitask":   ( 30,  -2, "left",   "center"),
         "oilpainting": (  0,  11, "center", "bottom"),
         "places_lt":   (  8, -16, "left",   "top"),
         "minc2500":    (  0, -16, "center", "top"),
         "food101":     (  0,  11, "center", "bottom")}


def collect() -> pd.DataFrame:
    """Um ponto por (corpus, backbone), com dispersao PAREADA por semente.

    O delta e calculado dentro da semente antes de agregar: fusao e backbone
    vem do mesmo treino, entao parear cancela a variacao de semente comum aos
    dois. Agregar cada lado primeiro e subtrair depois inflaria a barra.
    """
    rows = []
    for corpus, path in SOURCES.items():
        table = pd.read_csv(ROOT / path)
        for backbone, group in table.groupby("model"):
            delta = group.fusion_bacc - group.backbone_bacc
            gap = group.backbone_bacc - group.ours_bacc
            rows.append(dict(
                corpus=corpus, backbone=backbone, seeds=len(group),
                gap=gap.mean(), gap_sd=gap.std(),
                delta=delta.mean(), delta_sd=delta.std(),
                positivas=int((delta > 0).sum())))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/decision_fusion_gap.png")
    parser.add_argument("--fit", action="store_true",
                        help="Desenha a reta de minimos quadrados. FORA por default "
                             "desde 07/set: com 14 pontos os residuos por corpus vao "
                             "a +2,05 (Multitask) e -1,90 (MINC), e ha duas inversoes "
                             "de monotonicidade -- NCT-CRC->Multitask e MINC->Food-101. "
                             "O Food-101 tem o MAIOR gap de todos (48pp) e mesmo assim "
                             "doi menos que o MINC (27pp). Uma reta faz a figura afirmar "
                             "linearidade que os dados negam.")
    args = parser.parse_args()

    points = collect()
    r, p = stats.pearsonr(points.gap, points.delta)
    rho, ps = stats.spearmanr(points.gap, points.delta)
    slope, intercept = np.polyfit(points.gap, points.delta, 1)
    crossing = -intercept / slope

    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    axis.set_axisbelow(True)
    axis.grid(True, linewidth=0.6, color="#e8e8e8")

    axis.axhline(0.0, color="#5a5a5a", linewidth=1.2, zorder=2)
    if args.fit:
        span = np.linspace(points.gap.min() - 3, points.gap.max() + 3, 100)
        axis.plot(span, slope * span + intercept, linestyle="--",
                  linewidth=1.5, color="#9a9a9a", zorder=3)

    for row in points.itertuples():
        marker, _ = MARKER[row.backbone]
        axis.errorbar(row.gap, row.delta, yerr=row.delta_sd,
                      fmt=marker, markersize=9, elinewidth=1.2, capsize=3,
                      color=COLOUR[row.corpus], markeredgecolor="white",
                      markeredgewidth=1.2, ecolor=COLOUR[row.corpus], zorder=5)

    # Rotulo direto por corpus, ancorado no topo da barra de erro mais alta do
    # par e deslocado em PONTOS, nao em unidades de dado: com deslocamento em
    # dados o rotulo do Multitask (o par mais alto) saia fora do eixo.
    for corpus, group in points.groupby("corpus"):
        dx, dy, ha, va = NUDGE.get(corpus, (0, 11, "center", "bottom"))
        band = group.delta_sd.fillna(0.0)
        anchor = ((group.delta + band).max() if dy > 0 else (group.delta - band).min())
        axis.annotate(LABEL[corpus], (group.gap.mean(), anchor),
                      textcoords="offset points", xytext=(dx, dy),
                      ha=ha, va=va, fontsize=9.5, color="#333333")

    # Folga no topo para o rotulo mais alto e para a caixa de r/p/n.
    low = (points.delta - points.delta_sd.fillna(0.0)).min()
    high = (points.delta + points.delta_sd.fillna(0.0)).max()
    axis.set_ylim(low - 1.6, high + 1.6)
    axis.set_xlim(points.gap.min() - 6, points.gap.max() + 7)

    handles = [plt.Line2D([], [], linestyle="", marker=m, markersize=8,
                          color="#666666", markeredgecolor="white",
                          markeredgewidth=1.0, label=name)
               for m, name in MARKER.values()]
    axis.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)

    axis.set_xlabel("vantagem do backbone sobre nós  (BAcc$_{ft}$ − BAcc$_{nós}$, pp)")
    axis.set_ylabel("ganho da fusão de decisão\n(BAcc$_{fusão}$ − BAcc$_{ft}$, pp)")
    axis.set_title("A fusão paga enquanto os dois lados são comparáveis",
                   fontsize=11.5, pad=10)
    # Spearman em vez de Pearson: a relacao e monotonica-por-postos (rho -0,81)
    # e nao linear (r -0,71, com residuos de +-2pp). O cruzamento de zero NAO
    # entra na figura -- ele foi de 22,4 para 29,1pp ao passar de 8 para 14
    # pontos, entao nao e uma constante para citar.
    axis.annotate(f"Spearman ρ = {rho:.2f}   p = {ps:.4f}   n = {len(points)}",
                  xy=(0.975, 0.965), xycoords="axes fraction",
                  ha="right", va="top", fontsize=9, color="#555555")

    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#9a9a9a")

    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    points.to_csv(ROOT / "results/decision_fusion_gap_points.csv", index=False)
    print(points.round(2).to_string(index=False))
    print(f"\nPearson r={r:.3f} p={p:.4f} n={len(points)}")
    print(f"reta: delta = {slope:.4f}*gap + {intercept:.3f}, cruza zero em {crossing:.1f}pp")
    print(f"figura: {args.output}")


if __name__ == "__main__":
    main()
