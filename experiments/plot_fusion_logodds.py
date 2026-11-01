"""O ganho da fusao contra a vantagem do backbone, na escala de LOG-ODDS.

Irma de `plot_fusion_gap.py`, e existe por uma medicao: trocar o eixo x de
**diferenca em pontos percentuais** por **diferenca de log-odds** aperta a
relacao de forma substancial, nos dois estimadores de peso.

```
                      x = gap (pp)    x = dif. de log-odds
delta com w=0,5        rho -0,811          rho -0,903
delta cross-fit        rho -0,881          rho -0,965
```

**ORIGEM E CREDITO.** A sugestao veio de fora (12/out): *"pela teoria o eixo x
tambem seria uma razao, de erros ou de log-odds, e nao uma diferenca em pp"*. A
mesma sugestao propunha normalizar o eixo **y** pela margem do backbone, e essa
metade foi **REFUTADA** -- normalizar o y PIORA em todas as tres parametrizacoes
de x, e contra a razao de erros perde significancia (rho -0,327, p=0,25). O que
sobrevive e so o eixo x, e especificamente em log-odds: a razao de erros da
-0,864, PIOR que o gap em pp.

**POR QUE O SPEARMAN PODE MUDAR.** Nao e transformacao monotonica do gap, e e
por isso que ha reordenamento: a mesma diferenca em pp vale MAIS quando os dois
lados estao perto do teto (ou do piso). Dois pares demonstram, e o primeiro e o
que a figura torna visivel:

```
             nosso   back    gap     dif. log-odds   delta (RN50 / ViT)
nct_crc      87,46  94,58   7,12pp      0,917        +0,58 / +1,27
oilpainting  43,61  62,90  19,30pp      0,785        +1,69 / +1,25

places_lt    14,07  33,92  19,85pp      1,143        +0,18 / -0,73
oilpainting  43,61  62,90  19,30pp      0,785        +1,69 / +1,25
```

**O `nct_crc` tem gap 2,7x MENOR que o `oilpainting` em pp e log-odds
LIGEIRAMENTE MAIOR** -- porque 87,5 contra 94,6 perto do teto e uma distancia
maior do que 43,6 contra 62,9 no meio da escala. Os desfechos sao praticamente
os mesmos (+0,58/+1,27 contra +1,69/+1,25). Em pp o `nct_crc` fica sozinho la
na esquerda e vira residuo; em log-odds ele cai EM CIMA do `oilpainting`, que e
onde o desfecho diz que ele pertence. O segundo par e o espelho disso no piso:
`places_lt` e `oilpainting` empatam em pp (0,55 de diferenca, dentro do ruido
entre sementes) e o log-odds os separa por 46%, na direcao certa.

**O QUE A ESCALA NAO CONSERTA**: o vao de dado. Em pp nao ha ponto entre 27 e
48; em log-odds nao ha entre 1,32 e 2,33. Reparametrizar nao cria observacao, e
a proibicao de ajustar curva (§4.1f) continua valendo pelo mesmo motivo.

**RESSALVA QUE TEM DE IR NA LEGENDA, e ela e de interpretacao.** A BAcc e a
media dos recalls por classe, **nao** o parametro de um Bernoulli -- entao
`logit(BAcc)` aqui e uma transformacao de ESCALA (estabiliza variancia e
lineariza efeito multiplicativo perto do piso e do teto), nao a afirmacao de
que a BAcc e probabilidade de um evento binario. A literatura de combinacao de
previsoes de onde a sugestao vem (Bates & Granger 1969; Kuncheva) trabalha com
probabilidades de verdade. Nao vender a escala como derivacao teorica.

**E O n=14 NAO E n=14 INDEPENDENTE**: sao 7 corpora x 2 backbones, e os dois
backbones de um corpus compartilham o nosso lado inteiro. Por isso o script
imprime TAMBEM o rho ao nivel de corpus (n=7, media dos dois backbones), que e
a versao de unidades independentes. A alegacao so vale se ela sobreviver la.

Sem reta ajustada, pela mesma razao do §4.1f: nenhuma forma funcional e
identificavel com 7 corpora (validacao deixa-um-corpus-de-fora da reta 2,08pp
contra exponencial 2,07 -- ajusta melhor e nao preve melhor).
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
import sys
sys.path.insert(0, str(ROOT / "experiments"))
from plot_fusion_gap import COLOUR, LABEL, MARKER, SOURCES   # noqa: E402

EPS = 1e-6


def logit(p: np.ndarray | pd.Series) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def collect() -> pd.DataFrame:
    """Um ponto por (corpus, backbone). Tudo PAREADO POR SEMENTE antes de agregar.

    Vale para as duas escalas de x: o log-odds e calculado dentro da semente e
    so depois promediado, senao a barra do eixo x misturaria a variacao comum
    aos dois lados (mesma convencao do `plot_fusion_gap.collect`).
    """
    rows = []
    for corpus, path in SOURCES.items():
        table = pd.read_csv(ROOT / path)
        for backbone, g in table.groupby("model"):
            delta = g.fusion_bacc - g.backbone_bacc
            gap = g.backbone_bacc - g.ours_bacc
            dlo = logit(g.backbone_bacc) - logit(g.ours_bacc)
            rows.append(dict(
                corpus=corpus, backbone=backbone, seeds=len(g),
                nosso=g.ours_bacc.mean(), backbone_bacc=g.backbone_bacc.mean(),
                gap=gap.mean(), gap_sd=gap.std(),
                dlogodds=float(np.mean(dlo)), dlogodds_sd=float(np.std(dlo, ddof=1)),
                delta=delta.mean(), delta_sd=delta.std(),
                positivas=int((delta > 0).sum())))
    return pd.DataFrame(rows)


EIXO = {
    "gap": ("vantagem do backbone sobre nós  (BAcc$_{ft}$ − BAcc$_{nós}$, pp)",
            "diferença em pontos percentuais"),
    "dlogodds": ("vantagem do backbone sobre nós  "
                 "(logit BAcc$_{ft}$ − logit BAcc$_{nós}$)",
                 "diferença de log-odds"),
}
# (dx, dy em PONTOS, ha, va). O rijksdraw e o ponto MAIS A ESQUERDA nas duas
# escalas (gap de 1,1pp, log-odds de 0,04), entao rotulo centrado sai pela
# borda -- ele vai ancorado a esquerda e deslocado para dentro.
NUDGE = {"rijksdraw":   ( 14,   4, "left",   "bottom"),
         "nct_crc":     (  0, -16, "center", "top"),
         # o par do multitask fica na mesma altura do rotulo do rijksdraw; vai
         # para a DIREITA dos proprios pontos, onde o painel e vazio
         "multitask":   ( 16,  -2, "left",   "center"),
         "oilpainting": (  0,  11, "center", "bottom"),
         # a esquerda: a direita dele estao os pontos do minc2500
         "places_lt":   (-10, -14, "right",  "top"),
         "minc2500":    (  0, -16, "center", "top"),
         "food101":     (  0, -16, "center", "top")}


def painel(ax, pts: pd.DataFrame, eixo: str, titulo: bool, rotulos: bool) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.6, color="#e8e8e8")
    ax.axhline(0.0, color="#5a5a5a", linewidth=1.2, zorder=2)

    # Conector cinza por corpus: e ele que torna 14 pontos legiveis, agrupando
    # por coluna antes de o leitor ler qualquer cor (§4.1f item 7).
    for corpus, g in pts.groupby("corpus"):
        if len(g) == 2:
            ax.plot(g[eixo], g.delta, color="#c4c4c4", linewidth=1.0, zorder=3)

    for r in pts.itertuples():
        marker, _ = MARKER[r.backbone]
        ax.errorbar(getattr(r, eixo), r.delta, yerr=r.delta_sd,
                    fmt=marker, markersize=9, elinewidth=1.2, capsize=3,
                    color=COLOUR[r.corpus], markeredgecolor="white",
                    markeredgewidth=1.2, ecolor=COLOUR[r.corpus], zorder=5)

    if rotulos:
        for corpus, g in pts.groupby("corpus"):
            dx, dy, ha, va = NUDGE.get(corpus, (0, 11, "center", "bottom"))
            band = g.delta_sd.fillna(0.0)
            anc = (g.delta + band).max() if dy > 0 else (g.delta - band).min()
            ax.annotate(LABEL[corpus], (g[eixo].mean(), anc),
                        textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, fontsize=9.5, color="#333333")

    rho, ps = stats.spearmanr(pts[eixo], pts.delta)
    ax.annotate(f"Spearman ρ = {rho:.3f}   p = {ps:.4f}",
                xy=(0.975, 0.965), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#555555")
    lo = (pts.delta - pts.delta_sd.fillna(0.0)).min()
    hi = (pts.delta + pts.delta_sd.fillna(0.0)).max()
    ax.set_ylim(lo - 1.6, hi + 2.4)   # folga para o rotulo mais alto E a caixa de rho
    span = pts[eixo].max() - pts[eixo].min()
    ax.set_xlim(pts[eixo].min() - 0.10 * span, pts[eixo].max() + 0.12 * span)
    ax.set_xlabel(EIXO[eixo][0], fontsize=9.5)
    if titulo:
        ax.set_title(EIXO[eixo][1], fontsize=10.5, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--par", action="store_true",
                    help="dois paineis (pp | log-odds) -- o DIAGNOSTICO que "
                         "justifica a troca de escala, nao a figura do artigo")
    args = ap.parse_args()

    pts = collect()
    out = args.output or (ROOT / ("results/decision_fusion_scale_pair.png" if args.par
                                  else "results/decision_fusion_logodds.png"))

    if args.par:
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
        painel(axes[0], pts, "gap", True, True)
        painel(axes[1], pts, "dlogodds", True, False)
        axes[0].set_ylabel("ganho da fusão de decisão\n(BAcc$_{fusão}$ − BAcc$_{ft}$, pp)")
        fig.suptitle("A mesma relação em duas parametrizações do eixo x",
                     fontsize=11.5, y=0.99)
    else:
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        painel(ax, pts, "dlogodds", False, True)
        ax.set_ylabel("ganho da fusão de decisão\n(BAcc$_{fusão}$ − BAcc$_{ft}$, pp)")
        ax.set_title("A fusão paga enquanto os dois lados são comparáveis",
                     fontsize=11.5, pad=10)
        axes = [ax]

    h = [plt.Line2D([], [], linestyle="", marker=m, markersize=8, color="#666666",
                    markeredgecolor="white", markeredgewidth=1.0, label=n)
         for m, n in MARKER.values()]
    axes[0].legend(handles=h, loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)

    # ------------------------------------------------------------------ relato
    print(pts[["corpus", "backbone", "nosso", "backbone_bacc", "gap",
               "dlogodds", "delta", "delta_sd", "positivas"]]
          .round(3).to_string(index=False))
    print(f"\n{'escala de x':24s} {'Spearman (n=14)':>24s} {'Pearson (n=14)':>24s} "
          f"{'Spearman por CORPUS (n=7)':>27s}")
    corp = pts.groupby("corpus").mean(numeric_only=True)
    for eixo in ("gap", "dlogodds"):
        rho, ps = stats.spearmanr(pts[eixo], pts.delta)
        r, pr = stats.pearsonr(pts[eixo], pts.delta)
        rc, pc = stats.spearmanr(corp[eixo], corp.delta)
        print(f"{EIXO[eixo][1]:24s} {f'{rho:+.3f} (p={ps:.4f})':>24s} "
              f"{f'{r:+.3f} (p={pr:.4f})':>24s} {f'{rc:+.3f} (p={pc:.4f})':>27s}")
    pts.to_csv(ROOT / "results/decision_fusion_logodds_points.csv", index=False)
    print(f"\nfigura: {out}")


if __name__ == "__main__":
    main()
