"""A assimetria entre os dois eixos, com a dimensionalidade PAREADA.

Esta figura existe porque nenhuma outra do projeto mostra o argumento central.
O §5.1 mede o expoente subindo com o número de representações, mas naquela
grade `nº de representações` e `dimensionalidade` sobem juntos (r=+0,919), então
"mais features é melhor" explica os dados igualmente bem. O §5.3 e o §5.14
mostram o eixo de k plano, mas em outra figura e outro corpus de execução.

Aqui os dois painéis vêm da MESMA execução: mesma sonda LDA, mesmas 125 classes,
mesmo critério de seleção (`max-r2-strat`), mesmos folds e sementes. Só o pool
de origem muda.

  esquerda  eixo de DADOS, com k FIXO em 128 features nos dois braços.
            razão dos expoentes 1,62x -- a procedência muda muito a FORMA.
  direita   eixo de DIMENSIONALIDADE, com o treino inteiro.
            razão 1,12x -- quase nada.

  diferença de diferenças  +0,49 +- 0,25   p < 0,0001   15/15 unidades

É a discordância entre os painéis que "mais features é melhor" não explica: essa
leitura prevê o mesmo tipo de efeito nos dois eixos.

**O painel direito para em k=64 de propósito.** O pool de uma família tem 130
features, então em k=128 ele está no teto e a curva dele achata por
esgotamento, não por saturação -- o mesmo artefato que o §5.14 marca com uma
vertical em 130. Incluindo k=128 a razão sobe de 1,12x para 1,33x e a
assimetria fica inflada. A primeira versão desta figura cometia esse erro e o
título afirmava que as inclinações "coincidem", o que os próprios dados dela
contradiziam.

Cor: o par de `plot_selection_curve_pools.py` (§5.14), sem alteração, porque é o
mesmo contraste semântico -- oito famílias contra uma -- e o leitor deve carregar
o significado entre as duas figuras. Auditado contra os três dicromatismos
(Machado et al. 2009): pior caso dE 96,7, muito acima do piso de 12 do §7.18.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BANK, ONE = "#2a78d6", "#eb6834"
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
CAPTION = """
Provenance, not dimensionality, changes the data-scaling exponent.
MultitaskPainting100k, 99,816 paintings, 125 styles; LDA probe, 5 folds x 3
selection seeds. Both arms receive the SAME number of features, chosen by the
same redundancy-first stratified criterion fitted on training rows only; only
the pool they are drawn from differs -- the eight-family bank against the 130
dimensions of the descriptor family. Left: training size varies with the
feature budget held at 128; the exponents are 0.073 and 0.045, a ratio of
1.62x. Right: the feature budget varies with the training set held at 79,852;
the exponents are 0.088 and 0.079, a ratio of 1.12x. The difference between the
two ratios is +0.49 +- 0.25 (p < 0.0001, 15/15 units). The right panel stops at
64 features because the one-family pool holds 130 and flattens from exhaustion
beyond that, which would inflate the asymmetry. beta is the slope of balanced
error in log-log; bands are one SD across folds.
"""

POOLS = [("bank", "eight families", BANK, "o"),
         ("descriptors", "one family (descriptors)", ONE, "s")]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)


def exponent(x, y) -> float:
    """Inclinação do erro em log-log. `y` é BAcc em %, o erro é 100 - y."""
    return -np.polyfit(np.log(np.asarray(x, float)), np.log(100 - np.asarray(y, float)), 1)[0]


def panel_exponents(frame, xcol):
    out = {}
    for pool, *_ in POOLS:
        sub = frame[frame.pool.eq(pool)]
        mean = sub.groupby(xcol).bacc.mean()
        out[pool] = exponent(mean.index, mean.to_numpy())
    return out


def panel(ax, frame, xcol, fixed_label, title, xlabel):
    for pool, label, colour, marker in POOLS:
        sub = frame[frame.pool.eq(pool)]
        mean = sub.groupby(xcol).bacc.mean()
        sd = sub.groupby(xcol).bacc.std(ddof=1)
        err = 100 - mean
        ax.plot(mean.index, err, color=colour, marker=marker, markersize=8,
                linewidth=2.0, label=label, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.4)
        ax.fill_between(mean.index, err - sd, err + sd, color=colour, alpha=0.16,
                        linewidth=0, zorder=2)
        b = exponent(mean.index, mean.to_numpy())
        # Rótulo direto do expoente, junto da curva: é a quantidade que a figura
        # afirma, então não pode ficar só na legenda.
        ax.annotate(f"β = {b:.3f}", xy=(mean.index[-1], err.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=colour, fontsize=12, fontweight="bold", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    def _thousands(v, _):
        # `round(2500/1000)` e 2 (arredondamento bancario) e rotularia 2.500
        # como "2k" -- erro factual no eixo. Abaixo de 10k mantem a decimal.
        if v < 1000:
            return f"{v:g}"
        return f"{v/1000:.1f}k".replace(".0k", "k") if v < 10000 else f"{v/1000:.0f}k"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_thousands))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(sorted(frame[xcol].unique()))
    # Ticks explícitos: num intervalo estreito o localizador de log não emite
    # nenhum major e o eixo sai SEM número, defeito da primeira versão.
    ax.set_yticks([70, 75, 80, 85, 90])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(68, 95)
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=12, color=MUTED)
    ax.text(0.02, 0.03, fixed_label, transform=ax.transAxes, fontsize=10.5,
            color=MUTED, ha="left", va="bottom")
    style(ax)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=ROOT / "results/paired_dimensionality_scaling_125.csv")
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--output", type=Path,
                   default=ROOT / "results/paired_dimensionality_scaling.png")
    args = p.parse_args()

    d = pd.read_csv(args.input)
    d["train_size"] = d.train_size.replace({79853: 79852, 66260: 66259})
    full = int(d.train_size.max())

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharey=True)
    fig.patch.set_facecolor("white")

    left = d[d.budget.eq(args.budget)]
    right = d[d.train_size.eq(full) & d.budget.le(64)]
    left_exp = panel_exponents(left, "train_size")
    right_exp = panel_exponents(right, "budget")
    left_ratio = left_exp["bank"] / left_exp["descriptors"]
    right_ratio = right_exp["bank"] / right_exp["descriptors"]

    panel(axes[0], left, "train_size",
          f"features held fixed at {args.budget}",
          f"Data axis — provenance changes the slope ({left_ratio:.2f}×)",
          "training images")
    # k<=64: acima disso o pool de uma família (130 features) está no teto e
    # achata por esgotamento, o que INFLA a assimetria (§5.14 marca o mesmo).
    panel(axes[1], right, "budget",
          f"training set held fixed at {full:,}",
          f"Dimensionality axis — barely ({right_ratio:.2f}×)",
          "number of features given to the classifier")
    axes[0].set_ylabel("balanced error (%)", fontsize=12, color=MUTED)

    axes[0].legend(frameon=False, fontsize=11.5, loc="lower left",
                   bbox_to_anchor=(0.0, 0.10), labelcolor=INK)
    fig.suptitle("Provenance changes how fast the model learns from data, "
                 "far more than how it uses dimensions",
                 fontsize=15, fontweight="bold", color=INK, x=0.008, ha="left", y=0.99)
    # Sem rodape na figura: tres linhas de 9,5pt viram borrao na largura de
    # coluna dupla. A procedencia vai na LEGENDA do artigo -- o script a
    # imprime ao final para nao se perder. O que fica DENTRO dos paineis e
    # so o que a figura nao pode omitir: qual eixo esta fixado em cada um.
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"escrito: {args.output}")
    print("\nLEGENDA PARA O ARTIGO (o rodape saiu da figura de proposito):\n")
    print(CAPTION.strip())


if __name__ == "__main__":
    main()
