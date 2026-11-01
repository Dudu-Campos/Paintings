"""Afinar theta paga sozinho e para de pagar na fusao: as cinco views.

**O eixo y e o GANHO AO AFINAR, nao a amplitude da grade.** A versao anterior
usava range (max - min), e o range e ancorado pela PIOR configuracao -- que e
escolha de quem desenhou a grade, nao propriedade da view. Trocar 64 px por 32
px no Fourier aumentaria a barra dele sem que nada mudasse na representacao. E
grades de tamanhos diferentes (aqui 3 a 4 configuracoes pareadas) dao ranges
diferentes por construcao.

O ganho ao afinar e sempre theta* contra a MELHOR alternativa, que e bem
definida, e responde a pergunta do artigo: *nos nao afinamos theta; quanto
teriamos ganho se tivessemos?* Sozinho, ate +2,86pp. Dentro do banco, entre
-0,61 e +0,46.

De brinde ele carrega SINAL, que o range nao podia: o Hilbert ganha +2,14
sozinho e **perde 0,61** no banco -- anticorrelacao, nao so achatamento.

**Vies declarado:** a melhor alternativa e escolhida na mesma medicao que se
reporta, entao o ganho isolado e um TETO otimista, nao estimativa nao-enviesada.
Isso infla o lado isolado, que e o lado grande da alegacao -- ou seja, e
conservador para o argumento, mas tem de estar dito.

**Slopegraph, nao barras pareadas.** A afirmacao e um colapso, e barras lado a
lado o codificam como razao entre dois comprimentos, leitura de segunda ordem.
Com duas colunas a queda vira uma silhueta unica, e a excecao (Fourier) aparece
como a linha que nao desce.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import colour, display  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "results/theta_series_summary.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "results/theta_series.png")
    args = ap.parse_args()
    t = pd.read_csv(args.input).sort_values("iso_gain", ascending=False)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor("white")
    X = (0.0, 1.0)
    # Deslocamento vertical anti-colisao: com o eixo em ganho, valores perto de
    # zero caem sobre a propria linha de zero, e a direita as views colapsadas
    # ficam a menos de 0,1pp umas das outras.
    def declutter(values, floor=0.16):
        """Empilha para BAIXO, sempre contra a posicao JA OCUPADA.

        A versao anterior deslocava pelo VALOR BRUTO (`v + floor`), e com isso o
        rotulo empurrado podia aterrissar colado no vizinho de novo: o `SLIC`
        (+0,04) subia para 0,20 e ficava a 0,07 do `Gabor` (+0,13), que e
        exatamente a colisao que aparecia na figura. Descendo e cortando em
        `ultimo - floor` a separacao minima fica garantida por construcao, e a
        ORDEM se conserva -- que e o que importa num slopegraph.
        """
        out, ultimo = {}, None
        for i in np.argsort(values)[::-1]:
            v = float(values[i])
            out[i] = v if ultimo is None else min(v, ultimo - floor)
            ultimo = out[i]
        return out

    left_at = declutter(t.iso_gain.to_numpy())
    right_at = declutter(t.pipe_gain.to_numpy())
    for i, r in enumerate(t.itertuples()):
        c = colour(r.view)
        ax.plot(X, (r.iso_gain, r.pipe_gain), "-o", color=c, linewidth=2.6,
                markersize=10, markeredgecolor="white", markeredgewidth=1.6, zorder=3)
        ax.annotate(f"{display(r.view)}   {r.iso_gain:+.2f}", (X[0], left_at[i]),
                    xytext=(-12, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=12, color=c, fontweight="bold")
        # So rotula a direita quem esta separado o bastante para caber; os tres
        # colapsados ficavam empilhados em 0,20-0,30 e ilegiveis. O grupo recebe
        # uma anotacao unica abaixo.
        if True:
            ax.annotate(f"{r.pipe_gain:+.2f}", (X[1], right_at[i]), xytext=(12, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=12, color=c, fontweight="bold")

    # Linha de zero: com o eixo em GANHO ela e a fronteira de decisao -- acima
    # dela afinar theta teria pago, abaixo teria custado.
    ax.axhline(0.0, color="#8a8a8a", linewidth=1.2, zorder=1)
    ax.set_xlim(-0.62, 1.42)
    lo = min(t.iso_gain.min(), t.pipe_gain.min())
    hi = max(t.iso_gain.max(), t.pipe_gain.max())
    ax.set_ylim(lo - 0.45, hi + 0.45)
    ax.set_xticks(X)
    ax.set_xticklabels(["the view on its own", "the same θ inside PRISM"],
                       fontsize=13, color=INK)
    ax.set_ylabel("gain from tuning θ (pp of BAcc)", fontsize=12, color=MUTED)
    ax.set_title("Tuning θ pays for the view alone, and stops paying in the bank",
                 fontsize=14.5, fontweight="bold", color=INK, loc="left", pad=12)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"escrito: {args.output}")
    print("\nLEGENDA:\n" + CAPTION.strip())


CAPTION = """
Tuning theta pays for a view on its own and stops paying inside the bank. Each
line is one view: on the left, the balanced accuracy that the best alternative
theta buys over the production theta* when the view acts alone; on the right,
what that SAME theta buys once it is swapped into PRISM (`hand-8` e a
chave interna; §7.18 manda traduzir so na borda) under the production
BranchMLP, 5 folds. The best alternative is chosen on the reported measurement,
so the left value is an optimistic ceiling on what tuning could buy. The exception is Fourier, the block with the
largest leave-one-out value in the bank and therefore the least covered by the
other seven: 60% of its theta effect survives. SLIC at 256 px keeps its segment
count but shares only CKA 0.111 with theta* -- a representation 89% different --
and the bank moves 0.27 pp, so the flatness is not invariance of the transform.
Multitask, 82,824 paintings, 25 styles, LDA probe alone and the production
BranchMLP inside the bank, 5 folds.
"""

if __name__ == "__main__":
    main()
