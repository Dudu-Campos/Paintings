"""Uma curva de escala em dados por número de modalidades.

O resultado: o expoente do erro **cresce** com o número de modalidades
(−0,031 com uma, −0,074 com sete). Com uma modalidade só a curva satura --
30,28% em 40k e 30,27% em 63k, plana. Com sete ainda está subindo. Ou seja,
representações adicionais não apenas somam acurácia: elas tiram o modelo do
regime de saturação e o devolvem ao regime em que mais dados ainda pagam.

Eixo y em **erro balanceado**, a forma padrão das scaling laws -- acurácia
satura e deforma o expoente (ver `plot_cnn_vs_handdesigned_scaling.py`).

Cores: rampa sequencial de uma cor só, porque o número de modalidades é uma
grandeza ordenada, não categorias. Claro = poucas, escuro = muitas.

Saída: results/representation_data_grid.png
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
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d7dbe0"
NESTED_ORDER = ("Descriptors", "Felzenszwalb", "SLIC", "Line graph",
                "Hilbert", "Fourier", "Multiscale")


def thousands(value: float) -> str:
    text = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
    return f"{text}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peaks", type=Path,
                        default=ROOT / "outputs/representation_data_grid/peaks.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/representation_data_grid.png")
    parser.add_argument("--max-modalities", type=int, default=0,
                        help="0 = todos os degraus presentes no arquivo. O default "
                             "era 6, herdado da ordenação antiga do WikiArt em que o "
                             "7º degrau era o Multiscale (1.032d contra 57-152d das "
                             "outras, misturando salto de tamanho com adição de "
                             "conteúdo). Na grade do Multitask não há Multiscale: os "
                             "nove degraus são as oito modalidades visuais mais o "
                             "grafo de similaridade, e cortar em 6 esconde três deles "
                             "-- inclusive o grafo, que é o achado do §5.2. Degraus "
                             "sem avanço (Gabor, L(G_felz)) devem aparecer: omiti-los "
                             "sugeriria que toda modalidade move o expoente.")
    parser.add_argument("--corpus-note", default="WikiArt, 27 styles",
                        help="o rodapé descrevia o WikiArt fixo, e a mesma figura "
                             "já foi gerada sobre a grade do Multitask -- o texto "
                             "precisa acompanhar os dados")
    parser.add_argument("--paper", action="store_true",
                        help="remove título e rodapé: em paper a legenda "
                             "carrega essa informação e a duplicata rouba espaço")
    args = parser.parse_args()

    peaks = pd.read_csv(args.peaks)
    # A grade do WikiArt grava `n_train`; a do Multitask, `train_size`. Aceitar
    # as duas evita duplicar o script por causa de um nome de coluna.
    if "n_train" not in peaks.columns and "train_size" in peaks.columns:
        peaks = peaks.rename(columns={"train_size": "n_train"})
    # O caso "treino completo" grava o tamanho realizado, que varia um exemplo
    # entre folds -- agrupa para não virar dois pontos.
    requested = np.array([2500, 5000, 10000, 20000, 40000])
    def group(value: int) -> int:
        near = requested[np.abs(requested - value) <= 0.02 * value]
        return int(near[0]) if len(near) else int(round(value, -2))
    peaks["n_train"] = peaks.n_train.map(group)
    if args.max_modalities:
        peaks = peaks[peaks.n_modalities <= args.max_modalities]

    curves = (peaks.groupby(["n_modalities", "n_train"], as_index=False)
              .agg(bacc=("bacc", "mean"), std=("bacc", "std")))
    curves[["bacc", "std"]] *= 100
    counts = sorted(curves.n_modalities.unique())
    ramp = plt.get_cmap("Blues")

    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    exponents = {}
    rotulos: list[dict] = []
    for index, count in enumerate(counts):
        block = curves[curves.n_modalities == count].sort_values("n_train")
        error = 100 - block.bacc.to_numpy()
        colour = ramp(0.30 + 0.62 * index / max(len(counts) - 1, 1))
        axis.errorbar(block.n_train, error, yerr=block["std"].fillna(0),
                      color=colour, linewidth=2, marker="o", markersize=6,
                      markeredgecolor="white", markeredgewidth=1.1,
                      capsize=2.5, elinewidth=1, zorder=3 + index)
        slope, _ = np.polyfit(np.log(block.n_train.astype(float)), np.log(error), 1)
        exponents[count] = slope
        # O rótulo é resolvido depois: com deslocamento fixo, as curvas 7 e 8
        # (Gabor acrescenta ~0 ao expoente) terminam no mesmo lugar e os números
        # se sobrepõem -- a figura passava a ler como ~6 curvas em vez de 9.
        rotulos.append({"count": count, "y": float(error[-1]),
                        "x": float(block.n_train.iloc[-1]), "colour": colour})

    # Separa verticalmente os rótulos que colidiriam. Em log, uma folga de 1,6%
    # do valor basta para dois números de 9,5pt não se tocarem.
    rotulos.sort(key=lambda r: r["y"])
    folga = 0.016
    for anterior_r, atual_r in zip(rotulos, rotulos[1:]):
        minimo = anterior_r["y_plot" if "y_plot" in anterior_r else "y"] * (1 + folga)
        atual_r["y_plot"] = max(atual_r["y"], minimo)
        anterior_r.setdefault("y_plot", anterior_r["y"])
    for r in rotulos:
        r.setdefault("y_plot", r["y"])
        axis.annotate(f"{r['count']}", (r["x"], r["y_plot"]),
                      textcoords="offset points", xytext=(9, -3), fontsize=9,
                      color=r["colour"], fontweight="bold",
                      annotation_clip=False)

    # Acima da série do topo (m=1), não da de baixo -- lá colidia com os rótulos.
    top = curves[(curves.n_modalities == counts[0]) &
                 (curves.n_train == curves.n_train.max())]
    axis.annotate("modalities", (curves.n_train.max(), 100 - top.bacc.iloc[0]),
                  textcoords="offset points", xytext=(6, 16), fontsize=9,
                  color=MUTED, annotation_clip=False)

    axis.set_xscale("log"); axis.set_yscale("log")
    ticks = sorted(curves.n_train.unique())
    axis.set_xticks(ticks); axis.set_xticklabels([thousands(t) for t in ticks])
    axis.set_yticks([57, 62, 67, 72, 77])
    axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.minorticks_off()
    axis.grid(True, alpha=0.28, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xlabel("Training images", fontsize=10, color=MUTED)
    axis.set_ylabel("Balanced error (%)", fontsize=10, color=MUTED)
    if not args.paper:
        axis.set_title("More representations pull the model out of saturation",
                   fontsize=13, color=INK, loc="left", pad=12)

    # Painel do expoente: a mensagem quantitativa da figura.
    # Canto inferior esquerdo: como o erro cai com n, essa região fica vazia.
    inset = figure.add_axes((0.135, 0.245, 0.235, 0.265))
    inset.plot(list(exponents), [-v for v in exponents.values()],
               color=ramp(0.85), linewidth=1.8, marker="o", markersize=5,
               markeredgecolor="white", markeredgewidth=1)
    inset.set_yscale("log")
    inset.set_yticks([0.03, 0.04, 0.06, 0.08])
    inset.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    inset.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    inset.minorticks_off()
    inset.set_title("error exponent", fontsize=8.5, color=MUTED, pad=4)
    inset.set_xlabel("views", fontsize=7.5, color=MUTED, labelpad=1)
    inset.tick_params(colors=MUTED, labelsize=7)
    inset.grid(True, alpha=0.25, color=GRID, linewidth=0.7)
    inset.set_axisbelow(True)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        inset.spines[side].set_color(GRID)

    if not args.paper:

        figure.text(0.011, 0.038, args.corpus_note +
                " · LDA probe, peak over PCA components · 5 folds · "
                "training subsampled, test fold never reduced",
                fontsize=8.5, color=MUTED)
    if not args.paper:
        # A ordem vem do PRÓPRIO arquivo (coluna `modalities`, separada por "+"),
        # e não de uma constante: a grade do WikiArt e a do Multitask usam ordens
        # e conjuntos diferentes, e o rodapé fixo já anunciou "Multiscale" numa
        # figura em que ele não estava.
        cumulativo = peaks.groupby("n_modalities")["modalities"].first().sort_index()
        ordem, vistos = [], set()
        for texto in cumulativo:
            for parte in str(texto).split("+"):
                nome = parte.strip()
                if nome and nome not in vistos:
                    vistos.add(nome)
                    ordem.append(nome)
        figure.text(0.011, 0.012,
                "Modalities nested in Shapley order: " + " → ".join(ordem) + ".",
                fontsize=8.5, color=MUTED)
    figure.subplots_adjust(left=0.075, right=0.935, top=0.915, bottom=0.175)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white",
                   bbox_inches="tight" if args.paper else None)
    print(f"salvo: {args.output}")
    for count, slope in exponents.items():
        print(f"  m={count}: erro ∝ n^{slope:.4f}")


if __name__ == "__main__":
    main()
