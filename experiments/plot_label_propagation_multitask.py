"""Matriz de confusão da identidade representacional inferida (Multitask, 8 famílias).

Cada linha é a modalidade de origem de uma feature; cada coluna, a modalidade
que a propagação de rótulos no grafo de r² prevê para ela **sem ver o próprio
rótulo**. Diagonal alta significa que a modalidade tem assinatura de redundância
própria; confusão fora da diagonal significa que duas medem a mesma coisa por
caminhos diferentes.

Fica em painel único, e não nos dois do gráfico antigo do WikiArt: com oito
famílias em vez de quatro, o painel de barras repetiria a diagonal sem
acrescentar nada, e a informação nova está toda fora dela.
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
INK, MUTED, GRID = "#1f2933", "#6b7683", "#d8dee5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/multitask_label_propagation_r2/confusion.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multitask_label_propagation_confusion.png")
    parser.add_argument("--merge", default="none", choices=["none", "pairs"],
                        help="`pairs` agrupa Felzenszwalb+SLIC em 'Grafos de região' e "
                             "Line graph+L(G_felz) em 'Grafos de linha'. As confusões "
                             "principais não somem -- passam a ser ENTRE as duas "
                             "famílias, o que é mais legível. Agrupar as quatro numa "
                             "só absorveria o achado na diagonal.")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="Só a porcentagem em cada célula, tipografia maior e "
                             "sem barra de cor. A contagem sai porque a share da "
                             "linha é o que a figura afirma; o denominador continua "
                             "legível no rótulo do eixo y, e por isso o `(Nd)` de lá "
                             "vira OBRIGATÓRIO neste modo em vez de decorativo.")
    args = parser.parse_args()

    counts = pd.read_csv(args.input, index_col=0)
    if args.merge == "pairs":
        grupos = {"Felzenszwalb": "Region graphs", "SLIC": "Region graphs",
                  "Line graph": "Line graphs", "L(G_felz)": "Line graphs"}
        rotulo = {n: grupos.get(n, n) for n in counts.index}
        counts = (counts.rename(index=rotulo, columns=rotulo)
                  .groupby(level=0).sum().T.groupby(level=0).sum().T)
        ordem_alvo = ["Descriptors", "Region graphs", "Line graphs",
                      "Hilbert", "Fourier", "Gabor"]
        # Qualquer família fora da lista fixa entra no fim em vez de ser
        # descartada pelo reindex -- foi assim que o `Similarity graph` sumiu
        # da figura sem nenhum aviso, levando 82 features junto.
        ordem_alvo += [n for n in counts.index if n not in ordem_alvo]
        counts = counts.reindex(index=ordem_alvo, columns=ordem_alvo, fill_value=0)
    order = list(counts.index)
    totals = counts.sum(axis=1).to_numpy()
    fractions = counts.to_numpy() / np.maximum(totals[:, None], 1)

    cell_fs, tick_fs, label_fs = (18, 15, 16) if args.clean else (12, 12.5, 14)
    figure, axis = plt.subplots(figsize=(9.0, 7.4) if args.clean else (9.4, 7.8))
    image = axis.imshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(len(order)):
        for j in range(len(order)):
            n = int(counts.iloc[i, j])
            if n == 0:
                continue
            # Texto branco sobre célula escura: o limiar 0,55 é onde o azul do
            # colormap deixa de dar contraste com preto.
            texto = (f"{100 * fractions[i, j]:.0f}%" if args.clean
                     else f"{n}\n{100 * fractions[i, j]:.0f}%")
            axis.text(j, i, texto, ha="center", va="center", fontsize=cell_fs,
                      color="white" if fractions[i, j] > 0.55 else INK)

    etiquetas = [f"{name}\n({int(totals[i])}d)" for i, name in enumerate(order)]
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=32, ha="right", fontsize=tick_fs)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(etiquetas, fontsize=tick_fs)
    axis.set_xlabel("Family inferred by label propagation", fontsize=label_fs, color=MUTED)
    axis.set_ylabel("Source family", fontsize=label_fs, color=MUTED)
    axis.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_color(GRID)

    # Emoldura o bloco de geometria -- Felzenszwalb, SLIC e os dois grafos de
    # linha -- que é onde a confusão se concentra. Descriptors fica FORA: ele é
    # a primeira linha da matriz mas acerta 78%, e incluí-lo na moldura sugeriria
    # uma parentesco que a matriz não mostra.
    primeiro = "Felzenszwalb" if "Felzenszwalb" in order else "Region graphs"
    ultimo = "L(G_felz)" if "L(G_felz)" in order else "Line graphs"
    inicio = [i for i, n in enumerate(order) if n == primeiro]
    fim = [i for i, n in enumerate(order) if n == ultimo]
    if inicio and fim:
        a, b = inicio[0] - 0.5, fim[0] + 0.5
        axis.add_patch(plt.Rectangle((a, a), b - a, b - a, fill=False,
                                     edgecolor="#d63384", linewidth=2.0, alpha=0.85))

    # Sem barra de cor no modo clean: com TODA célula rotulada em porcentagem, a
    # escala vira codificação redundante e só rouba largura da matriz.
    if not args.clean:
        barra = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
        barra.set_label("share of row", fontsize=12.5, color=MUTED)
        barra.ax.tick_params(colors=MUTED, labelsize=11)

    if not args.paper:
        acerto = 100 * np.trace(counts.to_numpy()) / totals.sum()
        axis.set_title("Representational identity inferred from redundancy\n"
                       f"Multitask, {int(totals.sum())} features, {len(order)} families · "
                       f"held-out accuracy {acerto:.1f}% (chance {100 / len(order):.1f}%)",
                       fontsize=14, color=INK, pad=14)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white",
                   bbox_inches="tight" if args.paper else None)
    print(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
