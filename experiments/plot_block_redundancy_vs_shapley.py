"""Redundância medida × contribuição medida, por bloco (Multitask, 8 modalidades).

Figura do achado do §6: **redundância não prevê contribuição**. Eixo x é o R²
mediano de reconstruir cada bloco a partir dos outros sete (auditoria
não-supervisionada, nenhum rótulo entra); eixo y é o valor de Shapley exato
sobre 2^9 máscaras.

Os dois extremos carregam o argumento e ficam anotados: Felzenszwalb tem R²
0,919 -- é quase inteiramente reconstruível -- e ainda assim é o maior Shapley
da tabela; Descriptors tem R² 0,101, é praticamente ortogonal, e vale
essencialmente o mesmo.

Usa **mediana** e não média: o R² por feature não tem piso inferior, e blocos
com features quase constantes produzem médias de −9, que tornam o eixo ilegível
sem mudar a conclusão (r = +0,037 na média contra −0,062 na mediana).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]

# Cores por família, na convenção já usada nas figuras do projeto.
FAMILIA = {
    "SLIC": ("segmentação", "#d63384"), "Felzenszwalb": ("segmentação", "#d63384"),
    "Line graph": ("grafo de linha", "#8b3fa8"), "L(G_felz)": ("grafo de linha", "#8b3fa8"),
    "Descriptors": ("imagem", "#1baf7a"), "Gabor": ("textura", "#e8890c"),
    "Hilbert": ("curva", "#2b7bba"), "Fourier": ("espectral", "#5a6b7d"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/block_redundancy_multitask.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/block_redundancy_vs_shapley.png")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    x = data["r2_median"].to_numpy()
    y = data["shapley"].to_numpy()
    r, p = pearsonr(x, y)

    figure, axis = plt.subplots(figsize=(8.4, 5.6))
    vistos = set()
    for row in data.itertuples():
        familia, cor = FAMILIA.get(row.block, ("outro", "#888888"))
        axis.scatter(row.r2_median, row.shapley, s=190, color=cor, zorder=3,
                     edgecolor="white", linewidth=1.6,
                     label=familia if familia not in vistos else None)
        vistos.add(familia)

    # Rótulos deslocados à mão para não colidirem: os três de segmentação ficam
    # praticamente empilhados em x entre 0,92 e 0,93.
    deslocamento = {"SLIC": (0.004, 0.16), "L(G_felz)": (0.004, -0.26),
                    "Felzenszwalb": (-0.10, 0.20), "Gabor": (0.010, 0.0),
                    "Hilbert": (0.012, 0.0), "Fourier": (0.012, 0.0),
                    "Line graph": (0.012, 0.0), "Descriptors": (0.012, 0.0)}
    for row in data.itertuples():
        dx, dy = deslocamento.get(row.block, (0.012, 0.0))
        axis.annotate(f"{row.block}\n({int(row.dims)}d)",
                      (row.r2_median + dx, row.shapley + dy),
                      fontsize=10.5, va="center", linespacing=1.15)

    faixa = np.linspace(x.min() - 0.05, x.max() + 0.05, 10)
    coef = np.polyfit(x, y, 1)
    axis.plot(faixa, np.polyval(coef, faixa), "--", color="#b0b0b0", linewidth=1.4,
              zorder=1, label=f"ajuste linear (r = {r:+.3f}, p = {p:.2f})")

    axis.set_xlabel("Redundância — R² mediano de reconstruir o bloco a partir dos outros sete",
                    fontsize=12)
    axis.set_ylabel("Contribuição — valor de Shapley (p.p. de BAcc)", fontsize=12)
    axis.set_title("Redundância medida não prevê contribuição medida\n"
                   "Multitask, 25 classes, 8 modalidades visuais", fontsize=13)
    axis.tick_params(labelsize=11)
    axis.set_xlim(-0.05, 1.12)
    axis.set_ylim(y.min() - 0.7, y.max() + 0.7)
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(fontsize=10, loc="lower left", framealpha=0.95)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, facecolor="white")
    print(f"salvo: {args.output}  (r = {r:+.3f}, p = {p:.3f}, n = {len(x)})")


if __name__ == "__main__":
    main()
