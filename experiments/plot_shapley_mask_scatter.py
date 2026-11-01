"""Scatter BAcc x (n_views, n_families), 4096 mascaras do Shapley de 12 views,
coloridas por qual bloco especial (Similarity graph / Text) esta presente --
pedido do usuario para checar visualmente se a regressao do
`analyze_shapley_mask_regression.py` esta sendo carregada por outliers em vez
de refletir a forma real da nuvem de pontos.

Reusa a paleta do projeto (`experiments/view_names.py`) para as duas cores que
tem correspondente direto (Similarity graph, Text); as outras duas categorias
(nucleo puro, os dois juntos) usam cinza neutro e um terceiro tom da mesma
familia categorica, para nao inventar paleta nova (§7.18 e explicito: uma
unica fonte de cor no projeto).

Uso:
    python experiments/plot_shapley_mask_scatter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import colour  # noqa: E402

VIEWS = [
    "Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
    "Hilbert", "Fourier", "Gabor", "Random walk", "HVG",
    "Similarity graph", "Text",
]
FAMILY_OF = {
    "Felzenszwalb": "Region graphs", "SLIC": "Region graphs",
    "Line graph": "Region graphs", "L(G_felz)": "Region graphs",
    "Similarity graph": "Similarity graph",
    "Descriptors": "Image statistics",
    "Hilbert": "Space-filling curve", "Random walk": "Space-filling curve",
    "HVG": "Space-filling curve",
    "Gabor": "Spectral", "Fourier": "Spectral",
    "Text": "Text",
}
GRAPH_BIT = VIEWS.index("Similarity graph")
TEXT_BIT = VIEWS.index("Text")

CAT_COLOUR = {
    "nucleo": "#9a9a9a",
    "+ Graph": colour("Similarity graph"),
    "+ Text": colour("Text"),
    "+ ambos": "#5b3a8e",
}
CAT_ORDER = ["nucleo", "+ Graph", "+ Text", "+ ambos"]


def category(mask: int) -> str:
    g = bool((mask >> GRAPH_BIT) & 1)
    t = bool((mask >> TEXT_BIT) & 1)
    if g and t:
        return "+ ambos"
    if g:
        return "+ Graph"
    if t:
        return "+ Text"
    return "nucleo"


def n_families(mask: int) -> int:
    return len({FAMILY_OF[VIEWS[i]] for i in range(12) if (mask >> i) & 1})


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "multitask_shapley_twelve.csv")
    agg = df.groupby("mask", as_index=False).agg(
        n_views=("n_views", "first"), dimensions=("dimensions", "first"),
        bacc=("bacc", "mean"))
    agg["n_families"] = agg["mask"].map(n_families)
    agg["category"] = agg["mask"].map(category)

    rng = np.random.default_rng(42)
    jitter_x = lambda n: n + rng.uniform(-0.28, 0.28, size=len(n))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, xcol, title in [
        (axes[0], "n_views", "BAcc x número de views"),
        (axes[1], "n_families", "BAcc x número de famílias"),
    ]:
        for cat in CAT_ORDER:
            sub = agg[agg["category"] == cat]
            ax.scatter(jitter_x(sub[xcol].to_numpy(dtype=float)), sub["bacc"],
                      s=10, alpha=0.45, color=CAT_COLOUR[cat], label=cat,
                      edgecolors="none")
        x_full = agg[xcol].to_numpy(dtype=float)
        order = np.argsort(x_full)
        model = sm.OLS(agg["bacc"], sm.add_constant(agg[[xcol]])).fit()
        pred = model.predict(sm.add_constant(agg[[xcol]]))
        ax.plot(x_full[order], pred.to_numpy()[order], color="black",
               lw=1.5, ls="--", label=f"OLS (R²={model.rsquared:.3f})")
        ax.set_xlabel(xcol)
        ax.set_title(title)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("BAcc (%)")
    axes[1].legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("Multitask 125cl, Shapley 2¹² (4.096 máscaras, média de 5 folds) — "
                 "sonda LDA", fontsize=10, y=1.01)
    fig.tight_layout()
    out = ROOT / "results" / "shapley_mask_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"salvo em {out}")

    # Diagnostico adicional: boxplot por n_families, pra ver dispersao/outliers
    # dentro de cada contagem (o que o scatter jitterizado so sugere).
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    groups = [agg[agg["n_families"] == k]["bacc"].to_numpy() for k in range(7)]
    bp = ax2.boxplot(groups, positions=range(7), widths=0.6, showfliers=True,
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d8d8d8")
        patch.set_alpha(0.7)
    for i, g in enumerate(groups):
        ax2.scatter(rng.uniform(-0.15, 0.15, size=len(g)) + i, g, s=6,
                   alpha=0.3, color="#2a78d6", edgecolors="none")
    ax2.set_xlabel("n_families")
    ax2.set_ylabel("BAcc (%)")
    ax2.set_title("Distribuição de BAcc por número de famílias (todas as 4.096 máscaras)")
    ax2.grid(alpha=0.15)
    fig2.tight_layout()
    out2 = ROOT / "results" / "shapley_mask_boxplot.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"salvo em {out2}")

    agg.to_csv(ROOT / "results" / "shapley_mask_scatter_data.csv", index=False,
              float_format="%.4f")


if __name__ == "__main__":
    main()
