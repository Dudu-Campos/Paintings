"""As quatro curvas BAcc x log(dims) separadas por presenca de Graph/Text --
pergunta do usuario depois de ver que a inclinacao cai de ~6,3 para ~2,0
quando o Similarity graph esta presente, e o Text nao tem esse efeito.
Pequenos multiplos com eixos compartilhados para comparar inclinacao
diretamente (a mesma logica de `plot_standalone_by_classifier.py`: escala
livre esmagaria os paineis de inclinacao mais achatada).

Uso:
    python experiments/plot_shapley_mask_logdims_by_category.py
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
GRAPH_BIT = VIEWS.index("Similarity graph")
TEXT_BIT = VIEWS.index("Text")

PANELS = [
    ("núcleo (sem Graph nem Text)", "#9a9a9a"),
    ("+ Graph", colour("Similarity graph")),
    ("+ Text", colour("Text")),
    ("+ ambos", "#5b3a8e"),
]


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "multitask_shapley_twelve.csv")
    agg = df.groupby("mask", as_index=False).agg(
        dimensions=("dimensions", "first"), bacc=("bacc", "mean"))
    agg = agg[agg["dimensions"] > 0].copy()
    agg["log_dims"] = np.log(agg["dimensions"])
    mask_arr = agg["mask"].to_numpy()  # pandas 3.x: Series nao suporta >> direto

    has_graph = ((mask_arr >> GRAPH_BIT) & 1).astype(bool)
    has_text = ((mask_arr >> TEXT_BIT) & 1).astype(bool)
    subsets = {
        "núcleo (sem Graph nem Text)": agg[~has_graph & ~has_text],
        "+ Graph": agg[has_graph & ~has_text],
        "+ Text": agg[~has_graph & has_text],
        "+ ambos": agg[has_graph & has_text],
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), sharex=True, sharey=True)
    xmin, xmax = agg["log_dims"].min(), agg["log_dims"].max()
    x_line = np.linspace(xmin, xmax, 200)

    for ax, (label, colr) in zip(axes.flat, PANELS):
        sub = subsets[label]
        model = sm.OLS(sub["bacc"], sm.add_constant(sub[["log_dims"]])).fit()
        pred = model.predict(sm.add_constant(pd.DataFrame({"log_dims": x_line})))
        ax.scatter(sub["log_dims"], sub["bacc"], s=11, alpha=0.5, color=colr,
                  edgecolors="none")
        ax.plot(x_line, pred, color="black", lw=1.6, ls="--")
        ax.set_title(f"{label}  (n={len(sub)})", fontsize=10)
        ax.text(0.04, 0.95,
               f"BAcc ≈ {model.params['const']:.1f} + {model.params['log_dims']:.2f}·log(dims)\n"
               f"R²={model.rsquared:.3f}",
               transform=ax.transAxes, fontsize=8.5, va="top",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
        ax.grid(alpha=0.15)

    for ax in axes[1, :]:
        ax.set_xlabel("log(dimensões)")
    for ax in axes[:, 0]:
        ax.set_ylabel("BAcc (%)")

    fig.suptitle("Multitask 125cl, Shapley 2¹² — BAcc x log(dims) por presença de Graph/Text\n"
                 "(inclinação achata de ~6,3 para ~2,0 sempre que o Similarity graph entra)",
                 fontsize=10.5, y=1.00)
    fig.tight_layout()

    out = ROOT / "results" / "shapley_mask_logdims_by_category.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"salvo em {out}")


if __name__ == "__main__":
    main()
