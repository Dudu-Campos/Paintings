"""BAcc x log(dimensoes), 4096 mascaras do Shapley de 12 views -- pergunta do
usuario depois de ver que n_views/n_families perdem informacao que a
dimensao real carrega. `log(dims)` sozinho bate o modelo de 3 variaveis
(R2=0,601 contra 0,528), entao vale olhar a curva.

Mesma paleta por categoria (presenca de Graph/Text) do
`plot_shapley_mask_scatter.py`.

Uso:
    python experiments/plot_shapley_mask_logdims.py
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

CAT_COLOUR = {
    "nucleo": "#9a9a9a",
    "+ Graph": colour("Similarity graph"),
    "+ Text": colour("Text"),
    "+ ambos": "#5b3a8e",
}
CAT_ORDER = ["nucleo", "+ Graph", "+ Text", "+ ambos"]


def category(mask: int) -> str:
    g, t = bool((mask >> GRAPH_BIT) & 1), bool((mask >> TEXT_BIT) & 1)
    if g and t:
        return "+ ambos"
    if g:
        return "+ Graph"
    if t:
        return "+ Text"
    return "nucleo"


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "multitask_shapley_twelve.csv")
    agg = df.groupby("mask", as_index=False).agg(
        dimensions=("dimensions", "first"), bacc=("bacc", "mean"))
    agg["category"] = agg["mask"].map(category)
    agg = agg[agg["dimensions"] > 0].copy()  # log indefinido na mascara vazia (0 dims)
    agg["log_dims"] = np.log(agg["dimensions"])

    model = sm.OLS(agg["bacc"], sm.add_constant(agg[["log_dims"]])).fit()
    model_lin = sm.OLS(agg["bacc"], sm.add_constant(agg[["dimensions"]])).fit()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(42)
    for cat in CAT_ORDER:
        sub = agg[agg["category"] == cat]
        ax.scatter(sub["log_dims"], sub["bacc"], s=10, alpha=0.45,
                  color=CAT_COLOUR[cat], label=cat, edgecolors="none")

    x_line = np.linspace(agg["log_dims"].min(), agg["log_dims"].max(), 200)
    pred = model.predict(sm.add_constant(pd.DataFrame({"log_dims": x_line})))
    ax.plot(x_line, pred, color="black", lw=1.8, ls="--",
           label=f"OLS: BAcc ≈ {model.params['const']:.1f} + {model.params['log_dims']:.2f}·log(dims)\n"
                 f"R²={model.rsquared:.3f}  (vs. dims linear: R²={model_lin.rsquared:.3f})")

    # eixo superior com dims reais, nao log, pra leitura direta
    dim_ticks = [1, 10, 30, 100, 300, 600, 1138]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([np.log(d) for d in dim_ticks])
    ax2.set_xticklabels([str(d) for d in dim_ticks])
    ax2.set_xlabel("dimensões (escala real)")

    ax.set_xlabel("log(dimensões)")
    ax.set_ylabel("BAcc (%)")
    ax.set_title("Multitask 125cl, Shapley 2¹² (4.096 máscaras, média de 5 folds) — sonda LDA\n"
                "BAcc x log(número de features)", fontsize=10)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    out = ROOT / "results" / "shapley_mask_logdims.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"salvo em {out}")
    print(f"BAcc ~ a + b*log(dims): R2={model.rsquared:.4f} b={model.params['log_dims']:.3f} "
          f"a={model.params['const']:.3f}")
    print(f"BAcc ~ a + b*dims (linear): R2={model_lin.rsquared:.4f}")


if __name__ == "__main__":
    main()
