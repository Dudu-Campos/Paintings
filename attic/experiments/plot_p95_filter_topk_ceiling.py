"""BAcc(k) = a - b*k^(-c) sobre as curvas do experimento de filtro p95_r2
(evaluate_p95_redundancy_filter.py). Método de perfil sobre `a` (grade de
candidatos + regressao linear em log(a-BAcc) vs log(k) por fold), igual ao
achado 7 do CLAUDE.md -- mais estavel que curve_fit 3-parametros direto.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/p95_redundancy_filter_curve.csv"
OUTPUT = ROOT / "results/p95_filter_topk_ceiling.png"

COLORS = {"supervised": "#2a78d6", "random": "#1baf7a", "p95": "#e34948"}
LABELS = {"supervised": "Supervisionado (coef. de estrutura)", "random": "Aleatorio", "p95": "p95_r2"}


def fit_ceiling(k: np.ndarray, bacc: np.ndarray) -> tuple[float, float, float, float, bool]:
    """Retorna (a, b, c, r2, no_limite) via perfil sobre a."""
    upper = min(0.999, bacc.max() + 0.30)
    candidates = np.linspace(bacc.max() + 1e-4, upper, 4000)
    best = None
    for a in candidates:
        residual = a - bacc
        if np.any(residual <= 0):
            continue
        y = np.log(residual)
        x = np.log(k.astype(float))
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if best is None or r2 > best[3]:
            best = (a, np.exp(intercept), -slope, r2)
    a, b, c, r2 = best
    hit_limit = a >= upper - 1e-3
    return a, b, c, r2, hit_limit


def main() -> None:
    data = pd.read_csv(INPUT)
    methods = ["supervised", "random", "p95"]

    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    print("=== ajuste BAcc(k) = a - b*k^(-c), por metodo ===")
    for method in methods:
        subset = data[data["method"] == method]
        summary = subset.groupby("k")["balanced_accuracy"].mean().sort_index()
        k_values = summary.index.to_numpy()
        bacc_values = summary.to_numpy()

        axis.scatter(k_values, bacc_values * 100, s=28, color=COLORS[method],
                     zorder=4, label=LABELS[method])

        fold_fits = []
        for fold, rows in subset.groupby("fold"):
            rows = rows.sort_values("k")
            k_fold = rows["k"].to_numpy()
            bacc_fold = rows["balanced_accuracy"].to_numpy()
            a, b, c, r2, hit_limit = fit_ceiling(k_fold, bacc_fold)
            fold_fits.append((a, b, c, r2, hit_limit))

        a_vals = np.array([f[0] for f in fold_fits])
        c_vals = np.array([f[2] for f in fold_fits])
        n_hit_limit = sum(f[4] for f in fold_fits)
        print(f"{method:11s}: a={a_vals.mean()*100:.2f} +/- {a_vals.std(ddof=0)*100:.2f}%  "
              f"c={c_vals.mean():.3f} +/- {c_vals.std(ddof=0):.3f}  "
              f"({n_hit_limit}/5 folds bateram no limite da grade -- teto NAO identificavel se >0)")

        a_mean, b_mean, c_mean, _, _ = fit_ceiling(k_values, bacc_values)
        k_line = np.linspace(k_values.min(), k_values.max() * 1.6, 200)
        fitted = a_mean - b_mean * k_line ** (-c_mean)
        axis.plot(k_line, fitted * 100, color=COLORS[method], linewidth=1.4,
                  linestyle="--", alpha=0.85, zorder=3)
        axis.axhline(a_mean * 100, color=COLORS[method], linewidth=0.8,
                     linestyle=":", alpha=0.5, zorder=2)

    axis.set_xscale("log")
    axis.set_xlabel("k (numero de features selecionadas)", fontsize=10.5)
    axis.set_ylabel("Balanced Accuracy (%)", fontsize=10.5)
    axis.set_title("Curva de saturacao LDA: BAcc(k) por metodo de selecao\n"
                    "(banco rico de 1.642 features, tracejado = ajuste a - b*k^(-c))",
                    fontsize=11.5, fontweight="bold")
    axis.legend(frameon=False, fontsize=9.5, loc="lower right")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, facecolor="white")
    print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
