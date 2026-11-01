"""Regride BAcc contra (n_views, n_families, n_features/dims) sobre as 4.096
mascaras do Shapley de 12 views (§5.9c), pergunta do usuario: um modelo linear
tao simples consegue prever o desempenho de uma coalizao so pela sua
"forma" (tamanho, cobertura de familia, dimensao), sem saber QUAIS views
estao dentro?

Fonte: results/multitask_shapley_twelve.csv (20.480 linhas = 4.096 mascaras x
5 folds, BranchMLP, LDA -- ver CLAUDE.md §5.9c). O bit i do mask corresponde a
VIEWS[i], na MESMA ordem que `evaluate_shapley_twelve_views.py` usa para
construir o mask (`mask >> i & 1`):

    VIEWS = list(HAND8) + DORMANT + ["Similarity graph", "Text"]
          = Descriptors, Felzenszwalb, SLIC, Line graph, L(G_felz), Hilbert,
            Fourier, Gabor, Random walk, HVG, Similarity graph, Text

Familias (taxonomia de MEDIDA do §5.9c/§7.18, nao o `view_names.FAMILIES` que
agrupa por padrao de erro):
    Region graphs       Felzenszwalb, SLIC, Line graph, L(G_felz)
    Similarity graph     Similarity graph
    Image statistics     Descriptors
    Space-filling curve  Hilbert, Random walk, HVG
    Spectral             Gabor, Fourier
    Text                 Text

`dimensions` no CSV ja e a dimensao REALIZADA da mascara (soma dos dims reais
das views presentes, nao um proxy) -- reusada direto, sem recalcular.

Uso:
    python experiments/analyze_shapley_mask_regression.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]

VIEWS = [
    "Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
    "Hilbert", "Fourier", "Gabor", "Random walk", "HVG",
    "Similarity graph", "Text",
]
assert len(VIEWS) == 12

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
assert set(FAMILY_OF) == set(VIEWS)
N_FAMILIES_TOTAL = len(set(FAMILY_OF.values()))


def families_in_mask(mask: int) -> int:
    present = {FAMILY_OF[VIEWS[i]] for i in range(len(VIEWS)) if (mask >> i) & 1}
    return len(present)


def fit_report(df: pd.DataFrame, label: str, predictors: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(df[predictors])
    y = df["bacc"]
    model = sm.OLS(y, X).fit()
    print(f"\n=== {label} (n={len(df)}) ===")
    print(model.summary().tables[0])
    print(model.summary().tables[1])
    return model


def main() -> None:
    path = ROOT / "results" / "multitask_shapley_twelve.csv"
    df = pd.read_csv(path)
    print(f"carregado {path}: {len(df)} linhas (esperado 4096 mascaras x 5 folds = 20480)")
    assert len(df) == 4096 * 5

    df["n_families"] = df["mask"].map(families_in_mask)
    # conferencia: mask=0 -> 0 familias; mask=4095 (todas) -> 6 familias, 12 views
    full = df[df["mask"] == 4095]
    empty = df[df["mask"] == 0]
    assert (full["n_families"] == N_FAMILIES_TOTAL).all(), "mascara cheia nao bate 6 familias"
    assert (empty["n_families"] == 0).all() and (empty["n_views"] == 0).all()
    print(f"conferencia: mascara cheia tem {N_FAMILIES_TOTAL} familias "
          f"({full['n_views'].iloc[0]} views, {full['dimensions'].iloc[0]} dims), "
          f"mascara vazia tem BAcc={empty['bacc'].iloc[0]:.2f} (acaso=100/125={100/125:.2f})")

    # agregando por mascara (media sobre os 5 folds) -- evita pseudo-replicacao:
    # n_views/n_families/dimensions sao IDENTICOS nos 5 folds da mesma mascara,
    # so o bacc varia por ruido de fold. Regredir nas 20.480 linhas cruas trata
    # esse ruido como 5 observacoes independentes da relacao X->Y, inflando a
    # confianca aparente sem acrescentar informacao sobre a FORMA da relacao.
    agg = df.groupby("mask", as_index=False).agg(
        n_views=("n_views", "first"), n_families=("n_families", "first"),
        dimensions=("dimensions", "first"), bacc=("bacc", "mean"),
        bacc_std=("bacc", "std"))
    print(f"\nagregado por mascara: {len(agg)} pontos "
          f"(desvio entre folds, mediana: {agg['bacc_std'].median():.3f}pp)")

    # colinearidade entre os tres preditores, antes de confiar nos coeficientes
    corr = agg[["n_views", "n_families", "dimensions"]].corr()
    print("\ncorrelacao entre preditores:")
    print(corr.round(3))

    # 1) cada preditor sozinho
    for p in ["n_views", "n_families", "dimensions"]:
        fit_report(agg, f"so {p}", [p])

    # 2) os tres juntos
    full_model = fit_report(agg, "n_views + n_families + dimensions",
                            ["n_views", "n_families", "dimensions"])

    # 3) familias sozinhas vs. views sozinhas -- qual das duas "contagens de
    # cobertura" explica mais, e o que a outra acrescenta por cima
    m_views = sm.OLS(agg["bacc"], sm.add_constant(agg[["n_views"]])).fit()
    m_views_fam = sm.OLS(agg["bacc"], sm.add_constant(agg[["n_views", "n_families"]])).fit()
    print(f"\nso n_views:               R2={m_views.rsquared:.4f}")
    print(f"n_views + n_families:     R2={m_views_fam.rsquared:.4f}  "
          f"(ganho {m_views_fam.rsquared - m_views.rsquared:+.4f}, "
          f"p do n_families={m_views_fam.pvalues['n_families']:.2e})")

    # 4) residuo do modelo completo x dimensoes -- ha algo sistematico sobrevivendo?
    agg["resid"] = full_model.resid
    print(f"\ndesvio padrao do residuo: {agg['resid'].std():.3f}pp "
          f"(desvio do bacc bruto: {agg['bacc'].std():.3f}pp)")

    out = ROOT / "results" / "shapley_mask_regression.csv"
    agg.to_csv(out, index=False, float_format="%.4f")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
