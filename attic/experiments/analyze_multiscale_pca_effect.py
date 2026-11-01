"""Multiscale nativo (1.032d) contra Multiscale comprimido (PCA-91) na grade.

O achado 20 registra que Multiscale, entrando como sétima modalidade em
dimensão nativa, **atrapalha em n=20.000** e faz o expoente do erro parar de
crescer -- e atribui isso ao salto de tamanho (1.032d contra 57-152d das
outras seis). Isso era inferência, não medição: nunca se separou "conteúdo
ruim" de "bloco grande demais".

Este script separa. As colunas são as mesmas nos dois braços; muda só a
dimensão em que elas entram, comprimidas a 91 componentes -- a **média das
outras seis** (Descriptors 130, Fourier 152, SLIC 71, Felzenszwalb 71,
Line graph 63, Hilbert 57), PCA ajustado só no treino de cada fold.

    anomalia some     era tamanho; m=7 entra na figura e a série monotônica
                      do expoente ganha um ponto
    anomalia fica     era conteúdo; a exclusão passa a ter justificativa
                      medida em vez de argumento de tamanho

Saída: results/multiscale_pca_effect.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
REQUESTED = np.array([2500, 5000, 10000, 20000, 40000])


def bucket(value: int) -> int:
    near = REQUESTED[np.abs(REQUESTED - value) <= 0.02 * value]
    return int(near[0]) if len(near) else 63365


def peaks_of(path: Path) -> pd.DataFrame:
    """Pico sobre k por (m, fold, n_train) -- para o MLP k é sempre 0."""
    detail = pd.read_csv(path)
    detail["n_train"] = detail.n_train.map(bucket)
    index = detail.groupby(["n_modalities", "fold", "n_train"]).bacc.idxmax()
    return detail.loc[index, ["n_modalities", "fold", "n_train", "bacc", "dimensions"]]


def exponent(curve: pd.Series) -> tuple[float, float]:
    """Expoente da lei de escala em **erro**, não em acurácia (achado 18)."""
    x = np.log(curve.index.to_numpy(dtype=float))
    y = np.log(100 - curve.to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (intercept + slope * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return slope, r2


def report(name: str, native: Path, squeezed: Path, rows: list) -> None:
    if not native.exists() or not squeezed.exists():
        print(f"\n### {name}: faltando {native if not native.exists() else squeezed}")
        return
    left, right = peaks_of(native), peaks_of(squeezed)
    print(f"\n{'=' * 66}\n### {name}\n{'=' * 66}")

    # 1. m=7 nativo vs comprimido, tamanho a tamanho.
    a = left[left.n_modalities == 7].set_index(["fold", "n_train"]).bacc * 100
    b = right[right.n_modalities == 7].set_index(["fold", "n_train"]).bacc * 100
    six = left[left.n_modalities == 6].set_index(["fold", "n_train"]).bacc * 100
    if a.empty:
        print("  (sem braço nativo de m=7 neste classificador)")
    else:
        print(f"\n{'n_train':>9}  {'m=6':>7}  {'m=7 nat':>8}  {'m=7 pca':>8}  "
              f"{'nat−6':>7}  {'pca−6':>7}  {'p(pca−6)':>9}")
        for n in sorted({n for _, n in a.index}):
            fa = a.xs(n, level="n_train"); fb = b.xs(n, level="n_train")
            fs = six.xs(n, level="n_train")
            da, db = (fa - fs).dropna(), (fb - fs).dropna()
            p = ttest_rel(fb.loc[db.index] , fs.loc[db.index])[1] if len(db) > 2 else np.nan
            print(f"{n:>9,}  {fs.mean():>7.2f}  {fa.mean():>8.2f}  {fb.mean():>8.2f}  "
                  f"{da.mean():>+7.2f}  {db.mean():>+7.2f}  {p:>9.4f}")
            rows.append({"classifier": name, "n_train": n, "bacc_m6": fs.mean(),
                         "bacc_m7_native": fa.mean(), "bacc_m7_pca": fb.mean(),
                         "delta_native": da.mean(), "delta_pca": db.mean(), "p_pca": p})

    # 2. Expoente do erro: a série monotônica volta a crescer em m=7?
    print(f"\n{'m':>3}  {'expoente (nativo)':>18}  {'expoente (pca)':>16}")
    for count in sorted(right.n_modalities.unique()):
        cn = left[left.n_modalities == count].groupby("n_train").bacc.mean() * 100
        cs = right[right.n_modalities == count].groupby("n_train").bacc.mean() * 100
        en = f"{-exponent(cn)[0]:.4f}" if len(cn) >= 3 else "  --  "
        es = f"{-exponent(cs)[0]:.4f}" if len(cs) >= 3 else "  --  "
        mark = "  <-- muda só aqui" if count == 7 else ""
        print(f"{count:>3}  {en:>18}  {es:>16}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multiscale_pca_effect.csv")
    args = parser.parse_args()
    out = ROOT / "outputs"
    rows: list = []
    report("LDA", out / "representation_data_grid/fold_results.csv",
           out / "representation_data_grid_mspca/fold_results.csv", rows)
    report("MLP", out / "representation_data_grid_mlp/fold_results.csv",
           out / "representation_data_grid_mspca_mlp/fold_results.csv", rows)
    if rows:
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nsalvo: {args.output}")


if __name__ == "__main__":
    main()
