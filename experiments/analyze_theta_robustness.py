"""Perda de pior caso ao trocar theta*, integrada sobre Theta_f por interpolação.

O quantil de uma grade é estatística da GRADE, não da transformação: pôr 5
níveis de resolução (eixo que quase não importa) e 4 de bandas (eixo que
importa) dilui a distribuição e melhora o p05 sem que nada da representação
mude. A saída é interpolar -- ajustar a superfície de resposta, amostrar
uniformemente na caixa Theta_f declarada, e tirar o quantil daí.

Isso só é legítimo se a superfície for de fato bem descrita pelo ajuste, então
o script **valida por leave-one-out** antes de usar: 6 coeficientes para 20
pontos é folgado o bastante para superajustar, e o R² dentro da amostra não
detecta isso. Se o RMSE de LOO não for muito menor que o desvio dos dados, o
interpolador não é confiável e o número de grade é o melhor disponível.

Duas estatísticas, porque uma só não captura a alegação:

  p05 do Δ com sinal   perda de pior caso a 95% ao trocar theta* por um theta
                       admissível qualquer -- o limite superior do risco.
  massa |Δ| <= eps     fração de Theta_f que custa quase nada -- a concentração.

Juntas dizem "95% dos theta custam no máximo X, e Y% custam quase nada", que é
a alegação da seção de theta-robustez quantificada.

**Não extrapola.** A caixa é o envelope convexo dos pontos medidos; com
curvatura significativa o ajuste quadrático diverge rápido fora do suporte.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAR = (128, 8)


def design(size: np.ndarray, bands: np.ndarray, order: str) -> np.ndarray:
    s, b = np.log2(size), np.log2(bands)
    columns = [np.ones_like(s), s, b]
    if order in ("interaction", "quadratic"):
        columns.append(s * b)
    if order == "quadratic":
        columns += [s ** 2, b ** 2]
    return np.column_stack(columns)


def loo_rmse(size, bands, y, order) -> float:
    """Leave-one-out: o R² dentro da amostra não vê superajuste com 6
    coeficientes em 20 pontos."""
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        X = design(size[keep], bands[keep], order)
        beta, *_ = np.linalg.lstsq(X, y[keep], rcond=None)
        errors.append(y[i] - design(size[i:i + 1], bands[i:i + 1], order) @ beta)
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/theta_hilbert.csv")
    parser.add_argument("--metric", default="lda_bacc")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if args.metric not in data.columns:
        for extra in ("results/theta_hilbert_mlp.csv", "results/theta_hilbert_pipeline_grid.csv"):
            path = ROOT / extra
            if path.exists():
                other = pd.read_csv(path)
                if args.metric in other.columns:
                    data = data.merge(other[["n_size", "n_bands", args.metric]],
                                      on=["n_size", "n_bands"], how="inner")
                    break
    size = data.n_size.to_numpy(float)
    bands = data.n_bands.to_numpy(float)
    star = data[(data.n_size == STAR[0]) & (data.n_bands == STAR[1])][args.metric].iloc[0]
    delta = data[args.metric].to_numpy(float) - star

    print(f"{args.metric}: {len(data)} configurações, theta* = {star:.2f}\n")
    print(f"{'ajuste':<12}{'R² amostra':>12}{'RMSE LOO':>11}{'sd dos dados':>14}")
    chosen, best = None, np.inf
    for order in ("additive", "interaction", "quadratic"):
        X = design(size, bands, order)
        beta, *_ = np.linalg.lstsq(X, delta, rcond=None)
        r2 = 1 - np.var(delta - X @ beta) / np.var(delta)
        rmse = loo_rmse(size, bands, delta, order)
        print(f"{order:<12}{r2:>12.4f}{rmse:>11.3f}{delta.std():>14.3f}")
        if rmse < best:
            chosen, best = order, rmse
    print(f"\nescolhido por LOO: {chosen}  (RMSE {best:.3f}pp contra sd {delta.std():.3f}pp)")
    if best > 0.6 * delta.std():
        print("  AVISO: o interpolador explica pouco; preferir os números de grade")

    X = design(size, bands, chosen)
    beta, *_ = np.linalg.lstsq(X, delta, rcond=None)

    # Caixa = envelope dos pontos medidos. Amostragem uniforme em LOG, que é a
    # escala em que o parâmetro é escolhido na prática (potências de 2).
    grid_s = np.linspace(np.log2(size.min()), np.log2(size.max()), args.samples)
    grid_b = np.linspace(np.log2(bands.min()), np.log2(bands.max()), args.samples)
    mesh_s, mesh_b = np.meshgrid(grid_s, grid_b)
    surface = design(2 ** mesh_s.ravel(), 2 ** mesh_b.ravel(), chosen) @ beta

    print(f"\ncaixa Theta_f: n_size [{int(size.min())}, {int(size.max())}] x "
          f"n_bands [{int(bands.min())}, {int(bands.max())}], "
          f"{args.samples}x{args.samples} pontos uniformes em log")
    print(f"{'':>22}{'interpolado':>13}{'contagem de grade':>20}")
    raw = delta[~((size == STAR[0]) & (bands == STAR[1]))]
    for label, value, reference in (
            ("p05 (pior caso 95%)", np.percentile(surface, 5), np.percentile(raw, 5)),
            ("mediana", np.median(surface), np.median(raw)),
            ("p95", np.percentile(surface, 95), np.percentile(raw, 95)),
            ("mínimo", surface.min(), raw.min())):
        print(f"  {label:<20}{value:>+13.2f}{reference:>+20.2f}")
    for eps in args.epsilons:
        print(f"  massa |Δ| <= {eps:<8.2f}{100 * (np.abs(surface) <= eps).mean():>12.1f}%"
              f"{100 * (np.abs(raw) <= eps).mean():>19.1f}%")


if __name__ == "__main__":
    main()
