#!/usr/bin/env python3
"""Quao firme e a inclinacao das curvas log-log do artigo (13/out).

Pergunta do usuario: *"acha importante colocar como estimamos a angulacao das
curvas de loglog, ja que essa estimacao e sempre problematica?"* -- e sim, porque
no §5.1 o expoente **e** o resultado, nao um detalhe de ajuste.

**O QUE TODOS OS PRODUTORES FAZEM HOJE:** `-polyfit(log x, log y, 1)[0]` --
minimos quadrados sobre os logs, sem peso, com todos os pontos. Isso tem tres
fragilidades conhecidas, e este script mede as tres em vez de argumentar:

  1. **Nenhuma incerteza no PROPRIO expoente.** As tabelas imprimem quatro casas
     decimais de um numero ajustado em 4 a 8 pontos. Se a comparacao e ENTRE
     expoentes (que e o caso do §5.1 e do §5.3), ignorar o erro de cada um pode
     transformar chacoalho de ajuste em "tendencia".
  2. **Sensibilidade a FAIXA.** O §9.7 descartou os dois primeiros pontos da
     curva de custo DEPOIS de ver o grafico (R² 0,985 contra 0,83 com todos).
     Esta declarado la, mas sem uma regra no metodo e escolha post-hoc.
  3. **OLS nos logs assume ruido multiplicativo** e ignora a barra de cada ponto.
     Para taxa de erro e defensavel; para `100 - BAcc` perto do teto, menos.

O script devolve, por curva: o expoente, o erro padrao analitico da inclinacao,
o IC95, e a amplitude deixa-um-ponto-de-fora. Quando a curva vem em familia
(varios subconjuntos), testa tambem a TENDENCIA entre os expoentes reamostrando
cada um dentro do proprio erro -- que e o teste que o §5.1 precisa e nao tem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def inclinacao(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Inclinacao de OLS nos logs e o erro padrao ANALITICO dela.

    `se = sqrt( SSE/(n-2) / Sxx )` -- a formula fechada; com 4-8 pontos o
    bootstrap seria pior, porque reamostrar 5 pontos devolve menos informacao
    que a forma fechada sob o mesmo modelo de ruido.
    """
    lx, ly = np.log(x), np.log(y)
    b, a = np.polyfit(lx, ly, 1)
    res = ly - (b * lx + a)
    se = np.sqrt((res ** 2).sum() / (len(lx) - 2) / ((lx - lx.mean()) ** 2).sum())
    return b, se


def analisa(nome: str, grupos: dict[str, tuple[np.ndarray, np.ndarray]],
            sinal: int, rotulo_x: str) -> pd.DataFrame:
    print("\n" + "=" * 96)
    print(f"{nome}   (x = {rotulo_x})")
    print("=" * 96)
    print(f"{'condicao':22s} {'pts':>4s} {'expoente':>9s} {'SE':>8s} {'IC95':>18s} "
          f"{'sem 1o pt':>10s} {'sem ultimo':>11s} {'ampl. LOO':>10s}")
    linhas = []
    for k, (x, y) in grupos.items():
        o = np.argsort(x)
        x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
        b, se = inclinacao(x, y)
        t = stats.t.ppf(0.975, len(x) - 2)
        loo = [sinal * inclinacao(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
        e = sinal * b
        print(f"{k[:22]:22s} {len(x):4d} {e:9.4f} {se:8.4f} "
              f"[{e - t * se:+.4f},{e + t * se:+.4f}] {loo[0]:10.4f} {loo[-1]:11.4f} "
              f"{max(loo) - min(loo):10.4f}")
        linhas.append({"condicao": k, "pontos": len(x), "expoente": e, "se": se,
                       "ic_lo": e - t * se, "ic_hi": e + t * se,
                       "loo_min": min(loo), "loo_max": max(loo),
                       "loo_amplitude": max(loo) - min(loo)})
    return pd.DataFrame(linhas)


def tendencia(d: pd.DataFrame, eixo: np.ndarray, rot: str, rng) -> None:
    """A tendencia ENTRE expoentes, propagando o erro de cada um."""
    b, se = d.expoente.values, d.se.values
    r, p = stats.pearsonr(eixo, b)
    sim = np.array([stats.pearsonr(eixo, b + rng.normal(0, se))[0] for _ in range(4000)])
    amp = b.max() - b.min()
    print(f"\n  tendencia contra {rot}:  r = {r:+.3f}  p = {p:.4f}  n = {len(b)}")
    print(f"  amplitude {amp:.4f}  /  SE tipico {np.median(se):.4f}  =  "
          f"{amp / np.median(se):.1f}x")
    print(f"  r propagando o erro de cada expoente: mediana {np.median(sim):+.3f}  "
          f"IC95 [{np.percentile(sim, 2.5):+.3f}, {np.percentile(sim, 97.5):+.3f}]  "
          f"P(mesmo sinal) = {np.mean(np.sign(sim) == np.sign(r)):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/exponent_robustness.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    tudo = []

    # -- §5.1: erro contra TAMANHO DE TREINO, por numero de representacoes -----
    d = pd.read_csv(ROOT / "results/representation_data_grid_multitask.csv")
    d["err"] = 100 - 100 * d.bacc
    g = d.groupby(["n_modalities", "train_size"]).err.mean().reset_index()
    gr = {f"{m} views": (s.train_size.values, s.err.values)
          for m, s in g.groupby("n_modalities")}
    t = analisa("§5.1  o expoente cresce com o numero de representacoes",
                gr, -1, "tamanho de treino")
    tendencia(t, np.array(sorted(g.n_modalities.unique())), "n de views", rng)
    t["curva"] = "5.1 dados"; tudo.append(t)

    # -- §5.3: o CONTROLE NEGATIVO -- erro contra COMPONENTES DE PCA ----------
    #
    # AS DUAS GRANULARIDADES, e o descasamento e o achado. O numero publicado
    # (r = -0,58, p = 0,13) vem do arquivo de 125 CLASSES; o §5.1 roda em 25 e
    # o §7.7 registra que a grade de 125 nunca foi rodada. Logo o controle
    # negativo NAO esta pareado com o que ele controla -- e no eixo de
    # componentes o sinal INVERTE entre as granularidades.
    for sfx, rot in (("", "25 classes -- PAREADO com o §5.1"),
                     ("_125", "125 classes -- o PUBLICADO")):
        d = pd.read_csv(ROOT / f"results/pca_curve_by_subset_multitask{sfx}.csv")
        d["err"] = 100 - 100 * d.bacc
        g = d.groupby(["n_modalities", "k"]).err.mean().reset_index()
        gr = {f"{m} views": (s.k.values, s.err.values) for m, s in g.groupby("n_modalities")}
        t = analisa(f"§5.3  CONTROLE NEGATIVO: expoente contra componentes ({rot})",
                    gr, -1, "componentes de PCA")
        tendencia(t, np.array(sorted(g.n_modalities.unique())), "n de views", rng)
        t["curva"] = f"5.3 componentes{sfx}"; tudo.append(t)

    # -- §5.14: erro contra numero de FEATURES, por pool ----------------------
    gr = {}
    for rot, f in (("banco 639d", "redundancy_selection_curve"),
                   ("Descriptors 130d", "redundancy_selection_descriptors")):
        d = pd.read_csv(ROOT / f"results/{f}.csv")
        d = d[d.order.eq("max-r2-strat")]
        d["err"] = 100 - 100 * d.bacc if d.bacc.max() <= 1 else 100 - d.bacc
        g = d.groupby("n_features").err.mean().reset_index()
        g = g[g.n_features <= 128]          # faixa onde OS DOIS pools tem folga
        gr[rot] = (g.n_features.values, g.err.values)
    t = analisa("§5.14  curva de selecao (faixa <=128, onde os dois pools tem folga)",
                gr, -1, "n de features")
    t["curva"] = "5.14 selecao"; tudo.append(t)

    # -- §9.7: BAcc contra CUSTO, e o efeito de descartar os 2 primeiros ------
    d = pd.read_csv(ROOT / "results/cost_ordered_multitask.csv")
    g = d.groupby(["step", "cost_hours"]).bacc.mean().reset_index()
    g["b"] = 100 * g.bacc if g.bacc.max() <= 1 else g.bacc
    gr = {"os 8 pontos": (g.cost_hours.values, g.b.values),
          "do 3o em diante": (g.cost_hours.values[2:], g.b.values[2:])}
    t = analisa("§9.7  elasticidade BAcc x custo -- a faixa foi escolhida POST-HOC",
                gr, +1, "CPU-horas")
    t["curva"] = "9.7 custo"; tudo.append(t)

    out = pd.concat(tudo, ignore_index=True)
    out.to_csv(args.output, index=False, float_format="%.5f")
    print(f"\n[{args.output}]")


if __name__ == "__main__":
    main()
