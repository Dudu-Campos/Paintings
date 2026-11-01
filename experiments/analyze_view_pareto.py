"""Fronteira de Pareto sobre TODAS as combinacoes de views, e quais nunca ajudam.

Isto nao custa GPU nenhuma: as 2^12 = 4.096 mascaras x 5 folds do
`evaluate_shapley_twelve_views.py` ja SAO a enumeracao completa. O §9.7 declara
como divida que "a fronteira exigiria os 255 subconjuntos sob o MLP (~16 h)" --
sob a sonda LDA ela ja esta paga.

Tres saidas, e a terceira e a que responde "quais views nao ajudam":

1. **Fronteira por dimensao** -- para cada orcamento de colunas, o melhor
   subconjunto. Eixo sem premissa de custo.
2. **Fronteira por custo de extracao** (CPU-h, custos do §9.7 com RAG
   compartilhado). O grafo e o texto NAO entram nesta, por moeda diferente: o
   grafo custa treino e depende dos outros blocos existirem, o texto e GPU.
3. **Distribuicao do marginal sobre as 2.048 coalizoes** de cada view. O Shapley
   e a MEDIA disso; a distribuicao mostra se uma view e ruim sempre ou so em
   media, que e a diferenca entre "aposentar" e "depende do contexto".

Ressalva herdada: sonda LDA. O §8 tem seis casos de "LDA extrai, MLP destroi",
entao isto ORDENA subconjuntos, nao prevê o BranchMLP.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_shapley_twelve_views import VIEWS  # noqa: E402

# Custo de extracao por view, em minutos (§9.7, RAG compartilhado). `Similarity
# graph` e `Text` ficam de fora: moeda diferente (treino e GPU).
COST_MINUTES = {"Hilbert": 3.2, "Gabor": 3.6, "Fourier": 7.0, "Descriptors": 48.2,
                "SLIC": 81.6, "Line graph": 12.7, "Felzenszwalb": 193.25,
                "L(G_felz)": 7.7}
# Dimensoes REAIS de cada bloco. O grafo e o texto tem dimensao como qualquer
# outro (2C+32 = 282 em 125 classes, e 125 escores); a primeira versao deste
# script os punha em 0 porque eles nao custam EXTRACAO, e isso e outro eixo --
# o resultado era um ponto de "0d" na fronteira que na verdade tinha 407d.
DIMS = {"Descriptors": 130, "Felzenszwalb": 69, "SLIC": 69, "Line graph": 63,
        "L(G_felz)": 63, "Hilbert": 57, "Fourier": 134, "Gabor": 32,
        "Random walk": 57, "HVG": 57,
        "Similarity graph": 2 * 125 + 32, "Text": 125}


def frontier(points: pd.DataFrame, x: str, y: str = "bacc") -> pd.DataFrame:
    """Pontos nao dominados: nenhum outro tem x menor-ou-igual E y maior."""
    ordered = points.sort_values([x, y], ascending=[True, False])
    keep, best = [], -np.inf
    for row in ordered.itertuples():
        value = getattr(row, y)
        if value > best:
            keep.append(row.Index)
            best = value
    return points.loc[keep].sort_values(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/multitask_shapley_twelve.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/view_pareto.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.source)
    n = len(VIEWS)
    # Media sobre os folds: a fronteira e sobre o valor esperado, nao sobre um
    # fold. O desvio vai junto para que a legenda possa declarar a barra.
    grouped = raw.groupby("mask").bacc.agg(["mean", "std", "count"])
    if (grouped["count"] != 5).any():
        raise SystemExit("mascaras com menos de 5 folds -- fronteira nao e comparavel")
    table = grouped.reset_index().rename(columns={"mean": "bacc", "std": "sd"})

    members = {m: [VIEWS[i] for i in range(n) if m >> i & 1] for m in table["mask"]}
    table["views"] = table["mask"].map(lambda m: "|".join(members[m]))
    table["n_views"] = table["mask"].map(lambda m: bin(m).count("1"))
    table["dimensions"] = table["mask"].map(
        lambda m: sum(DIMS[v] for v in members[m]))
    # Custo so e definido para subconjuntos sem grafo e sem texto.
    extractable = table["mask"].map(
        lambda m: all(v in COST_MINUTES for v in members[m]))
    table["cost_hours"] = np.where(
        extractable,
        table["mask"].map(lambda m: sum(COST_MINUTES.get(v, 0.0)
                                        for v in members[m])) / 60.0,
        np.nan)

    print(f"{len(table):,} combinacoes, 5 folds cada\n")

    for axis_name, subset in (("dimensions", table[table.n_views > 0]),
                              ("cost_hours", table[extractable & (table.n_views > 0)])):
        front = frontier(subset.dropna(subset=[axis_name]), axis_name)
        print(f"=== fronteira por {axis_name} ({len(front)} pontos) ===")
        for row in front.itertuples():
            budget = getattr(row, axis_name)
            unit = "d" if axis_name == "dimensions" else "h"
            print(f"  {budget:8.2f}{unit}  BAcc {row.bacc:5.2f} ± {row.sd:4.2f}  "
                  f"[{row.n_views}] {row.views}")
        print()

    # Quais views aparecem na fronteira, e quais nunca.
    front_dims = frontier(table[table.n_views > 0], "dimensions")
    on_front = {v for row in front_dims.itertuples() for v in row.views.split("|")}
    print("=== presenca na fronteira por dimensao ===")
    for view in VIEWS:
        share = np.mean([view in row.views.split("|") for row in front_dims.itertuples()])
        flag = "" if view in on_front else "   <== NUNCA"
        print(f"  {view:<18} em {100*share:5.1f}% dos pontos da fronteira{flag}")

    # Distribuicao do marginal sobre as 2.048 coalizoes que nao contem a view.
    print("\n=== marginal sobre TODAS as coalizoes (o Shapley e a media disto) ===")
    value = dict(zip(table["mask"], table.bacc))
    stats = []
    for i, view in enumerate(VIEWS):
        bit = 1 << i
        deltas = np.array([value[m | bit] - value[m] for m in value if not m & bit])
        stats.append(dict(view=view, media=deltas.mean(), mediana=np.median(deltas),
                          p5=np.percentile(deltas, 5), p95=np.percentile(deltas, 95),
                          frac_positiva=100 * (deltas > 0).mean(), n=len(deltas)))
    summary = pd.DataFrame(stats).sort_values("media", ascending=False)
    print(summary.round(2).to_string(index=False))

    table.to_csv(args.output, index=False)
    summary.to_csv(args.output.with_name("view_marginal_distribution.csv"), index=False)
    print(f"\nsalvo: {args.output} e view_marginal_distribution.csv")


if __name__ == "__main__":
    main()
