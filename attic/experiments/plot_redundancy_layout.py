"""Layout dirigido por força do grafo de r², colorido por família.

Esta figura foi descartada uma vez e restaurada depois, e a razão importa.

Ela foi julgada pelo Rand ajustado contra os **sete blocos** do pipeline, deu
0,107 e eu concluí que não mostrava estrutura. A métrica é que estava errada:
SLIC, Felzenszwalb, Line graph e L(G_felz) são todos grafo de regiões, e cobrar
do layout que os separasse era cobrar uma distinção que existe na nossa
engenharia e não no dado. Contra as **famílias** o mesmo agrupamento sobe para
0,194 e a razão de quase-duplicatas triplica.

O que o layout mostra, e mostra bem: os quatro grafos colapsam num blob único,
Fourier fica isolado, Hilbert forma nós compactos e separados, e Descriptors tem
estrutura própria. Colorir por família em vez de por bloco transforma isso de
"falhou em separar" em "separou o que existe".

Limiar em r² ≥ 0,3: acima disso a maioria dos nós fica sem aresta e o
`spring_layout` os espalha ao acaso (ruído de posicionamento parecendo ruído
estrutural); muito abaixo, o grafo vira um componente gigante e colapsa. Os nós
sem parceiro saem do painel e viram um número na legenda -- "esta feature não
tem quase-duplicata" é informação, não sujeira.

Saída: results/redundancy_layout.png e _paper.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED = 42
FAMILIES = {
    "Region graphs": ["SLIC", "Felzenszwalb", "Line graph", "L(G_felz)"],
    "Image statistics": ["Descriptors"],
    "Space-filling curve": ["Hilbert"],
    "Spectral": ["Fourier"],
}
COLOURS = {"Region graphs": "#eb6834", "Image statistics": "#2a78d6",
           "Space-filling curve": "#1baf7a", "Spectral": "#4a3aa7",
           "Text (external)": "#e34948"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--text-cache", type=Path, default=None,
                        help="inclui o bloco Text (27 escores OOF). É a única "
                             "família não-visual, logo o melhor teste de "
                             "sanidade do layout; e é fold-dependente por "
                             "construção, então usa o fold 1 e não entra em "
                             "nenhum número final do paper.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/redundancy_layout.png")
    args = parser.parse_args()

    blocks = [b for names in FAMILIES.values() for b in names]
    cohort = load_blocks(args.dataset, blocks, allow_external=False)
    pieces, family = [], []
    for name, members in FAMILIES.items():
        for block in members:
            pieces.append(cohort.blocks[block])
            family += [name] * cohort.blocks[block].shape[1]
    if args.text_cache and args.text_cache.exists():
        import pandas as pd
        frame = (pd.read_parquet(args.text_cache).drop_duplicates("stem")
                 .set_index("stem"))
        columns = [c for c in frame.columns if c.startswith("score_")]
        aligned = frame.reindex(cohort.stems)[columns]
        if aligned.isna().any().any():
            log("  Text: stems não batem, bloco descartado")
        else:
            pieces.append(aligned.to_numpy(dtype=np.float64))
            family += ["Text (external)"] * len(columns)
            log(f"  Text incluído: {len(columns)} colunas (fold 1, OOF)")

    values = np.hstack(pieces).astype(np.float64)
    family = np.array(family)
    keep = values.std(axis=0) > 1e-9
    values, family = values[:, keep], family[keep]
    standard = (values - values.mean(0)) / values.std(0)
    matrix = (standard.T @ standard / len(standard)) ** 2
    log(f"{values.shape[1]} features · {len(FAMILIES)} famílias")

    rows, columns = np.triu_indices(len(family), 1)
    strong = matrix[rows, columns] >= args.threshold
    edges = [(int(a), int(b), float(w)) for a, b, w in
             zip(rows[strong], columns[strong], matrix[rows, columns][strong])]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    isolated = len(family) - graph.number_of_nodes()
    layout = nx.spring_layout(graph, weight="weight", seed=SEED, iterations=250,
                              k=2.2 / np.sqrt(max(graph.number_of_nodes(), 1)))
    nodes = np.array(sorted(graph.nodes))
    points = np.array([layout[node] for node in nodes])
    index = {node: position for position, node in enumerate(nodes)}
    kinds = family[nodes]
    log(f"{len(edges):,} arestas · {isolated} sem parceiro")

    for with_title in (True, False):
        figure, axis = plt.subplots(figsize=(9.4, 8.4))
        for a, b, weight in edges:
            pair = [index[a], index[b]]
            axis.plot(points[pair, 0], points[pair, 1], zorder=1, color="#8a8a8a",
                      linewidth=0.32,
                      alpha=0.05 + 0.30 * (weight - args.threshold)
                      / (1 - args.threshold + 1e-9))
        for name in COLOURS:
            mask = kinds == name
            if not mask.any():
                continue
            axis.scatter(points[mask, 0], points[mask, 1], s=26, zorder=3,
                         color=COLOURS[name], edgecolor="white", linewidth=0.5,
                         label=f"{name} — {int(mask.sum())} of "
                               f"{int((family == name).sum())}")
        # Fora do eixo: com a cadeia de histograma esticando o quadro, qualquer
        # posição interna cobre nós reais.
        axis.legend(frameon=False, fontsize=9.5, loc="upper left",
                    bbox_to_anchor=(0.0, 1.10), ncol=2,
                    handletextpad=0.4, borderpad=0.3)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.text(0.5, -0.02,
                  "the layout is given only the r² edges — it never sees which "
                  f"family a feature belongs to\n{isolated} features have no "
                  f"partner above r² = {args.threshold:g} and are not drawn",
                  transform=axis.transAxes, ha="center", fontsize=8.8, color=MUTED)
        if with_title:
            figure.suptitle("Redundancy structure emerges without labels\n"
                            "the four region-graph blocks collapse into one; "
                            "spectral and curve stay apart",
                            fontsize=12.5, color=TEXT)
            figure.tight_layout(rect=(0, 0.03, 1, 0.93))
            path = args.output
        else:
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            path = args.output.with_name(args.output.stem + "_paper.png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        log(f"salvo: {path}")


if __name__ == "__main__":
    main()
