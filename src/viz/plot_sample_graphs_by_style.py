"""
plot_sample_graphs_by_style.py — Desenha um grafo RAG aleatório de cada
estilo, posicionando nós nas coordenadas (X,Y) reais e coloridos pela
cor Lab média — para inspeção visual de quão distintas (ou parecidas)
as estruturas de grafo são entre estilos diferentes.

Cada nó é desenhado como um círculo cujo raio é proporcional à
area_frac (fração de área que o componente ocupa na imagem), e as
arestas RAG (já materializadas no .coarse) são desenhadas como linhas
finas conectando os centróides.

Uso:
    python plot_sample_graphs_by_style.py \
        --coarse_dir ./coarsening_output_v3/coarse \
        --meta       /exp/duducampos/Paintings/meta/meta.csv \
        --output     ./plots \
        --n_cols     6 \
        --seed       42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage.color import lab2rgb

from lda_graph_features import read_coarse2, load_meta


def lab_to_rgb_safe(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lab = np.stack([L, a, b], axis=-1)
    rgb = lab2rgb(lab)
    return np.clip(rgb, 0, 1)


def draw_graph(ax, nodes: np.ndarray, edges: np.ndarray, title: str):
    n_nodes = len(nodes)

    if n_nodes == 0:
        ax.set_title(f'{title}\n(grafo vazio)', fontsize=9)
        ax.axis('off')
        return

    X, Y = nodes['X'], nodes['Y']
    colors = lab_to_rgb_safe(nodes['L'], nodes['a'], nodes['b'])

    area = np.maximum(nodes['area_frac'], 1e-6)
    sizes = 800 * np.sqrt(area)

    if len(edges) > 0:
        max_edges_draw = 3000
        if len(edges) > max_edges_draw:
            idx = np.random.default_rng(0).choice(
                len(edges), max_edges_draw, replace=False)
            edges_draw = edges[idx]
        else:
            edges_draw = edges

        segments = []
        for e in edges_draw:
            s, d = int(e['src']), int(e['dst'])
            if s < n_nodes and d < n_nodes:
                segments.append([(X[s], Y[s]), (X[d], Y[d])])

        lc = LineCollection(segments, colors='gray', linewidths=0.3, alpha=0.25)
        ax.add_collection(lc)

    ax.scatter(X, Y, s=sizes, c=colors, edgecolors='black',
               linewidths=0.15, alpha=0.9, zorder=3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n{n_nodes:,} nós  {len(edges):,} arestas', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--meta',       default='')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--n_cols',     type=int, default=6)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--max_styles', type=int, default=30)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.meta)
    paths = list(Path(args.coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse encontrados: {len(paths):,}')

    by_style: dict[str, list[Path]] = {}
    for p in paths:
        style = meta.get(p.stem, 'unknown')
        if style == 'unknown':
            continue
        by_style.setdefault(style, []).append(p)

    styles = sorted(by_style.keys())[:args.max_styles]
    print(f'{len(styles)} estilos encontrados (mostrando até {args.max_styles})')

    rng = np.random.default_rng(args.seed)
    n_cols = args.n_cols
    n_rows = int(np.ceil(len(styles) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2, n_rows*3.2))
    axes = np.array(axes).reshape(-1)

    for i, style in enumerate(styles):
        candidates = by_style[style]
        chosen = None
        for _ in range(min(10, len(candidates))):
            p = candidates[rng.integers(len(candidates))]
            data = read_coarse2(str(p))
            if data is not None and len(data['nodes']) > 0:
                chosen = (p, data)
                break

        ax = axes[i]
        if chosen is None:
            ax.set_title(f'{style}\n(sem grafo .coarse)', fontsize=9)
            ax.axis('off')
            continue

        p, data = chosen
        draw_graph(ax, data['nodes'], data['edges'], style)

    for j in range(len(styles), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    p = out / 'sample_graphs_by_style.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSalvo: {p}')


if __name__ == '__main__':
    main()
