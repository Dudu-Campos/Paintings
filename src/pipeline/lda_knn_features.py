"""
lda_knn_features.py — Grafo k-NN sobre os mesmos nós do RAG.

Em vez de adjacência espacial direta (RAG, fronteira física entre
componentes), conecta cada nó aos k=10 vizinhos mais próximos no
espaço de atributos (cor Lab + posição XY normalizada).

Isso captura uma relação estrutural diferente e potencialmente
complementar ao RAG: similaridade de cor+posição independente de
adjacência física — por exemplo, duas regiões de céu azul em lados
opostos da pintura, que o RAG nunca conectaria (não são adjacentes),
mas que são extremamente similares em atributo.

Reaproveita read_coarse2, extract_graph_features, extract_topo_features,
build_nx_graph de lda_graph_features.py — mesma extração de features
básicas+topológicas, só trocando a fonte das arestas.

Uso:
    python lda_knn_features.py \
        --coarse_dir ./coarsening_output_v2/coarse \
        --meta       /exp/duducampos/Paintings/meta/meta.csv \
        --output     ./plots \
        --min_count  30 \
        --n_sample   3000 \
        --k          10 \
        --acc_rag_basic 0.180 \
        --acc_rag_topo  0.172
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from lda_graph_features import (
    read_coarse2, load_meta, select_balanced_paths,
    extract_graph_features, extract_topo_features,
    GRAPH_FEATURE_COLS, TOPO_FEATURE_COLS,
    run_lda,
)


# ── Construção do grafo k-NN sobre os nós existentes ──────────────────────────

def build_knn_edges(nodes: np.ndarray, k: int = 10,
                     color_weight: float = 1.0,
                     spatial_weight: float = 1.0) -> np.ndarray:
    """
    Constrói arestas k-NN no espaço (L, a, b, X, Y) ponderado.

    Retorna array estruturado igual a dt_edge (src, dst, color_dist,
    spatial_dist), para reaproveitar extract_graph_features/
    extract_topo_features sem modificação.
    """
    n = len(nodes)
    if n < 2:
        return np.zeros(0, dtype=[('src','<u4'),('dst','<u4'),
                                   ('color_dist','<f4'),('spatial_dist','<f4')])

    k_eff = min(k, n - 1)

    color_feats   = np.column_stack([nodes['L'], nodes['a'], nodes['b']])
    spatial_feats = np.column_stack([nodes['X'], nodes['Y']])

    color_std   = color_feats.std(axis=0)
    color_std[color_std == 0] = 1.0
    spatial_std = spatial_feats.std(axis=0)
    spatial_std[spatial_std == 0] = 1.0

    combined = np.column_stack([
        color_weight   * color_feats   / color_std,
        spatial_weight * spatial_feats / spatial_std,
    ])

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(combined)
    dist, idx = nbrs.kneighbors(combined)

    src_list, dst_list, cdist_list, sdist_list = [], [], [], []
    seen = set()
    for i in range(n):
        for j_pos in range(1, k_eff + 1):
            j = idx[i, j_pos]
            pair = (min(i,j), max(i,j))
            if pair in seen:
                continue
            seen.add(pair)
            src_list.append(pair[0])
            dst_list.append(pair[1])
            cdist_list.append(float(np.linalg.norm(
                color_feats[pair[0]] - color_feats[pair[1]])))
            sdist_list.append(float(np.linalg.norm(
                spatial_feats[pair[0]] - spatial_feats[pair[1]])))

    edges = np.zeros(len(src_list),
                      dtype=[('src','<u4'),('dst','<u4'),
                             ('color_dist','<f4'),('spatial_dist','<f4')])
    edges['src']          = src_list
    edges['dst']          = dst_list
    edges['color_dist']   = cdist_list
    edges['spatial_dist'] = sdist_list
    return edges


# ── Coleta do dataset com grafo k-NN ──────────────────────────────────────────

def build_knn_feature_dataset(coarse_dir: str, meta_path: str,
                               n_sample: int = 3000, k: int = 10,
                               seed: int = 42,
                               include_topo: bool = True,
                               n_workers: int = 1,
                               balanced_min: int = 0,
                               balanced_max: int = 0,
                               max_nodes: int = 100_000) -> pd.DataFrame:
    meta = load_meta(meta_path)
    paths = list(Path(coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse encontrados: {len(paths):,}')

    if balanced_min > 0 and balanced_max > 0:
        paths = select_balanced_paths(paths, meta, balanced_min, balanced_max, seed)
    elif n_sample > 0 and len(paths) > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), n_sample, replace=False)
        paths = [paths[i] for i in idx]

    if n_workers <= 1:
        rows = _process_paths_sequential(paths, meta, k, include_topo, max_nodes)
    else:
        rows = _process_paths_parallel(paths, meta, k, include_topo, n_workers, max_nodes)

    print(f'\nProcessados: {len(rows):,}  falhas: {len(paths)-len(rows)}')
    return pd.DataFrame(rows)


def _process_one_knn(args) -> dict | None:
    path, meta_style, k, include_topo, max_nodes = args
    data = read_coarse2(str(path))
    if data is None:
        return None

    nodes = data['nodes']
    if max_nodes > 0 and len(nodes) > max_nodes:
        return None
    knn_edges = build_knn_edges(nodes, k=k)
    feats = extract_graph_features(nodes, knn_edges)

    if include_topo:
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        if len(knn_edges) > 0:
            weighted = [(int(e['src']), int(e['dst']),
                         {'weight': 1.0/(1.0+float(e['color_dist']))})
                        for e in knn_edges]
            G.add_edges_from(weighted)
        topo_feats = extract_topo_features(G)
        feats.update(topo_feats)

    feats['stem']  = Path(path).stem
    feats['style'] = meta_style
    return feats


def _process_paths_sequential(paths, meta, k, include_topo, max_nodes) -> list:
    rows = []
    for i, p in enumerate(paths):
        r = _process_one_knn((p, meta.get(p.stem, 'unknown'), k, include_topo, max_nodes))
        if r is not None:
            rows.append(r)
        if (i+1) % 200 == 0:
            print(f'  {i+1:>7}/{len(paths):,}...', end='\r')
    return rows


def _process_paths_parallel(paths, meta, k, include_topo, n_workers, max_nodes) -> list:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    args = [(p, meta.get(p.stem, 'unknown'), k, include_topo, max_nodes) for p in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_one_knn, a): a for a in args}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
            if i % 200 == 0 or i == len(paths):
                print(f'  [{i:>7}/{len(paths):,}]  ok={len(rows):,}',
                      end='\r', flush=True)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--meta',       default='')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--min_count',  type=int, default=30)
    parser.add_argument('--n_sample',   type=int, default=3000)
    parser.add_argument('--balanced_min', type=int, default=0,
                        help='Seleção balanceada por classe (replica '
                             'protocolo CNN: min=31, max=100)')
    parser.add_argument('--balanced_max', type=int, default=0)
    parser.add_argument('--k',          type=int, default=10)
    parser.add_argument('--workers',    type=int, default=1,
                        help='Processos paralelos — recomendado para '
                             'datasets grandes, já que Louvain/clustering '
                             'em grafos k-NN densos é mais caro que no RAG')
    parser.add_argument('--acc_curve',      type=float, default=0.150)
    parser.add_argument('--acc_rag_basic',  type=float, default=0.180)
    parser.add_argument('--acc_rag_topo',   type=float, default=0.172)
    parser.add_argument('--include_topo', action='store_true', default=False,
                        help='Inclui features topológicas (mais caro). '
                             'Por default só extrai features básicas '
                             '(n_nodes, n_edges, density, área, distâncias).')
    parser.add_argument('--max_nodes', type=int, default=100_000,
                        help='Descarta imagens com mais que este número de '
                             'nós no grafo materializado (grafos degenerados '
                             'por eps* muito baixo). 0 desativa o filtro.')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = build_knn_feature_dataset(args.coarse_dir, args.meta,
                                     args.n_sample, args.k,
                                     include_topo=args.include_topo,
                                     n_workers=args.workers,
                                     balanced_min=args.balanced_min,
                                     balanced_max=args.balanced_max,
                                     max_nodes=args.max_nodes)
    df = df[df['style'] != 'unknown'].copy()

    counts = df['style'].value_counts()
    valid  = counts[counts >= args.min_count].index
    df     = df[df['style'].isin(valid)]
    print(f'\nApós filtro min_count={args.min_count}: '
          f'{len(df):,} imagens  {df["style"].nunique()} estilos')

    df.to_parquet(out / 'knn_features.parquet', index=False)
    y = df['style'].values

    X_basic = df[GRAPH_FEATURE_COLS].values.astype(float)
    for j in range(X_basic.shape[1]):
        mask = np.isnan(X_basic[:, j])
        if mask.any():
            X_basic[mask, j] = np.nanmedian(X_basic[:, j])

    acc_knn_basic, _ = run_lda(
        X_basic, y, str(out), label=f'knn_k{args.k}_basic',
        title_extra=f'LDA — k-NN (k={args.k}) básico, '
                     f'{len(GRAPH_FEATURE_COLS)} features')

    results = {
        'Curva N(ε) — 36 features':              args.acc_curve,
        'RAG básico — 15 features':               args.acc_rag_basic,
    }
    if args.acc_rag_topo > 0:
        results['RAG + topológicas — 23 features'] = args.acc_rag_topo
    results[f'k-NN (k={args.k}) básico — '
            f'{len(GRAPH_FEATURE_COLS)} features'] = acc_knn_basic

    if args.include_topo:
        all_cols = GRAPH_FEATURE_COLS + TOPO_FEATURE_COLS
        X_full = df[all_cols].values.astype(float)
        for j in range(X_full.shape[1]):
            mask = np.isnan(X_full[:, j])
            if mask.any():
                X_full[mask, j] = np.nanmedian(X_full[:, j])

        acc_knn_full, _ = run_lda(
            X_full, y, str(out), label=f'knn_k{args.k}_full',
            title_extra=f'LDA — k-NN (k={args.k}) + topológicas, '
                         f'{len(all_cols)} features')
        results[f'k-NN (k={args.k}) + topológicas — '
                f'{len(all_cols)} features'] = acc_knn_full

    print('\n' + '='*70)
    print(f'{"Método":<52} {"Acurácia":>12}')
    print('-'*70)
    for name, acc in results.items():
        print(f'{name:<52} {acc:>12.3f}')
    print('='*70)


if __name__ == '__main__':
    main()
