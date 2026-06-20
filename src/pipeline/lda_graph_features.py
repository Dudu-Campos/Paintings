"""
lda_graph_features.py — Features do grafo materializado (ε fixo) + LDA.

Compara o poder discriminativo de estilo entre:
  (a) features agregadas do GRAFO ÚNICO materializado em ε=15
      (poucas features, mas estruturalmente ricas — topologia real)
  (b) as 36 features da CURVA N(ε) já testadas (mais valores, mas
      cada um é só um escalar por threshold)

Features do grafo extraídas a partir de NodeData/EdgeData:
  - n_nodes, n_edges, density (E / (N*(N-1)/2))
  - estatísticas de área dos nós: gini, entropy, mean, std, max_frac
  - estatísticas de color_dist das arestas: mean, std, max
  - estatísticas de spatial_dist das arestas: mean, std
  - grau médio e std (proxy de densidade local)

Uso:
    python lda_graph_features.py \
        --coarse_dir ./coarsening_output_v2/coarse \
        --meta       /exp/duducampos/Paintings/meta/meta.csv \
        --output     ./plots \
        --min_count  30 \
        --n_sample   3000
"""

import argparse
import csv
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_community
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


# ── Dtypes (espelham coarse2.cpp) ─────────────────────────────────────────────

dt_percol = np.dtype([
    ('epsilon', '<f4'), ('n_components', '<u4'), ('giant_size', '<u4'),
    ('giant_frac', '<f4'), ('entropy', '<f4'), ('mean_size', '<f4'),
    ('gini', '<f4'), ('n_singletons_frac', '<f4'),
])

dt_node = np.dtype([
    ('L', '<f4'), ('a', '<f4'), ('b', '<f4'),
    ('X', '<f4'), ('Y', '<f4'),
    ('area_frac', '<f4'), ('area_px', '<u4'),
])

dt_edge = np.dtype([
    ('src', '<u4'), ('dst', '<u4'),
    ('color_dist', '<f4'), ('spatial_dist', '<f4'),
])


# ── Leitura ───────────────────────────────────────────────────────────────────

def read_coarse2(path: str) -> dict | None:
    try:
        with open(path, 'rb') as f:
            mode = struct.unpack('<I', f.read(4))[0]
            n_thresholds = struct.unpack('<I', f.read(4))[0]
            eps_star = struct.unpack('<f', f.read(4))[0] if mode == 1 else None
            f.read(n_thresholds * dt_percol.itemsize)

            nodes = np.zeros(0, dtype=dt_node)
            edges = np.zeros(0, dtype=dt_edge)
            if mode == 1:
                nn = struct.unpack('<I', f.read(4))[0]
                nodes = np.frombuffer(f.read(nn * dt_node.itemsize),
                                       dtype=dt_node).copy()
                ne = struct.unpack('<I', f.read(4))[0]
                edges = np.frombuffer(f.read(ne * dt_edge.itemsize),
                                       dtype=dt_edge).copy()
        if mode != 1 or len(nodes) == 0:
            return None
        return {'eps_star': eps_star, 'nodes': nodes, 'edges': edges}
    except Exception:
        return None


def read_coarse2_full(path: str) -> dict | None:
    """
    Lê o .coarse COMPLETO, incluindo a curva de percolação por threshold
    (steps) com entropy/giant_frac/gini/n_components — diferente de
    read_coarse2, que descarta os steps e só retorna nodes/edges do
    grafo materializado em eps_star.

    Útil para calcular eps_star_entropy (ponto de queda na curva de
    entropia), sem precisar reprocessar as imagens do zero — a curva
    já está salva no arquivo, só não era parseada antes.
    """
    try:
        with open(path, 'rb') as f:
            mode = struct.unpack('<I', f.read(4))[0]
            n_thresholds = struct.unpack('<I', f.read(4))[0]
            eps_star = struct.unpack('<f', f.read(4))[0] if mode == 1 else None
            steps = np.frombuffer(f.read(n_thresholds * dt_percol.itemsize),
                                   dtype=dt_percol).copy()

            nodes = np.zeros(0, dtype=dt_node)
            edges = np.zeros(0, dtype=dt_edge)
            if mode == 1:
                nn = struct.unpack('<I', f.read(4))[0]
                nodes = np.frombuffer(f.read(nn * dt_node.itemsize),
                                       dtype=dt_node).copy()
                ne = struct.unpack('<I', f.read(4))[0]
                edges = np.frombuffer(f.read(ne * dt_edge.itemsize),
                                       dtype=dt_edge).copy()
        return {'mode': mode, 'eps_star': eps_star, 'steps': steps,
                'nodes': nodes, 'edges': edges}
    except Exception:
        return None


# ── Features do grafo ──────────────────────────────────────────────────────────

def gini_coefficient(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_graph_features(nodes: np.ndarray, edges: np.ndarray) -> dict:
    n_nodes = len(nodes)
    n_edges = len(edges)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    density = n_edges / max_edges

    area = nodes['area_frac']
    area_gini = gini_coefficient(area)
    area_entropy = shannon_entropy(area)
    area_max = float(area.max()) if len(area) else 0.0

    color_dist = edges['color_dist'] if n_edges > 0 else np.array([0.0])
    spatial_dist = edges['spatial_dist'] if n_edges > 0 else np.array([0.0])

    if n_edges > 0:
        deg = np.zeros(n_nodes)
        np.add.at(deg, edges['src'], 1)
        np.add.at(deg, edges['dst'], 1)
    else:
        deg = np.zeros(n_nodes)

    area_mean  = float(area.mean()) if len(area) else 0.0
    area_std   = float(area.std())  if len(area) else 0.0
    deg_mean   = float(deg.mean())  if n_nodes else 0.0
    deg_max    = float(deg.max())   if n_nodes else 0.0

    area_sorted = np.sort(area)[::-1]
    area_second = float(area_sorted[1]) if len(area_sorted) > 1 else 0.0

    return {
        'n_nodes':              n_nodes,
        'n_edges':              n_edges,
        'density':              density,
        'area_gini':            area_gini,
        'area_entropy':         area_entropy,
        'area_max_frac':        area_max,
        'area_mean':            area_mean,
        'area_std':             area_std,
        'color_dist_mean':      float(color_dist.mean()),
        'color_dist_std':       float(color_dist.std()),
        'color_dist_max':       float(color_dist.max()),
        'spatial_dist_mean':    float(spatial_dist.mean()),
        'spatial_dist_std':     float(spatial_dist.std()),
        'degree_mean':          deg_mean,
        'degree_std':           float(deg.std()) if n_nodes else 0.0,
        # ── features de forma estrutural ─────────────────────────────────────
        'degree_max':           deg_max,
        'hub_degree_ratio':     deg_max / deg_mean if deg_mean > 0 else 0.0,
        'n_leaf_frac':          float((deg == 1).sum() / n_nodes) if n_nodes else 0.0,
        'degree_gini':          gini_coefficient(deg),
        'area_second_max_frac': area_second,
        'area_cv':              area_std / area_mean if area_mean > 0 else 0.0,
    }


GRAPH_FEATURE_COLS = [
    'n_nodes', 'n_edges', 'density',
    'area_gini', 'area_entropy', 'area_max_frac', 'area_mean', 'area_std',
    'color_dist_mean', 'color_dist_std', 'color_dist_max',
    'spatial_dist_mean', 'spatial_dist_std',
    'degree_mean', 'degree_std',
]

STRUCT_FEATURE_COLS = [
    'degree_max', 'hub_degree_ratio', 'n_leaf_frac',
    'degree_gini', 'area_second_max_frac', 'area_cv',
]

_PER_WINDOW_COLS = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + ['spatial_density_cv', 'spatial_density_max_block_frac']
PYRAMID_GRID     = 2
PYRAMID_FEATURE_COLS = [
    f'q{r}{c}_{feat}'
    for r in range(PYRAMID_GRID)
    for c in range(PYRAMID_GRID)
    for feat in _PER_WINDOW_COLS
]  # 4 × 23 = 92 features


def extract_pyramid_features(nodes: np.ndarray, edges: np.ndarray,
                              grid: int = PYRAMID_GRID) -> dict:
    """
    Divide a imagem em grid×grid janelas pelo centróide de cada nó (X, Y
    normalizados para [0,1]).  Para cada janela extrai os mesmos 23 features
    globais (GRAPH_FEATURE_COLS + STRUCT + SPATIAL), prefixados com 'q{r}{c}_'.
    Arestas só são incluídas se AMBOS os nós estiverem na mesma janela.
    Coordenadas são normalizadas para [0,1] dentro da janela antes do
    spatial_density_cv, evitando CV artificialmente alto por nós concentrados
    numa sub-região do espaço global.
    """
    step   = 1.0 / grid
    result = {}
    n_full = len(nodes)
    zero_window = {f: 0.0 for f in _PER_WINDOW_COLS}

    for r in range(grid):
        for c in range(grid):
            prefix = f'q{r}{c}_'
            if n_full == 0:
                result.update({prefix + k: v for k, v in zero_window.items()})
                continue

            x_min, x_max = c * step, (c + 1) * step
            y_min, y_max = r * step, (r + 1) * step

            mask = ((nodes['X'] >= x_min) & (nodes['X'] < x_max) &
                    (nodes['Y'] >= y_min) & (nodes['Y'] < y_max))
            w_nodes = nodes[mask]

            if len(w_nodes) == 0:
                result.update({prefix + k: v for k, v in zero_window.items()})
                continue

            # Remap edge indices para o subconjunto local
            old_to_new = np.full(n_full, -1, dtype=np.int64)
            old_to_new[np.where(mask)[0]] = np.arange(len(w_nodes))

            if len(edges) > 0:
                s_new = old_to_new[edges['src']]
                d_new = old_to_new[edges['dst']]
                keep  = (s_new >= 0) & (d_new >= 0)
                w_edges       = edges[keep].copy()
                w_edges['src'] = s_new[keep]
                w_edges['dst'] = d_new[keep]
            else:
                w_edges = edges

            feats = extract_graph_features(w_nodes, w_edges)

            # Normaliza coords para [0,1] dentro da janela antes do spatial CV
            w_norm = w_nodes.copy()
            w_norm['X'] = (w_nodes['X'] - x_min) / step
            w_norm['Y'] = (w_nodes['Y'] - y_min) / step
            feats.update(spatial_density_cv(w_norm))

            result.update({prefix + k: v for k, v in feats.items()})

    return result


# ── Remoção do componente gigante e recálculo completo ───────────────────────

def _union_find_components(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Retorna o id de componente (root) de cada nó, via union-find."""
    parent = np.arange(n_nodes)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(int(e['src']), int(e['dst']))

    return np.array([find(i) for i in range(n_nodes)])


def remove_giant_component(nodes: np.ndarray, edges: np.ndarray,
                            mode: str = 'dominant_node'
                            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Remove o nó/estrutura dominante do grafo, recalculando tudo sobre
    o que resta.

    mode='dominant_node' (default): remove apenas o ÚNICO nó com maior
    area_frac (o "hub" — tipicamente o fundo homogêneo). Os demais nós
    mantêm suas arestas entre si, preservando a estrutura de
    conectividade real do que resta. Esse é o modo correto quando o
    grafo RAG forma um único componente conexo (caso comum/universal
    no dataset real) — remover o COMPONENTE inteiro nesse caso apagaria
    o grafo todo, o que é degenerado e inútil.

    mode='component': remove o maior COMPONENTE CONEXO inteiro (pode
    remover múltiplos nós se vários estiverem fundidos no hub). Só use
    se o grafo tiver múltiplos componentes de tamanho razoável — caso
    contrário todos os nós acabam sendo removidos.

    Retorna (nodes_sem_hub, edges_remapeadas, info).
    """
    n_nodes = len(nodes)
    if n_nodes == 0:
        return nodes, edges, {'giant_size': 0, 'giant_frac': 0.0, 'n_remaining': 0}

    if mode == 'dominant_node':
        giant_idx = int(np.argmax(nodes['area_frac']))
        giant_size = 1
        keep_idx = np.array(
            [i for i in range(n_nodes) if i != giant_idx], dtype=np.int64)
    else:  # mode == 'component'
        roots = _union_find_components(n_nodes, edges)
        unique_roots, sizes = np.unique(roots, return_counts=True)
        giant_root = unique_roots[np.argmax(sizes)]
        giant_size = int(sizes.max())
        keep_idx = np.where(roots != giant_root)[0].astype(np.int64)

    # Remapear índices antigos -> novos (só para os nós mantidos)
    old_to_new = -np.ones(n_nodes, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx))

    nodes_new = nodes[keep_idx]

    if len(edges) > 0:
        src_new = old_to_new[edges['src']]
        dst_new = old_to_new[edges['dst']]
        edge_keep_mask = (src_new >= 0) & (dst_new >= 0)
        edges_new = edges[edge_keep_mask].copy()
        edges_new['src'] = src_new[edge_keep_mask]
        edges_new['dst'] = dst_new[edge_keep_mask]
    else:
        edges_new = edges

    info = {
        'giant_size':   giant_size,
        'giant_frac':   float(giant_size / n_nodes),
        'n_remaining':  int(len(keep_idx)),
    }
    return nodes_new, edges_new, info


def extract_graph_features_no_giant(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """
    Remove o componente gigante e recalcula TODAS as features básicas
    (extract_graph_features) sobre o subgrafo resultante — não apenas
    as novas features de componentes, mas as 15 originais também,
    já que todas (médias, desvios, gini) eram contaminadas pela escala
    do hub.
    """
    nodes_ng, edges_ng, info = remove_giant_component(nodes, edges)
    feats = extract_graph_features(nodes_ng, edges_ng)
    feats['giant_removed_frac'] = info['giant_frac']  # quanto foi removido
    return feats


GRAPH_FEATURE_COLS_NO_GIANT = GRAPH_FEATURE_COLS + ['giant_removed_frac']


# ── Feature de distribuição espacial — densidade por quadrante ──────────────

def spatial_density_cv(nodes: np.ndarray, grid_size: int = 3) -> dict:
    """
    Divide a imagem (coordenadas X,Y em [0,1]) em uma grade grid_size x
    grid_size, conta quantos nós cada bloco recebe, e retorna o
    coeficiente de variação (CV = std/mean) dessa contagem entre blocos.

    Captura algo que estatísticas agregadas globais (n_nodes, density,
    etc) não capturam: O QUANTO a complexidade estrutural está
    concentrada em certas regiões da pintura vs espalhada uniformemente.

    CV baixo  -> densidade de nós uniforme pelo espaço (ex: pointillism
                 com textura repetitiva por toda a tela)
    CV alto   -> densidade muito desigual (ex: um canto detalhado,
                 resto quase vazio — comum em retratos com fundo liso)
    """
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {'spatial_density_cv': 0.0, 'spatial_density_max_block_frac': 0.0}

    X = np.clip(nodes['X'], 0, 0.999999)
    Y = np.clip(nodes['Y'], 0, 0.999999)

    bin_x = (X * grid_size).astype(int)
    bin_y = (Y * grid_size).astype(int)
    block_idx = bin_y * grid_size + bin_x

    counts = np.bincount(block_idx, minlength=grid_size*grid_size).astype(float)
    mean_c = counts.mean()
    std_c  = counts.std()

    cv = float(std_c / mean_c) if mean_c > 0 else 0.0
    max_block_frac = float(counts.max() / n_nodes) if n_nodes > 0 else 0.0

    return {
        'spatial_density_cv':           cv,
        'spatial_density_max_block_frac': max_block_frac,
    }


SPATIAL_FEATURE_COLS = ['spatial_density_cv', 'spatial_density_max_block_frac']


# ── Features de cor dos nós (Lab ponderado por area_frac) ────────────────────

COLOR_FEATURE_COLS = [
    'L_mean_w', 'L_std_w',
    'a_mean_w', 'a_std_w',
    'b_mean_w', 'b_std_w',
    'chroma_mean_w', 'chroma_std_w',
    'warm_frac',
]


def extract_color_features(nodes: np.ndarray) -> dict:
    """
    Estatísticas de cor Lab ponderadas pela área de cada região (area_frac).
    Regiões maiores contribuem proporcionalmente mais para a impressão visual
    da paleta — a média simples superestimaria regiões minúsculas.

    warm_frac: fração da área coberta por regiões com b > 0 (amarelo/quente
    domina sobre azul/frio no eixo b de Lab).
    chroma = sqrt(a²+b²) — saturação total, independente de hue.
    """
    zero = {k: 0.0 for k in COLOR_FEATURE_COLS}
    if len(nodes) == 0:
        return zero

    w = nodes['area_frac'].astype(np.float64)
    w_sum = w.sum()
    if w_sum <= 0:
        return zero
    w = w / w_sum  # pesos normalizados

    L = nodes['L'].astype(np.float64)
    a = nodes['a'].astype(np.float64)
    b = nodes['b'].astype(np.float64)
    C = np.sqrt(a**2 + b**2)

    def wmean(x):
        return float(np.sum(w * x))

    def wstd(x):
        mu = np.sum(w * x)
        return float(np.sqrt(np.sum(w * (x - mu)**2)))

    return {
        'L_mean_w':      wmean(L),
        'L_std_w':       wstd(L),
        'a_mean_w':      wmean(a),
        'a_std_w':       wstd(a),
        'b_mean_w':      wmean(b),
        'b_std_w':       wstd(b),
        'chroma_mean_w': wmean(C),
        'chroma_std_w':  wstd(C),
        'warm_frac':     float(np.sum(w[b > 0])),
    }


# ── Componentes conexos do grafo RAG materializado ───────────────────────────

def connected_components_stats(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """
    Analisa a estrutura de componentes conexos do grafo RAG materializado
    (não a curva de percolação por threshold — o grafo final em ε=15).

    Testa a hipótese: o grafo é dominado por um único componente gigante
    (ex: um "fundo" homogêneo conectando quase tudo), com o resto sendo
    fragmentos residuais — nesse caso, features agregadas sobre o grafo
    inteiro mascaram a estrutura real, que está "fora" do gigante.
    """
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {
            'n_components': 0, 'giant_frac': 0.0,
            'giant_to_second_ratio': 0.0, 'n_components_size_gt1': 0,
        }

    # Union-Find simples para achar componentes conexos
    parent = np.arange(n_nodes)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(int(e['src']), int(e['dst']))

    roots = np.array([find(i) for i in range(n_nodes)])
    _, sizes = np.unique(roots, return_counts=True)
    sizes_sorted = np.sort(sizes)[::-1]

    giant_frac = float(sizes_sorted[0] / n_nodes)
    second = float(sizes_sorted[1]) if len(sizes_sorted) > 1 else 0.0
    giant_to_second_ratio = float(sizes_sorted[0] / second) if second > 0 else float(n_nodes)

    return {
        'n_components':          int(len(sizes_sorted)),
        'giant_frac':            giant_frac,
        'giant_to_second_ratio': giant_to_second_ratio,
        'n_components_size_gt1': int((sizes_sorted > 1).sum()),
    }


COMPONENT_FEATURE_COLS = [
    'n_components', 'giant_frac', 'giant_to_second_ratio', 'n_components_size_gt1',
]

# ── Features topológicas avançadas (via networkx) ────────────────────────────

TOPO_FEATURE_COLS = [
    'modularity', 'n_communities',
    'clustering_coef_mean', 'clustering_coef_std',
    'assortativity_degree',
    'avg_path_length_approx', 'diameter_approx',
    'spectral_gap',
]

MAX_NODES_FOR_PATH      = 5_000  # limite para path_length/diameter aproximados
MAX_NODES_FOR_SPECTRAL  = 5_000  # limite para spectral gap (via eigsh esparso)
MAX_NODES_FOR_LOUVAIN   = 30_000 # Louvain sem limite trava em grafos grandes
MAX_NODES_FOR_CLUSTERING = 30_000 # clustering via nx também sem limite
PATH_SAMPLE_SIZE        = 100    # amostragem de pares para aproximar path length


def build_nx_graph(nodes: np.ndarray, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    if len(edges) > 0:
        # Peso = inverso da distância cromática (mais similar = aresta "mais forte")
        weighted = [(int(e['src']), int(e['dst']),
                     {'weight': 1.0 / (1.0 + float(e['color_dist']))})
                    for e in edges]
        G.add_edges_from(weighted)
    return G


def extract_topo_features(G: nx.Graph) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    feats = {k: 0.0 for k in TOPO_FEATURE_COLS}

    if n_nodes < 3 or n_edges < 2:
        return feats

    # ── Modularidade via Louvain ──────────────────────────────────────────────
    if n_nodes <= MAX_NODES_FOR_LOUVAIN:
        try:
            communities = nx_community.louvain_communities(
                G, weight='weight', seed=42)
            feats['modularity']    = float(nx_community.modularity(
                G, communities, weight='weight'))
            feats['n_communities'] = len(communities)
        except Exception:
            pass

    # ── Clustering coefficient ────────────────────────────────────────────────
    if n_nodes <= MAX_NODES_FOR_CLUSTERING:
        try:
            cc = nx.clustering(G, weight='weight')
            cc_vals = np.array(list(cc.values()))
            feats['clustering_coef_mean'] = float(cc_vals.mean())
            feats['clustering_coef_std']  = float(cc_vals.std())
        except Exception:
            pass

    # ── Assortatividade de grau ──────────────────────────────────────────────
    try:
        if n_edges > 1:
            feats['assortativity_degree'] = float(
                nx.degree_assortativity_coefficient(G))
    except Exception:
        pass

    # ── Path length / diâmetro aproximados (amostragem para grafos grandes) ──
    try:
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc)

        if H.number_of_nodes() <= MAX_NODES_FOR_PATH:
            nodes_list = list(H.nodes())
            sample_n = min(PATH_SAMPLE_SIZE, len(nodes_list))
            rng = np.random.default_rng(42)
            sample_nodes = rng.choice(nodes_list, sample_n, replace=False)

            lengths = []
            ecc = []
            for s in sample_nodes:
                sp = nx.single_source_shortest_path_length(H, s)
                lengths.extend(sp.values())
                ecc.append(max(sp.values()))

            feats['avg_path_length_approx'] = float(np.mean(lengths))
            feats['diameter_approx']        = float(np.max(ecc))
    except Exception:
        pass

    # ── Spectral gap (Laplaciano normalizado, via eigsh esparso) ────────────
    try:
        if 5 <= n_nodes <= MAX_NODES_FOR_SPECTRAL:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            L = nx.normalized_laplacian_matrix(G, weight='weight')
            L = csr_matrix(L, dtype=np.float64)
            # eigsh esparso pega só os k menores autovalores — muito mais
            # rápido que decompor a matriz densa inteira (que é O(n^3))
            k = min(4, n_nodes - 2)
            if k >= 3:
                eigvals = eigsh(L, k=k, which='SM',
                                 return_eigenvectors=False)
                eigvals_sorted = np.sort(eigvals)
                if len(eigvals_sorted) > 2:
                    feats['spectral_gap'] = float(
                        eigvals_sorted[2] - eigvals_sorted[1])
    except Exception:
        pass

    return feats


# ── Meta ──────────────────────────────────────────────────────────────────────

def load_meta(meta_path: str) -> dict:
    lookup = {}
    if not meta_path:
        return lookup
    try:
        with open(meta_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row.get('filename', '')).stem
                lookup[stem] = row.get('style', 'unknown').strip()
    except FileNotFoundError:
        print(f'AVISO: meta não encontrado em {meta_path}')
    return lookup


def select_balanced_paths(paths: list[Path], meta: dict,
                           min_per_class: int = 31,
                           max_per_class: int = 100,
                           seed: int = 42) -> list[Path]:
    """
    Seleciona arquivos de forma balanceada por estilo, replicando o
    protocolo usado na avaliação da CNN (min=31, max=100 por classe).

    Classes com menos que min_per_class amostras disponíveis são
    descartadas inteiramente (não dá para "balancear para cima").
    Classes com mais que max_per_class são sub-amostradas aleatoriamente.
    """
    rng = np.random.default_rng(seed)

    by_style: dict[str, list[Path]] = {}
    for p in paths:
        style = meta.get(p.stem, 'unknown')
        if style == 'unknown':
            continue
        by_style.setdefault(style, []).append(p)

    selected = []
    n_dropped_classes = 0
    for style, style_paths in by_style.items():
        if len(style_paths) < min_per_class:
            n_dropped_classes += 1
            continue
        if len(style_paths) > max_per_class:
            idx = rng.choice(len(style_paths), max_per_class, replace=False)
            style_paths = [style_paths[i] for i in idx]
        selected.extend(style_paths)

    n_classes_kept = len(by_style) - n_dropped_classes
    print(f'Seleção balanceada: {n_classes_kept} estilos mantidos '
          f'(>= {min_per_class} amostras), {n_dropped_classes} descartados '
          f'(< {min_per_class}), {len(selected):,} arquivos selecionados')
    return selected


# ── Coleta do dataset de features de grafo ────────────────────────────────────

def build_graph_feature_dataset(coarse_dir: str, meta_path: str,
                                 n_sample: int = 0,
                                 seed: int = 42,
                                 include_topo: bool = False,
                                 pyramid: bool = False,
                                 balanced_min: int = 0,
                                 balanced_max: int = 0,
                                 remove_giant: bool = False,
                                 max_nodes: int = 100_000) -> pd.DataFrame:
    """
    balanced_min/balanced_max > 0 ativa seleção balanceada por estilo
    (replicando o protocolo da CNN: min=31, max=100), aplicada ANTES de
    qualquer amostragem aleatória por n_sample e antes da extração de
    features — garante que todas as classes elegíveis estejam
    representadas de forma justa, em vez de depender de quem caiu na
    amostra aleatória.

    remove_giant=True remove o maior componente conexo (tipicamente o
    "fundo" homogêneo que domina a maioria das pinturas) ANTES de
    calcular qualquer feature — todas as 15 features básicas são
    recalculadas sobre o subgrafo resultante, não só as estatísticas
    de componentes. Isso evita que a escala do hub domine médias e
    desvios das features agregadas.
    """
    meta = load_meta(meta_path)
    paths = list(Path(coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse encontrados: {len(paths):,}')

    if balanced_min > 0 and balanced_max > 0:
        paths = select_balanced_paths(paths, meta, balanced_min, balanced_max, seed)
    elif n_sample > 0 and len(paths) > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), n_sample, replace=False)
        paths = [paths[i] for i in idx]

    rows = []
    failed = 0
    for i, p in enumerate(paths):
        data = read_coarse2(str(p))
        if data is None:
            failed += 1
            continue

        if max_nodes > 0 and len(data['nodes']) > max_nodes:
            failed += 1
            continue

        if remove_giant:
            nodes_ng, edges_ng, giant_info = remove_giant_component(
                data['nodes'], data['edges'])
            feats = extract_graph_features(nodes_ng, edges_ng)
            feats['giant_removed_frac'] = giant_info['giant_frac']
            feats.update(spatial_density_cv(nodes_ng))
            feats.update(extract_color_features(nodes_ng))
            if include_topo:
                G = build_nx_graph(nodes_ng, edges_ng)
                feats.update(extract_topo_features(G))
            if pyramid:
                feats.update(extract_pyramid_features(nodes_ng, edges_ng))
        else:
            feats = extract_graph_features(data['nodes'], data['edges'])
            feats.update(spatial_density_cv(data['nodes']))
            feats.update(extract_color_features(data['nodes']))
            if include_topo:
                G = build_nx_graph(data['nodes'], data['edges'])
                feats.update(extract_topo_features(G))
            if pyramid:
                feats.update(extract_pyramid_features(data['nodes'], data['edges']))

        feats['stem'] = p.stem
        feats['style'] = meta.get(p.stem, 'unknown')
        rows.append(feats)
        if (i+1) % 500 == 0:
            print(f'  {i+1:>7}/{len(paths):,}...', end='\r')

    print(f'\nProcessados: {len(rows):,}  falhas: {failed}')
    return pd.DataFrame(rows)


# ── LDA ───────────────────────────────────────────────────────────────────────

def run_lda(X: np.ndarray, y, output: str,
            label: str, title_extra: str = '') -> tuple[float, float]:
    y = np.array(y, dtype=str)  # garantir numpy array puro
    styles   = np.unique(y)
    n_styles = len(styles)
    n_comps  = min(n_styles - 1, X.shape[1])

    uniform_priors = np.ones(n_styles) / n_styles
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(n_components=n_comps,
                                              priors=uniform_priors)),
    ])
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    print(f'[{label}] BAcc CV (5-fold): {scores.mean():.3f} ± {scores.std():.3f}')

    pipe.fit(X, y)
    X_lda = pipe.named_steps['lda'].transform(
        pipe.named_steps['scaler'].transform(X))
    var = pipe.named_steps['lda'].explained_variance_ratio_

    cmap   = cm.tab20 if n_styles <= 20 else cm.gist_ncar
    colors = {s: cmap(i/n_styles) for i, s in enumerate(styles)}

    fig, ax = plt.subplots(figsize=(11, 7))
    for style in styles:
        mask = y == style
        ax.scatter(X_lda[mask,0],
                   X_lda[mask,1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                   c=[colors[style]], alpha=0.3, s=8, linewidths=0)
    for style in styles:
        mask = y == style
        cx = X_lda[mask,0].mean()
        cy = X_lda[mask,1].mean() if X_lda.shape[1] > 1 else 0
        ax.scatter(cx, cy, c=[colors[style]], s=70,
                   edgecolors='white', linewidth=1.3, zorder=6)
        ax.annotate(style, (cx,cy), fontsize=6,
                    xytext=(3,3), textcoords='offset points',
                    color=colors[style], fontweight='bold')

    ax.set_xlabel(f'LD1 ({var[0]*100:.1f}%)', fontsize=11)
    if len(var) > 1:
        ax.set_ylabel(f'LD2 ({var[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'{title_extra}\nAcurácia CV: {scores.mean():.3f} ± '
                  f'{scores.std():.3f}  ({n_styles} estilos)', fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    p = Path(output) / f'lda_{label}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')

    return float(scores.mean()), float(scores.std())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--meta',       default='')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--min_count',  type=int, default=30)
    parser.add_argument('--n_sample',   type=int, default=3000)
    parser.add_argument('--balanced_min', type=int, default=0,
                        help='Se >0 (junto com balanced_max), seleciona '
                             'classes balanceadas — min amostras por '
                             'classe para ser incluída, replicando '
                             'protocolo da CNN (min=31, max=100)')
    parser.add_argument('--balanced_max', type=int, default=0,
                        help='Máximo de amostras por classe (sub-amostra '
                             'classes maiores) — usar junto com balanced_min')
    parser.add_argument('--acc_curve',  type=float, default=0.150,
                        help='Acurácia já obtida com as 36 features da curva N(ε)')
    parser.add_argument('--include_topo', action='store_true',
                        help='Inclui features topológicas (modularidade, '
                             'clustering, assortatividade, path length, '
                             'spectral gap) — mais caro computacionalmente')
    parser.add_argument('--remove_giant', action='store_true',
                        help='Remove o maior componente conexo (tipicamente '
                             'o "fundo" homogêneo) ANTES de calcular '
                             'qualquer feature — todas as features são '
                             'recalculadas sobre o subgrafo resultante, '
                             'evitando viés de escala do hub dominante')
    parser.add_argument('--max_nodes', type=int, default=100_000,
                        help='Descarta imagens cujo grafo materializado '
                             'tenha mais que este número de nós — grafos '
                             'muito grandes indicam eps* muito baixo '
                             '(coarsening insuficiente) e contaminam as '
                             'features com sinal espúrio de complexidade. '
                             '0 desativa o filtro.')
    parser.add_argument('--pyramid', action='store_true',
                        help='Extrai features em pirâmide espacial 2×2: '
                             'divide a imagem em 4 janelas por centróide '
                             'de nó e computa as 23 features por janela '
                             '(4×23=92 features). Salvo em '
                             'graph_features_pyramid.parquet.')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = build_graph_feature_dataset(args.coarse_dir, args.meta, args.n_sample,
                                      include_topo=args.include_topo,
                                      pyramid=args.pyramid,
                                      balanced_min=args.balanced_min,
                                      balanced_max=args.balanced_max,
                                      remove_giant=args.remove_giant,
                                      max_nodes=args.max_nodes)
    df = df[df['style'] != 'unknown'].copy()

    # Se já passou pela seleção balanceada, min_count é redundante mas
    # inofensivo (todas as classes já satisfazem balanced_min)
    counts = df['style'].value_counts()
    valid  = counts[counts >= args.min_count].index
    df     = df[df['style'].isin(valid)]
    print(f'\nApós filtro min_count={args.min_count}: '
          f'{len(df):,} imagens  {df["style"].nunique()} estilos')

    parquet_name = 'graph_features_pyramid.parquet' if args.pyramid else 'graph_features.parquet'
    df.to_parquet(out / parquet_name, index=False)
    y = df['style'].values

    def _prep(cols):
        X = df[cols].values.astype(float)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = np.nanmedian(X[:, j])
        return X

    # ── LDA com features básicas (15) ────────────────────────────────────────
    X_basic = _prep(GRAPH_FEATURE_COLS)
    acc_basic, std_basic = run_lda(
        X_basic, y, str(out), label='graph_basic',
        title_extra=f'LDA — {len(GRAPH_FEATURE_COLS)} features básicas do grafo')

    # ── LDA básicas + estruturais + cor (15+8+9=32) ──────────────────────────
    all_struct_cols = GRAPH_FEATURE_COLS + STRUCT_FEATURE_COLS + SPATIAL_FEATURE_COLS
    all_full_cols   = all_struct_cols + COLOR_FEATURE_COLS
    X_struct = _prep(all_struct_cols)
    X_full   = _prep(all_full_cols)
    acc_struct, std_struct = run_lda(
        X_struct, y, str(out), label='graph_struct',
        title_extra=f'LDA — {len(all_struct_cols)} features (básicas+estruturais)')
    acc_full, std_full = run_lda(
        X_full, y, str(out), label='graph_full',
        title_extra=f'LDA — {len(all_full_cols)} features (básicas+estruturais+cor)')

    print(f'\n── Comparação ablação ────────────────────────────────────────────')
    print(f'  Básicas      ({len(GRAPH_FEATURE_COLS):>2} feat): BAcc {acc_basic:.4f} ± {std_basic:.4f}')
    print(f'  + Estrutural ({len(all_struct_cols):>2} feat): BAcc {acc_struct:.4f} ± {std_struct:.4f}  Δ={acc_struct-acc_basic:+.4f}')
    print(f'  + Cor        ({len(all_full_cols):>2} feat): BAcc {acc_full:.4f} ± {std_full:.4f}  Δ={acc_full-acc_struct:+.4f}')
    print(f'──────────────────────────────────────────────────────────────────')

    results = {
        'Curva N(ε) — 36 features (12 thresholds)': args.acc_curve,
        f'Grafo básico ε — {len(GRAPH_FEATURE_COLS)} features': acc_basic,
        f'Grafo + estrutural — {len(all_struct_cols)} features': acc_struct,
        f'Grafo + estrutural + cor — {len(all_full_cols)} features': acc_full,
    }

    # ── LDA com features básicas + topológicas ──────────────────────────────
    if args.include_topo:
        all_cols = GRAPH_FEATURE_COLS + TOPO_FEATURE_COLS
        X_full = df[all_cols].values.astype(float)
        for j in range(X_full.shape[1]):
            mask = np.isnan(X_full[:, j])
            if mask.any():
                X_full[mask, j] = np.nanmedian(X_full[:, j])

        acc_full, std_full = run_lda(
            X_full, y, str(out), label='graph_full_topo',
            title_extra=f'LDA — {len(all_cols)} features '
                         f'(básicas + topológicas)')

        results[f'Grafo básico+topológico — {len(all_cols)} features'] = acc_full

        # Também testar SÓ as features topológicas, isoladas
        X_topo = df[TOPO_FEATURE_COLS].values.astype(float)
        for j in range(X_topo.shape[1]):
            mask = np.isnan(X_topo[:, j])
            if mask.any():
                X_topo[mask, j] = np.nanmedian(X_topo[:, j])

        acc_topo, std_topo = run_lda(
            X_topo, y, str(out), label='graph_topo_only',
            title_extra=f'LDA — {len(TOPO_FEATURE_COLS)} features '
                         f'topológicas isoladas')
        results[f'Só topológicas — {len(TOPO_FEATURE_COLS)} features'] = acc_topo

    print('\n' + '='*65)
    print(f'{"Método":<48} {"Acurácia":>12}')
    print('-'*65)
    for name, acc in results.items():
        print(f'{name:<48} {acc:>12.3f}')
    print('='*65)


if __name__ == '__main__':
    main()
