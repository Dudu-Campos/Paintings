"""
storage.py — Schema de armazenamento para grafos de pintura.

Hierarquia:
    HDF5:    /{painting_id}/{graph_type}/level_{k}/nodes|edges|edge_weights
    Parquet: painting_id | graph_type | level | features...
    CSV:     índice mestre de paintings
"""

import hashlib
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── IDs canônicos ─────────────────────────────────────────────────────────────

def make_painting_id(image_path: str) -> str:
    """
    ID determinístico a partir do conteúdo da imagem.
    Formato: {artist}_{md5[:12]}
    Independente de nome de arquivo ou ordem de processamento.
    """
    path = Path(image_path)
    with open(path, 'rb') as f:
        content_hash = hashlib.md5(f.read()).hexdigest()[:12]
    # Convenção WikiArt / MultitaskPainting100K: pai do arquivo = artista
    artist = path.parent.name.lower().replace(' ', '_')
    return f'{artist}_{content_hash}'


# ── Grafos em HDF5 ────────────────────────────────────────────────────────────

def save_graph(h5_path: str,
               painting_id: str,
               graph_type: str,
               level: int,
               nodes: np.ndarray,
               edges: np.ndarray,
               edge_weights: np.ndarray,
               metadata: dict = {}) -> None:
    """
    Salva um grafo em HDF5.

    Args:
        nodes:        float32 (N, F) — F features por nó
        edges:        int32   (2, E) — pares [source, target]
        edge_weights: float32 (E,)  — peso de cada aresta
        metadata:     atributos extras salvos como attrs do grupo
    """
    key = f'{painting_id}/{graph_type}/level_{level}'

    with h5py.File(h5_path, 'a') as f:
        if key in f:
            del f[key]

        grp = f.create_group(key)
        grp.create_dataset('nodes', data=nodes.astype(np.float32),
                           compression='gzip', compression_opts=4)
        grp.create_dataset('edges', data=edges.astype(np.int32),
                           compression='gzip', compression_opts=4)
        grp.create_dataset('edge_weights',
                           data=edge_weights.astype(np.float32),
                           compression='gzip', compression_opts=4)

        grp.attrs['n_nodes'] = int(nodes.shape[0])
        grp.attrs['n_edges'] = int(edges.shape[1])
        grp.attrs['graph_type'] = graph_type
        grp.attrs['level'] = level
        for k, v in metadata.items():
            grp.attrs[k] = v


def load_graph(h5_path: str,
               painting_id: str,
               graph_type: str,
               level: int) -> dict:
    """Carrega nós, arestas e metadados de um grafo."""
    key = f'{painting_id}/{graph_type}/level_{level}'
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f'Grafo não encontrado: {key}')
        grp = f[key]
        return {
            'nodes':        grp['nodes'][:],
            'edges':        grp['edges'][:],
            'edge_weights': grp['edge_weights'][:],
            'metadata':     dict(grp.attrs),
        }


def list_graphs(h5_path: str,
                painting_id: Optional[str] = None) -> list[dict]:
    """
    Lista todos os grafos disponíveis.
    Se painting_id for None, lista tudo.
    """
    results = []
    with h5py.File(h5_path, 'r') as f:
        pids = [painting_id] if painting_id else list(f.keys())
        for pid in pids:
            if pid not in f:
                continue
            for gtype in f[pid].keys():
                for level_key in f[pid][gtype].keys():
                    level = int(level_key.split('_')[1])
                    attrs = dict(f[pid][gtype][level_key].attrs)
                    results.append({
                        'painting_id': pid,
                        'graph_type':  gtype,
                        'level':       level,
                        **attrs,
                    })
    return results


# ── Features em Parquet ───────────────────────────────────────────────────────

# Schema base — extensível: novas colunas são adicionadas com NaN nas linhas antigas
FEATURE_COLUMNS = [
    # chave composta
    'painting_id', 'graph_type', 'level',
    # metadados da pintura
    'artist', 'style', 'movement',
    # estrutura do grafo
    'n_nodes', 'n_edges', 'mean_degree', 'density',
    # propriedades topológicas
    'clustering_coeff', 'modularity', 'spectral_gap', 'diameter',
    # qualidade de reconstrução
    'ssim', 'mse',
    # distribuição de grau
    'degree_exponent',   # α da lei de potência (se aplicável)
    'degree_gini',       # desigualdade da distribuição de grau
]


def append_features(parquet_path: str, row: dict) -> None:
    """
    Adiciona uma linha de features ao Parquet.
    Compatível com schema evolutivo — novas colunas são aceitas.
    """
    path = Path(parquet_path)

    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame(columns=FEATURE_COLUMNS)

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_parquet(path, compression='snappy', index=False)


def load_features(parquet_path: str,
                  painting_id: Optional[str] = None,
                  graph_type: Optional[str] = None,
                  level: Optional[int] = None) -> pd.DataFrame:
    """Carrega features com filtros opcionais."""
    df = pd.read_parquet(parquet_path)
    if painting_id:
        df = df[df['painting_id'] == painting_id]
    if graph_type:
        df = df[df['graph_type'] == graph_type]
    if level is not None:
        df = df[df['level'] == level]
    return df


# ── Índice mestre em CSV ──────────────────────────────────────────────────────

INDEX_COLUMNS = [
    'painting_id', 'artist', 'style', 'movement',
    'source_path', 'width', 'height',
]


def register_painting(index_path: str,
                       painting_id: str,
                       metadata: dict) -> None:
    """Registra uma pintura no índice mestre. Idempotente."""
    path = Path(index_path)

    if path.exists():
        index = pd.read_csv(path)
        if painting_id in index['painting_id'].values:
            return  # já registrada
    else:
        index = pd.DataFrame(columns=INDEX_COLUMNS)

    row = {'painting_id': painting_id, **metadata}
    index = pd.concat([index, pd.DataFrame([row])], ignore_index=True)
    index.to_csv(path, index=False)


def load_index(index_path: str) -> pd.DataFrame:
    return pd.read_csv(index_path)
