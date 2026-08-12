"""
lda_delaunay_mst_features.py — Extrai features de Delaunay e MST para o WikiArt.

Lê os mesmos .coarse files do RAG, mas constrói grafos diferentes:
  Delaunay: triangulação de Delaunay sobre centróides (X, Y) — geometria espacial
  MST:      árvore geradora mínima no espaço Lab (ΔE) — hierarquia da paleta

Aplica o mesmo pipeline de features do RAG (32 feat por grafo).

Uso:
    python src/pipeline/lda_delaunay_mst_features.py \
        --coarse_dir outputs/wikiart_coarsening/coarse \
        --meta       data/artemis/wikiart_meta.csv \
        --output     outputs/wikiart_features \
        --workers    128
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay as _Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from lda_graph_features import (
    read_coarse2, extract_graph_features, extract_color_features,
    spatial_density_cv, remove_giant_component,
)

MAX_NODES = 100_000


def _make_edge_array(src_dst, color_dist, spatial_dist):
    dt = np.dtype([('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4')])
    arr = np.zeros(len(src_dst), dtype=dt)
    arr['src']          = [e[0] for e in src_dst]
    arr['dst']          = [e[1] for e in src_dst]
    arr['color_dist']   = color_dist
    arr['spatial_dist'] = spatial_dist
    return arr


def build_delaunay_edges(nodes):
    if len(nodes) < 3:
        return np.zeros(0, dtype=np.dtype([
            ('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4')]))
    XY  = np.column_stack([nodes['X'], nodes['Y']])
    Lab = np.column_stack([nodes['L'], nodes['a'], nodes['b']])
    try:
        tri = _Delaunay(XY)
    except Exception:
        return np.zeros(0, dtype=np.dtype([
            ('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4')]))
    edge_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = sorted([simplex[i], simplex[(i+1)%3]])
            edge_set.add((a, b))
    if not edge_set:
        return np.zeros(0, dtype=np.dtype([
            ('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4')]))
    src_dst = list(edge_set)
    s = np.array([e[0] for e in src_dst])
    d = np.array([e[1] for e in src_dst])
    cdist = np.sqrt(((Lab[s] - Lab[d])**2).sum(axis=1)).astype(np.float32)
    sdist = np.sqrt(((XY[s]  - XY[d])**2).sum(axis=1)).astype(np.float32)
    return _make_edge_array(src_dst, cdist, sdist)


def build_mst_edges(nodes, rag_edges):
    """MST sobre o grafo RAG ponderado por color_dist — O(E log V), sem all-pairs."""
    n = len(nodes)
    if n < 2 or len(rag_edges) == 0:
        return np.zeros(0, dtype=np.dtype([
            ('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4')]))
    XY = np.column_stack([nodes['X'], nodes['Y']])
    s  = rag_edges['src'].astype(np.int32)
    d  = rag_edges['dst'].astype(np.int32)
    cd = rag_edges['color_dist'].astype(np.float32)
    sparse = csr_matrix((cd, (s, d)), shape=(n, n))
    # symmetrize
    sparse = sparse + sparse.T
    mst = minimum_spanning_tree(sparse)
    cx  = mst.tocoo()
    src_dst = list(zip(cx.row.tolist(), cx.col.tolist()))
    if not src_dst:
        return np.zeros(0, dtype=np.dtype([
            ('src','<u4'),('dst','<u4'),('color_dist','<f4'),('spatial_dist','<f4')]))
    si = np.array([e[0] for e in src_dst])
    di = np.array([e[1] for e in src_dst])
    cd_out = cx.data.astype(np.float32)
    sd_out = np.sqrt(((XY[si] - XY[di])**2).sum(axis=1)).astype(np.float32)
    return _make_edge_array(src_dst, cd_out, sd_out)


def _process_one(args):
    path, meta_style = args
    try:
        data = read_coarse2(str(path))
        if data is None:
            return None, None
        nodes = data['nodes']
        if len(nodes) == 0 or len(nodes) > MAX_NODES:
            return None, None

        # remove nó dominante antes de construir os grafos complementares
        nodes_ng, edges_ng, _ = remove_giant_component(nodes, data['edges'])
        if len(nodes_ng) < 3:
            return None, None

        del_edges = build_delaunay_edges(nodes_ng)
        mst_edges = build_mst_edges(nodes_ng, edges_ng)

        color_f  = extract_color_features(nodes_ng)
        spat_f   = spatial_density_cv(nodes_ng)

        del_f = extract_graph_features(nodes_ng, del_edges)
        del_f.update(color_f); del_f.update(spat_f)
        del_f['stem'] = path.stem; del_f['style'] = meta_style

        mst_f = extract_graph_features(nodes_ng, mst_edges)
        mst_f.update(color_f); mst_f.update(spat_f)
        mst_f['stem'] = path.stem; mst_f['style'] = meta_style

        return del_f, mst_f
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coarse_dir', required=True)
    ap.add_argument('--meta',       required=True)
    ap.add_argument('--output',     required=True)
    ap.add_argument('--workers',    type=int, default=32)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    style_map = dict(zip(meta['stem'], meta['style']))

    paths = sorted(Path(args.coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse: {len(paths):,}  workers: {args.workers}')

    del_rows, mst_rows = [], []
    done = failed = 0
    worker_args = [(p, style_map.get(p.stem, 'unknown')) for p in paths]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, a): a for a in worker_args}
        for fut in as_completed(futs):
            done += 1
            d, m = fut.result()
            if d is None:
                failed += 1
            else:
                del_rows.append(d)
                mst_rows.append(m)
            if done % 2000 == 0:
                print(f'  {done:>7}/{len(paths):,}  ok={len(del_rows):,}  fail={failed}',
                      flush=True)

    print(f'\nProcessados: {len(del_rows):,}  falhas: {failed}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(del_rows).to_parquet(out / 'delaunay_features.parquet', index=False)
    pd.DataFrame(mst_rows).to_parquet(out / 'mst_features.parquet',      index=False)
    print(f'Salvo: {out}/delaunay_features.parquet  mst_features.parquet')


if __name__ == '__main__':
    main()
