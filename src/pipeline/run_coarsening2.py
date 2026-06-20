"""
run_coarsening2.py — Pipeline para coarse2.cpp (análise de percolação).

Uso:
    # Só features (rápido)
    python run_coarsening2.py \
        --image_dir /path/to/raw \
        --coarse_bin ./coarse2 \
        --output_dir ./output \
        --meta meta.csv \
        --workers 64

    # Features + grafo em ε*=10
    python run_coarsening2.py \
        --image_dir /path/to/raw \
        --coarse_bin ./coarse2 \
        --output_dir ./output \
        --meta meta.csv \
        --workers 64 \
        --graph \
        --epsilon_star 10.0
"""

import argparse
import csv
import struct
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Thresholds (espelho do C++) ───────────────────────────────────────────────

EPSILONS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0]

# ── Dtypes ────────────────────────────────────────────────────────────────────

dt_percol = np.dtype([
    ('epsilon',           '<f4'),
    ('n_components',      '<u4'),
    ('giant_size',        '<u4'),
    ('giant_frac',        '<f4'),
    ('entropy',           '<f4'),
    ('mean_size',         '<f4'),
    ('gini',              '<f4'),
    ('n_singletons_frac', '<f4'),
])

dt_node = np.dtype([
    ('L',         '<f4'),
    ('a',         '<f4'),
    ('b',         '<f4'),
    ('X',         '<f4'),
    ('Y',         '<f4'),
    ('area_frac', '<f4'),
    ('area_px',   '<u4'),
])

dt_edge = np.dtype([
    ('src',          '<u4'),
    ('dst',          '<u4'),
    ('color_dist',   '<f4'),
    ('spatial_dist', '<f4'),
])

# ── Leitura ───────────────────────────────────────────────────────────────────

def read_coarse2(path: str) -> dict:
    """
    Lê arquivo .coarse do coarse2.cpp.
    Retorna dict com 'mode', 'eps_star', 'steps', 'nodes', 'edges'.
    """
    with open(path, 'rb') as f:
        mode         = struct.unpack('<I', f.read(4))[0]
        n_thresholds = struct.unpack('<I', f.read(4))[0]
        eps_star     = struct.unpack('<f', f.read(4))[0] if mode == 1 else None

        steps = np.frombuffer(
            f.read(n_thresholds * dt_percol.itemsize), dtype=dt_percol).copy()

        nodes = np.zeros(0, dtype=dt_node)
        edges = np.zeros(0, dtype=dt_edge)
        if mode == 1:
            nn    = struct.unpack('<I', f.read(4))[0]
            nodes = np.frombuffer(
                f.read(nn * dt_node.itemsize), dtype=dt_node).copy()
            ne    = struct.unpack('<I', f.read(4))[0]
            edges = np.frombuffer(
                f.read(ne * dt_edge.itemsize), dtype=dt_edge).copy()

    return {'mode': mode, 'eps_star': eps_star,
            'steps': steps, 'nodes': nodes, 'edges': edges}


# ── Features da curva ─────────────────────────────────────────────────────────

def extract_features(data: dict, stem: str,
                      style: str, artist: str, genre: str,
                      coarse_path: str) -> dict:
    """
    Extrai features da curva de percolação para o index.parquet.
    Uma linha por imagem — colunas por threshold para giant_frac, n_comp, gini.
    """
    steps = data['steps']
    row   = {
        'stem':   stem,
        'style':  style,
        'artist': artist,
        'genre':  genre,
        'coarse': coarse_path,
        'mode':   data['mode'],
    }

    for s in steps:
        eps = s['epsilon']
        tag = f'e{eps:.0f}'.replace('.', '_')
        row[f'giant_frac_{tag}'] = float(s['giant_frac'])
        row[f'n_comp_{tag}']     = int(s['n_components'])
        row[f'gini_{tag}']       = float(s['gini'])

    # ε* de percolação empírica — onde giant_frac cruza 0.5
    eps_percol = None
    for s in steps:
        if s['giant_frac'] >= 0.5:
            eps_percol = float(s['epsilon'])
            break
    row['eps_percol'] = eps_percol  # None se nunca atingiu 0.5

    # Se grafo disponível
    if data['mode'] == 1:
        row['kstar_n_nodes'] = len(data['nodes'])
        row['kstar_n_edges'] = len(data['edges'])

    return row


# ── Meta CSV ──────────────────────────────────────────────────────────────────

def load_meta(meta_path: str) -> dict:
    lookup = {}
    if not meta_path:
        return lookup
    try:
        with open(meta_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row.get('filename', '')).stem
                lookup[stem] = {
                    'artist': row.get('artist', 'unknown').strip(),
                    'style':  row.get('style',  'unknown').strip(),
                    'genre':  row.get('genre',  'unknown').strip(),
                }
        print(f'Meta: {len(lookup):,} entradas')
    except FileNotFoundError:
        print(f'AVISO: meta não encontrado em {meta_path}')
    return lookup


def load_eps_star_lookup(eps_star_path: str) -> dict:
    """
    Carrega um parquet com colunas (stem, eps_star_*) e retorna dict
    stem -> valor de eps_star. Aceita qualquer nome de coluna que
    comece com 'eps_star' (eps_star_ssim, eps_star_entropy, etc) —
    detecta automaticamente, já que diferentes critérios (SSIM,
    entropia, joelho de N(ε)) podem ser usados para calibrar o ε*
    individual de cada imagem.

    Usado para materializar o grafo no ponto crítico individual de cada
    imagem, em vez de um valor fixo global.
    """
    lookup = {}
    if not eps_star_path:
        return lookup
    try:
        df = pd.read_parquet(eps_star_path)
        eps_cols = [c for c in df.columns if c.startswith('eps_star')]
        if not eps_cols:
            print(f'AVISO: nenhuma coluna eps_star_* encontrada em '
                  f'{eps_star_path}. Colunas disponíveis: {list(df.columns)}')
            return lookup
        col = eps_cols[0]
        if len(eps_cols) > 1:
            print(f'AVISO: múltiplas colunas eps_star_* encontradas '
                  f'({eps_cols}), usando "{col}"')
        lookup = dict(zip(df['stem'], df[col]))
        print(f'{col} lookup: {len(lookup):,} entradas  '
              f'(mediana={df[col].median():.2f})')
    except Exception as e:
        print(f'AVISO: falha ao carregar eps_star_path ({eps_star_path}): {e}')
    return lookup


# ── Worker ────────────────────────────────────────────────────────────────────

def process_one(args: tuple) -> dict | None:
    (image_path, coarse_bin, output_dir, meta_lookup,
     save_graph, eps_star_lookup, eps_star_default) = args
    p    = Path(image_path)
    stem = p.stem

    m      = meta_lookup.get(stem, {})
    style  = m.get('style',  p.parent.parent.name)
    artist = m.get('artist', p.parent.name)
    genre  = m.get('genre',  'unknown')

    # ε* individual desta imagem — vem do eps_star_ssim_index.parquet.
    # Cada pintura tem seu próprio ponto crítico de fidelidade perceptual;
    # usa o default só se a imagem não tiver entrada no lookup (não processada
    # na etapa de SSIM, ou falha naquela etapa).
    epsilon_star = eps_star_lookup.get(stem, eps_star_default)

    out_dir = Path(output_dir) / 'coarse' / style.replace(' ', '_').replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{stem}.coarse'

    if not out_path.exists() or out_path.stat().st_size < 16:
        cmd = [coarse_bin, str(image_path), str(out_path)]
        if save_graph:
            cmd += ['--graph', '--epsilon_star', str(epsilon_star)]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0 or not out_path.exists():
                return None
        except Exception:
            return None

    try:
        data = read_coarse2(str(out_path))
        row = extract_features(data, stem, style, artist, genre, str(out_path))
        row['epsilon_star_used'] = epsilon_star
        row['eps_star_from_lookup'] = stem in eps_star_lookup
        return row
    except Exception:
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_dataset(image_dir: str, coarse_bin: str, output_dir: str,
                meta_path: str = '', n_workers: int = 8,
                save_graph: bool = False,
                eps_star_path: str = '',
                eps_star_default: float = 10.0,
                extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> pd.DataFrame:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    index_path = Path(output_dir) / 'index.parquet'

    meta = load_meta(meta_path)
    eps_star_lookup = load_eps_star_lookup(eps_star_path) if save_graph else {}

    paths = sorted(
        [str(p) for p in Path(image_dir).rglob('*')
         if p.suffix.lower() in extensions],
        key=lambda p: Path(p).stat().st_size,
        reverse=True  # LPT scheduling
    )
    print(f'Imagens: {len(paths):,}  workers: {n_workers}  '
          f'mode: {"graph" if save_graph else "features"}')
    if save_graph:
        missing = sum(1 for p in paths if Path(p).stem not in eps_star_lookup)
        print(f'  ε* individual disponível para '
              f'{len(paths)-missing:,}/{len(paths):,} imagens  '
              f'({missing:,} usarão default={eps_star_default})')

    args = [(p, coarse_bin, output_dir, meta, save_graph,
             eps_star_lookup, eps_star_default)
            for p in paths]

    results = []
    failed  = 0
    checkpoint_every = 1000

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process_one, a): a[0] for a in args}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                results.append(r)
            else:
                failed += 1
            if i % checkpoint_every == 0 or i == len(paths):
                print(f'  [{i:>7}/{len(paths):,}]  '
                      f'ok={len(results):,}  fail={failed}',
                      end='\r', flush=True)
                # Checkpoint incremental — permite inspecionar resultados
                # parciais sem esperar o dataset completo terminar
                pd.DataFrame(results).to_parquet(index_path, index=False)

    print(f'\nConcluído: {len(results):,} OK  {failed} falhas')

    df = pd.DataFrame(results)
    df.to_parquet(index_path, index=False)
    print(f'Index: {index_path}  ({len(df):,} linhas)')
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',        required=True)
    parser.add_argument('--coarse_bin',       default='./coarse2')
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--meta',             default='')
    parser.add_argument('--workers',          type=int,   default=8)
    parser.add_argument('--graph',            action='store_true')
    parser.add_argument('--eps_star_path',    default='',
                        help='Caminho para eps_star_ssim_index.parquet '
                             '(coluna stem, eps_star_ssim) — usa ε* '
                             'individual por imagem em vez de valor fixo')
    parser.add_argument('--eps_star_default', type=float, default=10.0,
                        help='ε* a usar quando a imagem não está no lookup')
    args = parser.parse_args()

    run_dataset(
        image_dir        = args.image_dir,
        coarse_bin        = args.coarse_bin,
        output_dir        = args.output_dir,
        meta_path         = args.meta,
        n_workers         = args.workers,
        save_graph        = args.graph,
        eps_star_path     = args.eps_star_path,
        eps_star_default  = args.eps_star_default,
    )


if __name__ == '__main__':
    main()
