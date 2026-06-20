"""
lda_combined_features.py — Combina features de RAG e k-NN sobre os
mesmos nós, testando se a concatenação supera cada representação isolada.

Para cada imagem:
  1. Lê o grafo materializado (.coarse, mode=1) — nós já existentes
  2. Extrai features básicas a partir das arestas RAG (já no arquivo)
  3. Constrói arestas k-NN (k=10) sobre os mesmos nós
  4. Extrai features básicas a partir das arestas k-NN
  5. Concatena os dois vetores de 15 features → 30 features totais

Compara contra cada representação isolada e contra a curva N(ε).

Uso:
    python lda_combined_features.py \
        --coarse_dir ./coarsening_output_v2/coarse \
        --meta       /exp/duducampos/Paintings/meta/meta.csv \
        --output     ./plots \
        --min_count  30 \
        --n_sample   3000 \
        --k          10 \
        --workers    32 \
        --acc_curve      0.150 \
        --acc_rag_basic  0.180 \
        --acc_knn_basic  0.202
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lda_graph_features import (
    read_coarse2, load_meta, select_balanced_paths,
    extract_graph_features, GRAPH_FEATURE_COLS, run_lda,
)
from lda_knn_features import build_knn_edges


# ── Features combinadas (RAG + k-NN) ──────────────────────────────────────────

RAG_COLS = [f'rag_{c}' for c in GRAPH_FEATURE_COLS]
KNN_COLS = [f'knn_{c}' for c in GRAPH_FEATURE_COLS]
COMBINED_FEATURE_COLS = RAG_COLS + KNN_COLS


def _process_one_combined(args) -> dict | None:
    path, meta_style, k = args
    data = read_coarse2(str(path))
    if data is None:
        return None

    nodes, rag_edges = data['nodes'], data['edges']

    rag_feats = extract_graph_features(nodes, rag_edges)
    knn_edges = build_knn_edges(nodes, k=k)
    knn_feats = extract_graph_features(nodes, knn_edges)

    row = {}
    for c in GRAPH_FEATURE_COLS:
        row[f'rag_{c}'] = rag_feats[c]
        row[f'knn_{c}'] = knn_feats[c]

    row['stem']  = Path(path).stem
    row['style'] = meta_style
    return row


def build_combined_dataset(coarse_dir: str, meta_path: str,
                            n_sample: int = 3000, k: int = 10,
                            seed: int = 42, n_workers: int = 1,
                            balanced_min: int = 0,
                            balanced_max: int = 0) -> pd.DataFrame:
    meta = load_meta(meta_path)
    paths = list(Path(coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse encontrados: {len(paths):,}')

    if balanced_min > 0 and balanced_max > 0:
        paths = select_balanced_paths(paths, meta, balanced_min, balanced_max, seed)
    elif n_sample > 0 and len(paths) > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), n_sample, replace=False)
        paths = [paths[i] for i in idx]

    args = [(p, meta.get(p.stem, 'unknown'), k) for p in paths]

    rows = []
    if n_workers <= 1:
        for i, a in enumerate(args):
            r = _process_one_combined(a)
            if r is not None:
                rows.append(r)
            if (i+1) % 200 == 0:
                print(f'  {i+1:>7}/{len(paths):,}...', end='\r')
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_one_combined, a): a for a in args}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                if r is not None:
                    rows.append(r)
                if i % 200 == 0 or i == len(paths):
                    print(f'  [{i:>7}/{len(paths):,}]  ok={len(rows):,}',
                          end='\r', flush=True)

    print(f'\nProcessados: {len(rows):,}  falhas: {len(paths)-len(rows)}')
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--meta',       default='')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--min_count',  type=int, default=30)
    parser.add_argument('--n_sample',   type=int, default=3000)
    parser.add_argument('--balanced_min', type=int, default=0)
    parser.add_argument('--balanced_max', type=int, default=0)
    parser.add_argument('--k',          type=int, default=10)
    parser.add_argument('--workers',    type=int, default=1)
    parser.add_argument('--acc_curve',     type=float, default=0.150)
    parser.add_argument('--acc_rag_basic', type=float, default=0.180)
    parser.add_argument('--acc_knn_basic', type=float, default=0.202)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = build_combined_dataset(args.coarse_dir, args.meta,
                                 args.n_sample, args.k,
                                 n_workers=args.workers,
                                 balanced_min=args.balanced_min,
                                 balanced_max=args.balanced_max)
    df = df[df['style'] != 'unknown'].copy()

    counts = df['style'].value_counts()
    valid  = counts[counts >= args.min_count].index
    df     = df[df['style'].isin(valid)]
    print(f'\nApós filtro min_count={args.min_count}: '
          f'{len(df):,} imagens  {df["style"].nunique()} estilos')

    df.to_parquet(out / 'combined_features.parquet', index=False)
    y = df['style'].values

    X = df[COMBINED_FEATURE_COLS].values.astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])

    acc_combined, std_combined = run_lda(
        X, y, str(out), label='rag_knn_combined',
        title_extra=f'LDA — RAG + k-NN (k={args.k}) combinados, '
                     f'{len(COMBINED_FEATURE_COLS)} features')

    results = {
        'Curva N(ε) — 36 features':                args.acc_curve,
        'RAG básico — 15 features':                 args.acc_rag_basic,
        f'k-NN (k={args.k}) básico — 15 features':  args.acc_knn_basic,
        f'RAG + k-NN combinados — '
        f'{len(COMBINED_FEATURE_COLS)} features':   acc_combined,
    }

    print('\n' + '='*70)
    print(f'{"Método":<52} {"Acurácia":>12}')
    print('-'*70)
    for name, acc in results.items():
        print(f'{name:<52} {acc:>12.3f}')
    print('='*70)

    best_isolated = max(args.acc_rag_basic, args.acc_knn_basic)
    if acc_combined > best_isolated:
        print(f'\n→ Combinação SUPERA ambas as representações isoladas '
              f'(+{(acc_combined-best_isolated)*100:.1f}pp) — '
              f'RAG e k-NN capturam informação complementar.')
    else:
        print(f'\n→ Combinação NÃO supera a melhor representação isolada — '
              f'pode haver redundância entre RAG e k-NN, ou efeito de '
              f'dimensionalidade no LDA com poucas amostras por classe.')


if __name__ == '__main__':
    main()
