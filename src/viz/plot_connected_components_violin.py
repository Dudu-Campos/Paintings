"""
plot_connected_components_violin.py — Estrutura de componentes conexos
do grafo RAG materializado (ε=15), por estilo.

Testa a hipótese: o grafo é dominado por um componente gigante (um
"fundo" homogêneo conectando quase tudo), com a estrutura real
(quantos fragmentos, qual o tamanho relativo deles) escondida atrás
desse hub — o que explicaria por que features agregadas sobre o
grafo inteiro discriminam mal entre estilos.

Gera violin plots para:
  - n_components            (quantos componentes conexos existem)
  - giant_frac               (fração de nós no maior componente)
  - giant_to_second_ratio    (razão maior / segundo maior componente)
  - n_components_size_gt1   (componentes "reais", excluindo singletons)

Uso:
    python plot_connected_components_violin.py \
        --coarse_dir ./coarsening_output_v2/coarse \
        --meta       /exp/duducampos/Paintings/meta/meta.csv \
        --output     ./plots \
        --min_count  30 \
        --max_styles 25 \
        --n_sample   3000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lda_graph_features import read_coarse2, load_meta, connected_components_stats


def build_components_dataset(coarse_dir: str, meta_path: str,
                              n_sample: int = 3000,
                              seed: int = 42) -> pd.DataFrame:
    meta = load_meta(meta_path)
    paths = list(Path(coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse encontrados: {len(paths):,}')

    if n_sample > 0 and len(paths) > n_sample:
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
        feats = connected_components_stats(data['nodes'], data['edges'])
        feats['stem']  = p.stem
        feats['style'] = meta.get(p.stem, 'unknown')
        rows.append(feats)
        if (i+1) % 500 == 0:
            print(f'  {i+1:>7}/{len(paths):,}...', end='\r')

    print(f'\nProcessados: {len(rows):,}  falhas: {failed}')
    return pd.DataFrame(rows)


def plot_violin(df: pd.DataFrame, output: str, feature_col: str,
                 min_count: int = 30, max_styles: int = 25,
                 log_scale: bool = False):
    df = df[df['style'] != 'unknown'].copy()
    counts = df['style'].value_counts()
    valid  = counts[counts >= min_count].index
    df     = df[df['style'].isin(valid)]

    order = (df.groupby('style')[feature_col]
               .median()
               .sort_values(ascending=False)
               .head(max_styles)
               .index.tolist())
    sub = df[df['style'].isin(order)].copy()

    if log_scale:
        sub[feature_col] = np.log10(sub[feature_col].clip(lower=1))

    cmap   = cm.RdYlBu_r
    colors = [cmap(i/max(1,len(order)-1)) for i in range(len(order))]

    fig, ax = plt.subplots(figsize=(10, max(5, len(order)*0.4)))

    data  = [sub[sub['style']==s][feature_col].dropna().values for s in order]
    parts = ax.violinplot(data, positions=range(len(order)),
                           vert=False, showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c); pc.set_alpha(0.75)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(1.5)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    xlabel = f'log10({feature_col})' if log_scale else feature_col
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(f'Distribuição de {feature_col} por estilo '
                 f'(grafo RAG, ε=15)', fontsize=11)
    ax.grid(axis='x', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    p = Path(output) / f'{feature_col}_violin.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')

    summary = (df[df['style'].isin(order)]
               .groupby('style')[feature_col]
               .agg(['count','median','mean','std'])
               .sort_values('median', ascending=False)
               .round(3))
    summary.to_csv(Path(output) / f'{feature_col}_summary.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--meta',       default='')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--min_count',  type=int, default=30)
    parser.add_argument('--max_styles', type=int, default=25)
    parser.add_argument('--n_sample',   type=int, default=3000)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = build_components_dataset(args.coarse_dir, args.meta, args.n_sample)
    df.to_parquet(out / 'connected_components_features.parquet', index=False)

    print('\n=== Estatísticas globais (todas as imagens, antes do filtro) ===')
    print(df[['n_components','giant_frac','giant_to_second_ratio',
              'n_components_size_gt1']].describe().round(3).to_string())

    plot_violin(df, str(out), 'n_components',
                args.min_count, args.max_styles, log_scale=True)
    plot_violin(df, str(out), 'giant_frac',
                args.min_count, args.max_styles)
    plot_violin(df, str(out), 'giant_to_second_ratio',
                args.min_count, args.max_styles, log_scale=True)
    plot_violin(df, str(out), 'n_components_size_gt1',
                args.min_count, args.max_styles, log_scale=True)


if __name__ == '__main__':
    main()
