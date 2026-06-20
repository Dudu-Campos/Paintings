"""
plot_spatial_density_violin.py — Violin plot da feature spatial_density_cv
(coeficiente de variação da contagem de nós por quadrante) por estilo.

Extrai a feature diretamente dos .coarse via spatial_density_cv (de
lda_graph_features.py), sem depender de um parquet já existente —
roda standalone sobre o coarse_dir.

Uso:
    python plot_spatial_density_violin.py \
        --coarse_dir ./coarsening_output_v2/coarse \
        --meta       /exp/duducampos/Paintings/meta/meta.csv \
        --output     ./plots \
        --grid_size  3 \
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

from lda_graph_features import read_coarse2, load_meta, spatial_density_cv


def build_spatial_density_dataset(coarse_dir: str, meta_path: str,
                                   grid_size: int = 3,
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
        feats = spatial_density_cv(data['nodes'], grid_size=grid_size)
        feats['stem']  = p.stem
        feats['style'] = meta.get(p.stem, 'unknown')
        rows.append(feats)
        if (i+1) % 1000 == 0:
            print(f'  {i+1:>7}/{len(paths):,}...', end='\r')

    print(f'\nProcessados: {len(rows):,}  falhas: {failed}')
    return pd.DataFrame(rows)


def plot_violin(df: pd.DataFrame, output: str, feature_col: str,
                 min_count: int = 30, max_styles: int = 25):
    df = df[df['style'] != 'unknown'].copy()
    counts = df['style'].value_counts()
    valid  = counts[counts >= min_count].index
    df     = df[df['style'].isin(valid)]

    order = (df.groupby('style')[feature_col]
               .median()
               .sort_values(ascending=False)
               .head(max_styles)
               .index.tolist())
    sub = df[df['style'].isin(order)]

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
    ax.set_xlabel(f'{feature_col}  (CV da contagem de nós por quadrante)',
                   fontsize=10)
    ax.set_title('Distribuição da densidade espacial de nós por estilo\n'
                  'Alto = estrutura concentrada em regiões específicas  |  '
                  'Baixo = estrutura espalhada uniformemente', fontsize=11)
    ax.grid(axis='x', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    p = Path(output) / f'{feature_col}_violin.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')

    summary = (sub.groupby('style')[feature_col]
               .agg(['count','median','mean','std'])
               .sort_values('median', ascending=False)
               .round(3))
    summary.to_csv(Path(output) / f'{feature_col}_summary.csv')
    print(f'\nTop 15 ({feature_col} mediano):')
    print(summary.head(15).to_string())
    print(f'\nBottom 15 ({feature_col} mediano):')
    print(summary.tail(15).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--meta',       default='')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--grid_size',  type=int, default=3)
    parser.add_argument('--min_count',  type=int, default=30)
    parser.add_argument('--max_styles', type=int, default=25)
    parser.add_argument('--n_sample',   type=int, default=3000)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = build_spatial_density_dataset(
        args.coarse_dir, args.meta, args.grid_size, args.n_sample)
    df.to_parquet(out / 'spatial_density_features.parquet', index=False)

    plot_violin(df, str(out), 'spatial_density_cv',
                args.min_count, args.max_styles)
    plot_violin(df, str(out), 'spatial_density_max_block_frac',
                args.min_count, args.max_styles)


if __name__ == '__main__':
    main()
