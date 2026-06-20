"""
plot_eps_perf_vs_cost.py — Acurácia LDA vs custo do grafo por epsilon.

Uso:
    python plot_eps_perf_vs_cost.py \
        --index  ./coarsening_output_v4/index.parquet \
        --summary ./plots_v4/lda_eps_sweep_summary.csv \
        --output  ./plots_v4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EPSILONS    = [2, 4, 6, 8, 10]
N_TOTAL     = 72993          # imagens no dataset completo
DT_NODE_B   = 7 * 4         # 28 bytes por nó
DT_EDGE_B   = 4 * 4         # 16 bytes por aresta


def compute_costs(idx: pd.DataFrame, cap: int = 100) -> pd.DataFrame:
    top30 = idx['style'].value_counts().head(30).index
    pieces = [g.sample(min(len(g), cap), random_state=42)
              for _, g in idx[idx['style'].isin(top30)].groupby('style')]
    sub = pd.concat(pieces)

    # Ratio n_edges/n_nodes a partir dos grafos materializados em eps_star
    ratio = (sub['kstar_n_edges'] / sub['kstar_n_nodes'].clip(lower=1)).median()

    rows = []
    for eps in EPSILONS:
        nn_mean = sub[f'n_comp_e{eps}'].mean()
        ne_mean = nn_mean * ratio
        gb = (nn_mean * DT_NODE_B + ne_mean * DT_EDGE_B) * N_TOTAL / 1e9
        rows.append({'epsilon': eps, 'avg_n_nodes': nn_mean, 'cost_gb': gb})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',   required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output',  required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    idx = pd.read_parquet(args.index)
    n_dataset = len(idx)   # tamanho real do dataset para o label
    costs = compute_costs(idx)

    summ = pd.read_csv(args.summary)
    # Pegar só as linhas de grafo isolado por epsilon numérico
    perf = (summ[(summ['mode'] == 'graph') & (summ['epsilon'] != 'combined')]
            .copy())
    perf['epsilon'] = perf['epsilon'].astype(int)
    perf = perf.sort_values('epsilon')

    merged = costs.merge(perf[['epsilon', 'acc_mean', 'acc_std']], on='epsilon')

    eps      = merged['epsilon'].values
    acc      = merged['acc_mean'].values * 100
    acc_err  = merged['acc_std'].values * 100
    cost_gb  = merged['cost_gb'].values
    n_nodes  = merged['avg_n_nodes'].values

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    COLOR_ACC  = '#2563EB'
    COLOR_COST = '#DC2626'

    # Custo (eixo direito, log)
    ax2.fill_between(eps, cost_gb, alpha=0.12, color=COLOR_COST)
    line_cost, = ax2.plot(eps, cost_gb, 'o-', color=COLOR_COST, lw=2.5,
                          markersize=8, label='Storage cost (GB)')
    ax2.set_yscale('log')
    ax2.set_ylabel(f'Estimated storage cost (GB, log scale)\nfor {n_dataset:,}-image dataset',
                   color=COLOR_COST, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=COLOR_COST, labelsize=12)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f'{v:.0f}'))

    # Acurácia (eixo esquerdo)
    ax1.errorbar(eps, acc, yerr=acc_err, fmt='s-', color=COLOR_ACC, lw=2.5,
                 markersize=8, capsize=5, label='LDA accuracy (graph)')
    ax1.set_xlabel('Threshold ε (ΔE CIE-Lab)', fontsize=14)
    ax1.set_ylabel('5-fold CV accuracy (%)', color=COLOR_ACC, fontsize=13)
    ax1.tick_params(axis='y', labelcolor=COLOR_ACC, labelsize=12)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_xticks(eps)
    ax1.set_ylim(0, max(acc) * 1.35)

    # Legenda das curvas (canto superior direito)
    lines = [
        line_cost,
        plt.Line2D([0], [0], marker='s', color=COLOR_ACC, lw=2.5,
                   markersize=8, label='LDA accuracy (graph)'),
    ]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc='upper right', fontsize=12, framealpha=0.95, edgecolor='#D1D5DB')

    # Tabela ε / avg nodes / cost — text box no canto inferior esquerdo
    col_w = 10
    header = f"{'ε':^{col_w}}{'avg nodes':^{col_w}}{'cost':^{col_w}}"
    sep    = '─' * (col_w * 3)
    rows   = '\n'.join(
        f"{str(e):^{col_w}}{f'{n/1e3:.0f}k':^{col_w}}{f'{c:.0f} GB':^{col_w}}"
        for e, n, c in zip(eps, n_nodes, cost_gb)
    )
    table_text = f"{header}\n{sep}\n{rows}"

    ax1.text(0.02, 0.03, table_text,
             transform=ax1.transAxes, fontsize=11,
             va='bottom', ha='left',
             fontfamily='monospace', color='#1F2937',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#D1D5DB', lw=0.9,
                       alpha=0.95))

    ax1.set_title(
        'Performance vs. graph storage cost by ε\n'
        '(15 graph features, LDA, 30 styles × 100 images)',
        fontsize=13)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')
    ax1.set_axisbelow(True)

    plt.tight_layout()
    p = out / 'eps_perf_vs_cost.png'
    plt.savefig(p, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')

    # Tabela resumo
    print('\nε   acc(%)   n_nodes_médio   custo_GB')
    for _, r in merged.iterrows():
        print(f"{r['epsilon']:2.0f}  {r['acc_mean']*100:.1f}±{r['acc_std']*100:.1f}"
              f"   {r['avg_n_nodes']:>10.0f}   {r['cost_gb']:>8.1f}")


if __name__ == '__main__':
    main()
