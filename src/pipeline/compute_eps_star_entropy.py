"""
compute_eps_star_entropy.py — Calcula ε*_entropy por imagem: o ponto
onde a curva de entropia da distribuição de área (decrescente com ε)
sai do platô "rico" e entra em colapso.

Reaproveita os .coarse já materializados — a curva completa (incluindo
entropy por threshold) já está salva no arquivo, só não era lida antes
por read_coarse2 (que descarta os steps). Usa read_coarse2_full.

Mesmo método de detecção de joelho (Kneedle) já usado em
diagnose_knee.py para a curva N(ε), mas aplicado a entropy(ε) — captura
onde a "riqueza estrutural" do grafo degenera, em vez de onde a
contagem de componentes degenera. É um critério não supervisionado,
sem precisar de SSIM (fidelidade perceptual) nem de rótulo de estilo.

Uso:
    python compute_eps_star_entropy.py \
        --coarse_dir ./coarsening_output_v2/coarse \
        --output     ./eps_star_entropy_output \
        --n_sample   0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lda_graph_features import read_coarse2_full


def find_knee_decreasing(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float(x[-1]) if len(x) else 0.0, 0.0

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    diff = y_norm - (1 - x_norm)
    idx_knee = int(np.argmax(np.abs(diff)))
    knee_strength = float(np.abs(diff[idx_knee]))
    return float(x[idx_knee]), knee_strength


def compute_eps_star_entropy_one(steps: np.ndarray) -> dict | None:
    if len(steps) < 3:
        return None

    eps = steps['epsilon'].astype(float)
    ent = steps['entropy'].astype(float)

    if ent.max() <= 0 or eps[0] <= 0:
        return None

    log_eps = np.log(eps)
    eps_knee, strength = find_knee_decreasing(log_eps, ent)

    return {
        'eps_star_entropy':       float(np.exp(eps_knee)),
        'knee_strength_entropy': strength,
        'entropy_at_eps1':        float(ent[0]),
        'entropy_at_max_eps':     float(ent[-1]),
    }


def build_eps_star_entropy_dataset(coarse_dir: str, n_sample: int = 0,
                                    seed: int = 42) -> pd.DataFrame:
    paths = list(Path(coarse_dir).rglob('*.coarse'))
    print(f'Arquivos .coarse encontrados: {len(paths):,}')

    if n_sample > 0 and len(paths) > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), n_sample, replace=False)
        paths = [paths[i] for i in idx]

    rows = []
    failed = 0
    for i, p in enumerate(paths):
        data = read_coarse2_full(str(p))
        if data is None or len(data['steps']) == 0:
            failed += 1
            continue
        r = compute_eps_star_entropy_one(data['steps'])
        if r is None:
            failed += 1
            continue
        r['stem'] = p.stem
        rows.append(r)
        if (i+1) % 1000 == 0:
            print(f'  {i+1:>7}/{len(paths):,}...', end='\r')

    print(f'\nProcessados: {len(rows):,}  falhas: {failed}')
    return pd.DataFrame(rows)


def plot_distribution(df: pd.DataFrame, output: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(df['eps_star_entropy'], bins=40, color='darkorange', alpha=0.8)
    ax.axvline(df['eps_star_entropy'].median(), color='red', linestyle='--',
               label=f"mediana={df['eps_star_entropy'].median():.1f}")
    ax.axvline(15, color='gray', linestyle=':', label='ε=15 (fixo, anterior)')
    ax.set_xlabel('ε*_entropy')
    ax.set_ylabel('Número de imagens')
    ax.set_title('Distribuição de ε*_entropy individual por imagem')
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.hist(df['knee_strength_entropy'], bins=40, color='steelblue', alpha=0.8)
    ax.set_xlabel('Força do joelho (entropia)')
    ax.set_ylabel('Número de imagens')
    ax.set_title('Quão bem definido é o ponto de colapso')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    p = Path(output) / 'eps_star_entropy_distribution.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse_dir', required=True)
    parser.add_argument('--output',     required=True)
    parser.add_argument('--n_sample',   type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = build_eps_star_entropy_dataset(args.coarse_dir, args.n_sample)

    print('\n=== Estatísticas de ε*_entropy ===')
    print(df['eps_star_entropy'].describe().round(2).to_string())
    print(f'\nMediana: {df["eps_star_entropy"].median():.2f}')
    print(f'% imagens com ε*_entropy < 15 (eps fixo atual seria TARDE demais): '
          f'{(df["eps_star_entropy"] < 15).mean()*100:.1f}%')
    print(f'% imagens com ε*_entropy > 15 (eps fixo atual seria CEDO demais): '
          f'{(df["eps_star_entropy"] > 15).mean()*100:.1f}%')

    df.to_parquet(out / 'eps_star_entropy_index.parquet', index=False)
    plot_distribution(df, str(out))


if __name__ == '__main__':
    main()
