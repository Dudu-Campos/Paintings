"""
plot_graph_structure_vs_epsilon.py — Para uma única imagem, materializa
o grafo em vários valores de ε e plota como a ESTRUTURA do grafo evolui
(não a imagem reconstruída — isso já existe em viz_segmentation_progressive.py).

Métricas plotadas em função de ε:
  - n_components            (quantos fragmentos existem)
  - giant_frac               (fração de nós no maior componente/hub)
  - n_components_size_gt1   (fragmentos "reais", excluindo singletons)
  - mean_area_excl_giant     (tamanho médio dos fragmentos fora do hub)

A hipótese a testar: existe uma janela de ε onde a estrutura residual
(fora do hub) é rica — muitos fragmentos de tamanho razoável — e fora
dessa janela ela degenera: ou há ε pequeno demais (quase 1 nó por
pixel, estrutura = ruído) ou ε grande demais (hub já absorveu quase
tudo, poucos fragmentos triviais sobrando).

Uso:
    python plot_graph_structure_vs_epsilon.py \
        --image      "/exp/duducampos/Paintings/raw/images/1.jpg"\
        --coarse_bin ./coarse2 \
        --output     ./plots \
        --eps_values 1 2 3 4 6 8 10 12 15 18 22 27 33 40 50
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lda_graph_features import (
    read_coarse2, connected_components_stats,
    gini_coefficient, shannon_entropy,
)


def materialize_graph(image_path: str, coarse_bin: str, epsilon: float,
                       work_dir: str) -> dict | None:
    out_path = Path(work_dir) / f'tmp_eps_{epsilon:.2f}.coarse'
    cmd = [coarse_bin, str(image_path), str(out_path),
           '--graph', '--epsilon_star', str(epsilon)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not out_path.exists():
            print(f'  ε={epsilon}: FALHOU — {result.stderr.decode()[:200]}')
            return None
    except Exception as e:
        print(f'  ε={epsilon}: ERRO — {e}')
        return None

    data = read_coarse2(str(out_path))
    out_path.unlink(missing_ok=True)
    return data


def compute_structure_metrics(nodes: np.ndarray, edges: np.ndarray) -> dict:
    stats = connected_components_stats(nodes, edges)

    n_nodes = len(nodes)
    if n_nodes <= 1 or stats['n_components'] <= 1:
        mean_area_excl_giant = 0.0
    else:
        giant_idx = int(np.argmax(nodes['area_frac']))
        area_excl = np.delete(nodes['area_frac'], giant_idx)
        mean_area_excl_giant = float(area_excl.mean()) if len(area_excl) else 0.0

    # Entropia e Gini da distribuição de área — captura a "forma" da
    # distribuição de tamanhos de nó, independente de conectividade.
    # Hipótese: existe um ε onde a distribuição é mais "rica"
    # (entropia alta = muitos nós de tamanho intermediário e
    # diversificado) versus ε extremos (muitos nós minúsculos quase
    # uniformes, ou poucos nós já dominados por um gigante).
    area = nodes['area_frac']
    area_entropy = shannon_entropy(area) if n_nodes > 0 else 0.0
    area_gini    = gini_coefficient(area) if n_nodes > 0 else 0.0
    # Entropia normalizada pelo máximo possível (log2(n_nodes)) —
    # permite comparar "riqueza relativa" entre ε diferentes, já que
    # o número de nós muda drasticamente
    max_entropy = np.log2(max(n_nodes, 2))
    area_entropy_norm = float(area_entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        'n_nodes':                n_nodes,
        'n_components':            stats['n_components'],
        'giant_frac':              stats['giant_frac'],
        'n_components_size_gt1':  stats['n_components_size_gt1'],
        'mean_area_excl_giant':    mean_area_excl_giant,
        'area_entropy':            area_entropy,
        'area_entropy_norm':       area_entropy_norm,
        'area_gini':               area_gini,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      required=True)
    parser.add_argument('--coarse_bin', default='./coarse2')
    parser.add_argument('--output',     required=True)
    parser.add_argument('--eps_values', type=float, nargs='+',
                        default=[1,2,3,4,6,8,10,12,15,18,22,27,33,40,50])
    parser.add_argument('--work_dir',   default='/tmp')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Imagem: {args.image}')
    print(f'Testando {len(args.eps_values)} valores de ε...')

    rows = []
    for eps in args.eps_values:
        data = materialize_graph(args.image, args.coarse_bin, eps, args.work_dir)
        if data is None:
            continue
        m = compute_structure_metrics(data['nodes'], data['edges'])
        m['epsilon'] = eps
        rows.append(m)
        print(f'  ε={eps:>5.1f}  n_nodes={m["n_nodes"]:>7,}  '
              f'giant_frac={m["giant_frac"]:.3f}  '
              f'entropy_norm={m["area_entropy_norm"]:.3f}  '
              f'gini={m["area_gini"]:.3f}')

    if not rows:
        print('Nenhum valor de ε produziu grafo válido.')
        return

    eps_arr     = np.array([r['epsilon'] for r in rows])
    n_comp      = np.array([r['n_components'] for r in rows])
    giant_f     = np.array([r['giant_frac'] for r in rows])
    n_gt1       = np.array([r['n_components_size_gt1'] for r in rows])
    mean_area   = np.array([r['mean_area_excl_giant'] for r in rows])
    entropy_n   = np.array([r['area_entropy_norm'] for r in rows])
    gini        = np.array([r['area_gini'] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0,0]
    ax.plot(eps_arr, n_comp, 'o-', color='steelblue')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('ε'); ax.set_ylabel('n_components')
    ax.set_title('Número total de componentes (incl. singletons)')
    ax.grid(alpha=0.3, which='both')

    ax = axes[0,1]
    ax.plot(eps_arr, giant_f, 'o-', color='tomato')
    ax.set_xscale('log')
    ax.set_xlabel('ε'); ax.set_ylabel('giant_frac')
    ax.set_title('Fração de nós no maior componente (hub)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
    ax.grid(alpha=0.3)

    ax = axes[0,2]
    ax.plot(eps_arr, n_gt1, 'o-', color='seagreen')
    ax.set_xscale('log')
    ax.set_xlabel('ε'); ax.set_ylabel('n_components (tamanho > 1)')
    ax.set_title('Fragmentos "reais" — excluindo nós isolados')
    ax.grid(alpha=0.3)

    ax = axes[1,0]
    ax.plot(eps_arr, entropy_n, 'o-', color='darkorange')
    ax.set_xscale('log')
    ax.set_xlabel('ε'); ax.set_ylabel('entropia normalizada de área')
    ax.set_title('Riqueza da distribuição de tamanhos\n'
                  '(pico = ε onde a estrutura é mais diversificada)')
    ax.grid(alpha=0.3)
    best_eps = eps_arr[np.argmax(entropy_n)]
    ax.axvline(best_eps, color='red', linestyle='--', alpha=0.5,
               label=f'pico ε={best_eps:.1f}')
    ax.legend(fontsize=8)

    ax = axes[1,1]
    ax.plot(eps_arr, gini, 'o-', color='indigo')
    ax.set_xscale('log')
    ax.set_xlabel('ε'); ax.set_ylabel('Gini de área')
    ax.set_title('Desigualdade na distribuição de tamanhos')
    ax.grid(alpha=0.3)

    ax = axes[1,2]
    ax.plot(eps_arr, mean_area, 'o-', color='purple')
    ax.set_xscale('log')
    ax.set_xlabel('ε'); ax.set_ylabel('área média dos fragmentos (excl. hub)')
    ax.set_title('Substância dos fragmentos fora do hub')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Estrutura do grafo vs ε — {Path(args.image).name}',
                 fontsize=13)
    plt.tight_layout()
    p = out / f'structure_vs_epsilon_{Path(args.image).stem}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSalvo: {p}')

    best_idx = np.argmax(n_gt1)
    print(f'\nPico de fragmentos "reais" em ε={eps_arr[best_idx]:.1f} '
          f'(n_comp_gt1={n_gt1[best_idx]})')
    best_entropy_idx = np.argmax(entropy_n)
    print(f'Pico de entropia normalizada em ε={eps_arr[best_entropy_idx]:.1f} '
          f'(entropy_norm={entropy_n[best_entropy_idx]:.3f})')


if __name__ == '__main__':
    main()
