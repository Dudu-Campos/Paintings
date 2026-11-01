"""
Protótipo de visualização do Quadtree Self-Avoiding Walk.

Estrutura:
  - L níveis de quadtree (raiz = imagem inteira, folhas = células NxN)
  - Conexões horizontais: restritas ao mesmo bloco 2×2 do pai (evita
    grafo totalmente conectado por nível, que tornaria d(t) constante)
  - Walk: a cada célula, visita vizinhos horizontais não visitados em
    ordem cromática (menor ΔE Lab primeiro). Quando nenhum disponível,
    DESCE para os filhos (d_counter += 1).
  - Folha visitada → emite (L, a, b, d_counter, r, c)

d(t) captura: quantas descidas ocorreram antes de visitar a folha t.
Em regiões homogêneas, o walk percorre toda a vizinhança antes de descer
(as células são similares → são visitadas no mesmo passo). Em regiões
texturizadas, pode descer mais cedo se a vizinhança é cromaticamente
diversa e o walk segue a menor ΔE que o leva para longe.

Saída: outputs/figures/quadtree_saw.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color as skcolor

sys.setrecursionlimit(10000)

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'images'
OUT_PNG   = ROOT / 'outputs' / 'figures' / 'quadtree_saw.png'

MAX_LEVEL = 5   # 2^5 = 32 → folhas 4×4px, 1024 folhas (comparável a Hilbert 32×32)
SEED      = 42

STEMS = [
    ('75232', 'Color Field Painting — uniform'),
    ('86727', 'Abstract Expressionism — textured'),
]


def quadtree_saw(lab, max_level=MAX_LEVEL, seed=SEED):
    """
    Self-avoiding walk cromático no quadtree.
    Retorna lista de (L, a, b, d, row_leaf, col_leaf) para cada folha.
    """
    rng = np.random.default_rng(seed)
    H   = lab.shape[0]
    assert H == lab.shape[1] and (H & (H - 1)) == 0, "precisa de imagem quadrada, potência de 2"

    results   = []
    d_counter = [0]
    visited   = set()

    def cell_lab(level, r, c):
        s = H >> level
        return lab[r*s:(r+1)*s, c*s:(c+1)*s].mean(axis=(0, 1))

    def siblings(level, r, c):
        """4 vizinhos restritos ao mesmo bloco 2×2 do pai (conexões dentro do pai)."""
        if level == 0:
            return []
        # bloco do pai: canto superior-esquerdo = (r//2*2, c//2*2)
        pr, pc = (r // 2) * 2, (c // 2) * 2
        sib = [(pr + dr, pc + dc) for dr in range(2) for dc in range(2)
               if not (pr + dr == r and pc + dc == c)]
        return sib

    def dfs(level, r, c):
        key = (level, r, c)
        if key in visited:
            return
        visited.add(key)

        if level == max_level:
            L, a, b = cell_lab(level, r, c)
            results.append((L, a, b, d_counter[0], r, c))
            return

        # Vizinhos horizontais (irmãos no mesmo pai) não visitados,
        # ordenados por similaridade cromática crescente (menor ΔE primeiro)
        cur_lab = cell_lab(level, r, c)
        sibs    = [(nr, nc) for nr, nc in siblings(level, r, c)
                   if (level, nr, nc) not in visited]

        if sibs:
            dE = [np.sqrt(np.sum((cell_lab(level, nr, nc) - cur_lab) ** 2))
                  for nr, nc in sibs]
            # Soft ordering: sorteia com probabilidade ∝ exp(-dE/σ), σ = mediana
            sigma = max(float(np.median(dE)), 1.0)
            probs = np.exp(-np.array(dE) / sigma)
            probs = probs / probs.sum()
            order = rng.choice(len(sibs), size=len(sibs), replace=False, p=probs)
            for idx in order:
                nr, nc = sibs[idx]
                if (level, nr, nc) not in visited:
                    dfs(level, nr, nc)

        # Descida para os filhos
        d_counter[0] += 1
        children = [(2*r + dr, 2*c + dc) for dr in range(2) for dc in range(2)]
        # Ordena filhos por similaridade com célula atual (cromática)
        child_labs = [cell_lab(level + 1, cr, cc) for cr, cc in children]
        child_dE   = [np.sqrt(np.sum((cl - cur_lab) ** 2)) for cl in child_labs]
        sigma_c    = max(float(np.median(child_dE)), 1.0)
        probs_c    = np.exp(-np.array(child_dE) / sigma_c)
        probs_c    = probs_c / probs_c.sum()
        c_order    = rng.choice(len(children), size=len(children),
                                replace=False, p=probs_c)
        for idx in c_order:
            cr, cc = children[idx]
            dfs(level + 1, cr, cc)

    dfs(0, 0, 0)
    return results


def load_lab(stem, size=128):
    img  = Image.open(IMAGE_DIR / f'{stem}.jpg').convert('RGB')
    arr  = np.array(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.
    return skcolor.rgb2lab(arr).astype(np.float32)


def make_leaf_maps(results, max_level=MAX_LEVEL):
    """Constrói mapas 2D de ordem de visita e profundidade a partir dos resultados."""
    grid = 1 << max_level   # 2^max_level
    visit_map = np.zeros((grid, grid), dtype=np.float32)
    depth_map = np.zeros((grid, grid), dtype=np.float32)
    for t, (L, a, b, d, r, c) in enumerate(results):
        visit_map[r, c] = t
        depth_map[r, c] = d
    return visit_map, depth_map


def main():
    n_cols = len(STEMS)
    fig, axes = plt.subplots(4, n_cols,
                              figsize=(7 * n_cols, 16),
                              gridspec_kw={'height_ratios': [1.4, 1, 1, 1.2],
                                           'hspace': 0.35, 'wspace': 0.12})

    for col, (stem, label) in enumerate(STEMS):
        print(f'Processing {label}...', flush=True)
        lab  = load_lab(stem, size=128)
        res  = quadtree_saw(lab)
        print(f'  {len(res)} folhas visitadas', flush=True)

        visit_map, depth_map = make_leaf_maps(res)

        # Série temporal
        arr   = np.array(res)
        F_L, F_a, F_b, D = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        T     = len(res)
        t_ax  = np.arange(T) / T

        # Imagem original para display
        img   = Image.open(IMAGE_DIR / f'{stem}.jpg').convert('RGB')
        w, h  = img.size; side = min(w, h)
        img_sq = img.crop(((w-side)//2, (h-side)//2,
                            (w-side)//2+side, (h-side)//2+side))
        img_disp = np.array(img_sq.resize((256, 256), Image.BILINEAR)) / 255.

        # Linha 0 — imagem original
        axes[0, col].imshow(img_disp)
        axes[0, col].set_title(label, fontsize=13, fontweight='bold', pad=8)
        axes[0, col].axis('off')

        # Linha 1 — mapa de ordem de visita
        im1 = axes[1, col].imshow(visit_map, cmap='viridis', interpolation='nearest')
        axes[1, col].set_title('Visit order', fontsize=11, pad=5)
        axes[1, col].axis('off')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Linha 2 — mapa de profundidade d(t)
        im2 = axes[2, col].imshow(depth_map, cmap='hot', interpolation='nearest')
        axes[2, col].set_title('Depth d(t) at visit', fontsize=11, pad=5)
        axes[2, col].axis('off')
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

        # Linha 3 — séries temporais L, a, b e d(t)
        ax = axes[3, col]
        w_ = max(T // 64, 1)
        ker = np.ones(w_) / w_

        ax2 = ax.twinx()
        ax2.plot(t_ax, D, color='#95a5a6', lw=0.8, alpha=0.5, label='d(t)')
        ax2.set_ylabel('depth d(t)', fontsize=9, color='#7f8c8d')
        ax2.tick_params(labelsize=8, colors='#7f8c8d')

        ch_colors = ['#2c3e50', '#e74c3c', '#2980b9']
        ch_labels = ['L', 'a', 'b']
        for ch, (clr, lbl), sig in zip(range(3), zip(ch_colors, ch_labels),
                                        [F_L, F_a, F_b]):
            ax.plot(t_ax, np.convolve(sig, ker, mode='same'),
                    color=clr, lw=1.4, alpha=0.85, label=lbl)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Relative leaf visit order', fontsize=10)
        ax.set_ylabel('Lab value', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines[['top']].set_visible(False)

        if col == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=9, loc='upper center',
                      bbox_to_anchor=(0.5, -0.22), ncol=4)

    fig.suptitle(
        f'Quadtree SAW (L={MAX_LEVEL}, {1 << MAX_LEVEL}×{1 << MAX_LEVEL} leaves)\n'
        'Visit order, depth map d(t), and chromatic time series',
        fontsize=13, fontweight='bold', y=1.01)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Salvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
