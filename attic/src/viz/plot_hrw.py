"""
Visualização do Hierarchical Random Walk (hrw.py do browser Claude) em pinturas reais.

Algoritmo (hrw.py):
  - Walk 4-conectado no grid de pixels (leaf level)
  - Passo LIVRE se ΔE < tau → d=0
  - TRAP (nenhum vizinho passa no gate) → sobe quadtree para achar bloco
    com pixels não-visitados → d = níveis escalados
  - D[t] = custo de escape por passo (0=livre, k=escalou k níveis)

Layout por imagem (4 linhas × 2 colunas):
  - Original | depth_map (d por pixel)
  - d(t) série temporal | histograma de run lengths

Saída: outputs/figures/hrw_paintings.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from skimage import color as skcolor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from hrw import hierarchical_walk, depth_map as make_depth_map

IMAGE_DIR = ROOT / 'raw' / 'images'
OUT_PNG   = ROOT / 'outputs' / 'figures' / 'hrw_paintings.png'

MAX_LEVEL = 6    # 64×64 leaf grid (4×4 px each em 256×256)
TAU       = 10.0
SEED      = 7

STEMS = [
    ('75232', 'Color Field Painting\n(uniform red — expected: long runs, low d)'),
    ('86727', 'Abstract Expressionism\n(textured blue+red — expected: short runs, high d)'),
]


def load_lab(stem, size=None):
    side = 1 << MAX_LEVEL  # 64
    img  = Image.open(IMAGE_DIR / f'{stem}.jpg').convert('RGB')
    w, h = img.size; sq = min(w, h)
    img_sq = img.crop(((w-sq)//2, (h-sq)//2, (w-sq)//2+sq, (h-sq)//2+sq))
    arr    = np.array(img_sq.resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.
    lab    = skcolor.rgb2lab(arr).astype(np.float32)
    img_disp = np.array(img_sq.resize((256, 256), Image.BILINEAR)) / 255.
    return lab, img_disp


def stats(D, run_lengths):
    return {
        'mean_d':       float(D.mean()),
        'trap_rate':    float(np.mean(D > 0)),
        'max_d':        float(D.max()),
        'mean_run_len': float(np.mean(run_lengths)) if run_lengths else 0.,
        'med_run_len':  float(np.median(run_lengths)) if run_lengths else 0.,
    }


def main():
    n_cases = len(STEMS)

    fig = plt.figure(figsize=(13 * n_cases // n_cases * n_cases, 22))
    fig.set_size_inches(13, 22)
    outer = gridspec.GridSpec(n_cases, 1, figure=fig, hspace=0.55)

    print(f"{'image':>30} | {'mean_d':>7} | {'trap%':>6} | {'max_d':>5} | {'mean_run':>8} | {'med_run':>7}")
    print('-' * 80)

    for case_i, (stem, label) in enumerate(STEMS):
        print(f'Processing {stem}...', flush=True)
        lab, img_disp = load_lab(stem)
        F, D, path, run_lengths = hierarchical_walk(lab, MAX_LEVEL, TAU, seed=SEED)
        dm = make_depth_map(path, D, 1 << MAX_LEVEL)
        st = stats(D, run_lengths)

        title = label.split('\n')[0]
        print(f"{title:>30} | {st['mean_d']:>7.3f} | {st['trap_rate']*100:>5.1f}% | "
              f"{st['max_d']:>5.0f} | {st['mean_run_len']:>8.2f} | {st['med_run_len']:>7.1f}")

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[case_i],
            wspace=0.30, hspace=0.45,
            width_ratios=[1, 1.1, 1.1]
        )

        # ── col 0: imagem original ──────────────────────────────────────────
        ax_img = fig.add_subplot(inner[0, 0])
        ax_img.imshow(img_disp)
        ax_img.set_title(label, fontsize=11, fontweight='bold', pad=6)
        ax_img.axis('off')

        # ── col 1: depth map ────────────────────────────────────────────────
        ax_dm = fig.add_subplot(inner[0, 1])
        side  = 1 << MAX_LEVEL
        im = ax_dm.imshow(dm, cmap='hot', interpolation='nearest',
                          vmin=0, vmax=max(dm[~np.isnan(dm)].max(), 1))
        ax_dm.set_title(
            f'Escape depth d (mean={st["mean_d"]:.2f}, trap={st["trap_rate"]*100:.1f}%)',
            fontsize=10, pad=5)
        ax_dm.axis('off')
        plt.colorbar(im, ax=ax_dm, fraction=0.046, pad=0.04, label='d (levels climbed)')

        # ── col 2: visit order ──────────────────────────────────────────────
        visit_map = np.full((side, side), np.nan)
        for t, (r, c) in enumerate(path):
            visit_map[r, c] = t
        ax_vo = fig.add_subplot(inner[0, 2])
        im2 = ax_vo.imshow(visit_map, cmap='viridis', interpolation='nearest')
        ax_vo.set_title('Visit order', fontsize=10, pad=5)
        ax_vo.axis('off')
        plt.colorbar(im2, ax=ax_vo, fraction=0.046, pad=0.04)

        # ── d(t) time series ────────────────────────────────────────────────
        ax_dt = fig.add_subplot(inner[1, 0:2])
        T     = len(D)
        t_ax  = np.arange(T)
        # show raw d(t) as stem-like bars (sparse events)
        ax_dt.fill_between(t_ax, D, alpha=0.6, color='#e74c3c', linewidth=0)
        # smoothed trap density (running mean over 64-step window)
        w_sm = max(T // 64, 1)
        smooth_d = np.convolve(D, np.ones(w_sm)/w_sm, mode='same')
        ax_dt.plot(t_ax, smooth_d, color='#c0392b', lw=1.5, label=f'smoothed (w={w_sm})')
        ax_dt.set_xlim(0, T)
        ax_dt.set_xlabel('Visit index t', fontsize=10)
        ax_dt.set_ylabel('Escape depth d(t)', fontsize=10)
        ax_dt.set_title(f'd(t) — bursts = traps (max_d={st["max_d"]:.0f})', fontsize=10)
        ax_dt.legend(fontsize=8, loc='upper right')
        ax_dt.spines[['top', 'right']].set_visible(False)

        # ── run length histogram ─────────────────────────────────────────────
        ax_rl = fig.add_subplot(inner[1, 2])
        rl    = np.array(run_lengths)
        rl    = rl[rl > 0]
        if len(rl):
            bins = min(40, max(rl) + 1)
            ax_rl.hist(rl, bins=bins, color='#2980b9', alpha=0.8, edgecolor='white', lw=0.3)
        ax_rl.set_xlabel('Free-roam run length', fontsize=10)
        ax_rl.set_ylabel('Count', fontsize=10)
        ax_rl.set_title(
            f'Run lengths (mean={st["mean_run_len"]:.1f}, med={st["med_run_len"]:.0f})',
            fontsize=10)
        ax_rl.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f'Hierarchical Random Walk (τ={TAU}, L={MAX_LEVEL}, {1<<MAX_LEVEL}×{1<<MAX_LEVEL} grid)\n'
        'd=0: free step  |  d>0: escaped trap by climbing d quadtree levels',
        fontsize=13, fontweight='bold', y=1.01)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSalvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
