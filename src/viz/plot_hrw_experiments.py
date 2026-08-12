"""
Experimentos com o Hierarchical Random Walk (hrw.py + hrw_diag.py).

Experimento 1 — três estilos contrastantes (MAX_LEVEL=6):
  Color Field (uniforme) | Pointillism (pontos) | Abstract Exp. (textura densa)
  Métrica-chave: trap_content = trap_raw - geometric_floor
  (remove o custo de self-avoidance intrínseco ao SAW, isola o sinal cromático)

Experimento 2 — sweep de resolução (L=6,7,8) numa pintura texturizada:
  Mostra como trap_content evolui com resolução de grade.

Saída: outputs/figures/hrw_experiments.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from skimage import color as skcolor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))           # hrw.py
sys.path.insert(0, str(Path(__file__).parent.parent / 'baselines'))  # hrw_diag.py

from hrw import hierarchical_walk, depth_map as make_depth_map
from hrw_diag import geometric_floor, content_stats

IMAGE_DIR = ROOT / 'raw' / 'images'
OUT_PNG   = ROOT / 'outputs' / 'figures' / 'hrw_experiments.png'

TAU  = 10.0
SEED = 7

EXP1_STEMS = [
    ('75232', 'Color Field\n(uniform red)'),
    ('44172', 'Pointillism\n(dot texture)'),
    ('42954', 'Abstract Exp.\n(dense texture)'),
]
EXP2_STEM   = ('42954', 'Abstract Expressionism')
EXP2_LEVELS = [4,5,6]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_lab(stem, max_level):
    side = 1 << max_level
    img  = Image.open(IMAGE_DIR / f'{stem}.jpg').convert('RGB')
    w, h = img.size; sq = min(w, h)
    img_sq = img.crop(((w-sq)//2, (h-sq)//2, (w-sq)//2+sq, (h-sq)//2+sq))
    arr    = np.array(img_sq.resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.
    lab    = skcolor.rgb2lab(arr).astype(np.float32)
    img_disp = np.array(img_sq.resize((128, 128), Image.BILINEAR)) / 255.
    return lab, img_disp


def run_one(stem, max_level, floor_trap=None, floor_d=None, n_floor_seeds=5):
    lab, img_disp = load_lab(stem, max_level)
    side = 1 << max_level

    if floor_trap is None:
        floor_trap, floor_d = geometric_floor(side, max_level, n_seeds=n_floor_seeds)

    st = content_stats(lab, max_level, TAU, floor_trap, floor_d, seed=SEED)
    F, D, path, _ = hierarchical_walk(lab, max_level, TAU, seed=SEED)
    dm = make_depth_map(path, D, side)
    return img_disp, dm, F, D, st, floor_trap, floor_d


def draw_case(fig, gs_cell, img_disp, dm, D, st, floor_trap, title):
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_cell, hspace=0.50)

    # row 0: image + stats text
    ax_img = fig.add_subplot(inner[0])
    ax_img.imshow(img_disp)
    ax_img.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax_img.axis('off')

    stat_txt = (
        f"trap_raw  = {st['trap_raw']*100:.1f}%\n"
        f"floor     = {floor_trap*100:.1f}%\n"
        f"trap_cnt  = {st['trap_content']*100:+.1f}%\n"
        f"mean_d    = {st['mean_d_raw']:.3f}\n"
        f"rl_mean   = {st['rl_mean']:.1f}\n"
        f"frac_len1 = {st['frac_len1']*100:.1f}%"
    )
    ax_img.text(1.04, 0.5, stat_txt, transform=ax_img.transAxes,
                fontsize=8, va='center', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='#f8f9fa', ec='#cccccc'))

    # row 1: depth map
    ax_dm = fig.add_subplot(inner[1])
    vmax  = max(float(np.nanmax(dm)), 1)
    im    = ax_dm.imshow(dm, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    ax_dm.set_title('Escape depth d', fontsize=9, pad=3)
    ax_dm.axis('off')
    plt.colorbar(im, ax=ax_dm, fraction=0.046, pad=0.04)

    # row 2: d(t) smoothed
    ax_dt = fig.add_subplot(inner[2])
    T     = len(D)
    t_ax  = np.arange(T)
    ax_dt.fill_between(t_ax, D, alpha=0.4, color='#e74c3c', linewidth=0)
    w_sm  = max(T // 80, 1)
    smooth = np.convolve(D, np.ones(w_sm)/w_sm, mode='same')
    ax_dt.plot(t_ax, smooth, color='#c0392b', lw=1.2)
    ax_dt.set_xlim(0, T)
    ax_dt.set_ylabel('d(t)', fontsize=8)
    ax_dt.set_xlabel('t', fontsize=8)
    ax_dt.tick_params(labelsize=7)
    ax_dt.spines[['top', 'right']].set_visible(False)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Pre-compute geometric floors per level (image-independent)
    floors = {}
    for lv in sorted(set([6] + EXP2_LEVELS)):
        side = 1 << lv
        print(f'  geometric_floor L={lv} ({side}×{side})...', end='', flush=True)
        t0 = time.time()
        ft, fd = geometric_floor(side, lv, n_seeds=5)
        print(f' floor_trap={ft*100:.1f}%  floor_d={fd:.3f}  ({time.time()-t0:.1f}s)')
        floors[lv] = (ft, fd)

    fig = plt.figure(figsize=(18, 16))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55,
                              height_ratios=[1.05, 1.0])

    # ── Experiment 1 ─────────────────────────────────────────────────────────
    print('\n[Experiment 1] Three contrasting styles at L=6')
    exp1_gs = gridspec.GridSpecFromSubplotSpec(
        1, len(EXP1_STEMS), subplot_spec=outer[0], wspace=0.42)

    ft6, fd6 = floors[6]
    for col, (stem, label) in enumerate(EXP1_STEMS):
        print(f'  {stem} ({label.splitlines()[0]})...', flush=True)
        img_disp, dm, F, D, st, _, _ = run_one(
            stem, 6, floor_trap=ft6, floor_d=fd6)
        draw_case(fig, exp1_gs[col], img_disp, dm, D, st, ft6,
                  f'L=6  {label}')
        print(f'    trap_cnt={st["trap_content"]*100:+.1f}%  '
              f'rl_mean={st["rl_mean"]:.1f}  frac_len1={st["frac_len1"]*100:.1f}%')

    fig.text(0.02, 0.96, 'Exp 1 — Contrasting styles (τ=10, L=6)',
             fontsize=12, fontweight='bold', va='top')

    # ── Experiment 2 ─────────────────────────────────────────────────────────
    stem2, label2 = EXP2_STEM
    print(f'\n[Experiment 2] Resolution sweep — {label2}')
    exp2_gs = gridspec.GridSpecFromSubplotSpec(
        1, len(EXP2_LEVELS), subplot_spec=outer[1], wspace=0.42)

    for col, lv in enumerate(EXP2_LEVELS):
        ft, fd = floors[lv]
        print(f'  L={lv}...', flush=True)
        img_disp, dm, F, D, st, _, _ = run_one(stem2, lv, floor_trap=ft, floor_d=fd)
        side = 1 << lv
        draw_case(fig, exp2_gs[col], img_disp, dm, D, st, ft,
                  f'L={lv}  ({side}×{side} grid)\ntrap_cnt={st["trap_content"]*100:+.1f}%')
        print(f'    L={lv}: trap_cnt={st["trap_content"]*100:+.1f}%  '
              f'rl_mean={st["rl_mean"]:.1f}')

    fig.text(0.02, 0.49, f'Exp 2 — Resolution sweep on {label2} (τ=10)',
             fontsize=12, fontweight='bold', va='top')

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSalvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
