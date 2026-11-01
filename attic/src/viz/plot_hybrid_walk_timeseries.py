"""
Figura diagnóstica do hybrid walk (janela 5×5, decaimento espacial + cromático).
Mesmo layout do random walk: original | heatmap | série temporal.
Permite comparação visual direta de cobertura e estrutura do sinal.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color as skcolor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'baselines'))
from extract_hybrid_walk_features import (
    _precompute_transitions, _run_walk,
    N_SIZE, N_WALKS, WALK_LEN, SEED,
)

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'images'
OUT_PNG   = ROOT / 'outputs' / 'figures' / 'hybrid_walk_timeseries.png'

STEMS = [
    ('88580', 'Emil Nolde — Expressionism'),
    ('30387', 'Paul Gauguin — Impressionism'),
]


def run_walk_with_visits(stem):
    img  = Image.open(IMAGE_DIR / f'{stem}.jpg').convert('RGB')
    w, h = img.size; side = min(w, h)
    img_sq   = img.crop(((w-side)//2, (h-side)//2,
                          (w-side)//2+side, (h-side)//2+side))
    img_disp = np.array(img_sq.resize((256, 256), Image.BILINEAR)) / 255.0

    arr128   = np.array(img.resize((N_SIZE, N_SIZE), Image.BILINEAR),
                        dtype=np.float32) / 255.0
    lab      = skcolor.rgb2lab(arr128).astype(np.float32)
    lab_flat = lab.reshape(-1, 3)

    nb, probs = _precompute_transitions(lab_flat, N_SIZE, N_SIZE)

    rng    = np.random.default_rng(SEED)
    N      = len(lab_flat)
    signal = np.empty((N_WALKS * WALK_LEN, 3), dtype=np.float32)
    visits = np.zeros(N, dtype=np.int32)

    for w_idx in range(N_WALKS):
        curr = int(rng.integers(0, N))
        base = w_idx * WALK_LEN
        for t in range(WALK_LEN):
            signal[base + t] = lab_flat[curr]
            visits[curr] += 1
            nb_curr = nb[curr]; p_curr = probs[curr]
            valid = nb_curr >= 0
            vp = p_curr[valid]; vp = vp / vp.sum()
            vn = nb_curr[valid]
            curr = int(vn[rng.choice(len(vn), p=vp)])

    visit_map   = visits.reshape(N_SIZE, N_SIZE).astype(float)
    pct_covered = (visits > 0).mean() * 100
    return img_disp, visit_map, signal, pct_covered


def smooth(x, w=32):
    return np.convolve(x, np.ones(w)/w, mode='same')


def main():
    n_cols = len(STEMS)
    fig, axes = plt.subplots(3, n_cols,
                              figsize=(7 * n_cols, 13),
                              gridspec_kw={'height_ratios': [1.4, 1.4, 1],
                                           'hspace': 0.30, 'wspace': 0.12})

    ch_colors = ['#2c3e50', '#e74c3c', '#2980b9']
    ch_labels = ['L (luminance)', 'a (green→red)', 'b (blue→yellow)']

    for col, (stem, label) in enumerate(STEMS):
        print(f'Processing {label}...', flush=True)
        img_disp, visit_map, signal, pct = run_walk_with_visits(stem)

        axes[0, col].imshow(img_disp)
        axes[0, col].set_title(label, fontsize=14, fontweight='bold', pad=8)
        axes[0, col].axis('off')

        im = axes[1, col].imshow(visit_map, cmap='hot', interpolation='nearest')
        axes[1, col].set_title(f'Visit heatmap  ({pct:.1f}% pixels covered)',
                                fontsize=11, pad=6)
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04,
                     label='visit count')

        ax = axes[2, col]
        T  = signal.shape[0]
        t  = np.arange(T) / T
        for w_idx in range(1, N_WALKS):
            ax.axvline(w_idx * WALK_LEN / T, color='#cccccc', lw=0.5, zorder=0)
        for ch, (clr, lbl) in enumerate(zip(ch_colors, ch_labels)):
            ax.plot(t, smooth(signal[:, ch], w=32), color=clr, lw=1.4,
                    alpha=0.85, label=lbl)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Relative position along walk (restarts in gray)', fontsize=11)
        ax.set_ylabel('Lab channel value', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        if col == 0:
            ax.legend(fontsize=10, framealpha=0.85,
                      loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)

    fig.suptitle('Hybrid walk chromatic time series (5×5 window, spatial+chromatic decay)',
                 fontsize=14, fontweight='bold', y=1.01)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Salvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
