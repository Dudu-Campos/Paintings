"""
Figura: Hilbert curve como série temporal em duas pinturas contrastantes.

Layout (2 colunas, uma por pintura):
  Linha 1 — imagem original redimensionada
  Linha 2 — imagem colorida pelo índice Hilbert (path visual)
  Linha 3 — série temporal L, a, b ao longo da curva (suavizada)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from skimage import color as skcolor

ROOT      = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / 'raw' / 'images'
OUT_PNG   = ROOT / 'outputs' / 'figures' / 'hilbert_timeseries.png'
N_SIZE    = 128   # mesmo do pipeline

# Duas pinturas contrastantes: escolher manualmente ou por estilo
# Sugestão: uma fria/azulada e uma quente/dourada
STEMS = [
    ('88580', 'Emil Nolde — Expressionism'),    # cores vibrantes, saturadas
    ('30387', 'Paul Gauguin — Impressionism'),  # tons quentes, tropicais
]


def build_hilbert_indices(n):
    N = n * n
    d = np.arange(N, dtype=np.int64)
    x = np.zeros(N, dtype=np.int32)
    y = np.zeros(N, dtype=np.int32)
    s = 1
    while s < n:
        rx = ((d >> 1) & 1).astype(np.int32)
        ry = ((d & 1) ^ rx).astype(np.int32)
        mask0 = ry == 0
        flip  = mask0 & (rx == 1)
        x[flip] = s - 1 - x[flip]
        y[flip] = s - 1 - y[flip]
        x[mask0], y[mask0] = y[mask0].copy(), x[mask0].copy()
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return y, x  # rows, cols


def smooth(signal, w=64):
    """Moving average para tornar o sinal legível na figura."""
    kernel = np.ones(w) / w
    return np.convolve(signal, kernel, mode='same')


def make_hilbert_path_image(lab, rows, cols, n):
    """
    Coloriza cada pixel pela sua posição relativa na curva de Hilbert
    (índice 0→1 mapeado em colormap) para visualizar o caminho.
    """
    idx_norm = np.linspace(0, 1, n * n)
    cmap     = plt.get_cmap('plasma')
    colors   = cmap(idx_norm)[:, :3]   # (N², 3) RGB

    path_img = np.zeros((n, n, 3), dtype=np.float32)
    path_img[rows, cols] = colors
    return path_img


def load_and_process(stem):
    img_path = IMAGE_DIR / f'{stem}.jpg'
    img      = Image.open(img_path).convert('RGB')

    # Original (preserva aspect ratio, crop central para display quadrado)
    w, h     = img.size
    side     = min(w, h)
    left     = (w - side) // 2
    top      = (h - side) // 2
    img_sq   = img.crop((left, top, left + side, top + side))
    img_disp = np.array(img_sq.resize((256, 256), Image.BILINEAR)) / 255.0

    # Lab 128×128 para a curva
    img128   = np.array(img.resize((N_SIZE, N_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    lab      = skcolor.rgb2lab(img128).astype(np.float32)

    rows, cols = build_hilbert_indices(N_SIZE)
    signal     = lab[rows, cols, :]  # (16384, 3)

    path_img   = make_hilbert_path_image(lab, rows, cols, N_SIZE)

    return img_disp, path_img, signal, rows, cols


def main():
    rows_h, cols_h = build_hilbert_indices(N_SIZE)  # precalcula uma vez

    n_cols = len(STEMS)
    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 11),
                              gridspec_kw={'height_ratios': [1.6, 1], 'hspace': 0.25, 'wspace': 0.12})

    ch_colors = ['#2c3e50', '#e74c3c', '#2980b9']
    ch_labels = ['L (luminance)', 'a (green→red)', 'b (blue→yellow)']

    for col, (stem, label) in enumerate(STEMS):
        img_disp, path_img, signal, _, _ = load_and_process(stem)

        axes[0, col].imshow(img_disp)
        axes[0, col].set_title(label, fontsize=15, fontweight='bold', pad=10)
        axes[0, col].axis('off')

        ax = axes[1, col]
        T  = signal.shape[0]
        t  = np.arange(T) / T

        for ch, (clr, lbl) in enumerate(zip(ch_colors, ch_labels)):
            s = smooth(signal[:, ch], w=128)
            ax.plot(t, s, color=clr, lw=1.6, alpha=0.85, label=lbl)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Relative position along Hilbert curve', fontsize=13)
        ax.set_ylabel('Lab channel value', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        if col == 0:
            ax.legend(fontsize=12, framealpha=0.85,
                      loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3)

    fig.suptitle('Chromatic time series via Hilbert space-filling curve',
                 fontsize=15, fontweight='bold', y=1.01)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Salvo: {OUT_PNG}')


if __name__ == '__main__':
    main()
