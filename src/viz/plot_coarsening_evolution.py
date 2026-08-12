"""
Figura: evolução da percolação cromática de ε_min até ε*.

Para uma imagem-exemplo:
  - Linha 1: imagem original + RAG em 5 thresholds até ε*
  - Linha 2: curva de percolação (n_comp normalizado + giant_frac) com ε* marcado
"""

import subprocess, struct, tempfile, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
from skimage import color as skcolor

# ── Config ─────────────────────────────────────────────────────────────────
STEM      = '55424'          # James Tissot, Realism — contraste claro, referência visual
IMAGE_DIR = Path('./raw/images')
COARSE2   = './coarse2'
EPSILONS  = [2, 4, 6, 8, 10, 15, 22]   # ε values a mostrar (ε*=10 marcado)
EPS_STAR  = 10.0
ALL_EPS   = [2, 4, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40]
OUT_PNG   = Path('./outputs/figures/coarsening_evolution.png')
# ───────────────────────────────────────────────────────────────────────────

dt_percol = np.dtype([
    ('epsilon','<f4'),('n_components','<u4'),('giant_size','<u4'),
    ('giant_frac','<f4'),('entropy','<f4'),('mean_size','<f4'),
    ('gini','<f4'),('n_singletons_frac','<f4'),
])
dt_node = np.dtype([
    ('L','<f4'),('a','<f4'),('b','<f4'),
    ('X','<f4'),('Y','<f4'),
    ('area_frac','<f4'),('area_px','<u4'),
])
dt_edge = np.dtype([
    ('src','<u4'),('dst','<u4'),
    ('color_dist','<f4'),('spatial_dist','<f4'),
])


def run_coarse2(img_path: Path, eps: float, tmp_dir: Path):
    out = tmp_dir / f'{img_path.stem}_e{eps:.0f}.coarse'
    subprocess.run(
        [COARSE2, str(img_path), str(out), '--graph', '--epsilon_star', str(eps)],
        check=True, capture_output=True
    )
    return out


def read_coarse(path: Path):
    with open(path, 'rb') as f:
        mode         = struct.unpack('<I', f.read(4))[0]
        n_thresh     = struct.unpack('<I', f.read(4))[0]
        _eps_star    = struct.unpack('<f', f.read(4))[0] if mode == 1 else None
        steps        = np.frombuffer(f.read(n_thresh * dt_percol.itemsize), dtype=dt_percol).copy()
        nodes, edges = np.zeros(0, dtype=dt_node), np.zeros(0, dtype=dt_edge)
        if mode == 1:
            nn    = struct.unpack('<I', f.read(4))[0]
            nodes = np.frombuffer(f.read(nn * dt_node.itemsize), dtype=dt_node).copy()
            ne    = struct.unpack('<I', f.read(4))[0]
            edges = np.frombuffer(f.read(ne * dt_edge.itemsize), dtype=dt_edge).copy()
    return steps, nodes, edges


def lab_to_rgb(L, a, b):
    lab = np.array([[[L, a, b]]], dtype=np.float64)
    rgb = skcolor.lab2rgb(lab)[0, 0].clip(0, 1)
    return rgb


DISPLAY_SIZE = 400   # resolução de display — cKDTree opera nessa grade

def make_voronoi_image(nodes, max_nodes=3000):
    """
    Imagem segmentada via cKDTree: O(N log K), memória O(N+K).
    Usa resolução DISPLAY_SIZE para evitar matrizes gigantes.
    """
    from scipy.spatial import cKDTree
    n = len(nodes)
    if n == 0:
        return np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3))

    keep = np.argsort(nodes['area_px'])[::-1][:max_nodes] if n > max_nodes else np.arange(n)

    cy = nodes['Y'][keep] * DISPLAY_SIZE
    cx = nodes['X'][keep] * DISPLAY_SIZE
    colors_lab = np.stack([nodes['L'][keep], nodes['a'][keep], nodes['b'][keep]], axis=1)

    gy, gx = np.mgrid[0:DISPLAY_SIZE, 0:DISPLAY_SIZE]
    pts = np.column_stack([gy.ravel().astype(np.float32),
                           gx.ravel().astype(np.float32)])
    tree = cKDTree(np.column_stack([cy, cx]))
    _, labels = tree.query(pts, workers=1)

    seg_lab = colors_lab[labels].reshape(DISPLAY_SIZE, DISPLAY_SIZE, 3)
    return skcolor.lab2rgb(seg_lab).clip(0, 1)


def draw_rag(ax, img_rgb, nodes, edges, eps, eps_star):
    """Segmentação colorida + grafo com seleção e escala adaptativas por n."""
    D = DISPLAY_SIZE
    n = len(nodes)

    if n == 0:
        ax.imshow(img_rgb)
        ax.set_title(f'ε = {eps:.0f}\n(no graph)', fontsize=11)
        ax.axis('off')
        return

    seg = make_voronoi_image(nodes)
    ax.imshow(seg)

    # Nós a desenhar: adaptativo — poucos em ε baixo (muitos nós), mais em ε alto
    # Fórmula: ~4000/sqrt(n), capped em [15, 400]
    n_draw = int(np.clip(4000 / max(np.sqrt(n), 1), 15, 400))

    rng = np.random.default_rng(42)
    if n > n_draw:
        # Amostragem ponderada por área (mais provável exibir regiões maiores)
        weights = nodes['area_px'].astype(float)
        weights /= weights.sum()
        idx = rng.choice(n, size=n_draw, replace=False, p=weights)
    else:
        idx = np.arange(n)

    xs     = (nodes['X'][idx] * D).astype(float)
    ys     = (nodes['Y'][idx] * D).astype(float)
    colors = np.array([lab_to_rgb(nodes['L'][i], nodes['a'][i], nodes['b'][i]) for i in idx])

    # Arestas entre nós amostrados
    idx_set = set(idx.tolist())
    lw = np.clip(400 / max(len(idx_set), 1), 0.3, 1.0)
    for e in edges:
        if e['src'] in idx_set and e['dst'] in idx_set:
            x0 = nodes['X'][e['src']] * D; y0 = nodes['Y'][e['src']] * D
            x1 = nodes['X'][e['dst']] * D; y1 = nodes['Y'][e['dst']] * D
            ax.plot([x0, x1], [y0, y1], '-', color='white', lw=lw, alpha=0.55, zorder=1)

    # Raio: círculos cobrem ~15% da imagem total dividido igualmente
    target_r = np.sqrt((D ** 2) * 0.15 / (np.pi * max(len(idx), 1)))
    target_r = np.clip(target_r, 2.0, D * 0.10)
    areas = nodes['area_px'][idx].astype(float)
    a_med = np.median(areas) if len(areas) else 1.0
    r_scale = target_r / max(np.sqrt(a_med), 1.0)

    for x, y, a, c in zip(xs, ys, areas, colors):
        r = np.clip(np.sqrt(a) * r_scale, 1.5, D * 0.10)
        ax.add_patch(plt.Circle((x, y), r, color=c, ec='black', lw=0.4, zorder=2, alpha=0.9))

    is_star = abs(eps - eps_star) < 0.5
    star_str = '\n← ε*' if is_star else ''
    ax.set_title(f'ε = {eps:.0f}{star_str}\n{n:,} regions', fontsize=11,
                 fontweight='bold' if is_star else 'normal',
                 color='#c0392b' if is_star else 'black')
    ax.set_xlim(0, D); ax.set_ylim(D, 0)
    ax.axis('off')


def main():
    img_path = IMAGE_DIR / f'{STEM}.jpg'
    assert img_path.exists(), f'Imagem não encontrada: {img_path}'

    img_rgb = np.array(Image.open(img_path).convert('RGB')) / 255.0

    # ── Materializar grafos em cada ε ────────────────────────────────────
    tmp = Path(tempfile.mkdtemp())
    try:
        print('Materializando grafos...', flush=True)
        coarse_files = {}
        for eps in EPSILONS:
            print(f'  ε={eps}...', end=' ', flush=True)
            coarse_files[eps] = run_coarse2(img_path, eps, tmp)
            print('ok')

        # Lê curva de percolação do arquivo com menor ε (tem todos os thresholds)
        steps_full, _, _ = read_coarse(coarse_files[EPSILONS[0]])

        # Lê grafos de cada ε
        graphs = {}
        for eps in EPSILONS:
            _, nodes, edges = read_coarse(coarse_files[eps])
            graphs[eps] = (nodes, edges)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── Curva de percolação ───────────────────────────────────────────────
    eps_curve  = steps_full['epsilon'].astype(float)
    nc         = steps_full['n_components'].astype(float)
    nc_norm    = nc / max(nc[0], 1)
    gf         = steps_full['giant_frac'].astype(float)

    # ── Figura ────────────────────────────────────────────────────────────
    n_cols = len(EPSILONS) + 1   # +1 para imagem original
    fig = plt.figure(figsize=(2.3 * n_cols, 6.5))
    gs  = GridSpec(2, n_cols, figure=fig,
                   height_ratios=[1.6, 1],
                   hspace=0.35, wspace=0.06)

    # Linha 1 – imagem original + grafos
    ax0 = fig.add_subplot(gs[0, 0])
    # resize para display (imagem grande fica lenta)
    from PIL import Image as PILImage
    disp = np.array(PILImage.fromarray((img_rgb*255).astype(np.uint8))
                    .resize((400, 400), PILImage.BILINEAR)) / 255.0
    ax0.imshow(disp)
    ax0.set_title('Original', fontsize=12, fontweight='bold')
    ax0.axis('off')

    for col, eps in enumerate(EPSILONS, start=1):
        ax = fig.add_subplot(gs[0, col])
        nodes, edges = graphs[eps]
        draw_rag(ax, img_rgb, nodes, edges, eps, EPS_STAR)

    ax_perc = fig.add_subplot(gs[1, :])

    color_nc = '#2980b9'
    color_gf = '#e67e22'

    ax_perc.plot(eps_curve, nc_norm, 'o-', color=color_nc,
                 lw=1.8, ms=4, label='$n_{comp}$ (normalized)')
    ax_perc.set_ylabel('$n_{comp} / n_{comp}(ε=2)$', color=color_nc, fontsize=11)
    ax_perc.tick_params(axis='y', labelcolor=color_nc, labelsize=10)

    ax2 = ax_perc.twinx()
    ax2.plot(eps_curve, gf, 's--', color=color_gf,
             lw=1.8, ms=4, label='Giant component fraction')
    ax2.set_ylabel('Giant component fraction', color=color_gf, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_gf, labelsize=10)
    ax2.set_ylim(0, 1.05)

    ax_perc.axvline(EPS_STAR, color='#c0392b', lw=1.8, ls='--', zorder=5)
    ax_perc.text(EPS_STAR + 0.4, 0.85, f'ε* = {EPS_STAR:.0f}',
                 color='#c0392b', fontsize=11, va='top')

    for eps in EPSILONS:
        ax_perc.axvline(eps, color='gray', lw=0.6, ls=':', alpha=0.5)

    ax_perc.set_xlabel('Threshold ε (ΔE CIE-Lab)', fontsize=11)
    ax_perc.set_xlim(eps_curve[0] - 0.5, eps_curve[-1] + 0.5)
    ax_perc.set_xticks(ALL_EPS)
    ax_perc.set_xticklabels(ALL_EPS, fontsize=9)
    ax_perc.set_ylim(-0.02, 1.05)

    lines1, labels1 = ax_perc.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_perc.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center', bbox_to_anchor=(0.5, -0.18),
                   ncol=2, fontsize=11, framealpha=0.85)

    fig.suptitle(
        f'Chromatic percolation — James Tissot, Realism  (stem {STEM})',
        fontsize=13, y=0.98
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'\nSalvo: {OUT_PNG}')

    # ── Salvar painéis separados ───────────────────────────────────────────
    # Strip (linha 1 apenas)
    fig_strip = plt.figure(figsize=(2.3 * n_cols, 2.8))
    gs_s = GridSpec(1, n_cols, figure=fig_strip, wspace=0.06)
    ax0s = fig_strip.add_subplot(gs_s[0, 0])
    ax0s.imshow(disp); ax0s.set_title('Original', fontsize=12, fontweight='bold'); ax0s.axis('off')
    for col, eps in enumerate(EPSILONS, start=1):
        ax = fig_strip.add_subplot(gs_s[0, col])
        nodes, edges = graphs[eps]
        draw_rag(ax, img_rgb, nodes, edges, eps, EPS_STAR)
    fig_strip.suptitle(f'Chromatic percolation — James Tissot, Realism  (stem {STEM})',
                       fontsize=13, y=1.01)
    p_strip = OUT_PNG.parent / 'coarsening_strip.png'
    fig_strip.savefig(p_strip, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig_strip)
    print(f'Salvo: {p_strip}')

    # Curvas (linha 2 apenas)
    fig_curves, ax_c = plt.subplots(figsize=(9, 4))
    ax_c.plot(eps_curve, nc_norm, 'o-', color=color_nc,
              lw=1.8, ms=4, label='$n_{comp}$ (normalized)')
    ax_c.set_ylabel('$n_{comp} / n_{comp}(ε=2)$', color=color_nc, fontsize=11)
    ax_c.tick_params(axis='y', labelcolor=color_nc, labelsize=10)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(eps_curve, gf, 's--', color=color_gf,
               lw=1.8, ms=4, label='Giant component fraction')
    ax_c2.set_ylabel('Giant component fraction', color=color_gf, fontsize=11)
    ax_c2.tick_params(axis='y', labelcolor=color_gf, labelsize=10); ax_c2.set_ylim(0, 1.05)
    ax_c.axvline(EPS_STAR, color='#c0392b', lw=1.8, ls='--', zorder=5)
    ax_c.text(EPS_STAR + 0.4, 0.85, f'ε* = {EPS_STAR:.0f}', color='#c0392b', fontsize=11, va='top')
    for eps in EPSILONS:
        ax_c.axvline(eps, color='gray', lw=0.6, ls=':', alpha=0.5)
    ax_c.set_xlabel('Threshold ε (ΔE CIE-Lab)', fontsize=11)
    ax_c.set_xlim(eps_curve[0] - 0.5, eps_curve[-1] + 0.5)
    ax_c.set_xticks(ALL_EPS); ax_c.set_xticklabels(ALL_EPS, fontsize=9)
    ax_c.set_ylim(-0.02, 1.05)
    l1, lb1 = ax_c.get_legend_handles_labels(); l2, lb2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(l1+l2, lb1+lb2, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=11, framealpha=0.85)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    p_curves = OUT_PNG.parent / 'coarsening_curves.png'
    fig_curves.savefig(p_curves, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig_curves)
    print(f'Salvo: {p_curves}')

    plt.close(fig)


if __name__ == '__main__':
    main()