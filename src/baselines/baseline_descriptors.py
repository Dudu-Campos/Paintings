"""
baseline_descriptors.py — Baseline com descritores clássicos de imagem.

Descritores:
    1. Histograma Lab (30 bins × 3 canais = 90 features)
    2. Estatísticas globais de cor (média, std, skewness por canal = 9 features)
    3. LBP — Local Binary Pattern (~26 features de textura local)
    4. Haralick via GLCM (13 features de textura estatística)

Total: ~138 features — mais que as 36 de percolação.
Avaliação idêntica: LDA com 5-fold CV estratificado.

Uso:
    python baseline_descriptors.py \
        --image_dir /path/to/raw \
        --meta      /path/to/meta.csv \
        --output    ./plots \
        --min_count 30 \
        --n_sample  5000   # imagens por estilo se muito grande
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.transform import resize as sk_resize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

Image.MAX_IMAGE_PIXELS = None
TARGET_SIDE = 128   # resolução reduzida — descritores são rápidos mas 100k imagens é muito


# ── Descritores ───────────────────────────────────────────────────────────────

def descriptor_color_hist(image_lab: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Histograma normalizado de cada canal Lab — 90 features."""
    feats = []
    for c in range(3):
        h, _ = np.histogram(image_lab[:,:,c].ravel(), bins=n_bins, density=True)
        feats.append(h)
    return np.concatenate(feats)


def descriptor_color_stats(image_lab: np.ndarray) -> np.ndarray:
    """Média, std, skewness por canal Lab — 9 features."""
    from scipy.stats import skew
    feats = []
    for c in range(3):
        ch = image_lab[:,:,c].ravel().astype(float)
        feats.extend([ch.mean(), ch.std(), float(skew(ch))])
    return np.array(feats, dtype=np.float32)


def descriptor_lbp(image_gray: np.ndarray,
                    radius: int = 2, n_points: int = 16) -> np.ndarray:
    """
    Local Binary Pattern — histograma de padrões de textura local.
    ~n_points+2 features (padrões uniformes).
    """
    lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
    n_bins = n_points + 2
    h, _ = np.histogram(lbp.ravel(), bins=n_bins,
                         range=(0, n_bins), density=True)
    return h.astype(np.float32)


def descriptor_haralick(image_gray: np.ndarray) -> np.ndarray:
    """
    Haralick via GLCM — 13 propriedades de textura estatística.
    """
    # Quantizar para 64 níveis para velocidade
    img_q = (image_gray / 4).astype(np.uint8).clip(0, 63)
    distances  = [1, 2]
    angles     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_q, distances=distances, angles=angles,
                         levels=64, symmetric=True, normed=True)

    props = ['contrast','dissimilarity','homogeneity',
             'energy','correlation','ASM']
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop)  # (n_distances, n_angles)
        feats.extend([vals.mean(), vals.std()])

    # Entropia do GLCM
    p = glcm.ravel()
    p = p[p > 0]
    feats.append(float(-np.sum(p * np.log2(p))))

    return np.array(feats, dtype=np.float32)


def extract_descriptors(image_path: str) -> np.ndarray | None:
    """
    Extrai todos os descritores de uma imagem.
    Retorna vetor (138,) ou None se erro.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        W, H = img.size
        scale = TARGET_SIDE / max(W, H)
        if scale != 1.0:
            img = img.resize((max(1,int(W*scale)), max(1,int(H*scale))),
                              Image.LANCZOS)
        arr = np.array(img)
        lab = color.rgb2lab(arr)
        gray = color.rgb2gray(arr)

        return np.concatenate([
            descriptor_color_hist(lab),
            descriptor_color_stats(lab),
            descriptor_lbp(gray),
            descriptor_haralick(gray),
        ])
    except Exception:
        return None


# ── Meta CSV ──────────────────────────────────────────────────────────────────

def load_meta(meta_path: str) -> dict:
    lookup = {}
    try:
        with open(meta_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row.get('filename','')).stem
                lookup[stem] = {
                    'style':  row.get('style',  'unknown').strip(),
                    'artist': row.get('artist', 'unknown').strip(),
                    'genre':  row.get('genre',  'unknown').strip(),
                }
    except FileNotFoundError:
        print(f'AVISO: meta não encontrado em {meta_path}')
    return lookup


# ── Processar dataset ─────────────────────────────────────────────────────────

def build_dataset(image_dir: str, meta_path: str,
                   min_count: int = 30,
                   n_sample: int = 0,
                   extensions: tuple = ('.jpg','.jpeg','.png','.webp')
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Processa imagens e retorna (X, y) prontos para LDA.
    n_sample > 0 limita imagens por estilo (para datasets grandes).
    """
    meta = load_meta(meta_path)

    # Descobrir imagens
    paths = [p for p in Path(image_dir).rglob('*')
              if p.suffix.lower() in extensions]
    print(f'Imagens encontradas: {len(paths):,}')

    # Agrupar por estilo
    by_style: dict[str, list] = {}
    for p in paths:
        stem  = p.stem
        style = meta.get(stem, {}).get('style',
                         p.parent.parent.name)
        by_style.setdefault(style, []).append(p)

    # Filtrar min_count
    valid = {s: ps for s, ps in by_style.items() if len(ps) >= min_count}
    print(f'Estilos válidos (n≥{min_count}): {len(valid)}')

    # Amostrar se necessário
    if n_sample > 0:
        valid = {s: (random.sample(ps, min(n_sample, len(ps)))
                     if len(ps) > n_sample else ps)
                 for s, ps in valid.items()}

    total = sum(len(ps) for ps in valid.values())
    print(f'Total a processar: {total:,}')

    X_list, y_list = [], []
    done = 0
    for style, paths_s in valid.items():
        for p in paths_s:
            feat = extract_descriptors(str(p))
            if feat is not None:
                X_list.append(feat)
                y_list.append(style)
            done += 1
            if done % 500 == 0:
                print(f'  {done:>6}/{total:,}...', end='\r')

    print(f'\nProcessadas: {len(X_list):,}  falhas: {total-len(X_list)}')
    return np.array(X_list), np.array(y_list)


# ── LDA ───────────────────────────────────────────────────────────────────────

def run_lda(X: np.ndarray, y: np.ndarray,
             output: str, label: str = 'baseline') -> float:
    """LDA com 5-fold CV e plot 2D."""
    styles    = np.unique(y)
    n_styles  = len(styles)
    n_comps   = min(n_styles - 1, X.shape[1])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(n_components=n_comps)),
    ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'Acurácia CV (5-fold): {scores.mean():.3f} ± {scores.std():.3f}')

    # Ajuste completo para visualização
    pipe.fit(X, y)
    X_lda = pipe.named_steps['lda'].transform(
        pipe.named_steps['scaler'].transform(X))
    var = pipe.named_steps['lda'].explained_variance_ratio_

    # Plot
    cmap   = cm.tab20 if n_styles <= 20 else cm.gist_ncar
    colors = {s: cmap(i/n_styles) for i, s in enumerate(styles)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for style in styles:
        mask = y == style
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                   c=[colors[style]], alpha=0.3, s=8, linewidths=0)

    for style in styles:
        mask = y == style
        cx = X_lda[mask, 0].mean()
        cy = X_lda[mask, 1].mean()
        ax.scatter(cx, cy, c=[colors[style]], s=80,
                   edgecolors='white', linewidth=1.5, zorder=6)
        ax.annotate(style, (cx, cy), fontsize=6,
                    xytext=(3,3), textcoords='offset points',
                    color=colors[style], fontweight='bold')

    ax.set_xlabel(f'LD1 ({var[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'LD2 ({var[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'Baseline — descritores clássicos de imagem\n'
                  f'Acurácia CV: {scores.mean():.3f} ± {scores.std():.3f}  '
                  f'({n_styles} estilos)\n'
                  f'Histograma Lab + Estatísticas + LBP + Haralick (~138 features)',
                  fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    p = Path(output) / f'lda_{label}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')

    return float(scores.mean())


# ── Comparação final ──────────────────────────────────────────────────────────

def print_comparison(acc_baseline: float, acc_percolation: float,
                      n_styles: int):
    """Tabela resumo para o paper."""
    baseline_random = 1.0 / n_styles
    print('\n' + '='*55)
    print(f'{"Método":<30} {"Acurácia":>10} {"vs Baseline":>12}')
    print('-'*55)
    print(f'{"Aleatório":<30} {baseline_random:>10.3f} {"—":>12}')
    print(f'{"Descritores clássicos":<30} {acc_baseline:>10.3f} '
          f'{acc_baseline/baseline_random:>11.1f}x')
    print(f'{"Percolação cromática":<30} {acc_percolation:>10.3f} '
          f'{acc_percolation/baseline_random:>11.1f}x')
    print('='*55)
    print()
    if acc_percolation >= acc_baseline * 0.9:
        print('→ Percolação competitiva com descritores clássicos')
        print('  apesar de ser não-supervisionada e capturar')
        print('  estrutura multi-escala que descritores não capturam.')
    else:
        print('→ Descritores clássicos superiores em classificação,')
        print('  mas percolação captura dimensões estruturais distintas')
        print('  (ver heatmap de separabilidade).')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',    required=True)
    parser.add_argument('--meta',         default='')
    parser.add_argument('--output',       required=True)
    parser.add_argument('--min_count',    type=int,   default=30)
    parser.add_argument('--n_sample',     type=int,   default=100,
                        help='Máx imagens por estilo (0=sem limite)')
    parser.add_argument('--acc_percol',   type=float, default=0.15,
                        help='Acurácia do método de percolação (para comparação)')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    X, y = build_dataset(args.image_dir, args.meta,
                          args.min_count, args.n_sample)

    if len(X) == 0:
        print('Nenhuma imagem processada.')
        return

    # Salvar features para reuso
    np.save(out / 'baseline_X.npy', X)
    np.save(out / 'baseline_y.npy', y)
    print(f'Features salvas: X={X.shape}  y={y.shape}')

    acc = run_lda(X, y, str(out), label='baseline_descriptors')

    print_comparison(acc, args.acc_percol, len(np.unique(y)))


if __name__ == '__main__':
    main()
