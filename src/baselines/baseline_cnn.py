"""
baseline_cnn.py — Baseline com modelos pré-treinados de visão.

Extrai features via Global Average Pooling de modelos pré-treinados
em ImageNet, sem fine-tuning. Resize com letterbox preserva proporção
(consistente com a normalização usada no pipeline de percolação).

Estrutura extensível — adicionar modelo:
    1. Adicionar entrada em MODEL_REGISTRY
    2. Especificar input_size e preprocessing

Modelos disponíveis atualmente:
    - resnet50    (2048-d features, GAP antes da camada fc)

Modelos planejados (adicionar quando necessário):
    - vit_b_16    (vision transformer)
    - clip_vit    (CLIP — embedding multimodal)
    - dino_v2     (self-supervised, sem viés de classe ImageNet)

Uso:
    python baseline_cnn.py \
        --image_dir /path/to/raw \
        --meta      /path/to/meta.csv \
        --output    ./plots \
        --model     resnet50 \
        --min_count 30 \
        --n_sample  100 \
        --acc_percol 0.150 \
        --acc_baseline 0.173
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

Image.MAX_IMAGE_PIXELS = None


# ── Registro de modelos — adicionar novos aqui ────────────────────────────────

def _build_resnet50():
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model   = resnet50(weights=weights)
    # Remover camada fc — manter até o Global Average Pooling
    model.fc = nn.Identity()
    model.eval()
    preprocess = weights.transforms()  # resize+normalize padrão do torchvision
    return model, preprocess, 2048


MODEL_REGISTRY = {
    'resnet50': {
        'builder':    _build_resnet50,
        'input_size': 224,
        'feature_dim': 2048,
        'description': 'ResNet50 pré-treinada ImageNet-1k (sem fine-tuning), GAP',
    },
    # Adicionar futuros modelos aqui, por exemplo:
    # 'vit_b_16': {
    #     'builder': _build_vit_b_16,
    #     'input_size': 224,
    #     'feature_dim': 768,
    #     'description': 'Vision Transformer B/16 pré-treinado ImageNet-1k',
    # },
    # 'dino_v2': {
    #     'builder': _build_dino_v2,
    #     'input_size': 224,
    #     'feature_dim': 768,
    #     'description': 'DINOv2 self-supervised — sem viés de classe ImageNet',
    # },
}


# ── Letterbox resize — preserva proporção, consistente com pipeline percolação ──

def letterbox_resize(img: Image.Image, target_size: int) -> Image.Image:
    """
    Resize mantendo proporção + padding cinza até target_size × target_size.
    Evita distorção que resize direto introduziria.
    """
    w, h  = img.size
    scale = target_size / max(w, h)
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    img_resized = img.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    paste_x = (target_size - nw) // 2
    paste_y = (target_size - nh) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


# ── Extração de features ──────────────────────────────────────────────────────

def extract_cnn_features(image_paths: list[str],
                          model_name: str = 'resnet50',
                          device: str = 'cpu',
                          batch_size: int = 16) -> tuple[np.ndarray, list[int]]:
    """
    Extrai features CNN para uma lista de imagens.
    Retorna (features, valid_indices) — valid_indices mapeia para image_paths.
    """
    cfg = MODEL_REGISTRY[model_name]
    model, preprocess, feat_dim = cfg['builder']()
    model = model.to(device)

    features = []
    valid_idx = []

    batch_imgs  = []
    batch_idx   = []

    def flush_batch():
        if not batch_imgs:
            return
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            out = model(x)  # (B, feat_dim) após GAP + Identity
            features.append(out.cpu().numpy())
        valid_idx.extend(batch_idx)
        batch_imgs.clear()
        batch_idx.clear()

    t0 = time.time()
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            img = letterbox_resize(img, cfg['input_size'])
            tensor = preprocess(img)
            batch_imgs.append(tensor)
            batch_idx.append(i)
        except Exception:
            continue

        if len(batch_imgs) >= batch_size:
            flush_batch()

        if (i+1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1:>6}/{len(image_paths):,}]  '
                  f'{elapsed:.1f}s  {(i+1)/elapsed:.1f} img/s', flush=True)

    flush_batch()  # último batch parcial
    print()

    if not features:
        return np.zeros((0, feat_dim)), []

    return np.concatenate(features, axis=0), valid_idx


# ── Meta CSV ──────────────────────────────────────────────────────────────────

def load_meta(meta_path: str) -> dict:
    lookup = {}
    try:
        with open(meta_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row.get('filename','')).stem
                lookup[stem] = {'style': row.get('style','unknown').strip()}
    except FileNotFoundError:
        print(f'AVISO: meta não encontrado em {meta_path}')
    return lookup


# ── Dataset ───────────────────────────────────────────────────────────────────

def collect_paths(image_dir: str, meta_path: str,
                   min_count: int, n_sample: int,
                   extensions: tuple = ('.jpg','.jpeg','.png','.webp')
                   ) -> tuple[list[str], list[str]]:
    meta = load_meta(meta_path)
    paths = [p for p in Path(image_dir).rglob('*')
              if p.suffix.lower() in extensions]
    print(f'Imagens encontradas: {len(paths):,}')

    by_style: dict[str, list] = {}
    for p in paths:
        style = meta.get(p.stem, {}).get('style', p.parent.parent.name)
        by_style.setdefault(style, []).append(p)

    valid = {s: ps for s, ps in by_style.items() if len(ps) >= min_count}
    print(f'Estilos válidos (n≥{min_count}): {len(valid)}')

    if n_sample > 0:
        valid = {s: (random.sample(ps, min(n_sample, len(ps)))
                     if len(ps) > n_sample else ps)
                 for s, ps in valid.items()}

    all_paths, all_styles = [], []
    for style, ps in valid.items():
        for p in ps:
            all_paths.append(str(p))
            all_styles.append(style)

    print(f'Total selecionado: {len(all_paths):,}')
    return all_paths, all_styles


# ── LDA ───────────────────────────────────────────────────────────────────────

def run_lda(X: np.ndarray, y: np.ndarray, output: str,
            label: str, title_extra: str = '') -> float:
    styles   = np.unique(y)
    n_styles = len(styles)
    n_comps  = min(n_styles - 1, X.shape[1])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(n_components=n_comps)),
    ])
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'Acurácia CV (5-fold): {scores.mean():.3f} ± {scores.std():.3f}')

    pipe.fit(X, y)
    X_lda = pipe.named_steps['lda'].transform(
        pipe.named_steps['scaler'].transform(X))
    var = pipe.named_steps['lda'].explained_variance_ratio_

    cmap   = cm.tab20 if n_styles <= 20 else cm.gist_ncar
    colors = {s: cmap(i/n_styles) for i, s in enumerate(styles)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for style in styles:
        mask = y == style
        ax.scatter(X_lda[mask,0], X_lda[mask,1],
                   c=[colors[style]], alpha=0.3, s=8, linewidths=0)
    for style in styles:
        mask = y == style
        cx, cy = X_lda[mask,0].mean(), X_lda[mask,1].mean()
        ax.scatter(cx, cy, c=[colors[style]], s=80,
                   edgecolors='white', linewidth=1.5, zorder=6)
        ax.annotate(style, (cx,cy), fontsize=6,
                    xytext=(3,3), textcoords='offset points',
                    color=colors[style], fontweight='bold')

    ax.set_xlabel(f'LD1 ({var[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'LD2 ({var[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'{title_extra}\nAcurácia CV: {scores.mean():.3f} ± '
                  f'{scores.std():.3f}  ({n_styles} estilos)', fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    p = Path(output) / f'lda_{label}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {p}')
    return float(scores.mean())


# ── Comparação ────────────────────────────────────────────────────────────────

def print_comparison(results: dict, n_styles: int):
    baseline_random = 1.0 / n_styles
    print('\n' + '='*60)
    print(f'{"Método":<32} {"Acurácia":>10} {"vs Aleatório":>14}')
    print('-'*60)
    print(f'{"Aleatório":<32} {baseline_random:>10.3f} {"—":>14}')
    for name, acc in results.items():
        print(f'{name:<32} {acc:>10.3f} {acc/baseline_random:>13.1f}x')
    print('='*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',     required=True)
    parser.add_argument('--meta',          default='')
    parser.add_argument('--output',        required=True)
    parser.add_argument('--model',         default='resnet50',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--min_count',     type=int,   default=30)
    parser.add_argument('--n_sample',      type=int,   default=100)
    parser.add_argument('--device',        default='cpu')
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--acc_percol',    type=float, default=0.150)
    parser.add_argument('--acc_baseline',  type=float, default=0.173,
                        help='Acurácia dos descritores clássicos (já calculada)')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    paths, styles = collect_paths(args.image_dir, args.meta,
                                    args.min_count, args.n_sample)

    cfg = MODEL_REGISTRY[args.model]
    print(f'\nModelo: {args.model}  ({cfg["description"]})')
    print(f'Device: {args.device}')

    X, valid_idx = extract_cnn_features(
        paths, model_name=args.model,
        device=args.device, batch_size=args.batch_size)
    y = np.array([styles[i] for i in valid_idx])

    print(f'\nFeatures extraídas: X={X.shape}')
    np.save(out / f'cnn_{args.model}_X.npy', X)
    np.save(out / f'cnn_{args.model}_y.npy', y)

    acc = run_lda(X, y, str(out), label=f'cnn_{args.model}',
                  title_extra=f'Baseline CNN — {args.model} (GAP, sem fine-tuning)')

    print_comparison({
        'Descritores clássicos':              args.acc_baseline,
        f'CNN ({args.model}, sem fine-tune)': acc,
        'Percolação cromática':               args.acc_percol,
    }, n_styles=len(np.unique(y)))


if __name__ == '__main__':
    main()
