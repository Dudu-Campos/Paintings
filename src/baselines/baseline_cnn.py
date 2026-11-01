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

def _build_resnet50(pretrained: bool = True, seed: int = 42):
    """ResNet50 até o GAP. `pretrained=False` dá a âncora de transfer zero.

    A versão de pesos aleatórios existe para calibrar o eixo "quanto o
    pré-treino do ImageNet vale neste domínio": ela mede o que a *arquitetura*
    entrega sozinha — convolução, hierarquia, GAP — sem nenhum conhecimento
    transferido. A margem entre ela e a versão ImageNet é a utilidade do
    transfer naquele dataset, e é a coordenada que hoje falta para sustentar a
    premissa de que existem domínios onde transferir não ajuda.

    O pré-processamento vem do objeto de pesos nos dois casos de propósito:
    a única diferença entre os braços tem que ser o conteúdo dos pesos.
    """
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    if pretrained:
        model = resnet50(weights=weights)
    else:
        torch.manual_seed(seed)
        model = resnet50(weights=None)
    # Remover camada fc — manter até o Global Average Pooling
    model.fc = nn.Identity()
    model.eval()
    preprocess = weights.transforms()  # resize+normalize padrão do torchvision
    return model, preprocess, 2048


def _build_resnet101(pretrained: bool = True, seed: int = 42):
    """ResNet101 até o GAP — mesma saída de 2.048d que a ResNet50.

    Existe para separar duas coisas que a comparação com "uma CNN" mistura:
    profundidade da arquitetura e conteúdo dos pesos. Como a dimensão de saída é
    idêntica à da ResNet50, a diferença entre os dois braços não pode ser
    atribuída a dimensionalidade -- que é o confundidor que o projeto já mediu
    valer +10pp numa comparação mal controlada (achado do pico sobre k).
    """
    import torch
    from torchvision.models import resnet101, ResNet101_Weights
    weights = ResNet101_Weights.IMAGENET1K_V2
    if pretrained:
        model = resnet101(weights=weights)
    else:
        torch.manual_seed(seed)
        model = resnet101(weights=None)
    model.fc = nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, 2048


def _build_vit_b_16(pretrained: bool = True, seed: int = 42):
    """ViT-B/16 até o token CLS — 768d, sem a cabeça de classificação.

    Existe para separar **arquitetura** de **orçamento de supervisão**, que a
    comparação com "uma CNN" mistura. Os pesos são `IMAGENET1K_V1` (ImageNet-1k
    puro, 81,07% top-1) e não o `SWAG` (3,6 bilhões de imagens do Instagram,
    85,3%): contra a nossa ResNet50 congelada, a única coisa que muda assim é a
    família de arquitetura, não quanto rótulo externo entrou. Trocar para SWAG
    mede outra coisa -- e quebra o eixo de supervisão do projeto.

    `heads = Identity` devolve o token CLS de 768d, que é o análogo do GAP da
    ResNet: a representação imediatamente antes do classificador.

    Nota: o ViT do torchvision exige entrada de exatamente 224x224 (o
    embedding posicional é fixo), então não há a liberdade de resolução que as
    ConvNets têm.
    """
    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    if pretrained:
        model = vit_b_16(weights=weights)
    else:
        torch.manual_seed(seed)
        model = vit_b_16(weights=None)
    model.heads = nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, 768


def _build_vit_b_16_in21k(pretrained: bool = True, seed: int = 42):
    """ViT-B/16 pré-treinado em **ImageNet-21k** (Dosovitskiy et al., ICLR 2021).

    Existe porque a linha `Transformer [32]` da tabela do MSA-Net é exatamente
    este checkpoint, e comparar com o nosso `IMAGENET1K_V1` seria comparar
    orçamento de supervisão (1,3M de rótulos contra 14M), não arquitetura.

    **Armadilha:** o ViT do HuggingFace normaliza com média/desvio 0,5 em todos
    os canais, NÃO com os do ImageNet. Usar a normalização errada não levanta
    erro nenhum -- as features só saem piores. Por isso o `preprocess` vem daqui
    e não do chamador.

    A saída é o token CLS depois do layernorm final (768d), análogo ao GAP da
    ResNet: a representação imediatamente antes do classificador.
    """
    import torch
    from torchvision import transforms

    class _CLS(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values):
            return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

    from transformers import ViTModel
    name = "google/vit-base-patch16-224-in21k"
    if pretrained:
        backbone = ViTModel.from_pretrained(name, add_pooling_layer=False)
    else:
        from transformers import ViTConfig
        torch.manual_seed(seed)
        backbone = ViTModel(ViTConfig.from_pretrained(name), add_pooling_layer=False)
    model = _CLS(backbone)
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return model, preprocess, 768


def _build_vgg(depth: int):
    """VGG-{11,13,16,19} até o GAP do último mapa convolucional.

    **Todas as quatro terminam em 512 canais**, então o GAP dá 512d nas quatro:
    a profundidade varia e a dimensionalidade não. Isso importa porque o projeto
    já mediu dimensionalidade valer ~10pp numa comparação mal controlada -- numa
    escada de backbones ela seria o confundidor óbvio.

    Descarta-se o `classifier` (fc6/fc7 de 4096d): aquelas camadas são densas,
    específicas do ImageNet e mudariam a dimensão entre modelos.
    """
    def builder(pretrained: bool = True, seed: int = 42):
        import torch
        from torchvision import models as tv
        constructor = getattr(tv, f"vgg{depth}")
        weights_enum = getattr(tv, f"VGG{depth}_Weights").IMAGENET1K_V1
        if pretrained:
            network = constructor(weights=weights_enum)
        else:
            torch.manual_seed(seed)
            network = constructor(weights=None)
        model = nn.Sequential(network.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model.eval()
        return model, weights_enum.transforms(), 512
    return builder


def _build_resnet(depth: int, feature_dim: int):
    """ResNet-{18,34} até o GAP. Mesma família da 50/101, menos profundidade."""
    def builder(pretrained: bool = True, seed: int = 42):
        import torch
        from torchvision import models as tv
        constructor = getattr(tv, f"resnet{depth}")
        weights_enum = getattr(tv, f"ResNet{depth}_Weights").IMAGENET1K_V1
        if pretrained:
            model = constructor(weights=weights_enum)
        else:
            torch.manual_seed(seed)
            model = constructor(weights=None)
        model.fc = nn.Identity()
        model.eval()
        return model, weights_enum.transforms(), feature_dim
    return builder


MODEL_REGISTRY = {
    'vgg11': {'builder': _build_vgg(11), 'input_size': 224, 'feature_dim': 512,
              'description': 'VGG-11 ImageNet-1k congelada, GAP do último conv'},
    'vgg13': {'builder': _build_vgg(13), 'input_size': 224, 'feature_dim': 512,
              'description': 'VGG-13 ImageNet-1k congelada, GAP do último conv'},
    'vgg16': {'builder': _build_vgg(16), 'input_size': 224, 'feature_dim': 512,
              'description': 'VGG-16 ImageNet-1k congelada, GAP do último conv'},
    'vgg19': {'builder': _build_vgg(19), 'input_size': 224, 'feature_dim': 512,
              'description': 'VGG-19 ImageNet-1k congelada, GAP do último conv'},
    'resnet18': {'builder': _build_resnet(18, 512), 'input_size': 224, 'feature_dim': 512,
                 'description': 'ResNet18 ImageNet-1k congelada, GAP'},
    'resnet34': {'builder': _build_resnet(34, 512), 'input_size': 224, 'feature_dim': 512,
                 'description': 'ResNet34 ImageNet-1k congelada, GAP'},
    'vit_b_16_in21k': {
        'builder':    _build_vit_b_16_in21k,
        'input_size': 224,
        'feature_dim': 768,
        'description': 'ViT-B/16 pré-treinado ImageNet-21k (Dosovitskiy et al.), token CLS',
    },
    'vit_b_16': {
        'builder':    _build_vit_b_16,
        'input_size': 224,
        'feature_dim': 768,
        'description': 'ViT-B/16 pré-treinado ImageNet-1k (sem fine-tuning), token CLS',
    },
    'resnet101': {
        'builder':    _build_resnet101,
        'input_size': 224,
        'feature_dim': 2048,
        'description': 'ResNet101 pré-treinada ImageNet-1k (sem fine-tuning), GAP',
    },
    'resnet50': {
        'builder':    _build_resnet50,
        'input_size': 224,
        'feature_dim': 2048,
        'description': 'ResNet50 pré-treinada ImageNet-1k (sem fine-tuning), GAP',
    },
    # Adicionar futuros modelos aqui, por exemplo:
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
