"""Train an end-to-end ResNet-50 on an image classification dataset.

This script is deliberately separate from ``extract_cnn_full.py``.  The latter
only freezes an ImageNet encoder and evaluates its embeddings with LDA/MLP; it
does not measure a CNN trained for the WikiArt task.

The two supported conditions share the data, outer CV folds, augmentation,
optimizer, loss, and stopping rule.  Their only model-level difference is the
initialization:

    scratch   ResNet-50 with random weights
    frozen    ImageNet ResNet-50 backbone frozen; only the classification head trains
    imagenet  ResNet-50 initialized with ImageNet-1k weights and fine-tuned end-to-end

For each outer test fold, early stopping uses a stratified validation split
drawn *only* from that fold's training data.  This keeps the reported test-fold
metrics independent of model selection.

Example
-------
uv run python src/baselines/train_resnet_wikiart.py \
  --image-dir raw/wikiart --init scratch --output outputs/cnn_scratch
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility matters for a paired scratch/transfer comparison.  The
    # small throughput cost is preferable to run-to-run ambiguity here.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Letterbox:
    """Resize an image without changing its aspect ratio, then pad to a square."""

    def __init__(self, size: int, fill: tuple[int, int, int] = (128, 128, 128)):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return canvas


class WikiArtDataset(Dataset):
    def __init__(self, records: list[dict], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, record["label"], record["stem"]


@dataclass(frozen=True)
class TrainConfig:
    image_size: int
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    seed: int
    initialization: str
    checkpoint: str | None = None
    # Campos com default TÊM de vir por último num dataclass, senão o __init__
    # gerado levanta "non-default argument follows default argument".
    architecture: str = "resnet50"
    optimizer: str = "sgd"
    no_class_weight: bool = False
    label_smoothing: float = 0.0
    select_by: str = "bacc"
    warmup_epochs: int = 0
    grad_clip: float = 0.0
    # Receita de treino DO ZERO (nenhum destes muda o fine-tuning: os defaults
    # reproduzem exatamente o agendamento anterior). Do zero precisa de mais
    # regularizacao e de um otimizador melhor comportado que o fine-tuning --
    # la o backbone chega pronto e o risco e destruir o que ja existe.
    mixup_alpha: float = 0.0      # Zhang et al., ICLR 2018; 0 desliga
    no_wd_on_norm: bool = False   # Goyal et al. 2017: sem decay em BN e bias
    zero_init_residual: bool = False  # gamma da ultima BN de cada bloco em 0


def build_transforms(image_size: int, mode: str = "light"
                     ) -> tuple[transforms.Compose, transforms.Compose]:
    """`light` e o pre-processamento historico do projeto; `strong` e a receita
    padrao de fine-tuning de ViT.

    Por que `strong` existe: o `Letterbox` preserva proporcao e preenche o resto
    com cinza, e a coorte do Multitask gasta **24,7% da entrada em padding na
    mediana** (medido em 25/ago, 800 imagens) -- num ViT-B/16 sao ~48 dos 196
    patches em cinza constante, um quarto da sequencia de atencao. Somado a uma
    augmentacao de so flip + affine(10 graus), o ViT memoriza: loss de treino em
    0,062 com a validacao parada em 62,5 (IN-21k, epoca 13).

    `strong` NAO mexe no `Letterbox`, que segue sendo o caminho de todo o resto
    do projeto -- inclusive da extracao congelada, que nao usa augmentacao
    nenhuma e portanto nao e afetada por esta flag. O raio de alcance sao os
    modelos treinados de ponta a ponta.

    Treino e avaliacao mudam JUNTOS: treinar com crop e avaliar com letterbox
    poria o modelo diante de barras cinzas que ele nunca viu.
    """
    if mode == "strong":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_transform, eval_transform
    common = [Letterbox(image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_transform = transforms.Compose([
        Letterbox(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, transforms.Compose(common)


def _title_part(filename: str) -> str:
    """Return the title portion of WikiArt's ``artist_title.jpg`` filename."""
    return filename.partition("_")[2] or filename


def index_images(image_dir: Path) -> tuple[
    dict[tuple[str, str], Path], dict[tuple[str, str], list[Path]], dict[str, list[Path]],
]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    log(f"Indexing images under {image_dir}...")
    by_exact: dict[tuple[str, str], Path] = {}
    by_style_title: dict[tuple[str, str], list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in image_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = (path.parent.name, path.name)
        if key in by_exact:
            raise ValueError(f"Duplicate style/filename in image directory: {key}")
        by_exact[key] = path
        title_key = (path.parent.name, _title_part(path.name))
        by_style_title.setdefault(title_key, []).append(path)
        by_stem.setdefault(path.stem, []).append(path)
    log(f"Indexed {len(by_exact):,} images across {len({style for style, _ in by_exact})} style directories.")
    return by_exact, by_style_title, by_stem


def load_records(meta_path: Path, image_dir: Path, task: str, min_per_class: int,
                 restrict_stems: Path | None = None) -> tuple[list[dict], list[str]]:
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")
    meta = pd.read_csv(meta_path)
    required = {"stem", task}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    by_exact, by_style_title, by_stem = index_images(image_dir)
    columns = ["stem", task] + (["filename"] if "filename" in meta else [])
    meta = meta[columns].dropna(subset=["stem", task]).copy()
    meta["stem"] = meta["stem"].astype(str)
    if "filename" in meta:
        meta["filename"] = meta["filename"].astype(str)

    paths: list[Path] = []
    exact_matches = fallback_matches = 0
    unresolved: list[str] = []
    for row in meta.itertuples(index=False):
        style = str(getattr(row, task))
        filename = getattr(row, "filename", None)
        path = by_exact.get((style, filename)) if filename else None
        if path is not None:
            exact_matches += 1
        elif filename:
            candidates = by_style_title.get((style, _title_part(filename)), [])
            if len(candidates) > 1:
                artist = filename.partition("_")[0]
                scored = sorted(
                    ((SequenceMatcher(None, artist, candidate.name.partition("_")[0]).ratio(), candidate)
                     for candidate in candidates),
                    key=lambda item: item[0], reverse=True,
                )
                if len(scored) > 1 and scored[0][0] == scored[1][0]:
                    unresolved.append(f"ambiguous {style}/{filename}")
                    paths.append(None)
                    continue
                path = scored[0][1]
                fallback_matches += 1
            elif candidates:
                path = candidates[0]
                fallback_matches += 1
            else:
                candidates = by_stem.get(row.stem, [])
                if len(candidates) == 1:
                    path = candidates[0]
                    fallback_matches += 1
                else:
                    unresolved.append(f"{style}/{filename}")
                    paths.append(None)
                    continue
        else:
            candidates = by_stem.get(row.stem, [])
            if len(candidates) != 1:
                unresolved.append(f"{style}/{row.stem}")
                paths.append(None)
                continue
            path = candidates[0]
            exact_matches += 1
        paths.append(path)
    if unresolved:
        preview = "; ".join(unresolved[:5])
        raise ValueError(f"Could not resolve {len(unresolved):,} metadata images (first: {preview})")
    meta["path"] = paths
    log(f"Image matching: {exact_matches:,} exact, {fallback_matches:,} recovered from filename encoding.")
    if restrict_stems is not None:
        # A coorte da CNN (79.902) e a canônica (79.207) concordam em apenas
        # 20% da partição em folds; usar folds canônicos com uma CNN treinada na
        # dela poria 63.372 imagens de teste dentro do treino dela. Restringir
        # aqui é o que torna o ensemble de posteriores válido.
        wanted = set(pd.read_csv(restrict_stems)["stem"].astype(str))
        before = len(meta)
        meta = meta[meta["stem"].isin(wanted)].copy()
        # 34 stems do WikiArt aparecem duas vezes com estilos **diferentes**.
        # Sem desduplicar, a mesma imagem cai em treino e teste com rótulos
        # conflitantes. `keep="last"` reproduz o que `load_dataset` faz via
        # `dict(zip(...))`, então o rótulo é idêntico ao do lado hand-designed.
        meta = meta.drop_duplicates("stem", keep="last")
        log(f"Stem restriction: {before:,} -> {len(meta):,} images "
            f"({len(wanted):,} requested).")
        if len(meta) < 0.9 * len(wanted):
            raise ValueError("Restriction dropped too many images; check the stem list.")

    counts = meta[task].value_counts()
    retained_labels = counts[counts >= min_per_class].index
    meta = meta[meta[task].isin(retained_labels)].copy()
    if meta.empty:
        raise ValueError("No images remain after matching metadata and applying --min-per-class.")

    encoder = LabelEncoder()
    meta["label"] = encoder.fit_transform(meta[task].astype(str))
    records = [
        {"path": str(row.path), "stem": row.stem, "label": int(row.label)}
        for row in meta.itertuples(index=False)
    ]
    log(
        f"Dataset: {len(records):,} images, {len(encoder.classes_)} {task} classes "
        f"(minimum {min_per_class} images/class)."
    )
    return records, encoder.classes_.tolist()


def make_loader(records: list[dict], transform: transforms.Compose, *, batch_size: int,
                workers: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        WikiArtDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_model(initialization: str, n_classes: int,
                checkpoint: Path | None = None, architecture: str = "resnet50",
                zero_init_residual: bool = False) -> nn.Module:
    """ResNet50 ou ViT-B/16 com cabeça nova do tamanho do alvo.

    Os pesos do ViT são `IMAGENET1K_V1` (ImageNet-1k puro), e NÃO o `SWAG`
    (3,6 bilhões de imagens do Instagram): contra a ResNet50 `IMAGENET1K_V2` a
    única coisa que muda assim é a família de arquitetura, não o orçamento de
    supervisão -- que é o eixo do §4.1 do CLAUDE.md.
    """
    if architecture == "vit_b_16_in21k":
        # Replicação de Wang & Feng (Signal Proc.: Image Comm. 137, 2025,
        # 117309), cuja linha "Transformer [32]" dá 72,25% de accuracy no
        # Wikiart25 sob 5-fold 80/20 -- o mesmo protocolo do nosso §4. Eles
        # citam `google-research/vision_transformer`, cujos checkpoints padrão
        # são ImageNet-21k, e `orig_in21k` é exatamente esse peso.
        #
        # Isto **quebra de propósito** a paridade de supervisão do §4.1: 14M de
        # imagens e 21.843 classes contra 1,28M e 1.000 da ResNet50. Só existe
        # para replicar a referência, não para a tabela de arquitetura.
        import timm
        model = timm.create_model(
            "vit_base_patch16_224.orig_in21k",
            pretrained=initialization in {"imagenet", "frozen"},
            num_classes=n_classes)
        if checkpoint is not None:
            raise ValueError("--checkpoint só é suportado para resnet50")
        if initialization == "frozen":
            for name, parameter in model.named_parameters():
                if not name.startswith("head."):
                    parameter.requires_grad_(False)
        return model
    if architecture == "vit_b_16":
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if initialization in {"imagenet", "frozen"} else None
        model = vit_b_16(weights=weights)
        if checkpoint is not None:
            raise ValueError("--checkpoint só é suportado para resnet50")
        if initialization == "frozen":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        # `heads` é um Sequential com um único Linear chamado `head`.
        model.heads = nn.Sequential(nn.Linear(model.hidden_dim, n_classes))
        return model
    # `zero_init_residual` so faz sentido SEM pesos: ele zera o gamma da ultima
    # BN de cada bloco residual para que o bloco comece como identidade, o que
    # estabiliza as primeiras epocas do treino do zero. Aplicado sobre pesos
    # pre-treinados destruiria o que foi aprendido -- dai a guarda.
    if architecture == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if initialization in {"imagenet", "frozen"} else None
        if checkpoint is not None:
            raise ValueError("--checkpoint so e suportado para resnet50")
        model = resnet18(weights=weights,
                         zero_init_residual=zero_init_residual and weights is None)
        if initialization == "frozen":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model

    weights = ResNet50_Weights.IMAGENET1K_V2 if initialization in {"imagenet", "frozen"} else None
    model = resnet50(weights=weights,
                     zero_init_residual=zero_init_residual and weights is None)
    if checkpoint is not None:
        # Backbone pré-treinado por regressão nas features hand-designed. O `fc`
        # daquele treino tem a dimensão do alvo (607), não a das classes, então
        # entra com strict=False e é substituído logo abaixo -- as camadas
        # convolucionais, que são o que interessa, carregam normalmente.
        state = torch.load(checkpoint, map_location="cpu")
        # `strict=False` perdoa chave ausente, mas NÃO tamanho incompatível numa
        # chave presente: o `fc` do pré-treino tem 607 saídas contra as 1000 do
        # resnet50 recém-construído e o load levanta RuntimeError. Descarto o
        # `fc` explicitamente -- ele seria substituído logo abaixo de qualquer forma.
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        carried = len(state)
        log(f"Backbone carregado de {checkpoint}: {carried} tensores; "
            f"faltando={len(missing)} inesperados={len(unexpected)}")
    if initialization == "frozen":
        # Freeze parameters *before* replacing fc, so the new head remains trainable.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def class_weights(records: list[dict], n_classes: int, device: torch.device) -> torch.Tensor:
    labels = np.fromiter((record["label"] for record in records), dtype=np.int64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = 1.0 / counts
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true, predicted, stems, probabilities = [], [], [], []
    for images, labels, batch_stems in loader:
        logits = model(images.to(device, non_blocking=True))
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        # As probabilidades existem para o ensemble em nível de posterior, que é
        # o único protocolo de fusão que sobreviveu: embeddings de checkpoints
        # diferentes vivem em espaços rotacionados entre si (medido: −16,54pp),
        # enquanto a dimensão `c` de um posterior é a classe `c` em todo fold.
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        true.extend(labels.tolist())
        stems.extend(batch_stems)
    y_true = np.asarray(true, dtype=np.int64)
    y_pred = np.asarray(predicted, dtype=np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }, stems, y_true, y_pred, np.concatenate(probabilities).astype(np.float32)


def train_outer_fold(*, fold: int, train_records: list[dict], test_records: list[dict],
                     validation_records: list[dict] | None = None,
                     n_classes: int, train_transform: transforms.Compose,
                     eval_transform: transforms.Compose, config: TrainConfig,
                     workers: int, device: torch.device, output_dir: Path) -> dict:
    if validation_records is None:
        outer_labels = np.fromiter((r["label"] for r in train_records), dtype=np.int64)
        train_indices, early_stop_indices = train_test_split(
            np.arange(len(train_records)),
            test_size=config.val_fraction,
            random_state=config.seed + fold,
            stratify=outer_labels,
        )
        fit_records = [train_records[index] for index in train_indices]
        early_stop_records = [train_records[index] for index in early_stop_indices]
    else:
        fit_records = train_records
        early_stop_records = validation_records

    train_loader = make_loader(fit_records, train_transform, batch_size=config.batch_size,
                               workers=workers, shuffle=True, seed=config.seed + fold, device=device)
    stop_loader = make_loader(early_stop_records, eval_transform, batch_size=config.batch_size,
                              workers=workers, shuffle=False, seed=config.seed + fold, device=device)
    test_loader = make_loader(test_records, eval_transform, batch_size=config.batch_size,
                              workers=workers, shuffle=False, seed=config.seed + fold, device=device)

    model = build_model(config.initialization, n_classes,
                        Path(config.checkpoint) if config.checkpoint else None,
                        architecture=config.architecture,
                        zero_init_residual=config.zero_init_residual).to(device)
    # `--no-class-weight` existe para a comparação com a literatura do benchmark,
    # que reporta acurácia simples: treinar ponderado otimiza BAcc e SACRIFICA
    # acurácia (medido em 25/ago: +7,30pp de acurácia no hand-8 ao tirar a
    # ponderação). Selecionar por uma métrica e reportar outra é o mesmo erro na
    # etapa seguinte, então `--select-by` acompanha.
    weight = (None if config.no_class_weight
              else class_weights(fit_records, n_classes, device))
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=config.label_smoothing)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.no_wd_on_norm:
        # Weight decay em BN e bias encolhe parametros que NAO sao pesos de
        # conexao: o gamma da BN e um ganho e o bias um deslocamento, e penaliza-los
        # e regularizacao sem sentido geometrico (Goyal et al. 2017). `ndim <= 1`
        # pega gamma, beta e todos os bias de uma vez.
        decayed = [q for n, q in model.named_parameters()
                   if q.requires_grad and q.ndim > 1]
        undecayed = [q for n, q in model.named_parameters()
                     if q.requires_grad and q.ndim <= 1]
        optim_targets = [{"params": decayed, "weight_decay": config.weight_decay},
                         {"params": undecayed, "weight_decay": 0.0}]
        log(f"weight decay so nos {len(decayed)} tensores de rank>1; "
            f"{len(undecayed)} de BN/bias ficam sem decay")
    else:
        optim_targets = trainable_parameters
    if config.optimizer == "adamw":
        # SGD com lr=0,01 diverge num ViT; transformers querem AdamW e lr ~1e-4.
        optimizer = torch.optim.AdamW(optim_targets, lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(optim_targets, lr=config.learning_rate, momentum=0.9,
                                    nesterov=True, weight_decay=config.weight_decay)
    # Warmup linear + cosine é a receita de fine-tuning do repositório oficial do
    # ViT (Dosovitskiy et al., apêndice B.1.1), que é a via pela qual os autores
    # do MSA-Net obtiveram a baseline `Transformer [32]` de 72,25 -- o artigo
    # deles cita github.com/google-research/vision_transformer como fonte. Lá o
    # fine-tuning usa SGD momentum 0,9, weight decay ZERO, clipping de norma
    # global em 1,0 e busca de LR em {0,001, 0,003, 0,01, 0,03}.
    # `warmup_epochs=0` (padrão) reproduz exatamente o agendamento anterior.
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - config.warmup_epochs))
    if config.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=config.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cosine], milestones=[config.warmup_epochs])
    else:
        scheduler = cosine
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    # Gerador PROPRIO do mixup, semeado do config: usar o global faria a
    # sequencia depender de quantas chamadas de numpy aconteceram antes, e o
    # §8 ja registra o custo de uma semente que o chamador achava que controlava.
    mix_rng = np.random.default_rng(config.seed)

    best_state, best_epoch, best_stop_bacc, stale_epochs = None, 0, -1.0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.initialization == "frozen":
            # BatchNorm running statistics are part of the frozen backbone too.
            model.eval()
            model.fc.train()
        loss_sum, n_seen = 0.0, 0
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # MIXUP (Zhang et al., ICLR 2018). Com alpha=0 (padrao) nada disto
            # roda e o laco e byte a byte o anterior. A perda mistura os DOIS
            # rotulos com o mesmo lambda da imagem, o que preserva os pesos
            # 1/n_c e o label smoothing do `criterion` nos dois termos.
            lam, perm = 1.0, None
            if config.mixup_alpha > 0:
                lam = float(mix_rng.beta(config.mixup_alpha, config.mixup_alpha))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1.0 - lam) * images[perm]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
                if perm is not None:
                    loss = lam * loss + (1.0 - lam) * criterion(logits, labels[perm])
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                # Precisa desescalar antes de medir a norma, senão o clipping
                # opera sobre gradientes multiplicados pelo fator do AMP.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            n_seen += labels.size(0)
        scheduler.step()

        stop_metrics, _, _, _, _ = evaluate(model, stop_loader, device)
        # O rótulo do log acompanha `--select-by`: imprimir "stop_BAcc" enquanto
        # a variável carrega acurácia já causou uma comparação errada em 25/ago.
        stop_key = "balanced_accuracy" if config.select_by == "bacc" else "accuracy"
        stop_label = "stop_BAcc" if config.select_by == "bacc" else "stop_Acc"
        stop_bacc = stop_metrics[stop_key]
        if stop_bacc > best_stop_bacc:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_stop_bacc, stale_epochs = epoch, stop_bacc, 0
        else:
            stale_epochs += 1
        log(
            f"{config.initialization} fold {fold}  epoch {epoch:03d}  "
            f"loss={loss_sum / n_seen:.4f}  {stop_label}={stop_bacc:.4f}  "
            f"best={best_stop_bacc:.4f}@{best_epoch}"
        )
        if stale_epochs >= config.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics, stems, y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    checkpoint_path = output_dir / f"fold_{fold:02d}_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": n_classes,
        "config": asdict(config),
        "fold": fold,
        "best_epoch": best_epoch,
        "early_stop_balanced_accuracy": best_stop_bacc,
    }, checkpoint_path)
    pd.DataFrame({"stem": stems, "fold": fold, "y_true": y_true, "y_pred": y_pred}).to_csv(
        output_dir / f"fold_{fold:02d}_predictions.csv", index=False
    )
    # Posteriores alinhados aos stems da mesma linha do CSV acima.
    np.savez_compressed(output_dir / f"fold_{fold:02d}_posteriors.npz",
                        stems=np.asarray(stems, dtype=str), probabilities=y_prob,
                        y_true=y_true)
    log(
        f"{config.initialization} fold {fold} complete: "
        f"BAcc={test_metrics['balanced_accuracy']:.4f}  Acc={test_metrics['accuracy']:.4f}  "
        f"F1={test_metrics['macro_f1']:.4f}  best_epoch={best_epoch}"
    )
    return {"fold": fold, "best_epoch": best_epoch, "early_stop_bacc": best_stop_bacc, **test_metrics}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-dir", required=True, type=Path,
                        help="Image root; nested directories and flat directories are supported.")
    parser.add_argument("--meta", type=Path, default=root / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--task", default="style",
                        help="Target-column name in --meta (e.g. style, period, label).")
    parser.add_argument("--init", choices=["scratch", "frozen", "imagenet"], required=True)
    parser.add_argument("--model", dest="architecture", default="resnet50",
                        choices=["resnet18", "resnet50", "vit_b_16", "vit_b_16_in21k"])
    parser.add_argument("--require-cuda", action="store_true",
                        help="aborta se CUDA nao estiver disponivel, em vez de "
                             "cair para CPU em silencio")
    parser.add_argument("--mixup", type=float, default=0.0, dest="mixup_alpha",
                        help="alpha do Mixup; 0 (padrao) desliga")
    parser.add_argument("--no-wd-on-norm", action="store_true",
                        help="tira weight decay de BN e bias (Goyal et al. 2017)")
    parser.add_argument("--zero-init-residual", action="store_true",
                        help="gamma da ultima BN de cada bloco residual em zero; "
                             "so tem efeito com --init scratch")
    parser.add_argument("--optimizer", default=None, choices=["sgd", "adamw"],
                        help="padrão: sgd para resnet50, adamw para vit_b_16")
    parser.add_argument("--no-class-weight", action="store_true",
                        help="para comparação com a literatura, que reporta acurácia simples")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="rampa linear de LR; 0 reproduz o agendamento anterior")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="norma global maxima; 0 desliga. A receita do ViT usa 1.0")
    parser.add_argument("--select-by", default="bacc", choices=["bacc", "accuracy"],
                        help="métrica do early stopping; casar com a que será reportada")
    parser.add_argument("--only-fold", type=int, default=0,
                        help="roda só este fold. Necessário com --init-from: o "
                             "backbone é pré-treinado no treino de UM fold, e o "
                             "treino daquele fold contém as imagens de teste dos "
                             "outros -- usá-lo nos demais seria pré-treino tendo "
                             "visto o teste.")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="state_dict de backbone para carregar antes de treinar "
                             "(ex.: pré-treino por regressão nas features "
                             "hand-designed). Use com --init scratch: os pesos do "
                             "checkpoint substituem a inicialização aleatória.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--restrict-stems", type=Path, default=None,
                        help="CSV com coluna `stem`: restringe a coorte, para os "
                             "folds baterem com os do pipeline hand-designed.")
    parser.add_argument("--augment", default="light", choices=["light", "strong"],
                        help="light = Letterbox + flip + affine (historico do projeto); "
                             "strong = RandomResizedCrop + RandAugment no treino e "
                             "Resize+CenterCrop na avaliacao (receita padrao de ViT)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.03,
                        help="Shared learning rate for both initialization conditions.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-subsample", type=int, default=0,
                        help="Subamostra estratificada do treino de cada fold; 0 usa tudo. O fold de teste nunca encolhe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-split", action="store_true",
                        help="usa a coluna `split` do metadado: train/test e, "
                             "se existir, val para early stopping. Se não houver "
                             "val, separa validação estratificada dentro de train.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate metadata/image matching and exit before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be strictly between 0 and 1.")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    seed_everything(args.seed)
    # QUEDA SILENCIOSA PARA CPU: foi ela, e nao a GPU cair, que custou 3,5 dias
    # em 02-05/out. A RTX 3090 morreu as 03:28 (`CUDA error: unspecified launch
    # failure`), o `train` seguinte comecou em CPU sem uma linha de aviso, e so o
    # `config.json` registrava `"device": "cpu"` -- 86 min por epoca contra 68 s,
    # 76x mais lento, 35 nucleos de uma maquina compartilhada, por 200 epocas que
    # levariam 12 dias. E o modo de falha do §8: nada na saida denunciava.
    if not torch.cuda.is_available():
        log("!! CUDA INDISPONIVEL -- este treino rodaria em CPU")
        if args.require_cuda:
            raise SystemExit(
                "abortando por --require-cuda. Treino de CNN em CPU e ~76x mais "
                "lento; se e isso mesmo que voce quer, tire a flag.")
        log("!! seguindo em CPU assim mesmo (sem --require-cuda)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("WARNING: CUDA is unavailable; ResNet-50 CV training will be slow on CPU.")
    args.output.mkdir(parents=True, exist_ok=True)
    records, classes = load_records(args.meta, args.image_dir, args.task, args.min_per_class,
                                    args.restrict_stems)
    split_by_stem = None
    if args.official_split:
        meta_for_split = pd.read_csv(args.meta, dtype={"stem": str})
        if "split" not in meta_for_split.columns:
            raise ValueError("--official-split requested, but metadata has no `split` column")
        split_by_stem = dict(zip(meta_for_split["stem"].astype(str), meta_for_split["split"].astype(str)))
        missing_split = [record["stem"] for record in records if record["stem"] not in split_by_stem]
        if missing_split:
            raise ValueError(f"{len(missing_split):,} records have no official split")
    if args.dry_run:
        log("Dry run complete: dataset is ready for training.")
        return
    labels = np.fromiter((record["label"] for record in records), dtype=np.int64)
    min_count = np.bincount(labels).min()
    if min_count < args.folds:
        raise ValueError(f"Smallest retained class has {min_count} samples, fewer than {args.folds} folds.")

    config = TrainConfig(
        image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, learning_rate=args.lr, weight_decay=args.weight_decay,
        architecture=args.architecture,
        optimizer=args.optimizer or ("adamw" if args.architecture.startswith("vit_") else "sgd"),
        no_class_weight=args.no_class_weight, select_by=args.select_by,
        warmup_epochs=args.warmup_epochs, grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha, no_wd_on_norm=args.no_wd_on_norm,
        zero_init_residual=args.zero_init_residual,
        val_fraction=args.val_fraction, seed=args.seed, initialization=args.init,
        checkpoint=str(args.init_from) if args.init_from else None,
    )
    (args.output / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({
        # `augment` NAO estava aqui, e o §7.17 registra a divida: a procedencia
        # da transformacao de avaliacao ficava so na linha de comando do shell, e
        # o §8 mede que ela sozinha vale 4,91pp. Duas linhas, como dito la.
        **asdict(config), "augment": args.augment,
        "task": args.task, "folds": args.folds,
        "meta": str(args.meta), "image_dir": str(args.image_dir), "n_images": len(records),
        "n_classes": len(classes), "device": str(device),
    }, indent=2) + "\n")

    train_transform, eval_transform = build_transforms(args.image_size, args.augment)
    fold_results = []
    started = time.time()
    if args.official_split:
        assert split_by_stem is not None
        splits = np.asarray([split_by_stem[record["stem"]] for record in records], dtype=str)
        train_idx = np.flatnonzero(splits == "train")
        test_idx = np.flatnonzero(splits == "test")
        validation_idx = np.flatnonzero(splits == "val")
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"official split is empty (train={len(train_idx)}, test={len(test_idx)})")
        if args.only_fold not in (0, 1):
            raise ValueError("--official-split has a single fold; use --only-fold 1 or omit it")
        seed_everything(args.seed + 1)
        validation_records = [records[i] for i in validation_idx] if len(validation_idx) else None
        log(f"Starting {args.init} official split ({len(train_idx):,} train / "
            f"{len(validation_idx):,} val / {len(test_idx):,} test)")
        fold_results.append(train_outer_fold(
            fold=1,
            train_records=[records[i] for i in train_idx],
            validation_records=validation_records,
            test_records=[records[i] for i in test_idx],
            n_classes=len(classes), train_transform=train_transform, eval_transform=eval_transform,
            config=config, workers=args.workers, device=device, output_dir=args.output,
        ))
    else:
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(records)), labels), start=1):
            if args.only_fold and fold != args.only_fold:
                continue
            seed_everything(args.seed + fold)
            if args.train_subsample and args.train_subsample < len(train_idx):
                # Só o treino encolhe; o fold de teste fica intacto para que a BAcc
                # seja comparável entre tamanhos. Estratificado, semente por fold.
                train_idx, _ = train_test_split(
                    train_idx, train_size=args.train_subsample,
                    stratify=labels[train_idx], random_state=args.seed + fold)
            log(f"Starting {args.init} fold {fold}/{args.folds} ({len(train_idx):,} train / {len(test_idx):,} test)")
            fold_results.append(train_outer_fold(
                fold=fold,
                train_records=[records[i] for i in train_idx],
                test_records=[records[i] for i in test_idx],
                n_classes=len(classes), train_transform=train_transform, eval_transform=eval_transform,
                config=config, workers=args.workers, device=device, output_dir=args.output,
            ))

    results = pd.DataFrame(fold_results)
    results.to_csv(args.output / "fold_metrics.csv", index=False, float_format="%.6f")
    summary = {
        "initialization": args.init,
        "n_images": len(records),
        "n_classes": len(classes),
        "elapsed_seconds": round(time.time() - started, 1),
        "folds": fold_results,
        "mean": {metric: float(results[metric].mean()) for metric in ("balanced_accuracy", "accuracy", "macro_f1")},
        "std": {metric: float(results[metric].std(ddof=0)) for metric in ("balanced_accuracy", "accuracy", "macro_f1")},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(
        f"{args.init} finished in {summary['elapsed_seconds']:.0f}s: "
        f"BAcc={summary['mean']['balanced_accuracy']:.4f}±{summary['std']['balanced_accuracy']:.4f}, "
        f"Acc={summary['mean']['accuracy']:.4f}±{summary['std']['accuracy']:.4f}, "
        f"F1={summary['mean']['macro_f1']:.4f}±{summary['std']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
