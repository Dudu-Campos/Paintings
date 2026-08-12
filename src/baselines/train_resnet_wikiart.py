"""Train an end-to-end ResNet-50 on WikiArt, with or without ImageNet weights.

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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
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


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
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


def index_images(image_dir: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], list[Path]]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    log(f"Indexing images under {image_dir}...")
    by_exact: dict[tuple[str, str], Path] = {}
    by_style_title: dict[tuple[str, str], list[Path]] = {}
    for path in image_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = (path.parent.name, path.name)
        if key in by_exact:
            raise ValueError(f"Duplicate style/filename in image directory: {key}")
        by_exact[key] = path
        title_key = (path.parent.name, _title_part(path.name))
        by_style_title.setdefault(title_key, []).append(path)
    log(f"Indexed {len(by_exact):,} images across {len({style for style, _ in by_exact})} style directories.")
    return by_exact, by_style_title


def load_records(meta_path: Path, image_dir: Path, task: str, min_per_class: int) -> tuple[list[dict], list[str]]:
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")
    meta = pd.read_csv(meta_path)
    required = {"stem", "filename", task}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    by_exact, by_style_title = index_images(image_dir)
    meta = meta[["stem", "filename", task]].dropna().copy()
    meta["stem"] = meta["stem"].astype(str)
    meta["filename"] = meta["filename"].astype(str)

    paths: list[Path] = []
    exact_matches = fallback_matches = 0
    unresolved: list[str] = []
    for row in meta.itertuples(index=False):
        style = str(getattr(row, task))
        filename = row.filename
        path = by_exact.get((style, filename))
        if path is not None:
            exact_matches += 1
        else:
            candidates = by_style_title.get((style, _title_part(filename)), [])
            if not candidates:
                unresolved.append(f"{style}/{filename}")
                paths.append(None)  # Keep row/path indices aligned until filtering below.
                continue
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
            else:
                path = candidates[0]
            fallback_matches += 1
        paths.append(path)
    if unresolved:
        preview = "; ".join(unresolved[:5])
        raise ValueError(f"Could not resolve {len(unresolved):,} metadata images (first: {preview})")
    meta["path"] = paths
    log(f"Image matching: {exact_matches:,} exact, {fallback_matches:,} recovered from filename encoding.")
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


def build_model(initialization: str, n_classes: int) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if initialization in {"imagenet", "frozen"} else None
    model = resnet50(weights=weights)
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[str], np.ndarray, np.ndarray]:
    model.eval()
    true, predicted, stems = [], [], []
    for images, labels, batch_stems in loader:
        logits = model(images.to(device, non_blocking=True))
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        true.extend(labels.tolist())
        stems.extend(batch_stems)
    y_true = np.asarray(true, dtype=np.int64)
    y_pred = np.asarray(predicted, dtype=np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }, stems, y_true, y_pred


def train_outer_fold(*, fold: int, train_records: list[dict], test_records: list[dict],
                     n_classes: int, train_transform: transforms.Compose,
                     eval_transform: transforms.Compose, config: TrainConfig,
                     workers: int, device: torch.device, output_dir: Path) -> dict:
    outer_labels = np.fromiter((r["label"] for r in train_records), dtype=np.int64)
    train_indices, early_stop_indices = train_test_split(
        np.arange(len(train_records)),
        test_size=config.val_fraction,
        random_state=config.seed + fold,
        stratify=outer_labels,
    )
    fit_records = [train_records[index] for index in train_indices]
    early_stop_records = [train_records[index] for index in early_stop_indices]

    train_loader = make_loader(fit_records, train_transform, batch_size=config.batch_size,
                               workers=workers, shuffle=True, seed=config.seed + fold, device=device)
    stop_loader = make_loader(early_stop_records, eval_transform, batch_size=config.batch_size,
                              workers=workers, shuffle=False, seed=config.seed + fold, device=device)
    test_loader = make_loader(test_records, eval_transform, batch_size=config.batch_size,
                              workers=workers, shuffle=False, seed=config.seed + fold, device=device)

    model = build_model(config.initialization, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(fit_records, n_classes, device))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable_parameters, lr=config.learning_rate, momentum=0.9,
                                nesterov=True, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

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
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
                loss = criterion(logits, labels.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            n_seen += labels.size(0)
        scheduler.step()

        stop_metrics, _, _, _ = evaluate(model, stop_loader, device)
        stop_bacc = stop_metrics["balanced_accuracy"]
        if stop_bacc > best_stop_bacc:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_stop_bacc, stale_epochs = epoch, stop_bacc, 0
        else:
            stale_epochs += 1
        log(
            f"{config.initialization} fold {fold}  epoch {epoch:03d}  "
            f"loss={loss_sum / n_seen:.4f}  stop_BAcc={stop_bacc:.4f}  "
            f"best={best_stop_bacc:.4f}@{best_epoch}"
        )
        if stale_epochs >= config.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics, stems, y_true, y_pred = evaluate(model, test_loader, device)
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
    log(
        f"{config.initialization} fold {fold} complete: "
        f"BAcc={test_metrics['balanced_accuracy']:.4f}  Acc={test_metrics['accuracy']:.4f}  "
        f"F1={test_metrics['macro_f1']:.4f}  best_epoch={best_epoch}"
    )
    return {"fold": fold, "best_epoch": best_epoch, "early_stop_bacc": best_stop_bacc, **test_metrics}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-dir", required=True, type=Path, help="Root of WikiArt image folders.")
    parser.add_argument("--meta", type=Path, default=root / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--task", choices=["style", "artist"], default="style")
    parser.add_argument("--init", choices=["scratch", "frozen", "imagenet"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.03,
                        help="Shared learning rate for both initialization conditions.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("WARNING: CUDA is unavailable; ResNet-50 CV training will be slow on CPU.")
    args.output.mkdir(parents=True, exist_ok=True)
    records, classes = load_records(args.meta, args.image_dir, args.task, args.min_per_class)
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
        val_fraction=args.val_fraction, seed=args.seed, initialization=args.init,
    )
    (args.output / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({
        **asdict(config), "task": args.task, "folds": args.folds,
        "meta": str(args.meta), "image_dir": str(args.image_dir), "n_images": len(records),
        "n_classes": len(classes), "device": str(device),
    }, indent=2) + "\n")

    train_transform, eval_transform = build_transforms(args.image_size)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    started = time.time()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(records)), labels), start=1):
        seed_everything(args.seed + fold)
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
