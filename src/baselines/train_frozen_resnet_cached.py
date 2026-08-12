"""Fast frozen ImageNet ResNet-50 baseline via cached 2048-d embeddings.

The ImageNet backbone is evaluated once, with the canonical (non-augmented)
preprocessing.  Five-fold CV then trains only a fresh linear classification
head on the cached vectors.  This is the conventional frozen-encoder control.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet50_Weights, resnet50

from train_resnet_wikiart import build_transforms, load_records, make_loader, log, seed_everything


def extract_embeddings(records: list[dict], transform, cache_dir: Path, *, batch_size: int,
                       workers: int, device: torch.device) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / "resnet50_imagenet_v2_features.npy"
    stem_path = cache_dir / "stems.npy"
    label_path = cache_dir / "labels.npy"
    expected_stems = np.asarray([record["stem"] for record in records])
    expected_labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    if feature_path.exists() and stem_path.exists() and label_path.exists():
        cached_stems = np.load(stem_path, allow_pickle=True).astype(str)
        cached_labels = np.load(label_path)
        cached = np.load(feature_path, mmap_mode="r")
        if (cached.shape == (len(records), 2048) and np.array_equal(cached_stems, expected_stems)
                and np.array_equal(cached_labels, expected_labels)):
            log(f"Reusing cached ImageNet embeddings: {feature_path}")
            return cached
        log("Embedding cache does not match current records; rebuilding it.")

    loader = make_loader(records, transform, batch_size=batch_size, workers=workers,
                         shuffle=False, seed=42, device=device)
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                         shape=(len(records), 2048))
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, _, _ in loader:
            values = backbone(images.to(device, non_blocking=True)).float().cpu().numpy()
            features[offset:offset + len(values)] = values
            offset += len(values)
            if offset % 5000 < len(values) or offset == len(records):
                log(f"Cached embeddings: {offset:,}/{len(records):,} ({offset / (time.perf_counter() - started):.1f} images/s)")
    features.flush()
    np.save(stem_path, expected_stems)
    np.save(label_path, expected_labels)
    del backbone, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(feature_path, mmap_mode="r")


def train_head(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
               test_x: np.ndarray, test_y: np.ndarray, *, n_classes: int, seed: int,
               epochs: int, patience: int, learning_rate: float, weight_decay: float,
               batch_size: int, device: torch.device) -> tuple[dict, dict[str, torch.Tensor], np.ndarray]:
    seed_everything(seed)
    train_tensor = torch.from_numpy(np.asarray(train_x, dtype=np.float32)).to(device)
    val_tensor = torch.from_numpy(np.asarray(val_x, dtype=np.float32)).to(device)
    test_tensor = torch.from_numpy(np.asarray(test_x, dtype=np.float32)).to(device)
    train_labels = torch.from_numpy(train_y.astype(np.int64)).to(device)
    class_counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1.0 / class_counts) / (1.0 / class_counts).mean()
    model = nn.Linear(train_tensor.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                nesterov=True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size=batch_size, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))
    best_state, best_epoch, best_val, stale = None, 0, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            n_seen += len(labels)
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            val_prediction = model(val_tensor).argmax(1).cpu().numpy()
        val_bacc = balanced_accuracy_score(val_y, val_prediction)
        if val_bacc > best_val:
            best_state, best_epoch, best_val, stale = copy.deepcopy(model.state_dict()), epoch, float(val_bacc), 0
        else:
            stale += 1
        log(f"frozen_cached epoch {epoch:03d} loss={loss_sum/n_seen:.4f} stop_BAcc={val_bacc:.4f} best={best_val:.4f}@{best_epoch}")
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(test_tensor).argmax(1).cpu().numpy()
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "accuracy": float(accuracy_score(test_y, prediction)),
        "macro_f1": float(f1_score(test_y, prediction, average="macro", zero_division=0)),
        "best_epoch": best_epoch, "early_stop_bacc": best_val,
    }
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    return metrics, state, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cnn_frozen_cache"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256, help="Image embedding extraction batch size")
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this cached frozen baseline")
    seed_everything(args.seed)
    records, classes = load_records(args.meta, args.image_dir, "style", min_per_class=5)
    args.output.mkdir(parents=True, exist_ok=True)
    _, eval_transform = build_transforms(224)
    features = extract_embeddings(records, eval_transform, args.cache_dir, batch_size=args.batch_size,
                                  workers=args.workers, device=device)
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    stems = np.asarray([record["stem"] for record in records])
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    results = []
    started = time.time()
    for fold, (outer_train, outer_test) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        fit_train, validation = train_test_split(outer_train, test_size=args.val_fraction,
                                                  stratify=labels[outer_train], random_state=args.seed + fold)
        log(f"Starting frozen_cached fold {fold}/{args.folds} ({len(outer_train):,} train / {len(outer_test):,} test)")
        metrics, state, prediction = train_head(features[fit_train], labels[fit_train], features[validation], labels[validation],
                                                features[outer_test], labels[outer_test], n_classes=len(classes), seed=args.seed + fold,
                                                epochs=args.epochs, patience=args.patience, learning_rate=args.lr,
                                                weight_decay=args.weight_decay, batch_size=args.head_batch_size, device=device)
        torch.save({"head_state_dict": state, "classes": classes, "fold": fold, **metrics}, args.output / f"fold_{fold:02d}_best.pt")
        pd.DataFrame({"stem": stems[outer_test], "fold": fold, "y_true": labels[outer_test], "y_pred": prediction}).to_csv(args.output / f"fold_{fold:02d}_predictions.csv", index=False)
        results.append({"fold": fold, **metrics})
        log(f"frozen_cached fold {fold} complete: BAcc={metrics['balanced_accuracy']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}")
    result = pd.DataFrame(results)
    result.to_csv(args.output / "fold_metrics.csv", index=False, float_format="%.6f")
    summary = {
        "initialization": "imagenet_frozen_cached", "n_images": len(records), "n_classes": len(classes),
        "augmentation": "none; canonical embeddings cached once", "elapsed_seconds": round(time.time() - started, 1),
        "folds": results,
        "mean": {metric: float(result[metric].mean()) for metric in ("balanced_accuracy", "accuracy", "macro_f1")},
        "std": {metric: float(result[metric].std(ddof=0)) for metric in ("balanced_accuracy", "accuracy", "macro_f1")},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"frozen_cached finished in {summary['elapsed_seconds']:.0f}s: BAcc={summary['mean']['balanced_accuracy']:.4f}±{summary['std']['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
