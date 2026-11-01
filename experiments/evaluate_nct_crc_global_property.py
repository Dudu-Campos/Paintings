"""NCT-CRC global-property evaluation.

Four conditions, written incrementally:

* descriptors: global colour/texture descriptors only
* hand-8: the visual production bank without Multiscale/Text
* resnet: frozen ImageNet ResNet50 features
* fusion: hand-8 plus frozen ResNet50
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_full_factorial_ablation import scaled_splits  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
HAND8 = ("Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)", "Hilbert", "Fourier", "Gabor")
CONDITIONS = {
    "descriptors": ("Descriptors",),
    "hand-8": HAND8,
    "resnet": ("ResNet50",),
    "fusion": HAND8 + ("ResNet50",),
}


class WideBranchMLP(nn.Module):
    def __init__(self, sizes, n_classes, enc=512, head=(2048, 1024, 1024)):
        super().__init__()
        self.sizes = list(sizes)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, enc), nn.BatchNorm1d(enc), nn.GELU(), nn.Dropout(0.25))
            for size in sizes
        ])
        layers, previous = [], enc * len(sizes)
        for width in head:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, values):
        return self.head(torch.cat([
            encoder(part) for encoder, part in zip(self.encoders, torch.split(values, self.sizes, dim=1))
        ], dim=1))


def fit_mlp(blocks, labels, fit_idx, val_idx, test_idx, n_classes, seed, epochs, patience):
    train_x, val_x, test_x = scaled_splits(blocks, fit_idx, val_idx, test_idx)
    seed_everything(seed)
    model = WideBranchMLP([b.shape[1] for b in blocks], n_classes).to(DEVICE)
    counts = np.bincount(labels[fit_idx], minlength=n_classes).astype(np.float32)
    weights = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(labels[fit_idx]).long()),
        batch_size=4096,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_tensor = torch.from_numpy(val_x).to(DEVICE)
    test_tensor = torch.from_numpy(test_x).to(DEVICE)
    best, state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pred = model(val_tensor).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(labels[val_idx], pred)
        if score > best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(test_tensor).argmax(1).cpu().numpy()
    truth = labels[test_idx]
    return {
        "bacc": balanced_accuracy_score(truth, pred),
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, average="macro"),
        "best_val_bacc": best,
        "epochs_run": epoch,
    }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not path.exists(), float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nct_crc")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/nct_crc_global_property_mlp.csv")
    args = parser.parse_args()

    requested = sorted({name for condition in args.conditions for name in CONDITIONS[condition]}, key=list(HAND8 + ("ResNet50",)).index)
    cohort = load_blocks(args.dataset, requested, allow_external=True)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    log(f"{args.dataset}: {len(labels):,} images, {n_classes} classes")
    for name in requested:
        log(f"  {name:<12} {cohort.blocks[name].shape[1]:>4}d")

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.condition), int(r.fold)) for r in previous.itertuples()}
        log(f"resuming {len(done)} completed rows from {args.output}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    for fold, (train, test) in enumerate(splits[: args.folds], start=1):
        fit, val = train_test_split(train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
        for condition in args.conditions:
            if (condition, fold) in done:
                continue
            names = CONDITIONS[condition]
            blocks = [cohort.blocks[name] for name in names]
            started = time.perf_counter()
            metrics = fit_mlp(blocks, labels, fit, val, test, n_classes, SEED + fold, args.epochs, args.patience)
            row = {
                "dataset": args.dataset,
                "condition": condition,
                "fold": fold,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_classes": n_classes,
                "dimensions": int(sum(block.shape[1] for block in blocks)),
                "seconds": time.perf_counter() - started,
                **metrics,
            }
            append_row(args.output, row)
            log(f"fold {fold}/{args.folds} | {condition:<11} | {row['dimensions']:4d}d | "
                f"BAcc {100 * row['bacc']:.2f} | Acc {100 * row['accuracy']:.2f} | "
                f"F1 {100 * row['macro_f1']:.2f} | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    summary = detail.groupby("condition").agg(
        folds=("fold", "count"),
        dimensions=("dimensions", "first"),
        bacc_mean=("bacc", "mean"),
        bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
    ).reset_index()
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    log("\n" + summary.to_string(index=False))
    log(f"saved: {args.output}")
    log(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
