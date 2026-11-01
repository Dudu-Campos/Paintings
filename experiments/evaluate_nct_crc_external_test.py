"""NCT-CRC external patient test evaluation.

Train on NCT-CRC-HE-100K and report only on CRC-VAL-HE-7K. The external
validation split contains different patients, so this is the protocol we want
for the paper-level positive control.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED = 42
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


def scale_external(train_blocks, val_blocks, test_blocks):
    xs_train, xs_val, xs_test = [], [], []
    for train, val, test in zip(train_blocks, val_blocks, test_blocks):
        scaler = StandardScaler().fit(train)
        xs_train.append(scaler.transform(train).astype(np.float32))
        xs_val.append(scaler.transform(val).astype(np.float32))
        xs_test.append(scaler.transform(test).astype(np.float32))
    return np.hstack(xs_train), np.hstack(xs_val), np.hstack(xs_test)


def fit_mlp(train_blocks, val_blocks, test_blocks, y_train, y_val, y_test, n_classes, seed, epochs, patience):
    train_x, val_x, test_x = scale_external(train_blocks, val_blocks, test_blocks)
    seed_everything(seed)
    model = WideBranchMLP([b.shape[1] for b in train_blocks], n_classes).to(DEVICE)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_train).long()),
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
            pred_val = model(val_tensor).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(y_val, pred_val)
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
    return {
        "bacc": balanced_accuracy_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "best_val_bacc": best,
        "epochs_run": epoch,
    }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not path.exists(), float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", default="nct_crc")
    parser.add_argument("--test-dataset", default="nct_crc_val")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/nct_crc_external_patient_mlp.csv")
    args = parser.parse_args()

    requested = sorted({name for condition in args.conditions for name in CONDITIONS[condition]}, key=list(HAND8 + ("ResNet50",)).index)
    train_cohort = load_blocks(args.train_dataset, requested, allow_external=True)
    test_cohort = load_blocks(args.test_dataset, requested, allow_external=True)

    encoder = LabelEncoder().fit(train_cohort.labels)
    unknown = sorted(set(test_cohort.labels) - set(encoder.classes_))
    if unknown:
        raise ValueError(f"test labels absent from train: {unknown}")
    y_all = encoder.transform(train_cohort.labels)
    y_test = encoder.transform(test_cohort.labels)
    n_classes = len(encoder.classes_)
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_all)), test_size=0.1, stratify=y_all, random_state=SEED
    )

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = set(previous["condition"].astype(str))
        log(f"resuming {len(done)} completed conditions from {args.output}")

    for condition in args.conditions:
        if condition in done:
            continue
        names = CONDITIONS[condition]
        train_blocks = [train_cohort.blocks[name][fit_idx] for name in names]
        val_blocks = [train_cohort.blocks[name][val_idx] for name in names]
        test_blocks = [test_cohort.blocks[name] for name in names]
        started = time.perf_counter()
        metrics = fit_mlp(
            train_blocks, val_blocks, test_blocks,
            y_all[fit_idx], y_all[val_idx], y_test,
            n_classes, SEED, args.epochs, args.patience,
        )
        row = {
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "condition": condition,
            "n_train_fit": int(len(fit_idx)),
            "n_train_val": int(len(val_idx)),
            "n_test": int(len(y_test)),
            "n_classes": n_classes,
            "dimensions": int(sum(block.shape[1] for block in train_blocks)),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        append_row(args.output, row)
        log(f"external | {condition:<11} | {row['dimensions']:4d}d | "
            f"BAcc {100 * row['bacc']:.2f} | Acc {100 * row['accuracy']:.2f} | "
            f"F1 {100 * row['macro_f1']:.2f} | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    log("\n" + detail.to_string(index=False))
    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
