"""O encoder uniforme por bloco penaliza o backbone na fusao de features?

`WideBranchMLP` da a todo bloco a mesma largura de encoder (`enc`), qualquer que
seja a dimensao de entrada. Na condicao `fusion` do §7.15 isso EXPANDE o Gabor
de 32d em 16x e COMPRIME o ResNet50 de 2048d em 4x -- o backbone entra com 76,8%
das dimensoes brutas e sai com 11,1% da largura da cabeca.

Varre a largura do encoder DO BACKBONE mantendo a nossa em 512, mais um controle
denso sem branches. Se o budget for a restricao, alargar o ramo do backbone tem
de mover o numero; se nao mover, a alocacao uniforme esta absolvida.
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_full_factorial_ablation import scaled_splits  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

HEAD = (2048, 1024, 1024)


class VariableBranchMLP(nn.Module):
    """Como `WideBranchMLP`, mas com uma largura de encoder POR bloco."""

    def __init__(self, sizes, encoder_widths, n_classes, head=HEAD):
        super().__init__()
        self.sizes = list(sizes)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.25))
            for size, width in zip(sizes, encoder_widths)
        ])
        layers, previous = [], int(sum(encoder_widths))
        for width in head:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, values):
        return self.head(torch.cat([
            encoder(part) for encoder, part in zip(self.encoders, torch.split(values, self.sizes, dim=1))
        ], dim=1))


class DenseMLP(nn.Module):
    """Controle sem branches: concatenacao crua direto na cabeca."""

    def __init__(self, total, n_classes, head=HEAD):
        super().__init__()
        layers, previous = [], total
        for width in head:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, values):
        return self.head(values)


def fit(model, blocks, labels, train_idx, val_idx, test_idx, n_classes, seed, epochs, patience):
    train_x, val_x, test_x = scaled_splits(blocks, train_idx, val_idx, test_idx)
    model = model.to(DEVICE)
    counts = np.bincount(labels[train_idx], minlength=n_classes).astype(np.float32)
    inverse = 1.0 / np.maximum(counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(inverse / inverse.mean(), device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(labels[train_idx]).long()),
        batch_size=4096, shuffle=True, generator=torch.Generator().manual_seed(seed))
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
            score = balanced_accuracy_score(labels[val_idx], model(val_tensor).argmax(1).cpu().numpy())
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
        "bacc": 100 * balanced_accuracy_score(truth, pred),
        "accuracy": 100 * accuracy_score(truth, pred),
        "macro_f1": 100 * f1_score(truth, pred, average="macro"),
        "best_val_bacc": 100 * best,
        "epochs_run": epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/minc2500_fusion_budget.csv")
    args = parser.parse_args()

    names = list(HAND8) + ["ResNet50"]
    cohort = load_blocks("minc2500", tuple(names), allow_external=True)
    meta = pd.read_csv(ROOT / "data/minc2500_meta.csv", dtype={"stem": str})[["stem", "split"]]
    frame = pd.DataFrame({"stem": cohort.stems, "label": cohort.labels}).merge(
        meta, on="stem", how="left", validate="one_to_one")
    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    n_classes = len(np.unique(labels))
    where = frame["split"].to_numpy(str)
    train_idx = np.flatnonzero(where == "train")
    val_idx = np.flatnonzero(where == "val")
    test_idx = np.flatnonzero(where == "test")

    ours = [cohort.blocks[name] for name in HAND8]
    backbone = cohort.blocks["ResNet50"]
    sizes_ours = [b.shape[1] for b in ours]
    size_backbone = backbone.shape[1]
    log(f"minc2500: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
    log(f"  nossos 8 blocos: {sizes_ours} = {sum(sizes_ours)}d | ResNet50 {size_backbone}d")

    # (blocos, larguras de encoder ou None para o controle denso)
    plans = {
        "hand-8":                 (ours, [512] * 8),
        "resnet-only":            ([backbone], [512]),
        "resnet-only-2048":       ([backbone], [2048]),
        "fusion-uniform-512":     (ours + [backbone], [512] * 9),
        "fusion-backbone-1024":   (ours + [backbone], [512] * 8 + [1024]),
        "fusion-backbone-2048":   (ours + [backbone], [512] * 8 + [2048]),
        "fusion-backbone-4096":   (ours + [backbone], [512] * 8 + [4096]),
        "fusion-proportional":    (ours + [backbone],
                                   [max(64, int(round(4608 * d / (sum(sizes_ours) + size_backbone))))
                                    for d in sizes_ours + [size_backbone]]),
        "fusion-dense":           (ours + [backbone], None),
    }

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.seed)) for r in pd.read_csv(args.output).itertuples()}
        log(f"resuming {len(done)} rows from {args.output}")

    for condition, (blocks, widths) in plans.items():
        for seed in args.seeds:
            if (condition, seed) in done:
                continue
            seed_everything(seed)
            total = int(sum(b.shape[1] for b in blocks))
            if widths is None:
                model = DenseMLP(total, n_classes)
                head_input = total
            else:
                model = VariableBranchMLP([b.shape[1] for b in blocks], widths, n_classes)
                head_input = int(sum(widths))
            started = time.perf_counter()
            metrics = fit(model, blocks, labels, train_idx, val_idx, test_idx,
                          n_classes, seed, args.epochs, args.patience)
            share = (widths[-1] / head_input) if (widths and len(blocks) > len(sizes_ours)) else np.nan
            row = {"condition": condition, "seed": seed, "dimensions": total,
                   "head_input": head_input, "backbone_share": share,
                   "seconds": time.perf_counter() - started, **metrics}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(args.output, mode="a", index=False,
                                       header=not args.output.exists(), float_format="%.4f")
            log(f"{condition:24s} seed {seed} | {total:5d}d -> cabeca {head_input:5d} | "
                f"BAcc {metrics['bacc']:.2f} | {metrics['parameters'] / 1e6:.1f}M par | "
                f"{metrics['epochs_run']:3d} ep | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    log("\n=== resumo (media +- desvio sobre seeds) ===")
    order = list(plans)
    summary = detail.groupby("condition").agg(
        n=("seed", "size"), dims=("dimensions", "first"), width=("head_input", "first"),
        share=("backbone_share", "first"), bacc=("bacc", "mean"), std=("bacc", "std"),
        params=("parameters", "first")).reindex(order)
    for name, r in summary.iterrows():
        share = "  --  " if np.isnan(r.share) else f"{100 * r.share:5.1f}%"
        log(f"{name:24s} n={int(r.n)} {int(r.dims):5d}d -> {int(r.width):5d} "
            f"backbone={share} BAcc {r.bacc:6.2f} +- {r['std']:.2f}  {r.params / 1e6:5.1f}M")


if __name__ == "__main__":
    main()
