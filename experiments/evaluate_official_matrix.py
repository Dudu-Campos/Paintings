"""Matriz do artigo nos corpora de SPLIT OFICIAL (os quatro externos a arte).

Mesmas linhas de `evaluate_preprocess_effect_oilpainting.py`, que atende os tres
artisticos por validacao cruzada. A diferenca e so o protocolo: aqui ha uma
particao publicada (`train/val/test` no metadado, ou coorte de teste separada no
registry como o `nct_crc_val`), entao o analogo dos 5 folds sao 5 SEMENTES do
MLP sobre a mesma particao -- que e o que a grade de fine-tuning do §9.1 tambem
faz nestes datasets.

`nos` = hand-8 + grafo de similaridade (§3). Nunca inclui Text.
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_decision_fusion_official import prepare  # noqa: E402
from evaluate_minc2500_official import HAND8, WideBranchMLP  # noqa: E402
from evaluate_preprocess_effect_oilpainting import BACKBONES  # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features)
from evaluate_robustness_mlp_aug import BranchMLP, DEVICE, seed_everything  # noqa: E402
from registry import log  # noqa: E402

HEADS = {"narrow": BranchMLP, "wide": WideBranchMLP}
GRAPH = dict(n_neighbors=30, alpha=0.4, max_iter=100)


def side_block(path: Path, stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Um bloco externo (congelado ou classico) recortado por split."""
    values = np.load(path / ("cnn_full_features.npy" if (path / "cnn_full_features.npy").exists()
                             else "features.npy"))
    name = "cnn_full_stems.npy" if (path / "cnn_full_stems.npy").exists() else "stems.npy"
    index = pd.Index(np.load(path / name, allow_pickle=True).astype(str))
    out = {}
    for split, keys in stems.items():
        order = index.get_indexer(keys)
        if (order < 0).any():
            raise ValueError(f"{path}: {int((order < 0).sum())} stems ausentes em {split}")
        out[split] = values[order].astype(np.float32)
    return out


def fit(blocks, labels, n_classes, seed, epochs, patience, head, graph_source):
    parts, sizes = {"train": [], "val": [], "test": []}, []
    for block in blocks:
        scaler = StandardScaler().fit(block["train"])
        for split in parts:
            parts[split].append(scaler.transform(block[split]).astype(np.float32))
        sizes.append(block["train"].shape[1])

    if graph_source is not None:
        scaler = StandardScaler().fit(graph_source["train"])
        space = {s: scaler.transform(v).astype(np.float32) for s, v in graph_source.items()}
        oof, held = rich_similarity_features(
            space["train"], labels["train"], np.vstack([space["val"], space["test"]]),
            n_classes, seed=seed, **GRAPH)
        oof = prior_correct_rich(oof, labels["train"], n_classes)
        held = prior_correct_rich(held, labels["train"], n_classes)
        cut = len(labels["val"])
        for split, values in zip(("train", "val", "test"), (oof, held[:cut], held[cut:])):
            parts[split].append(values.astype(np.float32))
        sizes.append(oof.shape[1])

    data = {s: np.hstack(v) for s, v in parts.items()}
    seed_everything(seed)
    model = HEADS[head](sizes, n_classes).to(DEVICE)
    counts = np.bincount(labels["train"], minlength=n_classes).astype(np.float32)
    inverse = 1.0 / np.maximum(counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(inverse / inverse.mean(), device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(data["train"]),
                                      torch.from_numpy(labels["train"]).long()),
                        batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_tensor = torch.from_numpy(data["val"]).to(DEVICE)
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
            score = balanced_accuracy_score(labels["val"], model(val_tensor).argmax(1).cpu().numpy())
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
        pred = model(torch.from_numpy(data["test"]).to(DEVICE)).argmax(1).cpu().numpy()
    truth = labels["test"]
    return {"bacc": 100 * balanced_accuracy_score(truth, pred),
            "accuracy": 100 * accuracy_score(truth, pred),
            "macro_f1": 100 * f1_score(truth, pred, average="macro"),
            "dimensions": int(sum(sizes))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["food101", "nct_crc", "minc2500", "places_lt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--heads", nargs="+", default=["narrow"], choices=list(HEADS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.output = args.output or ROOT / f"results/{args.dataset}_final_table.csv"

    hand, labels, test_stems, classes, split_stems = prepare(args.dataset, tuple(HAND8))
    n_classes = len(classes)
    log(f"{args.dataset}: train={len(labels['train']):,} val={len(labels['val']):,} "
        f"test={len(labels['test']):,} | {n_classes} classes")

    hand_blocks = [{s: hand[s][i] for s in ("train", "val", "test")} for i in range(len(HAND8))]
    conditions = {}
    classical = ROOT / f"outputs/{args.dataset}_haralick_hog"
    if classical.exists():
        full = side_block(classical, split_stems)
        conditions["Haralick"] = ([{s: v[:, :13] for s, v in full.items()}], None)
        conditions["HOG"] = ([{s: v[:, 13:] for s, v in full.items()}], None)
    conditions["hand-8"] = (hand_blocks, None)
    graph_source = {s: np.hstack([b[s] for b in hand_blocks]) for s in ("train", "val", "test")}
    conditions["hand-8 + grafo"] = (hand_blocks, graph_source)
    for name, path in BACKBONES[args.dataset].items():
        if not (ROOT / path).exists():
            log(f"  pulando {name}: {path} ausente")
            continue
        block = side_block(ROOT / path, split_stems)
        conditions[name] = ([block], None)
        conditions[f"hand-8 + {name}"] = (hand_blocks + [block], None)
        conditions[f"hand-8 + grafo + {name}"] = (hand_blocks + [block], graph_source)

    done = set()
    if args.output.exists():
        done = {(str(r.head), str(r.condition), int(r.fold))
                for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    for head in args.heads:
        for name, (blocks, source) in conditions.items():
            for seed in args.seeds:
                if (head, name, seed) in done:
                    continue
                started = time.perf_counter()
                metrics = fit(blocks, labels, n_classes, seed, args.epochs, args.patience, head, source)
                row = {"head": head, "condition": name, "fold": seed,
                       "seconds": time.perf_counter() - started, **metrics}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                frame = pd.DataFrame([row])
                if args.output.exists():
                    frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
                frame.to_csv(args.output, mode="a", index=False,
                             header=not args.output.exists(), float_format="%.4f")
                log(f"{head:6s} | {name:34s} seed {seed} | {metrics['dimensions']:5d}d | "
                    f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    for head, block in detail.groupby("head"):
        log(f"\n=== {args.dataset}, split oficial, 5 sementes, MLP {head} ===")
        summary = block.groupby("condition").agg(dims=("dimensions", "first"), bacc=("bacc", "mean"),
                                                 sd=("bacc", "std"), acc=("accuracy", "mean"),
                                                 f1=("macro_f1", "mean")).sort_values("bacc")
        for name, r in summary.iterrows():
            log(f"  {name:34s} {int(r.dims):5d}d  BAcc {r.bacc:5.2f} +- {r.sd:4.2f}  "
                f"Acc {r.acc:5.2f}  F1 {r.f1:5.2f}")


if __name__ == "__main__":
    main()
