"""MINC-2500 official-split evaluation.

Uses split 1 from the MINC-2500 labels materialized in ``data/minc2500_meta.csv``:
train fits the models, val selects the MLP epoch, and test is reported once.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

# IMPORTA DA BIBLIOTECA, nao dos scripts de experimento. Era o que o §7bis
# pretendia ao extrair o pipeline de producao para `src/pipeline/` -- aqui a
# importacao antiga arrastava a cadeia
#   evaluate_full_factorial_ablation -> evaluate_multiscale_graph_fourier
#   -> extract_fourier_spectrum_features
# e o ultimo mudou para `src/extractors/dormant/` no §7quater, entao o import
# quebrou. O `sys.path` para `experiments/PRISM` tambem saiu: aquele diretorio
# deixou de existir no mesmo commit.
from protocol import scaled_splits  # noqa: E402
from models import DEVICE, seed_everything  # noqa: E402
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


def fit_lda(blocks, labels, train_idx, test_idx):
    values = np.hstack(blocks).astype(np.float32, copy=False)
    train_x = values[train_idx]
    test_x = values[test_idx]
    pipe_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(len(np.unique(labels))) / len(np.unique(labels)))
    train_x = scaler.fit_transform(pipe_imputer.fit_transform(train_x))
    test_x = scaler.transform(pipe_imputer.transform(test_x))
    pred = model.fit(train_x, labels[train_idx]).predict(test_x)
    truth = labels[test_idx]
    return {
        "bacc": balanced_accuracy_score(truth, pred),
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, average="macro"),
        "best_val_bacc": np.nan,
        "epochs_run": 0,
    }


def fit_mlp(blocks, labels, train_idx, val_idx, test_idx, n_classes, seed, epochs, patience):
    train_x, val_x, test_x = scaled_splits(blocks, train_idx, val_idx, test_idx)
    seed_everything(seed)
    model = WideBranchMLP([b.shape[1] for b in blocks], n_classes).to(DEVICE)
    counts = np.bincount(labels[train_idx], minlength=n_classes).astype(np.float32)
    weights = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(labels[train_idx]).long()),
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
        score = balanced_accuracy_score(labels[val_idx], pred_val)
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
    """Anexa uma linha, ABORTANDO se o cabecalho do arquivo nao casar.

    Sem a guarda, acrescentar uma coluna ao `row` produz um CSV com linhas de
    larguras diferentes que o pandas recusa a ler -- falha silenciosa na escrita
    e erro so na leitura, longe da causa. Aconteceu em 06/out ao acrescentar a
    coluna `seed` sobre um arquivo ja escrito sem ela (§8: nada na saida
    denunciava). Conserto do chamador: apagar o arquivo ou usar `--output` novo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            cabecalho = handle.readline().strip().split(",")
        if cabecalho != list(row):
            faltam = [c for c in row if c not in cabecalho]
            sobram = [c for c in cabecalho if c not in row]
            raise SystemExit(
                f"{path.name}: cabecalho existente nao casa com a linha nova "
                f"({len(cabecalho)} contra {len(row)} campos). "
                f"faltam no arquivo: {faltam} | sobram: {sobram}. "
                f"Apague o arquivo ou passe --output novo -- anexar produziria "
                f"um CSV ilegivel.")
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=not path.exists(), float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"], choices=["lda", "mlp"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/minc2500_official.csv")
    args = parser.parse_args()

    requested = sorted({name for condition in args.conditions for name in CONDITIONS[condition]}, key=list(HAND8 + ("ResNet50",)).index)
    cohort = load_blocks("minc2500", requested, allow_external=True)
    meta = pd.read_csv(ROOT / "data/minc2500_meta.csv")[["stem", "split"]]
    frame = pd.DataFrame({"stem": cohort.stems, "label": cohort.labels}).merge(meta, on="stem", how="left", validate="one_to_one")
    if frame["split"].isna().any():
        raise ValueError("some cohort stems are absent from data/minc2500_meta.csv")

    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    n_classes = len(np.unique(labels))
    where = frame["split"].to_numpy(str)
    train_idx = np.flatnonzero(where == "train")
    val_idx = np.flatnonzero(where == "val")
    test_idx = np.flatnonzero(where == "test")
    log(f"minc2500: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}, classes={n_classes}")
    for name in requested:
        log(f"  {name:<12} {cohort.blocks[name].shape[1]:>4}d")

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.classifier), str(r.condition)) for r in previous.itertuples()}
        log(f"resuming {len(done)} completed rows from {args.output}")

    for classifier in args.classifiers:
        for condition in args.conditions:
            if (classifier, condition) in done:
                continue
            names = CONDITIONS[condition]
            blocks = [cohort.blocks[name] for name in names]
            started = time.perf_counter()
            if classifier == "lda":
                metrics = fit_lda(blocks, labels, train_idx, test_idx)
            else:
                metrics = fit_mlp(blocks, labels, train_idx, val_idx, test_idx, n_classes, SEED, args.epochs, args.patience)
            row = {
                "dataset": "minc2500",
                "classifier": classifier,
                "condition": condition,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "n_classes": n_classes,
                "dimensions": int(sum(block.shape[1] for block in blocks)),
                "seconds": time.perf_counter() - started,
                **metrics,
            }
            append_row(args.output, row)
            log(f"{classifier:<3} | {condition:<11} | {row['dimensions']:4d}d | "
                f"BAcc {100 * row['bacc']:.2f} | Acc {100 * row['accuracy']:.2f} | "
                f"F1 {100 * row['macro_f1']:.2f} | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    log("\n" + detail.to_string(index=False))
    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
