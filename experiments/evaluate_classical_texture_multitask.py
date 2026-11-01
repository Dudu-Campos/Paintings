"""Baselines clássicas de textura isoladas no multitask125.

O §6 registra "HOG −1,21pp" e o §4 tem Haralick só no WikiArt -- que não existe
mais em disco. Aqui os dois saem no corpus vivo, com a decomposição inteira do
`Descriptors` (que já CONTÉM Haralick e LBP, §3) para separar duas leituras que
os números do WikiArt confundiam: o HOG é fraco, ou é redundante com o Haralick?

Sub-blocos do Descriptors pela ordem de `extract_descriptors_full.py:90`
--`(lab_hist 90, lab_stats 9, lbp 18, haralick 13)`.

Instrumento: LDA e MLP SEM branches (bloco único, branch não faz sentido),
5 folds, mesma convenção do §2.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_redundancy_selection_curve import PlainMLP  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
HAND8 = ("Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)", "Hilbert", "Fourier", "Gabor")
# (início, fim) dentro das 130 colunas de Descriptors
SUBBLOCKS = {
    "Lab histogram": (0, 90),
    "Lab statistics": (90, 99),
    "LBP": (99, 117),
    "Haralick/GLCM": (117, 130),
}


def fit_lda(values, labels, train_idx, test_idx, n_classes):
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(values[train_idx]))
    test_x = scaler.transform(imputer.transform(values[test_idx]))
    # Priors uniformes sao a convencao do projeto para BAcc: sem eles, em 125
    # classes desbalanceadas o LDA prediz cabeca e a BAcc cai de ~33 para ~13.
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                       priors=np.ones(n_classes) / n_classes)
    pred = model.fit(train_x, labels[train_idx]).predict(test_x)
    return pred


def fit_mlp(values, labels, train_idx, test_idx, n_classes, seed, epochs, patience):
    fit_idx, val_idx = train_test_split(train_idx, test_size=0.10, random_state=seed,
                                        stratify=labels[train_idx])
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(values[fit_idx])).astype(np.float32)
    val_x = scaler.transform(imputer.transform(values[val_idx])).astype(np.float32)
    test_x = scaler.transform(imputer.transform(values[test_idx])).astype(np.float32)

    seed_everything(seed)
    model = PlainMLP(train_x.shape[1], n_classes).to(DEVICE)
    counts = np.bincount(labels[fit_idx], minlength=n_classes).astype(np.float32)
    inverse = 1.0 / np.maximum(counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(inverse / inverse.mean(), device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x),
                                      torch.from_numpy(labels[fit_idx]).long()),
                        batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_tensor = torch.from_numpy(val_x).to(DEVICE)
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
        return model(torch.from_numpy(test_x).to(DEVICE)).argmax(1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hog-dir", type=Path, default=ROOT / "outputs/multitask_haralick_hog")
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "results/classical_texture_multitask125.csv")
    args = parser.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    descriptors = cohort.blocks["Descriptors"]
    if descriptors.shape[1] != 130:
        raise ValueError(f"Descriptors com {descriptors.shape[1]}d, esperado 130")

    hog_stems = np.load(args.hog_dir / "stems.npy", allow_pickle=True).astype(str)
    hog_matrix = np.load(args.hog_dir / "features.npy")
    order = pd.Index(hog_stems).get_indexer(stems)
    if (order < 0).any():
        raise ValueError(f"{int((order < 0).sum())} stems da coorte sem HOG extraído")
    hog_matrix = hog_matrix[order]
    log(f"multitask125: {len(stems):,} imagens, {n_classes} classes | "
        f"HOG {hog_matrix.shape[1]}d (13 Haralick + {hog_matrix.shape[1] - 13})")

    conditions = {name: descriptors[:, a:b] for name, (a, b) in SUBBLOCKS.items()}
    conditions["Descriptors (todo)"] = descriptors
    conditions["HOG"] = hog_matrix[:, 13:]
    conditions["Haralick + HOG"] = hog_matrix
    conditions["hand-8"] = cohort.matrix(HAND8)

    done = set()
    if args.output.exists():
        done = {(str(r.classifier), str(r.condition), int(r.fold))
                for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas de {args.output}")

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    for classifier in args.classifiers:
        for name, values in conditions.items():
            for number, (train_idx, test_idx) in enumerate(folds, 1):
                if (classifier, name, number) in done:
                    continue
                started = time.perf_counter()
                if classifier == "lda":
                    pred = fit_lda(values, labels, train_idx, test_idx, n_classes)
                else:
                    pred = fit_mlp(values, labels, train_idx, test_idx, n_classes,
                                   SEED, args.epochs, args.patience)
                truth = labels[test_idx]
                row = {"classifier": classifier, "condition": name, "fold": number,
                       "dims": int(values.shape[1]),
                       "bacc": 100 * balanced_accuracy_score(truth, pred),
                       "accuracy": 100 * accuracy_score(truth, pred),
                       "macro_f1": 100 * f1_score(truth, pred, average="macro"),
                       "seconds": time.perf_counter() - started}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row]).to_csv(args.output, mode="a", index=False,
                                           header=not args.output.exists(), float_format="%.4f")
                log(f"{classifier:3s} | {name:20s} | {row['dims']:5d}d | fold {number} | "
                    f"BAcc {row['bacc']:5.2f} | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    log("\n=== multitask125, 125 classes, 5 folds ===")
    for classifier, block in detail.groupby("classifier"):
        log(f"\n--- {classifier.upper()} ---")
        summary = block.groupby("condition").agg(
            dims=("dims", "first"), bacc=("bacc", "mean"), sd=("bacc", "std"),
            acc=("accuracy", "mean"), f1=("macro_f1", "mean")).sort_values("bacc")
        for name, r in summary.iterrows():
            log(f"  {name:20s} {int(r.dims):5d}d  BAcc {r.bacc:5.2f} +- {r.sd:4.2f}  "
                f"Acc {r.acc:5.2f}  F1 {r.f1:5.2f}")


if __name__ == "__main__":
    main()
