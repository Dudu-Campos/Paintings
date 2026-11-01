"""Importância por permutação condicional das modalidades no branch-MLP.

Treina o MLP de produção uma vez por fold e, no conjunto de teste intocado,
embaralha as linhas de **uma** modalidade de cada vez (mantendo todas as
outras alinhadas). A queda de BAcc mede se a fusão usa informação *única*
daquela modalidade, condicionada às demais -- é a mesma pergunta do Shapley,
mas a O(m) treinos em vez de O(2^m).

Diferença importante em relação ao Shapley exato do achado 1: aqui a rede
**não é retreinada** sem a modalidade, então o número mede
*insubstituibilidade na rede treinada*, não *contribuição marginal*. O projeto
já viu os dois divergirem (nota do achado 16: desligar Text derruba 6,46pp
apesar do Shapley modesto). Ler como complementar, não como substituto.

Motivação de 22/ago: Gabor entrou no pipeline (validado no Pandora) e nunca
passou por nenhuma medida de importância junto com as outras modalidades.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402
from evaluate_full_factorial_ablation import scaled_splits, SEED  # noqa: E402
from evaluate_robustness_mlp_aug import BranchMLP, DEVICE, seed_everything  # noqa: E402

N_FOLDS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def train_model(train_x, train_y, validation_x, validation_y, branch_sizes,
                n_classes, seed, max_epochs, patience, label_smoothing):
    """Mesma receita de `train_one`, mas devolve o modelo em vez de descartá-lo."""
    seed_everything(seed)
    model = BranchMLP(branch_sizes, n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed))
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE, non_blocking=True)

    best_score, best_state, stale = -1.0, None, 0
    for _epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation_tensor).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(validation_y, prediction)
        if score > best_score:
            best_score, best_state, stale = float(score), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    del validation_tensor
    return model


def score_matrix(model: BranchMLP, values: np.ndarray, labels: np.ndarray) -> float:
    with torch.no_grad():
        tensor = torch.from_numpy(values).to(DEVICE, non_blocking=True)
        prediction = model(tensor).argmax(1).cpu().numpy()
        del tensor
    return balanced_accuracy_score(labels, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50",),
                        help="modalidades fora da análise; ResNet sai por "
                             "padrão porque a alegação do paper é sem "
                             "supervisão externa")
    parser.add_argument("--repeats", type=int, default=10,
                        help="permutações por modalidade por fold")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/modality_permutation_importance.csv")
    args = parser.parse_args()

    meta_path, label_column = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / meta_path).dropna(subset=[label_column])
    meta["stem"] = meta["stem"].astype(str)
    meta["label"] = meta[label_column].astype(str)

    available = {}
    for name in BLOCKS:
        if name in args.exclude:
            continue
        block = load_block(args.dataset, name)
        if block is None:
            log(f"{name}: não extraído, pulando")
        else:
            available[name] = block
    names = list(available)

    frame = meta[["stem", "label"]]
    for name, block in available.items():
        frame = frame.merge(block, on="stem", how="inner")

    counts = frame["label"].value_counts()
    if args.classes_file:
        def normalize(value: str) -> str:
            return value.replace("_", " ").replace("-", " ").replace("ï", "i").strip().lower()
        wanted = {normalize(line) for line in args.classes_file.read_text().splitlines() if line.strip()}
        by_norm = {normalize(label): label for label in counts.index}
        missing = wanted - set(by_norm)
        if missing:
            raise ValueError(f"classes não encontradas: {missing}")
        frequent = [by_norm[name] for name in wanted]
    else:
        frequent = counts[counts >= args.min_per_class].index
    frame = frame[frame["label"].isin(frequent)]

    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    n_classes = len(np.unique(labels))
    blocks = [frame[[c for c in frame.columns if c.startswith(f"{name}_")]].to_numpy(dtype=np.float32)
              for name in names]
    for name, values in zip(names, blocks):
        bad = ~np.isfinite(values)
        if bad.any():
            medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
            log(f"  ! {name}: {int(bad.sum())} não-finitos imputados pela mediana")

    branch_sizes = [b.shape[1] for b in blocks]
    # Fatias de coluna na matriz concatenada, na mesma ordem dos branches.
    bounds, start = [], 0
    for size in branch_sizes:
        bounds.append((start, start + size))
        start += size
    log(f"{len(frame):,} imagens x {start}d em {len(blocks)} blocos, {n_classes} classes")
    log(f"blocos: {list(zip(names, branch_sizes))}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))

    rows = []
    for fold, (outer_train, outer_test) in enumerate(splits[:args.folds], start=1):
        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train],
            random_state=SEED + fold)
        train_x, validation_x, test_x = scaled_splits(blocks, fit, validation, outer_test)
        model = train_model(train_x, labels[fit], validation_x, labels[validation],
                            branch_sizes, n_classes, SEED + fold, args.epochs,
                            args.patience, args.label_smoothing)
        baseline = score_matrix(model, test_x, labels[outer_test])
        log(f"fold {fold}: baseline BAcc={100 * baseline:.2f}")

        generator = np.random.default_rng(SEED + fold)
        for name, (begin, end) in zip(names, bounds):
            drops = []
            for _repeat in range(args.repeats):
                permuted = test_x.copy()
                order = generator.permutation(len(permuted))
                permuted[:, begin:end] = permuted[order, begin:end]
                drops.append(baseline - score_matrix(model, permuted, labels[outer_test]))
            rows.append({"fold": fold, "modality": name,
                         "dimensions": end - begin, "baseline_bacc": baseline,
                         "drop_mean": float(np.mean(drops)),
                         "drop_std": float(np.std(drops))})
            log(f"  {name:<14} {end - begin:>5}d  queda={100 * np.mean(drops):>6.2f}pp")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.6f")

    summary = (result.groupby("modality")
               .agg(dimensions=("dimensions", "first"),
                    drop=("drop_mean", "mean"), std=("drop_mean", "std"))
               .sort_values("drop", ascending=False))
    print()
    print("Queda de BAcc ao permutar cada modalidade (pp), média sobre folds:")
    print((summary.assign(drop=summary["drop"] * 100, std=summary["std"] * 100)).round(2))
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
