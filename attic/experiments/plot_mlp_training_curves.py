"""Curvas de treino/validação do branch-MLP de produção, fold 1.

Nunca tínhamos olhado a loss do nosso MLP -- só a BAcc de validação usada
pro early stopping. `train_one` (evaluate_robustness_mlp_aug.py) ganhou
`track_history=True` pra registrar train_loss/validation_loss/validation_bacc
por época sem mudar o comportamento default de quem já chama a função.

Roda só o fold 1 (mais barato) na condição "todas hand-designed" do dataset
pedido, mesmo split de evaluate_external_modalities (StratifiedKFold, SEED=42,
90/10 fit/validation dentro do treino do fold, mesma seed SEED+fold).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402
from evaluate_full_factorial_ablation import scaled_splits, SEED  # noqa: E402
from evaluate_robustness_mlp_aug import train_one  # noqa: E402

N_FOLDS = 5


def log(message: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--exclude", nargs="+", default=("ResNet50", "Multiscale"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/mlp_training_curves")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

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
    log(f"blocos: {names}")

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
            log(f"  ! {name}: {int(bad.sum())} valores não-finitos imputados pela mediana")
    dims = sum(b.shape[1] for b in blocks)
    log(f"{len(frame):,} imagens x {dims}d em {len(blocks)} blocos, {n_classes} classes")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    outer_train, outer_test = splits[args.fold - 1]
    fit, validation = train_test_split(
        outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + args.fold,
    )
    train_x, validation_x, test_x = scaled_splits(blocks, fit, validation, outer_test)

    log(f"treinando fold {args.fold}: fit={len(fit):,} val={len(validation):,} test={len(outer_test):,}")
    metrics, _ = train_one(
        train_x, labels[fit], validation_x, labels[validation], test_x, labels[outer_test],
        [b.shape[1] for b in blocks], n_classes, SEED + args.fold, args.epochs, args.patience,
        track_history=True, label_smoothing=args.label_smoothing, weight_decay=args.weight_decay,
    )
    history = pd.DataFrame(metrics.pop("history"))
    log(f"final: bacc={metrics['bacc']:.4f} best_epoch={metrics['best_epoch']} "
        f"ran={len(history)} epochs")

    tag = f"{args.dataset}_fold{args.fold}_ls{args.label_smoothing:g}_wd{args.weight_decay:g}"
    history.to_csv(args.output / f"{tag}.csv", index=False)

    figure, (loss_axis, bacc_axis) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis in (loss_axis, bacc_axis):
        axis.set_facecolor("#fbfbfa")
        axis.grid(color="#dedede", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    loss_axis.plot(history["epoch"], history["train_loss"], color="#2a78d6", label="train", linewidth=1.6)
    loss_axis.plot(history["epoch"], history["validation_loss"], color="#e34948", label="validation", linewidth=1.6)
    loss_axis.axvline(metrics["best_epoch"], color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("cross-entropy loss (weighted)")
    loss_axis.set_title("Loss")
    loss_axis.legend(frameon=False, fontsize=9.5)

    bacc_axis.plot(history["epoch"], history["validation_bacc"] * 100, color="#1baf7a", linewidth=1.6)
    bacc_axis.axvline(metrics["best_epoch"], color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0,
                       label=f"best epoch ({metrics['best_epoch']})")
    bacc_axis.axhline(metrics["bacc"] * 100, color="#4a3aa7", linestyle=":", linewidth=1.2,
                       label=f"test BAcc ({100*metrics['bacc']:.2f}%)")
    bacc_axis.set_xlabel("epoch")
    bacc_axis.set_ylabel("Balanced Accuracy (%)")
    bacc_axis.set_title("Validation BAcc (early-stopping criterion)")
    bacc_axis.legend(frameon=False, fontsize=9.5)

    figure.suptitle(f"MLP training curves -- {tag}, {dims}d, {n_classes} classes", fontsize=12.5)
    figure.tight_layout()
    figure.savefig(args.output / f"{tag}.png", dpi=200, facecolor="white")
    log(f"saved: {args.output / f'{tag}.png'}")


if __name__ == "__main__":
    main()
