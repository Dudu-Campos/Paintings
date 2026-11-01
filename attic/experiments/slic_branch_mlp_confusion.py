"""OOF confusion analysis for the current SLIC-only branch-fusion MLP.

This reproduces the five outer folds and the internal early stopping used in
``slic_branch_mlp_fusion.py``.  Predictions are made only for each untouched
outer test fold.  Cached representations are used; no image or graph feature
is regenerated.
"""

from __future__ import annotations

import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_branch_mlp_fusion import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    BranchFusionMLP,
    seed_everything,
)
from slic_oof_stacking import load_dataset


OUTPUT_PNG = ROOT / "paper/slic_branch_mlp_confusion.png"
OUTPUT_RAW_CSV = ROOT / "paper/slic_branch_mlp_confusion_raw.csv"
OUTPUT_NORM_CSV = ROOT / "paper/slic_branch_mlp_confusion_normalized.csv"
OUTPUT_CLASS_CSV = ROOT / "paper/slic_branch_mlp_confusion_by_class.csv"
OUTPUT_PAIRS_CSV = ROOT / "paper/slic_branch_mlp_confusion_pairs.csv"
OUTPUT_PREDICTIONS_CSV = ROOT / "paper/slic_branch_mlp_oof_predictions.csv"


def train_and_predict(
    branch_sizes: list[int],
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    test_values: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    """Train exactly the current branch MLP and predict one outer test fold."""
    seed_everything(seed)
    model = BranchFusionMLP(branch_sizes, n_classes).to(DEVICE)
    class_counts = np.bincount(
        train_labels, minlength=n_classes
    ).astype(np.float32)
    class_weights = (1 / class_counts) / (1 / class_counts).mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_values),
            torch.from_numpy(train_labels).long(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )
    validation_tensor = torch.from_numpy(validation_values).to(
        DEVICE, non_blocking=True
    )
    test_tensor = torch.from_numpy(test_values).to(DEVICE, non_blocking=True)

    best_validation = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_values), batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(validation_tensor).argmax(dim=1).cpu().numpy()
            )
        validation_score = balanced_accuracy_score(
            validation_labels, validation_prediction
        )
        if validation_score > best_validation:
            best_validation = float(validation_score)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(test_tensor).argmax(dim=1).cpu().numpy()
    elapsed = time.perf_counter() - started

    del model, validation_tensor, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, best_epoch, elapsed


def save_tables(
    frame: pd.DataFrame,
    labels: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(labels))
    matrix = confusion_matrix(true, predicted, labels=indices)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(OUTPUT_RAW_CSV)
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(
        OUTPUT_NORM_CSV
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=indices, zero_division=0
    )
    class_rows = []
    pair_rows = []
    for source, source_label in enumerate(labels):
        errors = matrix[source].copy()
        errors[source] = 0
        target = int(errors.argmax())
        class_rows.append({
            "style": source_label,
            "support": int(support[source]),
            "correct": int(matrix[source, source]),
            "recall": float(recall[source]),
            "precision": float(precision[source]),
            "f1": float(f1[source]),
            "top_confusion": labels[target],
            "top_confusion_count": int(errors[target]),
            "top_confusion_rate": float(
                errors[target] / support[source] if support[source] else 0
            ),
        })
        for target, count in enumerate(matrix[source]):
            if source != target and count:
                pair_rows.append({
                    "true_style": source_label,
                    "predicted_style": labels[target],
                    "count": int(count),
                    "rate_within_true_style": float(count / support[source]),
                })
    by_class = pd.DataFrame(class_rows).sort_values("recall")
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["rate_within_true_style", "count"], ascending=False
    )
    by_class.to_csv(OUTPUT_CLASS_CSV, index=False)
    pairs.to_csv(OUTPUT_PAIRS_CSV, index=False)

    prediction_frame = pd.DataFrame({
        "stem": frame["stem"].astype(str),
        "fold": folds,
        "true_style": labels[true],
        "predicted_style": labels[predicted],
        "correct": true == predicted,
    })
    prediction_frame.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    return matrix, normalized, by_class, pairs


def make_plot(
    normalized: np.ndarray,
    labels: np.ndarray,
    by_class: pd.DataFrame,
    bacc: float,
    macro_f1: float,
) -> None:
    display_labels = [label.replace("_", " ") for label in labels]
    figure = plt.figure(figsize=(20, 13))
    grid = figure.add_gridspec(1, 2, width_ratios=[3.3, 1.15], wspace=0.3)
    matrix_axis = figure.add_subplot(grid[0, 0])
    recall_axis = figure.add_subplot(grid[0, 1])

    image = matrix_axis.imshow(
        normalized, cmap="Blues", vmin=0, vmax=1, aspect="equal"
    )
    matrix_axis.set_xticks(np.arange(len(labels)), display_labels)
    matrix_axis.set_yticks(np.arange(len(labels)), display_labels)
    matrix_axis.tick_params(axis="x", rotation=90, labelsize=8)
    matrix_axis.tick_params(axis="y", labelsize=8)
    matrix_axis.set_xlabel("Estilo predito")
    matrix_axis.set_ylabel("Estilo verdadeiro")
    matrix_axis.set_title(
        "Matriz de confusão normalizada por estilo verdadeiro\n"
        f"OOF 5-fold — balanced accuracy = {bacc:.3f}; macro-F1 = {macro_f1:.3f}"
    )
    colorbar = figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Fração dentro do estilo verdadeiro")

    ordered = by_class.sort_values("recall", ascending=True)
    bar_labels = [label.replace("_", " ") for label in ordered["style"]]
    colors = ["#d95f5f" if value < bacc else "#3977b8" for value in ordered["recall"]]
    recall_axis.barh(bar_labels, ordered["recall"], color=colors)
    recall_axis.axvline(
        bacc, color="#222222", linestyle="--", linewidth=1,
        label=f"média = {bacc:.3f}",
    )
    recall_axis.set_xlim(0, 1)
    recall_axis.set_xlabel("Recall")
    recall_axis.set_title("Recall por estilo")
    recall_axis.tick_params(axis="y", labelsize=8)
    recall_axis.grid(axis="x", alpha=0.2)
    recall_axis.legend(loc="lower right")

    figure.suptitle(
        "MLP ramificado — SLIC multigrafo + descritores + Hilbert + sentimento",
        fontsize=15,
        y=0.995,
    )
    figure.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame, groups = load_dataset()
    encoder = LabelEncoder()
    true = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    labels = encoder.classes_
    n_classes = len(labels)
    branch_sizes = [len(columns) for columns in groups.values()]
    columns = [column for group in groups.values() for column in group]
    values = frame[columns].to_numpy(dtype=np.float32)

    predicted = np.full(len(frame), -1, dtype=np.int64)
    folds = np.zeros(len(frame), dtype=np.int8)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    outer_splits = list(
        splitter.split(np.zeros(len(true), dtype=np.uint8), true)
    )
    log(
        f"Current branch MLP: {len(frame):,} images, {n_classes} styles, "
        f"device={DEVICE}"
    )
    for fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=true[outer_train],
            random_state=SEED + fold,
        )
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[train]).astype(np.float32)
        validation_values = scaler.transform(values[validation]).astype(
            np.float32
        )
        test_values = scaler.transform(values[outer_test]).astype(np.float32)
        fold_prediction, best_epoch, elapsed = train_and_predict(
            branch_sizes,
            train_values,
            true[train],
            validation_values,
            true[validation],
            test_values,
            n_classes,
            SEED + fold + 10_000,
        )
        predicted[outer_test] = fold_prediction
        folds[outer_test] = fold
        score = balanced_accuracy_score(true[outer_test], fold_prediction)
        log(
            f"Fold {fold}/{N_FOLDS}: BAcc={score:.4f}, "
            f"epoch={best_epoch}, {elapsed:.1f}s"
        )

    if np.any(predicted < 0) or np.any(folds == 0):
        raise RuntimeError("OOF predictions are incomplete")
    bacc = float(balanced_accuracy_score(true, predicted))
    macro_f1 = float(f1_score(true, predicted, average="macro"))
    accuracy = float(np.mean(true == predicted))
    _, normalized, by_class, pairs = save_tables(
        frame, labels, true, predicted, folds
    )
    make_plot(normalized, labels, by_class, bacc, macro_f1)

    log(
        f"OOF summary: accuracy={accuracy:.4f}, BAcc={bacc:.4f}, "
        f"macro-F1={macro_f1:.4f}"
    )
    log("Lowest recalls:\n" + by_class.head(8).to_string(index=False))
    log("Largest directional confusions:\n" + pairs.head(12).to_string(index=False))
    for path in (
        OUTPUT_PNG,
        OUTPUT_CLASS_CSV,
        OUTPUT_PAIRS_CSV,
        OUTPUT_PREDICTIONS_CSV,
        OUTPUT_RAW_CSV,
        OUTPUT_NORM_CSV,
    ):
        log(f"Saved: {path}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
