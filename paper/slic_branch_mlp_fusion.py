"""SLIC-only neural fusion: flat MLP versus representation-specific branches.

The outer test fold is never used for early stopping.  For each of the five
outer folds, 10% of that fold's training partition is reserved as an internal
validation set.  Both architectures use identical splits and optimization.
No image or graph extraction is performed.
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
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fusion_curve_slic import N_FOLDS, ROOT, SEED, log
from slic_oof_stacking import load_dataset


OUTPUT_CSV = ROOT / "paper/slic_branch_mlp_fusion_results.csv"
DETAIL_CSV = ROOT / "paper/slic_branch_mlp_fusion_folds.csv"
OUTPUT_PNG = ROOT / "paper/slic_branch_mlp_fusion.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BRANCH_WIDTH = 128


class FlatMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for width in (512, 512, 256, 128):
            layers.extend([
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout(0.35),
            ])
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class BranchFusionMLP(nn.Module):
    def __init__(
        self, branch_sizes: list[int], n_classes: int
    ) -> None:
        super().__init__()
        self.branch_sizes = branch_sizes
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, BRANCH_WIDTH),
                nn.BatchNorm1d(BRANCH_WIDTH),
                nn.GELU(),
                nn.Dropout(0.25),
            )
            for size in branch_sizes
        ])
        fused_width = len(branch_sizes) * BRANCH_WIDTH
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_width, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, n_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        branches = torch.split(values, self.branch_sizes, dim=1)
        embeddings = [
            encoder(branch) for encoder, branch in zip(self.encoders, branches)
        ]
        return self.fusion_head(torch.cat(embeddings, dim=1))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_and_evaluate(
    model_factory,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[float, int, float]:
    seed_everything(seed)
    model = model_factory().to(DEVICE)
    class_counts = np.bincount(
        train_labels, minlength=n_classes
    ).astype(np.float32)
    class_weights = (1 / class_counts) / (1 / class_counts).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights, dtype=torch.float32, device=DEVICE
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
    test_tensor = torch.from_numpy(test_values).to(
        DEVICE, non_blocking=True
    )

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
        test_prediction = model(test_tensor).argmax(dim=1).cpu().numpy()
    test_score = balanced_accuracy_score(test_labels, test_prediction)
    elapsed = time.perf_counter() - started

    del model, validation_tensor, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return float(test_score), best_epoch, elapsed


def save_summary(
    fold_rows: list[dict],
    parameters: dict[str, int],
) -> pd.DataFrame:
    detail = pd.DataFrame(fold_rows)
    detail.to_csv(DETAIL_CSV, index=False)
    rows = []
    for architecture in ("Flat MLP", "Branch fusion MLP"):
        subset = detail[detail["architecture"] == architecture].sort_values(
            "fold"
        )
        scores = subset["bacc"].to_numpy()
        epochs = subset["best_epoch"].to_numpy()
        rows.append({
            "architecture": architecture,
            "parameters": parameters[architecture],
            "bacc_mean": scores.mean(),
            "bacc_std": scores.std(),
            "best_epoch_mean": epochs.mean(),
            "elapsed_seconds": subset["elapsed_seconds"].sum(),
            **{
                f"fold_{index + 1}": score
                for index, score in enumerate(scores)
            },
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    return result


def make_plot(result: pd.DataFrame, fold_rows: list[dict]) -> None:
    detail = pd.DataFrame(fold_rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    pivot = detail.pivot(
        index="fold", columns="architecture", values="bacc"
    )
    for fold, row in pivot.iterrows():
        axes[0].plot(
            [0, 1],
            [row["Flat MLP"], row["Branch fusion MLP"]],
            "o-",
            color="#888888",
            alpha=0.65,
        )
        axes[0].text(
            1.03,
            row["Branch fusion MLP"],
            f"F{fold}",
            va="center",
            fontsize=8,
        )
    axes[0].set_xticks([0, 1], ["Flat MLP", "Branch fusion"])
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Paired outer folds")
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#777777", "#2a78d6"]
    bars = axes[1].bar(
        result["architecture"],
        result["bacc_mean"],
        yerr=result["bacc_std"],
        capsize=5,
        color=colors,
    )
    for bar, mean in zip(bars, result["bacc_mean"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}",
            ha="center",
            fontsize=10,
        )
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Mean ± fold standard deviation")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=12)
    figure.suptitle(
        "SLIC-only neural representation fusion — internal early stopping"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame, groups = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    branch_sizes = [len(columns) for columns in groups.values()]
    columns = [column for group in groups.values() for column in group]
    values = frame[columns].to_numpy(dtype=np.float32)

    flat_factory = lambda: FlatMLP(values.shape[1], n_classes)
    branch_factory = lambda: BranchFusionMLP(branch_sizes, n_classes)
    parameters = {
        "Flat MLP": parameter_count(flat_factory()),
        "Branch fusion MLP": parameter_count(branch_factory()),
    }
    log(
        f"SLIC-only cohort: {len(frame):,} images, {n_classes} styles, "
        f"device={DEVICE}"
    )
    log(
        "Branches: " + ", ".join(
            f"{name}={len(group)}" for name, group in groups.items()
        )
    )
    log(
        f"Parameters: flat={parameters['Flat MLP']:,}, "
        f"branch={parameters['Branch fusion MLP']:,}"
    )

    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    outer_splits = list(
        splitter.split(np.zeros(len(y), dtype=np.uint8), y)
    )
    fold_rows: list[dict] = []
    for fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        train, validation = train_test_split(
            outer_train,
            test_size=0.1,
            stratify=y[outer_train],
            random_state=SEED + fold,
        )
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[train]).astype(np.float32)
        validation_values = scaler.transform(values[validation]).astype(
            np.float32
        )
        test_values = scaler.transform(values[outer_test]).astype(np.float32)

        for architecture, factory, seed_offset in (
            ("Flat MLP", flat_factory, 0),
            ("Branch fusion MLP", branch_factory, 10_000),
        ):
            score, best_epoch, elapsed = train_and_evaluate(
                factory,
                train_values,
                y[train],
                validation_values,
                y[validation],
                test_values,
                y[outer_test],
                n_classes,
                SEED + fold + seed_offset,
            )
            fold_rows.append({
                "architecture": architecture,
                "fold": fold,
                "bacc": score,
                "best_epoch": best_epoch,
                "elapsed_seconds": elapsed,
                "n_outer_train": len(outer_train),
                "n_fit_train": len(train),
                "n_internal_validation": len(validation),
                "n_outer_test": len(outer_test),
            })
            pd.DataFrame(fold_rows).to_csv(DETAIL_CSV, index=False)
            log(
                f"Fold {fold}/{N_FOLDS} | {architecture:<17} "
                f"BAcc={score:.4f}, epoch={best_epoch}, {elapsed:.1f}s"
            )

    result = save_summary(fold_rows, parameters)
    make_plot(result, fold_rows)
    flat_score = result.loc[
        result["architecture"] == "Flat MLP", "bacc_mean"
    ].iloc[0]
    branch_score = result.loc[
        result["architecture"] == "Branch fusion MLP", "bacc_mean"
    ].iloc[0]
    log(f"Paired mean delta branch-flat: {branch_score - flat_score:+.4f}")
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {DETAIL_CSV}")
    log(f"Saved: {OUTPUT_PNG}")
    log(f"Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
