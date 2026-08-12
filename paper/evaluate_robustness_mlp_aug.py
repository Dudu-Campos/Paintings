"""Exploratory parent-grouped MLP test of modality-specific augmentation.

Both conditions are fit and early-stopped only on original paintings.  The
``selective_aug`` condition adds, for training parents only, three samples per
painting: SLIC flip, descriptor flip, and Hilbert JPEG. ``clean_repeat`` is a
fourfold copy of the canonical train set, controlling for the extra optimiser
updates in selective augmentation. Text sentiment remains the original
painting's feature vector in every augmented row.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLIC_METADATA = {
    "stem", "style", "segmentation", "slic_requested_segments",
    "slic_actual_segments", "slic_compactness", "slic_sigma",
    "slic_max_side", "elapsed_seconds_all_graphs",
}
SENTIMENT_COLUMNS = [
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
    "tb_polarity", "tb_subjectivity", "nrc_anger", "nrc_anticipation",
    "nrc_disgust", "nrc_fear", "nrc_joy", "nrc_negative", "nrc_positive",
    "nrc_sadness", "nrc_surprise", "nrc_trust",
]
RECIPES = {
    "clean": [("original", "original", "original")],
    "clean_repeat": [("original", "original", "original")] * 4,
    "selective_aug": [
        ("original", "original", "original"),
        ("hflip", "original", "original"),
        ("original", "hflip", "original"),
        ("original", "original", "jpeg_35"),
    ],
}


class BranchMLP(nn.Module):
    def __init__(self, branch_sizes: list[int], n_classes: int) -> None:
        super().__init__()
        self.branch_sizes = branch_sizes
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25)
            ) for size in branch_sizes
        ])
        width = 128 * len(branch_sizes)
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(128, n_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = [
            encoder(block) for encoder, block in zip(self.encoders, torch.split(values, self.branch_sizes, dim=1))
        ]
        return self.head(torch.cat(encoded, dim=1))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npy(values_path: Path, ids_path: Path) -> pd.DataFrame:
    values = np.load(values_path)
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    frame = pd.DataFrame(values, index=pd.Index(ids, name="sample_id"))
    if len(frame) != len(frame.index.unique()):
        raise ValueError(f"Duplicate IDs in {ids_path}")
    return frame


def load_feature_blocks(features_dir: Path, sentiment_path: Path) -> dict[str, pd.DataFrame]:
    graph = pd.read_parquet(features_dir / "slic" / "combined_features.parquet")
    graph_columns = [column for column in graph if column not in SLIC_METADATA]
    if len(graph_columns) != 71:
        raise ValueError(f"Expected 71 SLIC features; got {len(graph_columns)}")
    graph = graph.set_index("stem")[graph_columns]
    graph.index = graph.index.astype(str)
    sentiment = pd.read_parquet(sentiment_path, columns=["stem"] + SENTIMENT_COLUMNS)
    sentiment = sentiment.drop_duplicates("stem").set_index("stem")[SENTIMENT_COLUMNS]
    sentiment.index = sentiment.index.astype(str)
    return {
        "SLIC": graph,
        "Descriptors": load_npy(features_dir / "desc_full_features.npy", features_dir / "desc_full_stems.npy"),
        "Hilbert": load_npy(features_dir / "hilbert_features.npy", features_dir / "hilbert_stems.npy"),
        "Sentiment": sentiment,
    }


def make_values(
    parent_ids: np.ndarray,
    recipes: list[tuple[str, str, str]],
    variants: dict[str, pd.DataFrame],
    blocks: dict[str, pd.DataFrame],
    parent_stems: pd.Series,
    labels: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    values, targets = [], []
    for slic_variant, descriptor_variant, hilbert_variant in recipes:
        slic_ids = variants[slic_variant].loc[parent_ids, "sample_id"].to_numpy()
        descriptor_ids = variants[descriptor_variant].loc[parent_ids, "sample_id"].to_numpy()
        hilbert_ids = variants[hilbert_variant].loc[parent_ids, "sample_id"].to_numpy()
        stems = parent_stems.loc[parent_ids].to_numpy()
        values.append(np.hstack([
            blocks["SLIC"].loc[slic_ids].to_numpy(),
            blocks["Descriptors"].loc[descriptor_ids].to_numpy(),
            blocks["Hilbert"].loc[hilbert_ids].to_numpy(),
            blocks["Sentiment"].loc[stems].to_numpy(),
        ]))
        targets.append(labels.loc[parent_ids].to_numpy())
    return np.vstack(values).astype(np.float32), np.concatenate(targets)


def train_one(
    train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
    validation_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
    branch_sizes: list[int], n_classes: int, seed: int, max_epochs: int, patience: int,
) -> tuple[dict, np.ndarray]:
    seed_everything(seed)
    model = BranchMLP(branch_sizes, n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE, non_blocking=True)
    test_tensor = torch.from_numpy(test_x).to(DEVICE, non_blocking=True)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
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
            best_score, best_state, best_epoch, stale = float(score), copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(test_tensor).argmax(1).cpu().numpy()
    metrics = {
        "bacc": balanced_accuracy_score(test_y, prediction),
        "accuracy": accuracy_score(test_y, prediction),
        "macro_f1": f1_score(test_y, prediction, average="macro"),
        "validation_bacc": best_score, "best_epoch": best_epoch,
    }
    del model, validation_tensor, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/wikiart_robustness_1k/manifest.csv")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "outputs/wikiart_robustness_1k_features")
    parser.add_argument("--sentiment", type=Path, default=ROOT / "outputs/wikiart_features/sentiment_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/wikiart_robustness_1k_features/mlp_selective_aug")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    originals = manifest[manifest.variant == "original"].copy()
    if len(originals) != originals.parent_id.nunique():
        raise ValueError("Expected one original per parent")
    parent_stems = originals.set_index("parent_id")["stem"].astype(str)
    labels_text = originals.set_index("parent_id")["style"].astype(str)
    variants = {name: group.set_index("parent_id") for name, group in manifest.groupby("variant", sort=False)}
    required_variants = {variant for recipes in RECIPES.values() for recipe in recipes for variant in recipe}
    if not required_variants <= variants.keys():
        raise ValueError(f"Missing variants: {sorted(required_variants - variants.keys())}")
    blocks = load_feature_blocks(args.features_dir, args.sentiment)
    expected_ids = set(manifest.sample_id.astype(str))
    for name in ("SLIC", "Descriptors", "Hilbert"):
        if set(blocks[name].index.astype(str)) != expected_ids:
            raise ValueError(f"{name} IDs do not match the robustness manifest")
    has_sentiment = parent_stems.isin(blocks["Sentiment"].index)
    if not has_sentiment.all():
        log(
            f"Dropping {(~has_sentiment).sum():,} parents without an existing sentiment vector "
            "from both MLP conditions."
        )
        parent_stems = parent_stems[has_sentiment]
        labels_text = labels_text.loc[parent_stems.index]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_text.to_numpy())
    labels = pd.Series(y, index=labels_text.index)
    parent_ids = labels.index.to_numpy(dtype=str)
    if labels_text.value_counts().min() < args.folds:
        raise ValueError("A style has fewer than --folds usable parents after sentiment alignment")
    branch_sizes = [blocks[name].shape[1] for name in ("SLIC", "Descriptors", "Hilbert", "Sentiment")]
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    rows, prediction_rows = [], []
    log(f"MLP selective augmentation: {len(parent_ids):,} parents, {len(encoder.classes_)} styles, device={DEVICE}")
    log("Branches: " + ", ".join(f"{name}={size}" for name, size in zip(("SLIC", "Descriptors", "Hilbert", "Sentiment"), branch_sizes)))
    for fold, (outer_train, outer_test) in enumerate(splitter.split(parent_ids, y), start=1):
        train_index, validation_index = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=args.seed + fold
        )
        train_parents, validation_parents, test_parents = parent_ids[train_index], parent_ids[validation_index], parent_ids[outer_test]
        validation_x, validation_y = make_values(validation_parents, RECIPES["clean"], variants, blocks, parent_stems, labels)
        test_x, test_y = make_values(test_parents, RECIPES["clean"], variants, blocks, parent_stems, labels)
        for condition, recipes in RECIPES.items():
            train_x, train_y = make_values(train_parents, recipes, variants, blocks, parent_stems, labels)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x).astype(np.float32)
            validation_scaled = scaler.transform(validation_x).astype(np.float32)
            test_scaled = scaler.transform(test_x).astype(np.float32)
            started = time.perf_counter()
            metrics, prediction = train_one(
                train_x, train_y, validation_scaled, validation_y, test_scaled, test_y,
                branch_sizes, len(encoder.classes_), args.seed + fold, args.epochs, args.patience,
            )
            elapsed = time.perf_counter() - started
            row = {"fold": fold, "condition": condition, "train_rows": len(train_x), "elapsed_seconds": elapsed, **metrics}
            rows.append(row)
            prediction_rows.extend({
                "fold": fold, "condition": condition, "parent_id": parent_id,
                "y_true": encoder.classes_[truth], "y_pred": encoder.classes_[pred],
            } for parent_id, truth, pred in zip(test_parents, test_y, prediction, strict=True))
            log(f"fold {fold}/{args.folds} {condition}: BAcc={metrics['bacc']:.4f}, val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}, train={len(train_x):,}")

    detail = pd.DataFrame(rows)
    summary = detail.groupby("condition", as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"), best_epoch_mean=("best_epoch", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    )
    clean_bacc = summary.loc[summary.condition == "clean", "bacc_mean"].iat[0]
    summary["bacc_delta_from_clean"] = summary.bacc_mean - clean_bacc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.8, 4))
    bars = axis.bar(summary.condition, summary.bacc_mean, yerr=summary.bacc_std, capsize=5,
                    color=["#777777", "#aaaaaa", "#2a78d6"])
    for bar, row in zip(bars, summary.itertuples(index=False)):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{row.bacc_mean:.3f}", ha="center")
    axis.set_ylim(0, max(summary.bacc_mean + summary.bacc_std) + 0.05)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Exploratory MLP: selective modality augmentation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "bacc_comparison.png", dpi=180)
    log(f"Saved results to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
