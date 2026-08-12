"""Train a small edge-aware GraphSAGE classifier on exported SLIC RAG graphs.

This pilot deliberately uses no PyG dependency.  It batches sparse graphs by
concatenating their nodes and edges, applies three edge-aware message-passing
layers, then mean/max pools each painting into a learned 128-D embedding.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphDataset(Dataset):
    def __init__(
        self, index: pd.DataFrame, labels: np.ndarray, graph_dir: Path,
        aggregate_features: np.ndarray | None = None,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.labels = labels
        self.graph_dir = graph_dir
        self.aggregate_features = aggregate_features

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int) -> tuple[dict, int, np.ndarray | None]:
        stem = self.index.iloc[position].stem
        graph = torch.load(self.graph_dir / f"{stem}.pt", map_location="cpu", weights_only=False)
        features = None if self.aggregate_features is None else self.aggregate_features[position]
        return graph, int(self.labels[position]), features


def collate(batch: list[tuple[dict, int, np.ndarray | None]]) -> dict[str, torch.Tensor]:
    xs, edges, attrs, graph_ids, labels, aggregate = [], [], [], [], [], []
    offset = 0
    for graph_id, (graph, label, features) in enumerate(batch):
        x = torch.as_tensor(graph["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long) + offset
        xs.append(x)
        edges.append(edge_index)
        attrs.append(torch.as_tensor(graph["edge_attr"], dtype=torch.float32))
        graph_ids.append(torch.full((len(x),), graph_id, dtype=torch.long))
        labels.append(label)
        if features is not None:
            aggregate.append(torch.as_tensor(features, dtype=torch.float32))
        offset += len(x)
    result = {
        "x": torch.cat(xs), "edge_index": torch.cat(edges, dim=1), "edge_attr": torch.cat(attrs),
        "graph_id": torch.cat(graph_ids), "y": torch.tensor(labels, dtype=torch.long),
    }
    if aggregate:
        result["aggregate"] = torch.stack(aggregate)
    return result


class EdgeSAGELayer(nn.Module):
    def __init__(self, width: int, edge_width: int = 2) -> None:
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.message_projection = nn.Linear(width + edge_width, width)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.message_projection(torch.cat([x[source], edge_attr], dim=1))
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, target, messages)
        degree = torch.bincount(target, minlength=len(x)).clamp_min(1).to(x.dtype).unsqueeze(1)
        return torch.nn.functional.gelu(self.norm(self.self_projection(x) + aggregate / degree))


class SLICGraphSAGE(nn.Module):
    def __init__(self, n_classes: int, aggregate_dimensions: int = 0) -> None:
        super().__init__()
        width = 64
        self.input = nn.Sequential(nn.Linear(6, width), nn.BatchNorm1d(width), nn.GELU())
        self.layers = nn.ModuleList([EdgeSAGELayer(width) for _ in range(3)])
        self.dropout = nn.Dropout(0.20)
        self.graph_embedding = nn.Sequential(nn.Linear(width * 2, 128), nn.GELU(), nn.Dropout(0.25))
        self.aggregate_embedding = (
            nn.Sequential(
                nn.Linear(aggregate_dimensions, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            ) if aggregate_dimensions else None
        )
        fused_width = 256 if self.aggregate_embedding is not None else 128
        self.classifier = nn.Sequential(
            nn.Linear(fused_width, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(128, n_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input(batch["x"])
        for layer in self.layers:
            x = self.dropout(layer(x, batch["edge_index"], batch["edge_attr"]))
        graph_id = batch["graph_id"]
        n_graphs = int(graph_id.max()) + 1
        mean = torch.zeros(n_graphs, x.shape[1], device=x.device)
        mean.index_add_(0, graph_id, x)
        counts = torch.bincount(graph_id, minlength=n_graphs).clamp_min(1).to(x.dtype).unsqueeze(1)
        mean /= counts
        maximum = torch.stack([x[graph_id == graph].max(dim=0).values for graph in range(n_graphs)])
        embedding = self.graph_embedding(torch.cat([mean, maximum], dim=1))
        if self.aggregate_embedding is not None:
            embedding = torch.cat([embedding, self.aggregate_embedding(batch["aggregate"])], dim=1)
        return self.classifier(embedding), embedding


def move(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to(DEVICE, non_blocking=True) for key, value in batch.items()}


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(move(batch))
            values.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(values)


def train_fold(train: GraphDataset, validation: GraphDataset, test: GraphDataset, n_classes: int, seed: int,
               epochs: int, patience: int, batch_size: int, fold: int,
               checkpoint_path: Path | None = None, resume: bool = False,
               ) -> tuple[dict[str, float], np.ndarray]:
    seed_everything(seed)
    aggregate_dimensions = 0 if train.aggregate_features is None else train.aggregate_features.shape[1]
    model = SLICGraphSAGE(n_classes, aggregate_dimensions).to(DEVICE)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate,
                              pin_memory=DEVICE.type == "cuda", generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    counts = np.bincount(train.labels, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_score, best_state, best_epoch, stale, first_epoch = -1.0, None, 0, 0, 1
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_score, best_state = float(checkpoint["best_score"]), checkpoint["best_state"]
        best_epoch, stale = int(checkpoint["best_epoch"]), int(checkpoint["stale"])
        first_epoch = int(checkpoint["epoch"]) + 1
        log(f"fold {fold}/{N_FOLDS}: resuming after epoch {first_epoch - 1}; best val={best_score:.4f}")
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for batch in train_loader:
            batch = move(batch)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = criterion(logits, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        validation_prediction = predict(model, validation_loader)
        score = balanced_accuracy_score(validation.labels, validation_prediction)
        if score > best_score:
            best_score, best_state, best_epoch, stale = float(score), copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "fold": fold, "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "best_score": best_score,
                "best_state": best_state, "best_epoch": best_epoch, "stale": stale,
            }, checkpoint_path)
        log(
            f"fold {fold}/{N_FOLDS} epoch {epoch}/{epochs}: loss={total_loss / max(batches, 1):.4f}; "
            f"val={score:.4f}; best={best_score:.4f}@{best_epoch}; stale={stale}/{patience}"
        )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    prediction = predict(model, test_loader)
    metrics = {
        "bacc": balanced_accuracy_score(test.labels, prediction),
        "accuracy": accuracy_score(test.labels, prediction),
        "macro_f1": f1_score(test.labels, prediction, average="macro"),
        "validation_bacc": best_score, "best_epoch": best_epoch,
    }
    return metrics, prediction


def subset_dataset(
    index: pd.DataFrame, labels: np.ndarray, positions: np.ndarray, graph_dir: Path,
    aggregate_features: np.ndarray | None = None,
) -> GraphDataset:
    values = None if aggregate_features is None else aggregate_features[positions]
    return GraphDataset(index.iloc[positions].reset_index(drop=True), labels[positions], graph_dir, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/slic_gnn_pilot/graphs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/slic_gnn_pilot/results")
    parser.add_argument(
        "--aggregate-mode", choices=("none", "rag32", "multigraph71"), default="none",
        help="Optional aggregate branch fused with the GNN: original RAG 32 or full multigraph 71 features.",
    )
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1), default=1,
                        help="Pilot default is one fold; pass --only-fold only after choosing a fold.")
    parser.add_argument("--all-folds", action="store_true", help="Run final 5-fold CV instead of the one-fold pilot.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Save one resumable model/optimizer checkpoint after every epoch, per outer fold.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume each fold from its epoch checkpoint when available.")
    args = parser.parse_args()

    index = pd.read_csv(args.graph_dir.parent / "index.csv", dtype={"stem": str, "style": str})
    if index["style"].value_counts().min() < N_FOLDS:
        raise ValueError("Every style needs at least five exported graphs")
    encoder = LabelEncoder()
    labels = encoder.fit_transform(index["style"].to_numpy())
    aggregate_values = None
    if args.aggregate_mode != "none":
        aggregate = pd.read_parquet(ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
        metadata = {
            "stem", "style", "segmentation", "reference_n_nodes", "slic_requested_segments",
            "slic_actual_segments", "slic_compactness", "slic_sigma", "slic_max_side",
            "slic_enforce_connectivity", "elapsed_seconds", "elapsed_seconds_all_graphs",
        }
        all_columns = [column for column in aggregate if column not in metadata]
        columns = all_columns[:32] if args.aggregate_mode == "rag32" else all_columns
        expected = 32 if args.aggregate_mode == "rag32" else 71
        if len(columns) != expected:
            raise ValueError(f"Expected {expected} aggregate columns, found {len(columns)}")
        aggregate = index[["stem", "style"]].merge(
            aggregate[["stem", "style"] + columns], on=["stem", "style"], how="left", validate="one_to_one",
        )
        if aggregate[columns].isna().any().any():
            raise ValueError("A pilot graph is missing its aggregate SLIC feature row")
        aggregate_values = aggregate[columns].to_numpy(dtype=np.float32)
        log(f"Fusion enabled: GNN embedding + {len(columns)} aggregate SLIC features ({args.aggregate_mode})")
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, prediction_rows = [], []
    log(f"SLIC GraphSAGE: {len(index):,} graphs, {len(encoder.classes_)} styles, device={DEVICE}")
    for fold, (outer_train, outer_test) in enumerate(splitter.split(index, labels), start=1):
        if not args.all_folds and fold != args.only_fold:
            continue
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold,
        )
        # Fit the standardisation only inside this outer train split.
        scaled_aggregate = None
        if aggregate_values is not None:
            scaler = StandardScaler()
            scaled_aggregate = np.empty_like(aggregate_values)
            scaled_aggregate[fit_train] = scaler.fit_transform(aggregate_values[fit_train])
            scaled_aggregate[validation] = scaler.transform(aggregate_values[validation])
            scaled_aggregate[outer_test] = scaler.transform(aggregate_values[outer_test])
        started = time.perf_counter()
        metrics, prediction = train_fold(
            subset_dataset(index, labels, fit_train, args.graph_dir, scaled_aggregate),
            subset_dataset(index, labels, validation, args.graph_dir, scaled_aggregate),
            subset_dataset(index, labels, outer_test, args.graph_dir, scaled_aggregate),
            len(encoder.classes_), SEED + fold, args.epochs, args.patience, args.batch_size, fold,
            None if args.checkpoint_dir is None else args.checkpoint_dir / f"fold_{fold}.pt", args.resume,
        )
        rows.append({"fold": fold, "n_images": len(index), "elapsed_seconds": time.perf_counter() - started, **metrics})
        prediction_rows.extend({"fold": fold, "stem": stem, "true_style": encoder.classes_[truth],
                                "predicted_style": encoder.classes_[pred]}
                               for stem, truth, pred in zip(index.iloc[outer_test].stem, labels[outer_test], prediction, strict=True))
        log(f"fold {fold}/{N_FOLDS}: BAcc={metrics['bacc']:.4f}, val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}")
    detail = pd.DataFrame(rows)
    summary = detail.agg({"bacc": ["mean", "std"], "accuracy": "mean", "macro_f1": "mean", "elapsed_seconds": "sum"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv")
    log(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
