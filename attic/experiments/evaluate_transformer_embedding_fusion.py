"""Pilot late fusion with image, Hilbert-Transformer, and graph embeddings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from evaluate_robustness_mlp_aug import train_one
from evaluate_three_embedding_fusion import (
    N_FOLDS,
    ROOT,
    SEED,
    graph_embedding_for_cohort,
    load_cohort,
    scale_blocks,
)
from train_hilbert_transformer import (
    DEVICE,
    HilbertTransformer,
    SequenceDataset,
    train_fold as train_hilbert_fold,
)


CONDITIONS = {
    "Image CNN": ("image",),
    "Image CNN + Hilbert Transformer": ("image", "transformer"),
    "Image CNN + graph GNN": ("image", "graph"),
    "Image CNN + Hilbert Transformer + graph GNN": ("image", "transformer", "graph"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class IndexedSequences:
    """Expose a cohort-ordered view while retaining the original memmap."""

    def __init__(self, sequences: np.ndarray, positions: np.ndarray) -> None:
        self.sequences = sequences
        self.positions = positions
        self.shape = (len(positions),) + tuple(sequences.shape[1:])

    def __getitem__(self, position):
        return self.sequences[self.positions[position]]


def ensure_transformer_checkpoint(
    path: Path,
    sequences: np.ndarray,
    sequence_positions: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    test_rows: np.ndarray,
    fold: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict:
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        complete = (
            int(checkpoint.get("fold", -1)) == fold
            and "best_state" in checkpoint
            and int(checkpoint.get("stale", -1)) >= patience
            or int(checkpoint.get("epoch", -1)) >= epochs
        )
        if complete:
            log(f"fold {fold}: reusing graph-aligned Transformer checkpoint")
            return checkpoint
        log(f"fold {fold}: removing incomplete Transformer checkpoint")
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"fold {fold}: training graph-aligned Hilbert Transformer from scratch")
    train_hilbert_fold(
        IndexedSequences(sequences, sequence_positions), labels,
        train_rows, validation_rows, test_rows, int(labels.max()) + 1,
        SEED + fold, epochs, patience, batch_size, 64, fold, "transformer", path,
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_transformer_embedding(
    checkpoint: dict,
    sequences: np.ndarray,
    sequence_positions: np.ndarray,
    stems: np.ndarray,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if np.array_equal(cached["stems"].astype(str), stems):
                embedding = cached["embedding"].astype(np.float32, copy=False)
                if embedding.shape == (len(stems), 128):
                    log(f"reusing cached Transformer embeddings: {cache_path.name}")
                    return embedding

    dataset = SequenceDataset(
        sequences, np.zeros(len(sequences), dtype=np.int64), sequence_positions,
        np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=DEVICE.type == "cuda", persistent_workers=True,
    )
    n_classes = int(checkpoint["best_state"]["head.1.weight"].shape[0])
    model = HilbertTransformer(sequences.shape[1], n_classes, patch_size=64).to(DEVICE)
    model.load_state_dict(checkpoint["best_state"])
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for offset, (values, _) in enumerate(loader, start=1):
            values = values.to(DEVICE, non_blocking=True)
            tokens = model.patch(values.transpose(1, 2)).transpose(1, 2)
            cls = model.class_token.expand(len(tokens), -1, -1)
            encoded = model.encoder(torch.cat([cls, tokens], dim=1) + model.position)[:, 0]
            embeddings.append(model.head[0](encoded).cpu().numpy())
            if offset % 100 == 0:
                log(f"Transformer embeddings: {min(offset * batch_size, len(dataset)):,}/{len(dataset):,}")
    embedding = np.vstack(embeddings).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, stems=stems, embedding=embedding)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return embedding


def save_results(rows: list[dict], output_dir: Path) -> None:
    detail = pd.DataFrame(rows).sort_values(["fold", "condition"])
    detail.to_csv(output_dir / "fold_results.csv", index=False)
    summary = detail.groupby("condition", as_index=False).agg(
        folds_completed=("fold", "nunique"), n_images=("n_images", "first"),
        dimensions=("dimensions", "first"), branches=("branches", "first"),
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"),
        best_epoch_mean=("best_epoch", "mean"), elapsed_seconds=("elapsed_seconds", "sum"),
    )
    image_score = summary.loc[summary.condition.eq("Image CNN"), "bacc_mean"]
    summary["delta_from_image"] = summary.bacc_mean - image_score.iat[0]
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-index", type=Path, default=ROOT / "outputs/slic_gnn_full/index.csv")
    parser.add_argument("--graph-cache", type=Path, default=ROOT / "outputs/slic_gnn_full/fusion_cache")
    parser.add_argument("--sequence-dir", type=Path, default=ROOT / "outputs/hilbert_transformer_full")
    parser.add_argument("--image-cache", type=Path, default=ROOT / "outputs/cnn_frozen_cache")
    parser.add_argument("--image-checkpoints", type=Path, default=ROOT / "outputs/cnn_frozen_cached")
    parser.add_argument(
        "--transformer-checkpoints", type=Path,
        default=ROOT / "outputs/hilbert_transformer_graph_aligned/checkpoints",
    )
    parser.add_argument(
        "--transformer-cache", type=Path,
        default=ROOT / "outputs/hilbert_transformer_graph_aligned/embeddings",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/transformer_embedding_fusion",
    )
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1), default=1)
    parser.add_argument("--transformer-epochs", type=int, default=120)
    parser.add_argument("--transformer-patience", type=int, default=15)
    parser.add_argument("--transformer-batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=150)
    parser.add_argument("--mlp-patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    cohort, graph_labels, labels, sequences, image_block = load_cohort(
        args.graph_index, args.sequence_dir, args.image_cache, args.image_checkpoints
    )
    sequence_positions = cohort["sequence_position"].to_numpy(dtype=int)
    source_positions = cohort["graph_position"].to_numpy(dtype=int)
    source_to_row = np.full(len(graph_labels), -1, dtype=np.int32)
    source_to_row[source_positions] = np.arange(len(cohort), dtype=np.int32)
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    completed = {(int(row["fold"]), str(row["condition"])) for row in rows}

    for fold, (outer_train, outer_test) in enumerate(
        splitter.split(np.zeros(len(graph_labels), dtype=np.uint8), graph_labels), start=1
    ):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=graph_labels[outer_train], random_state=SEED + fold
        )
        row_splits = []
        for source_split in (fit_train, validation, outer_test):
            mapped = source_to_row[source_split]
            row_splits.append(mapped[mapped >= 0])
        train_rows, validation_rows, test_rows = row_splits
        checkpoint = ensure_transformer_checkpoint(
            args.transformer_checkpoints / f"fold_{fold}.pt", sequences,
            sequence_positions, labels, train_rows, validation_rows, test_rows,
            fold, args.transformer_epochs, args.transformer_patience,
            args.transformer_batch_size,
        )
        transformer_block = extract_transformer_embedding(
            checkpoint, sequences, sequence_positions, cohort.stem.to_numpy(dtype=str),
            args.transformer_cache / f"fold_{fold}_transformer_embedding.npz",
            args.embedding_batch_size,
        )
        graph_block = graph_embedding_for_cohort(cohort, args.graph_cache, fold)
        blocks = {"image": image_block, "transformer": transformer_block, "graph": graph_block}
        for condition, names in CONDITIONS.items():
            if (fold, condition) in completed:
                log(f"fold {fold}/{N_FOLDS} {condition}: already complete")
                continue
            selected = [blocks[name] for name in names]
            train_x, validation_x, test_x = scale_blocks(
                selected, train_rows, validation_rows, test_rows
            )
            started = time.perf_counter()
            metrics, _ = train_one(
                train_x, labels[train_rows], validation_x, labels[validation_rows],
                test_x, labels[test_rows], [block.shape[1] for block in selected],
                int(graph_labels.max()) + 1, SEED + fold, args.mlp_epochs, args.mlp_patience,
            )
            rows.append({
                "fold": fold, "condition": condition, "n_images": len(cohort),
                "n_fit_train": len(train_rows), "n_validation": len(validation_rows),
                "n_test": len(test_rows), "dimensions": sum(block.shape[1] for block in selected),
                "branches": len(selected), "elapsed_seconds": time.perf_counter() - started,
                **metrics,
            })
            completed.add((fold, condition))
            save_results(rows, args.output_dir)
            log(
                f"fold {fold}/{N_FOLDS} {condition}: BAcc={metrics['bacc']:.4f}; "
                f"val={metrics['validation_bacc']:.4f}; epoch={metrics['best_epoch']}"
            )
    if rows:
        save_results(rows, args.output_dir)
        print(pd.read_csv(args.output_dir / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
