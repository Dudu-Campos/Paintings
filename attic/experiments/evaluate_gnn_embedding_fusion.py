"""Test whether the learned SLIC-GNN graph branch complements visual features.

For each outer fold, graph embeddings come from that fold's saved GNN
checkpoint.  The checkpoint was selected without access to its outer-test
fold; the same original inner validation partition is reused for the fusion
MLP.  This avoids turning out-of-fold GNN embeddings from different models
into a leaky second-stage training set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from evaluate_robustness_mlp_aug import train_one
from train_slic_gnn_pilot import (
    DEVICE, N_FOLDS, ROOT, SEED, GraphDataset, SLICGraphSAGE, collate, move,
)


SLIC_DIR = ROOT / "outputs/wikiart_slic_features"
FEATURE_DIR = ROOT / "outputs/wikiart_features"
SLIC_METADATA = {
    "stem", "style", "segmentation", "reference_n_nodes", "slic_requested_segments",
    "slic_actual_segments", "slic_compactness", "slic_sigma", "slic_max_side",
    "slic_enforce_connectivity", "elapsed_seconds", "elapsed_seconds_all_graphs",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_npy(values_name: str, stems_name: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    values = np.load(FEATURE_DIR / values_name, mmap_mode="r")
    stems = np.load(FEATURE_DIR / stems_name, allow_pickle=True).astype(str)
    if len(stems) != len(pd.Index(stems).unique()):
        raise ValueError(f"Duplicate stems in {stems_name}")
    columns = [f"{prefix}_{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "stem", stems)
    return frame, columns


def load_visual_frame(index: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return the GNN cohort rows that have all non-graph visual features."""
    graph = pd.read_parquet(SLIC_DIR / "combined_features.parquet")
    slic_columns = [column for column in graph if column not in SLIC_METADATA]
    if len(slic_columns) != 71:
        raise ValueError(f"Expected 71 SLIC features, found {len(slic_columns)}")
    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    hilbert, hilbert_columns = load_npy(
        "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
    )
    base = index[["stem", "style"]].copy()
    base.insert(0, "source_position", np.arange(len(index), dtype=np.int32))
    frame = base.merge(
        graph[["stem", "style"] + slic_columns], on=["stem", "style"], how="inner", validate="one_to_one"
    )
    for addition in (descriptors, hilbert):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
    frame = frame.sort_values("source_position").reset_index(drop=True)
    return frame, {
        "SLIC multigraph": slic_columns,
        "Descriptors": descriptor_columns,
        "Hilbert": hilbert_columns,
    }


def load_graph_embedding(
    fold: int, graph_index: pd.DataFrame, labels: np.ndarray, graph_dir: Path,
    checkpoint_dir: Path, cache_dir: Path, batch_size: int,
) -> np.ndarray:
    """Encode one model's graph branch for the complete visual-feature cohort."""
    cache_path = cache_dir / f"fold_{fold}_graph_embedding.npz"
    expected_stems = graph_index["stem"].to_numpy(dtype=str)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_stems = cached["stems"].astype(str)
            if np.array_equal(cached_stems, expected_stems):
                embedding = cached["embedding"].astype(np.float32, copy=False)
                if embedding.shape == (len(graph_index), 128):
                    log(f"fold {fold}: reusing cached graph embeddings")
                    return embedding
        log(f"fold {fold}: cache does not match current cohort; rebuilding")

    checkpoint_path = checkpoint_dir / f"fold_{fold}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing GNN checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["best_state"]
    aggregate_key = "aggregate_embedding.0.weight"
    aggregate_dimensions = int(state[aggregate_key].shape[1]) if aggregate_key in state else 0
    n_classes = int(state["classifier.4.weight"].shape[0])
    model = SLICGraphSAGE(n_classes, aggregate_dimensions).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    dataset = GraphDataset(graph_index, labels, graph_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate,
        pin_memory=DEVICE.type == "cuda",
    )
    values = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset, batch in enumerate(loader, start=1):
            values.append(model.encode_graph(move(batch)).cpu().numpy())
            if offset % 100 == 0:
                done = min(offset * batch_size, len(dataset))
                log(f"fold {fold}: embeddings {done:,}/{len(dataset):,}")
    embedding = np.vstack(values).astype(np.float32, copy=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, stems=expected_stems, embedding=embedding)
    log(f"fold {fold}: extracted {len(embedding):,} graph embeddings in {time.perf_counter() - started:.1f}s")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return embedding


def scale_blocks(
    blocks: list[np.ndarray], train: np.ndarray, validation: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = [[], [], []]
    for block in blocks:
        scaler = StandardScaler()
        transformed[0].append(scaler.fit_transform(block[train]))
        transformed[1].append(scaler.transform(block[validation]))
        transformed[2].append(scaler.transform(block[test]))
    return tuple(np.hstack(values).astype(np.float32) for values in transformed)


def save_results(rows: list[dict], output_dir: Path) -> None:
    """Persist completed folds so an interrupted long run can resume safely."""
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "gnn_embedding_fusion_folds.csv", index=False)
    summary = detail.groupby("condition", as_index=False).agg(
        n_images=("n_images", "first"), feature_dimensions=("feature_dimensions", "first"),
        branches=("branches", "first"), bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"), best_epoch_mean=("best_epoch", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    )
    summary.to_csv(output_dir / "gnn_embedding_fusion_summary.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/slic_gnn_full/graphs")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/slic_gnn_full/checkpoints")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/slic_gnn_full/fusion_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--graph-batch-size", type=int, default=16)
    parser.add_argument("--restart", action="store_true", help="Discard prior partial fold results in output-dir.")
    args = parser.parse_args()

    full_index = pd.read_csv(args.graph_dir.parent / "index.csv", dtype={"stem": str, "style": str})
    encoder = LabelEncoder()
    full_labels = encoder.fit_transform(full_index["style"].to_numpy())
    frame, groups = load_visual_frame(full_index)
    source_positions = frame["source_position"].to_numpy(dtype=int)
    source_to_row = np.full(len(full_index), -1, dtype=int)
    source_to_row[source_positions] = np.arange(len(frame))
    visual_blocks = [frame[columns].to_numpy(dtype=np.float32) for columns in groups.values()]
    log(
        f"Fusion cohort: {len(frame):,}/{len(full_index):,} GNN graphs with SLIC, descriptors and Hilbert; "
        f"{len(encoder.classes_)} styles; device={DEVICE}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "gnn_embedding_fusion_folds.csv"
    if detail_path.exists() and not args.restart:
        rows = pd.read_csv(detail_path).to_dict("records")
        completed_folds = set(pd.DataFrame(rows).groupby("fold")["condition"].nunique().loc[lambda values: values == 4].index)
        log(f"Resuming with {len(completed_folds)} completed fold(s): {sorted(completed_folds)}")
    else:
        rows = []
        completed_folds = set()
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (outer_train, outer_test) in enumerate(splitter.split(full_index, full_labels), start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        if fold in completed_folds:
            log(f"fold {fold}: already complete; skipping")
            continue
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=full_labels[outer_train], random_state=SEED + fold,
        )
        graph_embedding = load_graph_embedding(
            fold, frame[["stem", "style"]], full_labels[source_positions], args.graph_dir,
            args.checkpoint_dir, args.cache_dir, args.graph_batch_size,
        )
        gnn_block = graph_embedding
        row_splits = []
        for positions in (fit_train, validation, outer_test):
            rows_for_split = source_to_row[positions]
            row_splits.append(rows_for_split[rows_for_split >= 0])
        train_rows, validation_rows, test_rows = row_splits
        labels = full_labels[source_positions]
        conditions = {
            "GNN graph embedding": [gnn_block],
            "Visual GIST": visual_blocks,
            "Non-SLIC GIST + GNN": visual_blocks[1:] + [gnn_block],
            "Visual GIST + GNN": visual_blocks + [gnn_block],
        }
        for name, blocks in conditions.items():
            train_x, validation_x, test_x = scale_blocks(blocks, train_rows, validation_rows, test_rows)
            branch_sizes = [block.shape[1] for block in blocks]
            started = time.perf_counter()
            metrics, prediction = train_one(
                train_x, labels[train_rows], validation_x, labels[validation_rows],
                test_x, labels[test_rows], branch_sizes, len(encoder.classes_), SEED + fold,
                args.epochs, args.patience,
            )
            score = balanced_accuracy_score(labels[test_rows], prediction)
            rows.append({
                "fold": fold, "condition": name, "n_images": len(frame),
                "feature_dimensions": sum(branch_sizes), "branches": len(branch_sizes),
                "elapsed_seconds": time.perf_counter() - started, "bacc": score, **metrics,
            })
            log(f"fold {fold}/{N_FOLDS} {name}: BAcc={score:.4f}; val={metrics['validation_bacc']:.4f}; epoch={metrics['best_epoch']}")
        save_results(rows, args.output_dir)
        completed_folds.add(fold)
    if rows:
        save_results(rows, args.output_dir)
    log(f"Saved fusion study to {args.output_dir}")


if __name__ == "__main__":
    main()
