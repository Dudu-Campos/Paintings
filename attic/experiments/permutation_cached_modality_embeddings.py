#!/usr/bin/env python3
"""Conditional permutation importance for cached modality embeddings.

Each outer-fold encoder is frozen before the fusion MLP is trained.  On that
fold's untouched test set, a modality's embedding is permuted across paintings
while all other embeddings remain aligned.  The BAcc drop is a conditional
importance measure: it asks whether the trained fusion uses information unique
to that modality beyond the others.

This is deliberately not a label-permutation significance test.  It is an OOF
feature-permutation analysis, and its null distribution is the score obtained
after breaking sample-to-embedding alignment in the test partition.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from evaluate_robustness_mlp_aug import BranchMLP, DEVICE, seed_everything
from evaluate_three_embedding_fusion import (
    N_FOLDS, ROOT, SEED, graph_embedding_for_cohort, load_cohort,
)


CONDITIONS = {
    "Image + graph": ("image", "graph"),
    "Image + graph + series": ("image", "graph", "series"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_series_cache(path: Path, expected_stems: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cached:
        stems = cached["stems"].astype(str)
        values = cached["embedding"].astype(np.float32, copy=False)
    if not np.array_equal(stems, expected_stems) or values.shape != (len(stems), 256):
        raise ValueError(f"Series cache does not match cohort: {path}")
    return values


def scale_blocks(
    blocks: list[np.ndarray], train: np.ndarray, validation: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = [[], [], []]
    for block in blocks:
        scaler = StandardScaler()
        transformed[0].append(scaler.fit_transform(block[train]))
        transformed[1].append(scaler.transform(block[validation]))
        transformed[2].append(scaler.transform(block[test]))
    return tuple(np.hstack(part).astype(np.float32) for part in transformed)


def train_fusion(
    train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray,
    branch_sizes: list[int], n_classes: int, seed: int, epochs: int, patience: int,
) -> BranchMLP:
    seed_everything(seed)
    model = BranchMLP(branch_sizes, n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE, non_blocking=True)
    best_score, best_state, stale = -1.0, None, 0
    for _epoch in range(1, epochs + 1):
        model.train()
        for values, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values.to(DEVICE, non_blocking=True)), targets.to(DEVICE, non_blocking=True))
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            score = balanced_accuracy_score(validation_y, model(validation_tensor).argmax(1).cpu().numpy())
        if score > best_score:
            best_score, best_state, stale = float(score), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: BranchMLP, values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(values).to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        return model(tensor).argmax(1).cpu().numpy()


def permuted_values(values: np.ndarray, branch_sizes: list[int], branch_index: int, permutation: np.ndarray) -> np.ndarray:
    start = sum(branch_sizes[:branch_index])
    stop = start + branch_sizes[branch_index]
    result = values.copy()
    result[:, start:stop] = values[permutation, start:stop]
    return result


def save(rows: list[dict], output_dir: Path) -> None:
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "permutation_folds.csv", index=False)
    baseline = detail.loc[detail.kind.eq("baseline"), ["fold", "condition", "bacc"]]
    permutations = detail.loc[detail.kind.eq("permutation")].copy()
    merged = permutations.merge(baseline, on=["fold", "condition"], suffixes=("_permuted", "_baseline"))
    merged["drop_pp"] = 100 * (merged.bacc_baseline - merged.bacc_permuted)
    merged.to_csv(output_dir / "permutation_draws.csv", index=False)
    summary = merged.groupby(["condition", "permuted_modality"], as_index=False).agg(
        folds=("fold", "nunique"), draws=("permutation", "count"),
        mean_drop_pp=("drop_pp", "mean"), std_drop_pp=("drop_pp", "std"),
        q05_drop_pp=("drop_pp", lambda values: values.quantile(0.05)),
        q95_drop_pp=("drop_pp", lambda values: values.quantile(0.95)),
        mean_permuted_bacc=("bacc_permuted", "mean"), mean_baseline_bacc=("bacc_baseline", "mean"),
    )
    summary.to_csv(output_dir / "permutation_summary.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-index", type=Path, default=ROOT / "outputs/slic_gnn_full/index.csv")
    parser.add_argument("--graph-cache", type=Path, default=ROOT / "outputs/slic_gnn_full/fusion_cache")
    parser.add_argument("--sequence-dir", type=Path, default=ROOT / "outputs/hilbert_transformer_full")
    parser.add_argument("--image-cache", type=Path, default=ROOT / "outputs/cnn_frozen_cache")
    parser.add_argument("--image-checkpoints", type=Path, default=ROOT / "outputs/cnn_frozen_cached")
    parser.add_argument("--series-cache", type=Path, default=ROOT / "outputs/modality_embedding_cache/series_cnn/embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/modality_embedding_permutation")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    cohort, graph_labels, labels, _sequences, image_block = load_cohort(
        args.graph_index, args.sequence_dir, args.image_cache, args.image_checkpoints
    )
    source_positions = cohort.graph_position.to_numpy(dtype=int)
    source_to_row = np.full(len(graph_labels), -1, dtype=np.int32)
    source_to_row[source_positions] = np.arange(len(cohort), dtype=np.int32)
    expected_stems = cohort.stem.to_numpy(dtype=str)
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "permutation_folds.csv"
    rows = [] if args.restart or not result_path.exists() else pd.read_csv(result_path).to_dict("records")
    completed = {(int(row["fold"]), str(row["condition"])) for row in rows if row["kind"] == "baseline"}

    for fold, (outer_train, outer_test) in enumerate(splitter.split(np.zeros(len(graph_labels)), graph_labels), 1):
        fit_train, validation = train_test_split(outer_train, test_size=0.1, stratify=graph_labels[outer_train], random_state=SEED + fold)
        train_rows, validation_rows, test_rows = [source_to_row[split][source_to_row[split] >= 0] for split in (fit_train, validation, outer_test)]
        graph_block = graph_embedding_for_cohort(cohort, args.graph_cache, fold)
        series_block = load_series_cache(args.series_cache / f"fold_{fold}_series_embedding.npz", expected_stems)
        blocks = {"image": image_block, "graph": graph_block, "series": series_block}
        for condition, names in CONDITIONS.items():
            if (fold, condition) in completed:
                log(f"fold {fold}/{N_FOLDS} {condition}: already complete")
                continue
            selected = [blocks[name] for name in names]
            train_x, validation_x, test_x = scale_blocks(selected, train_rows, validation_rows, test_rows)
            branch_sizes = [block.shape[1] for block in selected]
            model = train_fusion(train_x, labels[train_rows], validation_x, labels[validation_rows], branch_sizes, len(np.unique(labels)), SEED + fold, args.epochs, args.patience)
            baseline_prediction = predict(model, test_x)
            baseline_bacc = balanced_accuracy_score(labels[test_rows], baseline_prediction)
            rows.append({
                "fold": fold, "condition": condition, "kind": "baseline", "permuted_modality": "",
                "permutation": -1, "bacc": baseline_bacc, "accuracy": accuracy_score(labels[test_rows], baseline_prediction),
                "macro_f1": f1_score(labels[test_rows], baseline_prediction, average="macro"),
            })
            rng = np.random.default_rng(SEED * 10_000 + fold)
            for branch_index, name in enumerate(names):
                for permutation in range(args.permutations):
                    shuffled = permuted_values(test_x, branch_sizes, branch_index, rng.permutation(len(test_x)))
                    bacc = balanced_accuracy_score(labels[test_rows], predict(model, shuffled))
                    rows.append({
                        "fold": fold, "condition": condition, "kind": "permutation", "permuted_modality": name,
                        "permutation": permutation, "bacc": bacc, "accuracy": np.nan, "macro_f1": np.nan,
                    })
            save(rows, args.output_dir)
            log(f"fold {fold}/{N_FOLDS} {condition}: baseline BAcc={baseline_bacc:.4f}; permutations={args.permutations}")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if rows:
        save(rows, args.output_dir)
        print(pd.read_csv(args.output_dir / "permutation_summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
