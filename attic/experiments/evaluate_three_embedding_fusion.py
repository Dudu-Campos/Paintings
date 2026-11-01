"""Fuse frozen-image, Hilbert-CNN, and SLIC-GNN embeddings with an MLP.

The SLIC-GNN outer folds define the evaluation protocol because its saved
checkpoints were trained on the complete graph cohort.  A fresh Hilbert CNN is
trained on the matching fit/validation rows for every fold.  The ImageNet
ResNet representation is label-independent and remains frozen throughout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from evaluate_robustness_mlp_aug import train_one
from train_hilbert_transformer import (
    DEVICE,
    HilbertCNN1D,
    SequenceDataset,
    train_fold as train_hilbert_fold,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FOLDS = 5
CONDITIONS = {
    "Image CNN": ("image",),
    "Image CNN + series CNN": ("image", "series"),
    "Image CNN + graph GNN": ("image", "graph"),
    "Image CNN + series CNN + graph GNN": ("image", "series", "graph"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_key(stem: str, style: str) -> tuple[str, str]:
    return str(stem), str(style)


def unique_lookup(stems: np.ndarray, styles: np.ndarray, name: str) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for position, (stem, style) in enumerate(zip(stems, styles, strict=True)):
        key = make_key(stem, style)
        if key in lookup:
            raise ValueError(f"Duplicate {name} key: {key}")
        lookup[key] = position
    return lookup


def load_cohort(
    graph_index_path: Path,
    sequence_dir: Path,
    image_cache: Path,
    image_checkpoint_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph_index = pd.read_csv(graph_index_path, dtype={"stem": str, "style": str})
    graph_index.insert(0, "graph_position", np.arange(len(graph_index), dtype=np.int32))
    if graph_index.duplicated(["stem", "style"]).any():
        raise ValueError("Graph index contains duplicate (stem, style) pairs")

    sequence_stems = np.load(sequence_dir / "stems.npy", allow_pickle=True).astype(str)
    sequence_styles = np.load(sequence_dir / "styles.npy", allow_pickle=True).astype(str)
    sequences = np.load(sequence_dir / "sequences_lab_hilbert.npy", mmap_mode="r")
    if len(sequences) != len(sequence_stems) or sequences.shape[1:] != (16384, 3):
        raise ValueError(f"Invalid Hilbert sequence array: {sequences.shape}")
    sequence_lookup = unique_lookup(sequence_stems, sequence_styles, "sequence")

    image_stems = np.load(image_cache / "stems.npy", allow_pickle=True).astype(str)
    image_labels = np.load(image_cache / "labels.npy")
    image_values = np.load(image_cache / "resnet50_imagenet_v2_features.npy", mmap_mode="r")
    checkpoint = torch.load(
        image_checkpoint_dir / "fold_01_best.pt", map_location="cpu", weights_only=False
    )
    image_classes = np.asarray(checkpoint["classes"], dtype=str)
    image_styles = image_classes[image_labels]
    if image_values.shape != (len(image_stems), 2048):
        raise ValueError(f"Invalid image embedding array: {image_values.shape}")
    image_lookup = unique_lookup(image_stems, image_styles, "image")

    keys = list(zip(graph_index["stem"], graph_index["style"], strict=True))
    usable = np.fromiter(
        (key in sequence_lookup and key in image_lookup for key in keys),
        dtype=bool,
        count=len(keys),
    )
    cohort = graph_index.loc[usable].copy().reset_index(drop=True)
    cohort["sequence_position"] = [
        sequence_lookup[make_key(row.stem, row.style)] for row in cohort.itertuples()
    ]
    cohort["image_position"] = [
        image_lookup[make_key(row.stem, row.style)] for row in cohort.itertuples()
    ]
    encoder = LabelEncoder()
    graph_labels = encoder.fit_transform(graph_index["style"].to_numpy(dtype=str))
    cohort_labels = encoder.transform(cohort["style"].to_numpy(dtype=str))
    image_block = np.asarray(image_values[cohort["image_position"].to_numpy()], dtype=np.float32)
    log(
        f"Exact three-branch cohort: {len(cohort):,}/{len(graph_index):,} paintings; "
        f"image={image_block.shape[1]}, series=256, graph=128; classes={len(encoder.classes_)}"
    )
    return cohort, graph_labels, cohort_labels, sequences, image_block


def graph_embedding_for_cohort(cohort: pd.DataFrame, cache_dir: Path, fold: int) -> np.ndarray:
    cache_path = cache_dir / f"fold_{fold}_graph_embedding.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}; run experiments/evaluate_gnn_embedding_fusion.py first"
        )
    with np.load(cache_path, allow_pickle=False) as cached:
        stems = cached["stems"].astype(str)
        values = cached["embedding"].astype(np.float32, copy=False)
    if values.shape != (len(stems), 128) or len(stems) != len(np.unique(stems)):
        raise ValueError(f"Invalid graph embedding cache: {cache_path}")
    lookup = {stem: position for position, stem in enumerate(stems)}
    missing = [stem for stem in cohort["stem"] if stem not in lookup]
    if missing:
        raise ValueError(f"Graph embedding cache lacks {len(missing)} cohort stems")
    return values[[lookup[stem] for stem in cohort["stem"]]]


def ensure_series_checkpoint(
    checkpoint_path: Path,
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
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"best_state", "mean", "std", "fold"}
        complete = (
            required <= checkpoint.keys()
            and int(checkpoint["fold"]) == fold
            and (
                int(checkpoint.get("stale", -1)) >= patience
                or int(checkpoint.get("epoch", -1)) >= epochs
            )
        )
        if complete:
            log(f"fold {fold}: reusing complete graph-aligned series checkpoint")
            return checkpoint
        # A checkpoint may exist because an earlier pilot was interrupted.  It
        # is never a valid encoder for an OOF fusion study: retrain the fold.
        log(f"fold {fold}: replacing incomplete series checkpoint")
        checkpoint_path.unlink()

    # Present the memmap in cohort order without copying the 15 GB sequence array.
    class IndexedSequences:
        shape = (len(sequence_positions),) + tuple(sequences.shape[1:])

        def __getitem__(self, position):
            return sequences[sequence_positions[position]]

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"fold {fold}: training graph-aligned Hilbert CNN from scratch")
    train_hilbert_fold(
        IndexedSequences(), labels, train_rows, validation_rows, test_rows,
        int(labels.max()) + 1, SEED + fold, epochs, patience, batch_size, 64,
        fold, "cnn1d", checkpoint_path,
    )
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def extract_series_embedding(
    checkpoint: dict,
    sequences: np.ndarray,
    sequence_positions: np.ndarray,
    cache_path: Path,
    stems: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if np.array_equal(cached["stems"].astype(str), stems):
                values = cached["embedding"].astype(np.float32, copy=False)
                if values.shape == (len(stems), 256):
                    log(f"reusing cached series embeddings: {cache_path.name}")
                    return values

    dataset = SequenceDataset(
        sequences, np.zeros(len(sequences), dtype=np.int64), sequence_positions,
        np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=DEVICE.type == "cuda", persistent_workers=True,
    )
    model = HilbertCNN1D(int(checkpoint["best_state"]["head.1.weight"].shape[0])).to(DEVICE)
    model.load_state_dict(checkpoint["best_state"])
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for offset, (values, _) in enumerate(loader, start=1):
            features = model.features(values.to(DEVICE, non_blocking=True).transpose(1, 2))
            embeddings.append(features.mean(dim=2).cpu().numpy())
            if offset % 100 == 0:
                log(f"series embeddings: {min(offset * batch_size, len(dataset)):,}/{len(dataset):,}")
    result = np.vstack(embeddings).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, stems=stems, embedding=result)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def scale_blocks(
    blocks: list[np.ndarray], train: np.ndarray, validation: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed: list[list[np.ndarray]] = [[], [], []]
    for block in blocks:
        scaler = StandardScaler()
        transformed[0].append(scaler.fit_transform(block[train]))
        transformed[1].append(scaler.transform(block[validation]))
        transformed[2].append(scaler.transform(block[test]))
    return tuple(np.hstack(split).astype(np.float32, copy=False) for split in transformed)


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
    baseline = summary.loc[summary["condition"].eq("Image CNN"), ["folds_completed", "bacc_mean"]]
    if len(baseline) == 1:
        baseline_folds, baseline_score = baseline.iloc[0]
        summary["delta_from_image"] = np.where(
            summary["folds_completed"].eq(baseline_folds), summary["bacc_mean"] - baseline_score, np.nan
        )
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")

    comparisons = []
    baseline_rows = detail.loc[detail.condition.eq("Image CNN"), ["fold", "bacc"]].set_index("fold")
    for condition in CONDITIONS:
        if condition == "Image CNN":
            continue
        candidate = detail.loc[detail.condition.eq(condition), ["fold", "bacc"]].set_index("fold")
        paired = baseline_rows.join(candidate, lsuffix="_base", rsuffix="_candidate", how="inner")
        statistic, pvalue = (np.nan, np.nan)
        if len(paired) >= 2:
            statistic, pvalue = ttest_rel(paired.bacc_candidate, paired.bacc_base)
        comparisons.append({
            "condition": condition, "paired_folds": len(paired),
            "mean_delta_from_image": (paired.bacc_candidate - paired.bacc_base).mean(),
            "wins": int((paired.bacc_candidate > paired.bacc_base).sum()),
            "paired_t": statistic, "paired_p": pvalue,
        })
    pd.DataFrame(comparisons).to_csv(output_dir / "comparisons.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-index", type=Path, default=ROOT / "outputs/slic_gnn_full/index.csv")
    parser.add_argument("--graph-cache", type=Path, default=ROOT / "outputs/slic_gnn_full/fusion_cache")
    parser.add_argument("--sequence-dir", type=Path, default=ROOT / "outputs/hilbert_transformer_full")
    parser.add_argument("--image-cache", type=Path, default=ROOT / "outputs/cnn_frozen_cache")
    parser.add_argument("--image-checkpoints", type=Path, default=ROOT / "outputs/cnn_frozen_cached")
    parser.add_argument(
        "--series-checkpoints", type=Path,
        default=ROOT / "outputs/modality_embedding_cache/series_cnn/checkpoints",
    )
    parser.add_argument(
        "--series-cache", type=Path,
        default=ROOT / "outputs/modality_embedding_cache/series_cnn/embeddings",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/modality_embedding_fusion",
    )
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--series-epochs", type=int, default=120)
    parser.add_argument("--series-patience", type=int, default=15)
    parser.add_argument("--series-batch-size", type=int, default=64)
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
    detail_path = args.output_dir / "fold_results.csv"
    rows = [] if args.restart or not detail_path.exists() else pd.read_csv(detail_path).to_dict("records")
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
        for positions in (fit_train, validation, outer_test):
            mapped = source_to_row[positions]
            row_splits.append(mapped[mapped >= 0])
        train_rows, validation_rows, test_rows = row_splits
        checkpoint = ensure_series_checkpoint(
            args.series_checkpoints / f"fold_{fold}.pt", sequences, sequence_positions, labels,
            train_rows, validation_rows, test_rows, fold, args.series_epochs,
            args.series_patience, args.series_batch_size,
        )
        series_block = extract_series_embedding(
            checkpoint, sequences, sequence_positions,
            args.series_cache / f"fold_{fold}_series_embedding.npz",
            cohort["stem"].to_numpy(dtype=str), args.series_batch_size,
        )
        graph_block = graph_embedding_for_cohort(cohort, args.graph_cache, fold)
        blocks = {"image": image_block, "series": series_block, "graph": graph_block}
        for condition, block_names in CONDITIONS.items():
            if (fold, condition) in completed:
                log(f"fold {fold}/{N_FOLDS} {condition}: already complete")
                continue
            selected = [blocks[name] for name in block_names]
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
        summary = pd.read_csv(args.output_dir / "summary.csv")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
