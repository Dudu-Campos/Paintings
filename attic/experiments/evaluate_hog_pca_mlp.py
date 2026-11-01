#!/usr/bin/env python3
"""Evaluate canonical visual features fused with fold-local PCA of HOG."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_segmentations_pilot import FEATURE_COLUMNS  # noqa: E402
from evaluate_super_resolution_pilot import load_original  # noqa: E402
from model_evaluation import (  # noqa: E402
    BATCH_SIZE,
    DEVICE,
    MAX_EPOCHS,
    MLP,
    PATIENCE,
)


ROOT = Path(__file__).resolve().parents[1]
HOG_OFFSET = 13


def load_features(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    sample = pd.read_csv(args.sample)[["stem", "style"]].drop_duplicates("stem")
    visual, visual_columns = load_original(args.slic_features, args.image_features)
    if len(visual_columns) != 258:
        raise ValueError(f"expected 258 canonical visual features, got {len(visual_columns)}")

    line_raw = pd.read_parquet(args.line_features)
    line_raw_columns = [column for column in line_raw if column not in {"stem", "style"}]
    line_rename = {column: f"line_{column}" for column in line_raw_columns}
    line = line_raw[["stem"] + line_raw_columns].rename(columns=line_rename)
    line_columns = list(line_rename.values())
    if len(line_columns) != 63:
        raise ValueError(f"expected 63 line-graph features, got {len(line_columns)}")

    felz_raw = pd.read_parquet(args.felz_features)
    missing = set(FEATURE_COLUMNS) - set(felz_raw.columns)
    if missing:
        raise ValueError(f"missing Felzenszwalb features: {sorted(missing)}")
    felz_rename = {column: f"felz_{column}" for column in FEATURE_COLUMNS}
    felz = (
        felz_raw[["stem"] + FEATURE_COLUMNS]
        .drop_duplicates("stem")
        .rename(columns=felz_rename)
    )
    felz_columns = list(felz_rename.values())
    if len(felz_columns) != 71:
        raise ValueError(f"expected 71 Felzenszwalb features, got {len(felz_columns)}")

    frame = sample.merge(visual, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(line, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(felz, on="stem", how="inner", validate="one_to_one")
    if len(frame) != len(sample):
        raise RuntimeError(f"feature intersection lost rows: {len(frame)}/{len(sample)}")
    frame = frame.sort_values("stem").reset_index(drop=True)
    base_columns = visual_columns + line_columns + felz_columns
    if len(base_columns) != 392:
        raise ValueError(f"expected 392 fused features, got {len(base_columns)}")

    hog_stems = np.load(args.hog_stems)
    hog_values = np.load(args.hog_features, mmap_mode="r")
    if hog_values.shape != (len(frame), 1777):
        raise ValueError(f"unexpected Haralick+HOG shape: {hog_values.shape}")
    stem_lookup = {str(stem): index for index, stem in enumerate(hog_stems)}
    if len(stem_lookup) != len(hog_stems):
        raise ValueError("duplicate stems in HOG cache")
    missing_hog = [stem for stem in frame["stem"].astype(str) if stem not in stem_lookup]
    if missing_hog:
        raise RuntimeError(f"missing HOG vectors: {len(missing_hog)}")
    hog_order = np.asarray(
        [stem_lookup[stem] for stem in frame["stem"].astype(str)], dtype=np.int64
    )
    return frame, base_columns, hog_order


def evaluate_fold(
    base_values: np.ndarray,
    hog_values: np.ndarray,
    hog_order: np.ndarray,
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    components: int,
    seed: int,
) -> tuple[float, float, int]:
    train, validation = train_test_split(
        outer_train,
        test_size=0.1,
        stratify=labels[outer_train],
        random_state=seed + fold,
    )
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        iterated_power=4,
        random_state=seed + fold,
    )
    train_hog = np.asarray(hog_values[hog_order[train], HOG_OFFSET:], dtype=np.float32)
    train_hog = pca.fit_transform(train_hog).astype(np.float32)
    validation_hog = pca.transform(
        np.asarray(hog_values[hog_order[validation], HOG_OFFSET:], dtype=np.float32)
    ).astype(np.float32)
    test_hog = pca.transform(
        np.asarray(hog_values[hog_order[outer_test], HOG_OFFSET:], dtype=np.float32)
    ).astype(np.float32)

    train_values = np.concatenate((base_values[train], train_hog), axis=1)
    validation_values = np.concatenate((base_values[validation], validation_hog), axis=1)
    test_values = np.concatenate((base_values[outer_test], test_hog), axis=1)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_values).astype(np.float32)
    validation_values = scaler.transform(validation_values).astype(np.float32)
    test_values = scaler.transform(test_values).astype(np.float32)

    torch.manual_seed(seed + fold)
    model = MLP(train_values.shape[1], len(np.unique(labels))).to(DEVICE)
    counts = np.bincount(labels[train], minlength=len(np.unique(labels))).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_values), torch.from_numpy(labels[train]).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    validation_tensor = torch.from_numpy(validation_values).to(DEVICE)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_values, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(
                model(batch_values.to(DEVICE, non_blocking=True)),
                batch_labels.to(DEVICE, non_blocking=True),
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_tensor).argmax(1).cpu().numpy()
        validation_score = balanced_accuracy_score(labels[validation], validation_prediction)
        if validation_score > best_score:
            best_score = float(validation_score)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("MLP failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(test_values).to(DEVICE)).argmax(1).cpu().numpy()
    score = float(balanced_accuracy_score(labels[outer_test], prediction))
    explained = float(pca.explained_variance_ratio_.sum())
    return score, explained, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "results/line_graph_full_sample.csv"
    )
    parser.add_argument(
        "--slic-features", type=Path, default=ROOT / "outputs/wikiart_slic_features"
    )
    parser.add_argument(
        "--image-features", type=Path, default=ROOT / "outputs/wikiart_features"
    )
    parser.add_argument(
        "--line-features", type=Path,
        default=ROOT / "results/line_graph_full_features.parquet",
    )
    parser.add_argument(
        "--felz-features", type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument(
        "--hog-features", type=Path,
        default=ROOT / "outputs/haralick_hog_features/features.npy",
    )
    parser.add_argument(
        "--hog-stems", type=Path,
        default=ROOT / "outputs/haralick_hog_features/stems.npy",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "results/felzenszwalb_full_mlp_final_results.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/hog_pca128_mlp_results.csv"
    )
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame, base_columns, hog_order = load_features(args)
    base_values = frame[base_columns].to_numpy(dtype=np.float32)
    hog_values = np.load(args.hog_features, mmap_mode="r")
    labels = LabelEncoder().fit_transform(frame["style"].astype(str))
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    print(
        f"cohort={len(frame):,}; base={base_values.shape[1]}; HOG={hog_values.shape[1] - HOG_OFFSET}; "
        f"PCA={args.pca_components}; fused={base_values.shape[1] + args.pca_components}; "
        f"device={DEVICE}",
        flush=True,
    )

    fold_rows = []
    started = time.perf_counter()
    fold_path = args.output.with_name(args.output.stem + "_folds.csv")
    for fold, (outer_train, outer_test) in enumerate(splits, 1):
        fold_started = time.perf_counter()
        score, explained, best_epoch = evaluate_fold(
            base_values,
            hog_values,
            hog_order,
            labels,
            outer_train,
            outer_test,
            fold,
            args.pca_components,
            args.seed,
        )
        fold_rows.append({
            "fold": fold,
            "bacc": score,
            "pca_explained_variance": explained,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.perf_counter() - fold_started,
        })
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
        print(
            f"fold={fold}/5 bacc={score:.4f} PCA_variance={explained:.4f} "
            f"epoch={best_epoch} elapsed={time.perf_counter() - fold_started:.1f}s",
            flush=True,
        )

    scores = np.asarray([row["bacc"] for row in fold_rows], dtype=np.float64)
    baseline = pd.read_csv(args.baseline).iloc[0]
    baseline_scores = np.asarray(
        [baseline[f"fold_{fold}"] for fold in range(1, 6)], dtype=np.float64
    )
    delta = 100 * (scores - baseline_scores)
    row = {
        "classifier": "MLP",
        "condition": f"Visual 258 + line + Felzenszwalb + HOG PCA-{args.pca_components}",
        "n_images": len(frame),
        "n_styles": len(np.unique(labels)),
        "n_features": base_values.shape[1] + args.pca_components,
        "hog_raw_features": hog_values.shape[1] - HOG_OFFSET,
        "pca_components": args.pca_components,
        "pca_explained_variance_mean": float(
            np.mean([item["pca_explained_variance"] for item in fold_rows])
        ),
        "bacc_mean": float(scores.mean()),
        "bacc_std": float(scores.std()),
        "baseline_bacc_mean": float(baseline_scores.mean()),
        "delta_vs_baseline_pp": float(delta.mean()),
        "p_vs_baseline": float(ttest_rel(scores, baseline_scores).pvalue),
        "positive_folds": int(np.sum(delta > 0)),
        "elapsed_seconds": time.perf_counter() - started,
        **{f"fold_{fold}": float(score) for fold, score in enumerate(scores, 1)},
        **{f"delta_fold_{fold}_pp": float(value) for fold, value in enumerate(delta, 1)},
    }
    pd.DataFrame([row]).to_csv(args.output, index=False)
    print(pd.DataFrame([row]).to_string(index=False), flush=True)
    print(f"saved={args.output} folds={fold_path}", flush=True)


if __name__ == "__main__":
    main()
