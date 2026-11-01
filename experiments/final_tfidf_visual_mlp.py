"""Final no-augmentation WikiArt MLP: visual GIST branches with TF-IDF text.

The 27-dimensional text branch consists of label-aligned LinearSVC decision
scores.  Each score is made without access to its own label: OOF inside the
MLP fit split, then fit-split trained for validation and outer-test scoring.
No sentiment feature and no image augmentation are used.

With ``--quadrant-dir``, it additionally evaluates the same method with fixed
top-left, top-right, bottom-left and bottom-right views concatenated to the
global image representation.  The views retain their position as independent
MLP branches; they are not treated as extra labelled samples.

With ``--cnn-cache``, cached 2,048-dimensional frozen ImageNet ResNet-50
embeddings are evaluated as a separate branch and fused with the visual GIST
branches.  The encoder itself remains frozen.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_robustness_mlp_aug import train_one
from fusion_curve_slic import ROOT, SEED, SLIC_METADATA, load_npy
from tfidf_caption_style import (
    clean_caption, load_caption_map, make_classifier, make_stop_words, make_vectorizer,
)


from protocol import N_FOLDS  # noqa: E402,F401
SLIC_DIR = ROOT / "outputs/wikiart_slic_features"
FEATURE_DIR = ROOT / "outputs/wikiart_features"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")


def load_region_features(region_dir: Path, region: str) -> dict[str, pd.DataFrame]:
    """Load one quadrant and give every feature a position-specific name."""
    graph = pd.read_parquet(region_dir / region / "slic" / "combined_features.parquet")
    graph_columns = [column for column in graph if column not in SLIC_METADATA]
    if len(graph_columns) != 71:
        raise ValueError(f"Expected 71 SLIC features for {region}, found {len(graph_columns)}")

    def load_local(values_name: str, stems_name: str, prefix: str) -> pd.DataFrame:
        values = np.load(region_dir / region / values_name)
        stems = np.load(region_dir / region / stems_name, allow_pickle=True).astype(str)
        columns = [f"{region}_{prefix}_{index}" for index in range(values.shape[1])]
        values_frame = pd.DataFrame(values, columns=columns)
        values_frame.insert(0, "stem", stems)
        if values_frame["stem"].duplicated().any():
            raise ValueError(f"Duplicate {prefix} stems for {region}")
        return values_frame

    slic_columns = [f"{region}_slic_{column}" for column in graph_columns]
    slic = graph[["stem"] + graph_columns].drop_duplicates("stem").rename(
        columns=dict(zip(graph_columns, slic_columns, strict=True))
    )
    return {
        f"SLIC ({region})": slic,
        f"Descriptors ({region})": load_local(
            "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
        ),
        f"Hilbert ({region})": load_local(
            "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
        ),
    }


def load_dataset(
    quadrant_dir: Path | None = None, cnn_cache: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    graph = pd.read_parquet(SLIC_DIR / "combined_features.parquet")
    graph_columns = [column for column in graph if column not in SLIC_METADATA]
    if len(graph_columns) != 71:
        raise ValueError(f"Expected 71 SLIC features, found {len(graph_columns)}")
    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    hilbert, hilbert_columns = load_npy(
        "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
    )
    frame = graph[["stem"] + graph_columns].drop_duplicates("stem")
    for addition in (descriptors, hilbert):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
    metadata = pd.read_csv(ROOT / "data/artemis/wikiart_meta.csv")
    label_map = dict(zip(metadata["stem"].astype(str), metadata["style"].astype(str)))
    frame["label"] = frame["stem"].astype(str).map(label_map)
    captions = load_caption_map()
    frame["caption"] = frame["stem"].astype(str).map(captions)
    frame = frame.dropna(subset=["label", "caption"]).sort_values("stem").reset_index(drop=True)
    frame["caption"] = frame["caption"].map(clean_caption)
    if frame["caption"].eq("").any():
        raise ValueError("Found empty cleaned caption")
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= N_FOLDS].index)].reset_index(drop=True)
    groups = {
        "SLIC": graph_columns,
        "Descriptors": descriptor_columns,
        "Hilbert": hilbert_columns,
    }
    if quadrant_dir is not None:
        for region in REGIONS:
            region_blocks = load_region_features(quadrant_dir, region)
            for name, addition in region_blocks.items():
                columns = [column for column in addition if column != "stem"]
                frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
                groups[name] = columns
        frame = frame.sort_values("stem").reset_index(drop=True)
    if cnn_cache is not None:
        values_path = cnn_cache / "resnet50_imagenet_v2_features.npy"
        stems_path = cnn_cache / "stems.npy"
        values = np.load(values_path, mmap_mode="r")
        stems = np.load(stems_path, allow_pickle=True).astype(str)
        if values.ndim != 2 or values.shape[1] != 2048 or len(values) != len(stems):
            raise ValueError(f"Invalid frozen CNN cache in {cnn_cache}")
        # A few WikiArt filenames occur in two style directories.  The legacy
        # feature tables key rows only by stem, while the CNN cache has one
        # embedding per physical image.  Exclude these ambiguous pairs rather
        # than arbitrarily matching an embedding to the wrong painting/style.
        duplicated_stems = pd.Index(stems)[pd.Index(stems).duplicated(keep=False)].unique()
        cnn_columns = [f"frozen_resnet_{index}" for index in range(values.shape[1])]
        cnn = pd.DataFrame(values, columns=cnn_columns)
        cnn.insert(0, "stem", stems)
        if len(duplicated_stems):
            log(f"Dropping {len(duplicated_stems):,} ambiguous duplicated CNN stems from fusion cohort")
            cnn = cnn.loc[~cnn["stem"].isin(duplicated_stems)].copy()
        if cnn["stem"].duplicated().any():
            raise ValueError("Frozen CNN cache still contains duplicate stems")
        frame = frame.merge(cnn, on="stem", how="inner", validate="one_to_one")
        frame = frame.sort_values("stem").reset_index(drop=True)
        groups["Frozen ResNet-50"] = cnn_columns
    return frame, groups


def aligned_decision(classifier, values, n_classes: int) -> np.ndarray:
    decision = classifier.decision_function(values)
    result = np.zeros((values.shape[0], n_classes), dtype=np.float32)
    result[:, classifier.classes_.astype(int)] = decision
    return result


def fit_text_scores(
    fit_texts: np.ndarray, fit_labels: np.ndarray, target_texts: np.ndarray,
    stop_words: list[str], n_classes: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested OOF fit scores and final fit-split scores for target texts."""
    oof = np.zeros((len(fit_texts), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, validation in splitter.split(np.zeros(len(fit_labels), dtype=np.uint8), fit_labels):
        vectorizer = make_vectorizer(stop_words)
        classifier = make_classifier()
        classifier.set_params(max_iter=20_000)
        classifier.fit(vectorizer.fit_transform(fit_texts[train]), fit_labels[train])
        oof[validation] = aligned_decision(
            classifier, vectorizer.transform(fit_texts[validation]), n_classes
        )
    vectorizer = make_vectorizer(stop_words)
    classifier = make_classifier()
    classifier.set_params(max_iter=20_000)
    classifier.fit(vectorizer.fit_transform(fit_texts), fit_labels)
    return oof, aligned_decision(classifier, vectorizer.transform(target_texts), n_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final_tfidf_visual_mlp")
    parser.add_argument(
        "--quadrant-dir", type=Path,
        help="Directory produced by `make quadrant_features`; enables multiscale conditions.",
    )
    parser.add_argument(
        "--cnn-cache", type=Path,
        help="Cached frozen ResNet-50 embeddings; enables frozen-CNN fusion conditions.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir, args.cnn_cache)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    texts = frame["caption"].to_numpy(dtype=str)
    global_groups = {name: groups[name] for name in ("SLIC", "Descriptors", "Hilbert")}
    visual_columns = [column for columns in global_groups.values() for column in columns]
    visual = frame[visual_columns].to_numpy(dtype=np.float32)
    visual_sizes = [len(columns) for columns in global_groups.values()]
    cnn_columns = groups.get("Frozen ResNet-50", [])
    cnn = frame[cnn_columns].to_numpy(dtype=np.float32) if cnn_columns else None
    cnn_visual = (
        frame[visual_columns + cnn_columns].to_numpy(dtype=np.float32)
        if cnn_columns else None
    )
    cnn_visual_sizes = visual_sizes + [len(cnn_columns)] if cnn_columns else None
    spatial_groups = {
        name: columns for name, columns in groups.items() if name != "Frozen ResNet-50"
    }
    multiscale_columns = [column for columns in spatial_groups.values() for column in columns]
    multiscale = frame[multiscale_columns].to_numpy(dtype=np.float32)
    multiscale_sizes = [len(columns) for columns in spatial_groups.values()]
    stop_words = make_stop_words(encoder.classes_)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, predictions = [], []
    log(f"Final TF-IDF MLP: {len(frame):,} images, {len(encoder.classes_)} styles; no augmentation, no sentiment")
    log(f"Global branches={visual_sizes}; TF-IDF score branch={len(encoder.classes_)}")
    if args.quadrant_dir:
        log(f"Multiscale branches={multiscale_sizes}; global plus four fixed quadrants")
    if args.cnn_cache:
        log("Frozen CNN branch=2048; ImageNet encoder remains fixed")

    for fold, (outer_train, outer_test) in enumerate(splitter.split(visual, y), start=1):
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=SEED + fold
        )
        target_texts = np.concatenate([texts[validation], texts[outer_test]])
        fit_text, target_text = fit_text_scores(
            texts[fit_train], y[fit_train], target_texts,
            stop_words, len(encoder.classes_), SEED + fold,
        )
        validation_text = target_text[:len(validation)]
        test_text = target_text[len(validation):]
        conditions = [
            ("Visual GIST", visual[fit_train], visual[validation], visual[outer_test], visual_sizes),
            (
                "Visual GIST + TF-IDF",
                np.hstack([visual[fit_train], fit_text]),
                np.hstack([visual[validation], validation_text]),
                np.hstack([visual[outer_test], test_text]),
                visual_sizes + [len(encoder.classes_)],
            ),
        ]
        if args.quadrant_dir:
            conditions.extend([
                (
                    "Visual GIST multiscale", multiscale[fit_train], multiscale[validation],
                    multiscale[outer_test], multiscale_sizes,
                ),
                (
                    "Visual GIST multiscale + TF-IDF",
                    np.hstack([multiscale[fit_train], fit_text]),
                    np.hstack([multiscale[validation], validation_text]),
                    np.hstack([multiscale[outer_test], test_text]),
                    multiscale_sizes + [len(encoder.classes_)],
                ),
            ])
        if args.cnn_cache:
            assert cnn is not None and cnn_visual is not None and cnn_visual_sizes is not None
            conditions.extend([
                (
                    "Frozen ResNet-50 MLP", cnn[fit_train], cnn[validation], cnn[outer_test],
                    [len(cnn_columns)],
                ),
                (
                    "Visual GIST + frozen ResNet-50", cnn_visual[fit_train],
                    cnn_visual[validation], cnn_visual[outer_test], cnn_visual_sizes,
                ),
                (
                    "Visual GIST + frozen ResNet-50 + TF-IDF",
                    np.hstack([cnn_visual[fit_train], fit_text]),
                    np.hstack([cnn_visual[validation], validation_text]),
                    np.hstack([cnn_visual[outer_test], test_text]),
                    cnn_visual_sizes + [len(encoder.classes_)],
                ),
            ])
        for condition, train_x, validation_x, test_x, branch_sizes in conditions:
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x).astype(np.float32)
            validation_x = scaler.transform(validation_x).astype(np.float32)
            test_x = scaler.transform(test_x).astype(np.float32)
            started = time.perf_counter()
            metrics, prediction = train_one(
                train_x, y[fit_train], validation_x, y[validation], test_x, y[outer_test],
                branch_sizes, len(encoder.classes_), SEED + fold, args.epochs, args.patience,
            )
            rows.append({
                "fold": fold, "condition": condition, "n_images": len(frame),
                "n_fit_train": len(fit_train), "n_validation": len(validation), "n_test": len(outer_test),
                "elapsed_seconds": time.perf_counter() - started, **metrics,
            })
            predictions.extend({
                "fold": fold, "condition": condition, "stem": stem,
                "true_style": encoder.classes_[truth], "predicted_style": encoder.classes_[pred],
            } for stem, truth, pred in zip(frame.iloc[outer_test]["stem"], y[outer_test], prediction, strict=True))
            log(f"fold {fold}/{N_FOLDS} {condition}: BAcc={metrics['bacc']:.4f}, val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}")

    detail = pd.DataFrame(rows)
    summary = detail.groupby("condition", as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        validation_bacc_mean=("validation_bacc", "mean"), best_epoch_mean=("best_epoch", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    )
    visual_bacc = summary.loc[summary.condition.eq("Visual GIST"), "bacc_mean"].iat[0]
    summary["bacc_delta_from_global_visual"] = summary.bacc_mean - visual_bacc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)
    log(f"Saved results to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
