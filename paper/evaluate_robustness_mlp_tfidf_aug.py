"""Parent-grouped MLP augmentation evaluation with a nested TF-IDF text branch.

Text scores are 27-dimensional, label-aligned LinearSVC decision values.  For
each outer fold they are OOF within the MLP fitting partition, then fit only on
that partition to score internal validation and outer test.  This avoids using
a caption's own label to construct its train-time text representation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_robustness_mlp_aug import (
    DEVICE, RECIPES, SLIC_METADATA, BranchMLP, seed_everything, train_one,
)
from tfidf_caption_style import (
    clean_caption, load_caption_map, make_classifier, make_stop_words, make_vectorizer,
)


ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_npy(values_path: Path, ids_path: Path) -> pd.DataFrame:
    values = np.load(values_path)
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    frame = pd.DataFrame(values, index=pd.Index(ids, name="sample_id"))
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate IDs in {ids_path}")
    return frame


def load_visual_blocks(features_dir: Path) -> dict[str, pd.DataFrame]:
    graph = pd.read_parquet(features_dir / "slic" / "combined_features.parquet")
    columns = [column for column in graph if column not in SLIC_METADATA]
    if len(columns) != 71:
        raise ValueError(f"Expected 71 SLIC features; got {len(columns)}")
    graph = graph.set_index("stem")[columns]
    graph.index = graph.index.astype(str)
    return {
        "SLIC": graph,
        "Descriptors": load_npy(features_dir / "desc_full_features.npy", features_dir / "desc_full_stems.npy"),
        "Hilbert": load_npy(features_dir / "hilbert_features.npy", features_dir / "hilbert_stems.npy"),
    }


def aligned_decision(classifier, values, n_classes: int) -> np.ndarray:
    decision = classifier.decision_function(values)
    aligned = np.zeros((values.shape[0], n_classes), dtype=np.float32)
    aligned[:, classifier.classes_.astype(int)] = decision
    return aligned


def text_scores(
    fit_texts: np.ndarray, fit_labels: np.ndarray, target_texts: np.ndarray,
    stop_words: list[str], n_classes: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF scores for fit rows and fit-trained scores for target rows."""
    oof = np.zeros((len(fit_texts), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, validation in splitter.split(np.zeros(len(fit_labels), dtype=np.uint8), fit_labels):
        vectorizer = make_vectorizer(stop_words)
        classifier = make_classifier()
        classifier.set_params(max_iter=20_000)
        classifier.fit(vectorizer.fit_transform(fit_texts[train]), fit_labels[train])
        oof[validation] = aligned_decision(classifier, vectorizer.transform(fit_texts[validation]), n_classes)
    vectorizer = make_vectorizer(stop_words)
    classifier = make_classifier()
    classifier.set_params(max_iter=20_000)
    classifier.fit(vectorizer.fit_transform(fit_texts), fit_labels)
    target = aligned_decision(classifier, vectorizer.transform(target_texts), n_classes)
    return oof, target


def make_values(
    parent_ids: np.ndarray, recipes: list[tuple[str, str, str]], variants: dict[str, pd.DataFrame],
    blocks: dict[str, pd.DataFrame], labels: pd.Series, text_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrices, targets = [], []
    for slic_variant, descriptor_variant, hilbert_variant in recipes:
        slic_ids = variants[slic_variant].loc[parent_ids, "sample_id"].to_numpy()
        descriptor_ids = variants[descriptor_variant].loc[parent_ids, "sample_id"].to_numpy()
        hilbert_ids = variants[hilbert_variant].loc[parent_ids, "sample_id"].to_numpy()
        matrices.append(np.hstack([
            blocks["SLIC"].loc[slic_ids].to_numpy(),
            blocks["Descriptors"].loc[descriptor_ids].to_numpy(),
            blocks["Hilbert"].loc[hilbert_ids].to_numpy(),
            text_values,
        ]))
        targets.append(labels.loc[parent_ids].to_numpy())
    return np.vstack(matrices).astype(np.float32), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/wikiart_robustness_1k/manifest.csv")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "outputs/wikiart_robustness_1k_features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/wikiart_robustness_1k_features/mlp_tfidf_selective_aug")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    originals = manifest[manifest.variant.eq("original")].copy()
    parent_stems = originals.set_index("parent_id")["stem"].astype(str)
    labels_text = originals.set_index("parent_id")["style"].astype(str)
    variants = {name: frame.set_index("parent_id") for name, frame in manifest.groupby("variant", sort=False)}
    captions = load_caption_map()
    has_caption = parent_stems.isin(captions)
    if not has_caption.all():
        log(f"Dropping {(~has_caption).sum():,} parents without captions from both conditions.")
        parent_stems, labels_text = parent_stems[has_caption], labels_text[has_caption]
    caption_text = parent_stems.map(captions).map(clean_caption)
    if caption_text.eq("").any():
        raise ValueError("Empty cleaned caption")
    blocks = load_visual_blocks(args.features_dir)
    expected_ids = set(manifest.sample_id.astype(str))
    for name, block in blocks.items():
        if set(block.index.astype(str)) != expected_ids:
            raise ValueError(f"{name} IDs do not match manifest")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_text.to_numpy())
    labels = pd.Series(y, index=labels_text.index)
    parent_ids = labels.index.to_numpy(dtype=str)
    if labels_text.value_counts().min() < args.folds:
        raise ValueError("Insufficient examples in at least one class")
    texts = caption_text.loc[parent_ids].to_numpy(dtype=str)
    stop_words = make_stop_words(encoder.classes_)
    branch_sizes = [blocks[name].shape[1] for name in ("SLIC", "Descriptors", "Hilbert")] + [len(encoder.classes_)]
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    rows, predictions = [], []
    log(f"TF-IDF MLP augmentation: {len(parent_ids):,} parents, {len(encoder.classes_)} styles, device={DEVICE}")
    log("Branches: " + ", ".join(map(str, branch_sizes)))
    for fold, (outer_train, outer_test) in enumerate(splitter.split(parent_ids, y), start=1):
        fit_train, validation = train_test_split(
            outer_train, test_size=0.1, stratify=y[outer_train], random_state=args.seed + fold
        )
        train_parents, validation_parents, test_parents = parent_ids[fit_train], parent_ids[validation], parent_ids[outer_test]
        target_texts = np.concatenate([texts[validation], texts[outer_test]])
        train_text, target_scores = text_scores(
            texts[fit_train], y[fit_train], target_texts,
            stop_words, len(encoder.classes_), args.seed + fold,
        )
        validation_text = target_scores[:len(validation)]
        test_text = target_scores[len(validation):]
        validation_x, validation_y = make_values(validation_parents, RECIPES["clean"], variants, blocks, labels, validation_text)
        test_x, test_y = make_values(test_parents, RECIPES["clean"], variants, blocks, labels, test_text)
        for condition, recipes in RECIPES.items():
            # ``make_values`` emits one block per recipe, so each individual
            # text matrix must stay parent-aligned (not be pre-repeated).
            train_x, train_y = make_values(train_parents, recipes, variants, blocks, labels, train_text)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x).astype(np.float32)
            validation_scaled = scaler.transform(validation_x).astype(np.float32)
            test_scaled = scaler.transform(test_x).astype(np.float32)
            started = time.perf_counter()
            metrics, prediction = train_one(
                train_x, train_y, validation_scaled, validation_y, test_scaled, test_y,
                branch_sizes, len(encoder.classes_), args.seed + fold, args.epochs, args.patience,
            )
            rows.append({"fold": fold, "condition": condition, "train_rows": len(train_x), "elapsed_seconds": time.perf_counter() - started, **metrics})
            predictions.extend({"fold": fold, "condition": condition, "parent_id": parent_id,
                                "y_true": encoder.classes_[truth], "y_pred": encoder.classes_[pred]}
                               for parent_id, truth, pred in zip(test_parents, test_y, prediction, strict=True))
            log(f"fold {fold}/{args.folds} {condition}: BAcc={metrics['bacc']:.4f}, val={metrics['validation_bacc']:.4f}, epoch={metrics['best_epoch']}")

    detail = pd.DataFrame(rows)
    summary = detail.groupby("condition", as_index=False).agg(
        bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"), accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"), validation_bacc_mean=("validation_bacc", "mean"),
        best_epoch_mean=("best_epoch", "mean"), elapsed_seconds=("elapsed_seconds", "sum"),
    )
    clean_bacc = summary.loc[summary.condition.eq("clean"), "bacc_mean"].iat[0]
    summary["bacc_delta_from_clean"] = summary.bacc_mean - clean_bacc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)
    log(f"Saved results to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
