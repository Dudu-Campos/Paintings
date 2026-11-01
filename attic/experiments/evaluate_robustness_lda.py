"""Evaluate modality robustness on controlled WikiArt image perturbations.

Each outer fold is split at the *painting parent* level.  An LDA is fit only
on canonical/original images, then evaluated on the original and each derived
variant of the held-out parents.  Therefore no transformed counterpart of an
evaluation painting can enter training.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SLIC_METADATA = {
    "stem", "style", "segmentation", "slic_requested_segments",
    "slic_actual_segments", "slic_compactness", "slic_sigma",
    "slic_max_side", "elapsed_seconds_all_graphs",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_npy_features(values_path: Path, ids_path: Path) -> pd.DataFrame:
    values = np.load(values_path)
    sample_ids = np.load(ids_path, allow_pickle=True).astype(str)
    if len(values) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Invalid IDs in {ids_path}")
    return pd.DataFrame(values, index=pd.Index(sample_ids, name="sample_id"))


def load_modalities(features_dir: Path) -> dict[str, pd.DataFrame]:
    graph = pd.read_parquet(features_dir / "slic" / "combined_features.parquet")
    graph_columns = [column for column in graph.columns if column not in SLIC_METADATA]
    if len(graph_columns) != 71:
        raise ValueError(f"Expected 71 SLIC features, found {len(graph_columns)}")
    graph = graph.set_index("stem")[graph_columns]
    graph.index = graph.index.astype(str)
    if graph.index.has_duplicates:
        raise ValueError("Duplicate SLIC sample IDs")
    return {
        "SLIC multigraph": graph,
        "Descriptors": load_npy_features(
            features_dir / "desc_full_features.npy", features_dir / "desc_full_stems.npy"
        ),
        "Hilbert": load_npy_features(
            features_dir / "hilbert_features.npy", features_dir / "hilbert_stems.npy"
        ),
    }


def make_lda(classes: np.ndarray) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(priors=np.ones(len(classes)) / len(classes))),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data" / "wikiart_robustness" / "manifest.csv")
    parser.add_argument("--features-dir", type=Path,
                        default=ROOT / "outputs" / "wikiart_robustness_features")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "wikiart_robustness_features" / "lda_robustness")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"sample_id", "parent_id", "style", "variant"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["parent_id"] = manifest["parent_id"].astype(str)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest sample_id values must be unique")
    if manifest.duplicated(["parent_id", "variant"]).any():
        raise ValueError("Each parent must have at most one row per variant")

    originals = manifest[manifest["variant"] == "original"].copy()
    if len(originals) != originals["parent_id"].nunique():
        raise ValueError("Expected exactly one original sample per parent")
    variants = list(manifest["variant"].drop_duplicates())
    parent_labels = originals.set_index("parent_id")["style"]
    if parent_labels.value_counts().min() < args.folds:
        raise ValueError("At least one style has fewer parents than --folds")
    modalities = load_modalities(args.features_dir)
    expected_ids = set(manifest["sample_id"])
    for name, features in modalities.items():
        missing_ids = expected_ids - set(features.index)
        if missing_ids or len(features) != len(expected_ids):
            raise ValueError(f"{name}: feature IDs do not exactly match the manifest")

    variant_by_parent = {
        variant: manifest[manifest["variant"] == variant].set_index("parent_id").sort_index()
        for variant in variants
    }
    parent_ids = parent_labels.index.to_numpy(dtype=str)
    labels = parent_labels.to_numpy(dtype=str)
    classes = np.unique(labels)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    rows: list[dict] = []
    predictions: list[dict] = []
    log(f"LDA robustness: {len(parent_ids):,} parents, {len(classes)} styles, {args.folds} parent-level folds")

    for modality_name, features in modalities.items():
        log(f"{modality_name}: {features.shape[1]} features")
        for fold, (train_index, test_index) in enumerate(splitter.split(parent_ids, labels), start=1):
            train_parents = parent_ids[train_index]
            test_parents = parent_ids[test_index]
            train_ids = variant_by_parent["original"].loc[train_parents, "sample_id"].to_numpy()
            train_y = parent_labels.loc[train_parents].to_numpy()
            estimator = make_lda(classes)
            estimator.fit(features.loc[train_ids].to_numpy(), train_y)

            original_ids = variant_by_parent["original"].loc[test_parents, "sample_id"].to_numpy()
            original_values = features.loc[original_ids].to_numpy()
            original_prediction = estimator.predict(original_values)
            imputer = estimator.named_steps["imputer"]
            scaler = estimator.named_steps["scaler"]
            original_scaled = scaler.transform(imputer.transform(original_values))

            for variant in variants:
                test_rows = variant_by_parent[variant].loc[test_parents]
                sample_ids = test_rows["sample_id"].to_numpy()
                y_true = test_rows["style"].to_numpy(dtype=str)
                values = features.loc[sample_ids].to_numpy()
                prediction = estimator.predict(values)
                scaled = scaler.transform(imputer.transform(values))
                rms_shift = np.sqrt(np.mean((scaled - original_scaled) ** 2, axis=1))
                agreement = prediction == original_prediction
                rows.append({
                    "modality": modality_name,
                    "variant": variant,
                    "fold": fold,
                    "n_test": len(y_true),
                    "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
                    "accuracy": accuracy_score(y_true, prediction),
                    "macro_f1": f1_score(y_true, prediction, average="macro"),
                    "prediction_agreement_original": float(agreement.mean()),
                    "feature_rms_shift": float(rms_shift.mean()),
                })
                predictions.extend({
                    "modality": modality_name, "variant": variant, "fold": fold,
                    "sample_id": sample_id, "parent_id": parent_id, "style": style,
                    "y_true": truth, "y_pred": pred,
                    "agrees_with_original_prediction": bool(same), "feature_rms_shift": float(shift),
                } for sample_id, parent_id, style, truth, pred, same, shift in zip(
                    sample_ids, test_rows.index.to_numpy(), test_rows["style"].to_numpy(),
                    y_true, prediction, agreement, rms_shift, strict=True
                ))
            log(f"  fold {fold}: original BAcc={rows[-len(variants)]['balanced_accuracy']:.4f}")

    fold_results = pd.DataFrame(rows)
    summary = fold_results.groupby(["modality", "variant"], as_index=False).agg(
        folds=("fold", "count"),
        bacc_mean=("balanced_accuracy", "mean"), bacc_std=("balanced_accuracy", "std"),
        accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
        agreement_mean=("prediction_agreement_original", "mean"),
        feature_rms_shift_mean=("feature_rms_shift", "mean"),
    )
    original_bacc = summary[summary["variant"] == "original"].set_index("modality")["bacc_mean"]
    summary["bacc_delta_from_original"] = summary.apply(
        lambda row: row["bacc_mean"] - original_bacc[row["modality"]], axis=1
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(args.output_dir / "fold_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)

    pivot = summary.pivot(index="modality", columns="variant", values="bacc_delta_from_original") * 100
    figure, axis = plt.subplots(figsize=(8.5, 3.2))
    image = axis.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-30, vmax=5, aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, f"{pivot.iloc[row, column]:+.1f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axis, label="BAcc change vs. original (p.p.)")
    axis.set_title("LDA robustness: train on originals, test on controlled variants")
    figure.tight_layout()
    figure.savefig(args.output_dir / "bacc_delta_heatmap.png", dpi=180)
    log(f"Saved results to {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
