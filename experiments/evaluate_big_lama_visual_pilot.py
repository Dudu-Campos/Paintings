#!/usr/bin/env python3
"""Paired 5-class evaluation of Big-LaMa only in the handcrafted pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


SLIC_METADATA = {
    "stem", "style", "segmentation", "slic_requested_segments",
    "slic_actual_segments", "slic_compactness", "slic_sigma",
    "slic_max_side", "elapsed_seconds_all_graphs",
}


def load_npy(directory: Path, values_name: str, stems_name: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    values = np.load(directory / values_name)
    stems = np.load(directory / stems_name, allow_pickle=True).astype(str)
    columns = [f"{prefix}_{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "stem", stems)
    return frame.drop_duplicates("stem"), columns


def load_representation(directory: Path, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    graph = pd.read_parquet(directory / "slic" / "combined_features.parquet")
    graph_columns = [column for column in graph.columns if column not in SLIC_METADATA]
    if len(graph_columns) != 71:
        raise ValueError(f"Expected 71 graph features in {directory}, got {len(graph_columns)}")
    graph_rename = {column: f"{prefix}_slic_{column}" for column in graph_columns}
    frame = graph[["stem"] + graph_columns].drop_duplicates("stem").rename(columns=graph_rename)
    columns = list(graph_rename.values())

    descriptors, descriptor_columns = load_npy(
        directory, "desc_full_features.npy", "desc_full_stems.npy", f"{prefix}_descriptor"
    )
    hilbert, hilbert_columns = load_npy(
        directory, "hilbert_features.npy", "hilbert_stems.npy", f"{prefix}_hilbert"
    )
    for addition in (descriptors, hilbert):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
    columns.extend(descriptor_columns)
    columns.extend(hilbert_columns)
    if len(columns) != 258:
        raise ValueError(f"Expected 258 visual features in {directory}, got {len(columns)}")
    return frame, columns


def lda_scores(values: np.ndarray, labels: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    n_classes = len(np.unique(labels))
    scores = []
    for train, test in splits:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(
                n_components=min(n_classes - 1, values.shape[1]),
                priors=np.ones(n_classes) / n_classes,
            )),
        ])
        model.fit(values[train], labels[train])
        scores.append(float(balanced_accuracy_score(labels[test], model.predict(values[test]))))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/big_lama_visual_pilot_sample.csv"))
    parser.add_argument("--original", type=Path, default=Path("outputs/wikiart_slic_features").parent / "wikiart_features")
    parser.add_argument("--original-slic", type=Path, default=Path("outputs/wikiart_slic_features"))
    parser.add_argument("--variant-root", type=Path, default=Path("outputs/big_lama_visual_pilot/features"))
    parser.add_argument("--output", type=Path, default=Path("paper/big_lama_visual_pilot_results.csv"))
    parser.add_argument(
        "--classifiers", nargs="+", choices=("LDA", "XGBoost", "MLP"),
        default=("LDA", "XGBoost", "MLP"),
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)[["stem", "style"]]
    # Original graph and original descriptors/Hilbert live in separate directories.
    original_graph = pd.read_parquet(args.original_slic / "combined_features.parquet")
    graph_columns = [column for column in original_graph.columns if column not in SLIC_METADATA]
    graph_rename = {column: f"original_slic_{column}" for column in graph_columns}
    original = original_graph[["stem"] + graph_columns].drop_duplicates("stem").rename(columns=graph_rename)
    original_columns = list(graph_rename.values())
    descriptors, descriptor_columns = load_npy(
        args.original, "desc_full_features.npy", "desc_full_stems.npy", "original_descriptor"
    )
    hilbert, hilbert_columns = load_npy(
        args.original, "hilbert_features.npy", "hilbert_stems.npy", "original_hilbert"
    )
    for addition in (descriptors, hilbert):
        original = original.merge(addition, on="stem", how="inner", validate="one_to_one")
    original_columns.extend(descriptor_columns + hilbert_columns)

    representations = {"Original": (original, original_columns)}
    for name, folder in (
        ("Resize only", "resized"),
        ("Blank canvas", "blank_canvas"),
        ("Big-LaMa", "outpaint"),
    ):
        representations[name] = load_representation(args.variant_root / folder, folder)

    frame = sample.copy()
    conditions: dict[str, list[str]] = {}
    for name, (addition, columns) in representations.items():
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")
        conditions[name] = columns
    conditions["Original + Resize"] = conditions["Original"] + conditions["Resize only"]
    conditions["Original + Blank"] = conditions["Original"] + conditions["Blank canvas"]
    conditions["Original + Big-LaMa"] = conditions["Original"] + conditions["Big-LaMa"]
    frame = frame.sort_values("stem").reset_index(drop=True)
    if len(frame) != len(sample):
        raise RuntimeError(f"Feature intersection lost rows: {len(frame)}/{len(sample)}")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["style"].astype(str))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_fusion_classifiers import evaluate_mlp, evaluate_xgb

    evaluators = {
        "LDA": lambda values: lda_scores(values.astype(np.float64), labels, splits),
        "XGBoost": lambda values: evaluate_xgb(values.astype(np.float32), labels, splits),
        "MLP": lambda values: evaluate_mlp(values.astype(np.float32), labels, splits),
    }
    rows = []
    for classifier in args.classifiers:
        for name, columns in conditions.items():
            started = time.perf_counter()
            values = frame[columns].to_numpy(dtype=np.float32)
            scores = np.asarray(evaluators[classifier](values))
            rows.append({
                "condition": name,
                "classifier": classifier,
                "n_images": len(frame),
                "n_styles": len(encoder.classes_),
                "n_features": len(columns),
                "bacc_mean": scores.mean(),
                "bacc_std": scores.std(),
                "elapsed_seconds": time.perf_counter() - started,
                **{f"fold_{index + 1}": score for index, score in enumerate(scores)},
            })
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(
                f"{classifier:<8} {name:<24} "
                f"{scores.mean():.4f} ± {scores.std():.4f}",
                flush=True,
            )
    result = pd.DataFrame(rows)
    original_scores = result.set_index(["classifier", "condition"])["bacc_mean"]
    result["delta_vs_original_pp"] = [
        (row.bacc_mean - original_scores.loc[(row.classifier, "Original")]) * 100
        for row in result.itertuples()
    ]
    result["delta_vs_blank_pp"] = [
        (row.bacc_mean - original_scores.loc[(row.classifier, "Blank canvas")]) * 100
        for row in result.itertuples()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[["classifier", "condition", "n_features", "bacc_mean", "bacc_std", "delta_vs_original_pp", "delta_vs_blank_pp"]].to_string(index=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
