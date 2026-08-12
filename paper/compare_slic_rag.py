"""Paired comparison of percolation RAG and node-budget-matched SLIC graphs.

All methods are evaluated on the same image intersection and the exact same
stratified folds.  Each graph is represented by the same 32 aggregate features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "n_nodes", "n_edges", "density", "area_gini", "area_entropy",
    "area_max_frac", "area_mean", "area_std", "color_dist_mean",
    "color_dist_std", "color_dist_max", "spatial_dist_mean",
    "spatial_dist_std", "degree_mean", "degree_std", "degree_max",
    "hub_degree_ratio", "n_leaf_frac", "degree_gini",
    "area_second_max_frac", "area_cv", "spatial_density_cv",
    "spatial_density_max_block_frac", "L_mean_w", "L_std_w",
    "a_mean_w", "a_std_w", "b_mean_w", "b_std_w", "chroma_mean_w",
    "chroma_std_w", "warm_frac",
]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def load_method(name: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(FEATURE_COLS + ["stem"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates("stem").copy()
    return frame


def make_model(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    n_components=min(n_classes - 1, n_features),
                    priors=np.ones(n_classes) / n_classes,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag",
        type=Path,
        default=Path("outputs/wikiart_features/graph_features.parquet"),
    )
    parser.add_argument(
        "--slic", type=parse_named_path, action="append", required=True,
        help="SLIC result as NAME=PATH; may be repeated",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/results_slic_vs_rag.csv"),
    )
    args = parser.parse_args()

    methods = {"Percolation RAG": load_method("Percolation RAG", args.rag)}
    for name, path in args.slic:
        if name in methods:
            parser.error(f"duplicate method name: {name}")
        methods[name] = load_method(name, path)

    common_stems = set.intersection(*(set(frame["stem"].astype(str)) for frame in methods.values()))
    rag_frame = methods["Percolation RAG"]
    if "style" not in rag_frame.columns:
        raise ValueError("Percolation RAG is missing the canonical style column")
    cohort = (
        rag_frame[rag_frame["stem"].astype(str).isin(common_stems)][["stem", "style"]]
        .drop_duplicates("stem")
        .sort_values("stem")
        .reset_index(drop=True)
    )
    counts = cohort["style"].value_counts()
    valid_styles = counts[counts >= args.folds].index
    cohort = cohort[cohort["style"].isin(valid_styles)].reset_index(drop=True)
    if cohort.empty:
        raise ValueError("No shared images remain after applying the fold requirement")

    y = cohort["style"].to_numpy(dtype=str)
    n_classes = len(np.unique(y))
    print(
        f"Paired cohort: {len(cohort):,} images, {n_classes} styles, "
        f"{args.folds} identical folds"
    )

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.zeros(len(y)), y))
    rows: list[dict] = []
    score_by_method: dict[str, np.ndarray] = {}

    for name, frame in methods.items():
        indexed = frame.set_index(frame["stem"].astype(str))
        X = indexed.loc[cohort["stem"], FEATURE_COLS].to_numpy(dtype=np.float64)
        scores = []
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            model = make_model(n_classes, len(FEATURE_COLS))
            model.fit(X[train_idx], y[train_idx])
            prediction = model.predict(X[test_idx])
            score = balanced_accuracy_score(y[test_idx], prediction)
            scores.append(float(score))
            print(f"  {name:<24} fold {fold}: BAcc={score:.4f}")
        score_array = np.asarray(scores)
        score_by_method[name] = score_array
        rows.append(
            {
                "method": name,
                "n_images": len(cohort),
                "n_styles": n_classes,
                "n_features": len(FEATURE_COLS),
                "bacc_mean": score_array.mean(),
                "bacc_std": score_array.std(),
                **{f"fold_{i + 1}": value for i, value in enumerate(score_array)},
            }
        )

    rag_scores = score_by_method["Percolation RAG"]
    print("\nPaired result")
    print(f"  {'Percolation RAG':<24} {rag_scores.mean():.4f} ± {rag_scores.std():.4f}")
    for name, scores in score_by_method.items():
        if name == "Percolation RAG":
            continue
        difference = rag_scores - scores
        print(
            f"  {name:<24} {scores.mean():.4f} ± {scores.std():.4f}  "
            f"RAG-SLIC={difference.mean():+.4f}"
        )
        for row in rows:
            if row["method"] == name:
                row["rag_minus_method_mean"] = difference.mean()
                row["rag_minus_method_std"] = difference.std()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
