"""Paired evaluation of the four SLIC graph families and their concatenation."""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src" / "pipeline"))

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


# O esquema do bloco de multigrafo mora em src/pipeline/corpora.py desde
# 06/set. Nasceu aqui e era alcancado por uma cadeia de tres imports.
from corpora import (  # noqa: E402,F401
    RAG_FEATURES, KNN_FEATURES, EDGE_FEATURES,
)


def load(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["stem"] + columns)
    if frame["stem"].duplicated().any():
        frame = frame.drop_duplicates("stem")
    return frame.set_index(frame["stem"].astype(str))


def model(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-dir", type=Path,
        default=Path("outputs/wikiart_slic_features"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/results_slic_multigraph_full.csv"),
    )
    args = parser.parse_args()

    frames = {
        "SLIC-RAG": load(args.graph_dir / "graph_features.parquet", RAG_FEATURES),
        "SLIC-kNN": load(args.graph_dir / "knn_features.parquet", KNN_FEATURES),
        "SLIC-Delaunay": load(
            args.graph_dir / "delaunay_features.parquet", RAG_FEATURES
        ),
        "SLIC-MST": load(args.graph_dir / "mst_features.parquet", RAG_FEATURES),
    }

    common = set.intersection(*(set(frame.index) for frame in frames.values()))
    metadata = pd.read_csv("data/artemis/wikiart_meta.csv")
    labels = dict(zip(metadata["stem"].astype(str), metadata["style"].astype(str)))
    cohort = pd.DataFrame({"stem": sorted(common)})
    cohort["style"] = cohort["stem"].map(labels)
    cohort = cohort.dropna(subset=["style"]).copy()
    counts = cohort["style"].value_counts()
    cohort = cohort[cohort["style"].isin(counts[counts >= args.folds].index)]
    stems = cohort["stem"].astype(str).tolist()
    y = cohort["style"].astype(str).to_numpy()
    n_classes = len(np.unique(y))

    matrices = {
        "SLIC-RAG": frames["SLIC-RAG"].loc[stems, RAG_FEATURES].to_numpy(float),
        "SLIC-kNN": frames["SLIC-kNN"].loc[stems, KNN_FEATURES].to_numpy(float),
        "SLIC-Delaunay": frames["SLIC-Delaunay"].loc[stems, RAG_FEATURES].to_numpy(float),
        "SLIC-MST": frames["SLIC-MST"].loc[stems, RAG_FEATURES].to_numpy(float),
    }
    matrices["SLIC concat raw"] = np.hstack([
        matrices["SLIC-RAG"], matrices["SLIC-kNN"],
        matrices["SLIC-Delaunay"], matrices["SLIC-MST"],
    ])
    matrices["SLIC concat unique"] = np.hstack([
        matrices["SLIC-RAG"],
        frames["SLIC-kNN"].loc[stems, EDGE_FEATURES].to_numpy(float),
        frames["SLIC-Delaunay"].loc[stems, EDGE_FEATURES].to_numpy(float),
        frames["SLIC-MST"].loc[stems, EDGE_FEATURES].to_numpy(float),
    ])

    print(
        f"Paired cohort: {len(stems):,} images, {n_classes} styles, "
        f"{args.folds} identical folds"
    )
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.zeros(len(y)), y))
    rows = []
    for name, values in matrices.items():
        scores = []
        for fold, (train, test) in enumerate(splits, 1):
            estimator = model(n_classes, values.shape[1])
            estimator.fit(values[train], y[train])
            score = balanced_accuracy_score(y[test], estimator.predict(values[test]))
            scores.append(float(score))
            print(f"  {name:<21} fold {fold}: BAcc={score:.4f}")
        scores_array = np.asarray(scores)
        rows.append({
            "method": name,
            "n_images": len(stems),
            "n_styles": n_classes,
            "n_features": values.shape[1],
            "bacc_mean": scores_array.mean(),
            "bacc_std": scores_array.std(),
            **{f"fold_{i + 1}": value for i, value in enumerate(scores_array)},
        })

    result = pd.DataFrame(rows).sort_values("bacc_mean", ascending=False)
    print("\n", result[["method", "n_features", "bacc_mean", "bacc_std"]].to_string(index=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
