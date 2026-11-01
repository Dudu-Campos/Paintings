"""Greedy LDA fusion curve starting from the 71-feature SLIC multigraph."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(ROOT / "src" / "pipeline"))
BASELINE_DIR = ROOT / "outputs/wikiart_features"
SLIC_DIR = ROOT / "outputs/wikiart_slic_features"
OUTPUT_CSV = ROOT / "paper/fusion_curve_slic_results.csv"
OUTPUT_PNG = ROOT / "paper/fusion_curve_slic.png"
# SEED mora em src/pipeline/protocol.py desde 06/set; re-exportado porque
# este arquivo era a raiz da cadeia que 62 scripts seguiam para achá-lo.
from protocol import SEED  # noqa: E402,F401
N_FOLDS = 5

NRC_CATEGORIES = [
    "anger", "anticipation", "disgust", "fear", "joy", "negative",
    "positive", "sadness", "surprise", "trust",
]
SENTIMENT_FEATURES = [
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
    "tb_polarity", "tb_subjectivity",
] + [f"nrc_{category}" for category in NRC_CATEGORIES]

SLIC_METADATA = {
    "stem", "style", "segmentation", "slic_requested_segments",
    "slic_actual_segments", "slic_compactness", "slic_sigma",
    "slic_max_side", "elapsed_seconds_all_graphs",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_npy(feature_name: str, stem_name: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    values = np.load(BASELINE_DIR / feature_name)
    stems = np.load(BASELINE_DIR / stem_name, allow_pickle=True).astype(str)
    columns = [f"{prefix}_{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "stem", stems)
    return frame, columns


def make_model(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def evaluate(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, list[float]]:
    values = frame[columns].to_numpy(dtype=np.float64)
    scores = []
    n_classes = len(np.unique(y))
    for train, test in splits:
        estimator = make_model(n_classes, len(columns))
        estimator.fit(values[train], y[train])
        prediction = estimator.predict(values[test])
        scores.append(float(balanced_accuracy_score(y[test], prediction)))
    return float(np.mean(scores)), float(np.std(scores)), scores


def main() -> None:
    graph = pd.read_parquet(SLIC_DIR / "combined_features.parquet")
    graph_columns = [column for column in graph.columns if column not in SLIC_METADATA]
    if len(graph_columns) != 71:
        raise ValueError(f"Expected 71 SLIC features, found {len(graph_columns)}")

    hilbert, hilbert_columns = load_npy(
        "hilbert_features.npy", "hilbert_stems.npy", "hilbert"
    )
    random_walk, random_walk_columns = load_npy(
        "rw_features.npy", "rw_stems.npy", "rw"
    )
    hybrid_walk, hybrid_walk_columns = load_npy(
        "hybrid_walk_features.npy", "hybrid_walk_stems.npy", "hybrid"
    )
    descriptors, descriptor_columns = load_npy(
        "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
    )
    sentiment = pd.read_parquet(
        BASELINE_DIR / "sentiment_features.parquet",
        columns=["stem"] + SENTIMENT_FEATURES,
    ).drop_duplicates("stem")

    frame = graph[["stem"] + graph_columns]
    for addition in (hilbert, random_walk, hybrid_walk, descriptors, sentiment):
        frame = frame.merge(addition, on="stem", how="inner", validate="one_to_one")

    metadata = pd.read_csv(ROOT / "data/artemis/wikiart_meta.csv")
    labels = dict(zip(metadata["stem"].astype(str), metadata["style"].astype(str)))
    frame["label"] = frame["stem"].astype(str).map(labels)
    frame = frame.dropna(subset=["label"]).sort_values("stem").reset_index(drop=True)
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= N_FOLDS].index)].reset_index(drop=True)

    groups = {
        "SLIC multigraph": graph_columns,
        "Descriptors": descriptor_columns,
        "Hilbert": hilbert_columns,
        "Random walk": random_walk_columns,
        "Hybrid walk": hybrid_walk_columns,
        "Sentiment": SENTIMENT_FEATURES,
    }
    y = frame["label"].to_numpy(dtype=str)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y)), y))
    log(f"Dataset: {len(frame):,} images, {len(np.unique(y))} styles, identical folds")

    standalone = {}
    for name, columns in groups.items():
        mean, std, scores = evaluate(frame, columns, y, splits)
        standalone[name] = (mean, std)
        log(f"Standalone {name:<15} {len(columns):>3} feat: {mean:.4f} ± {std:.4f}")

    selected: list[str] = []
    selected_columns: list[str] = []
    remaining = list(groups)
    rows = [{
        "step": 0, "added_group": "Chance", "selected": "",
        "n_features": 0, "bacc_mean": 1 / len(np.unique(y)), "bacc_std": 0.0,
    }]
    while remaining:
        candidates = []
        for name in remaining:
            columns = selected_columns + groups[name]
            mean, std, scores = evaluate(frame, columns, y, splits)
            candidates.append((mean, name, std, scores))
            log(f"Candidate step {len(selected)+1}: +{name:<15} {mean:.4f} ± {std:.4f}")
        mean, name, std, scores = max(candidates, key=lambda item: item[0])
        selected.append(name)
        selected_columns.extend(groups[name])
        remaining.remove(name)
        rows.append({
            "step": len(selected), "added_group": name,
            "selected": " + ".join(selected), "n_features": len(selected_columns),
            "bacc_mean": mean, "bacc_std": std,
            **{f"fold_{index + 1}": value for index, value in enumerate(scores)},
        })
        log(f"Selected: {rows[-1]['selected']} -> {mean:.4f} ± {std:.4f}")

    result = pd.DataFrame(rows)
    for name, (mean, std) in standalone.items():
        result[f"standalone_{name}_mean"] = mean
        result[f"standalone_{name}_std"] = std
    result.to_csv(OUTPUT_CSV, index=False)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = result["step"].to_numpy()
    means = result["bacc_mean"].to_numpy()
    stds = result["bacc_std"].to_numpy()
    axis.plot(x, means, "o-", color="#2a78d6", linewidth=2.3, markersize=7)
    axis.fill_between(x, means - stds, means + stds, color="#2a78d6", alpha=0.16)
    for index, row in result.iterrows():
        label = "Chance" if index == 0 else f"+ {row['added_group']}\n({row['n_features']} feat)"
        axis.annotate(label, (row["step"], row["bacc_mean"]), xytext=(0, 10),
                      textcoords="offset points", ha="center", fontsize=8)
    axis.set_xlabel("Greedy fusion step")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("WikiArt style classification — SLIC multigraph feature fusion")
    axis.set_xticks(x)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=180)
    log(f"Saved: {OUTPUT_CSV}")
    log(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
