"""Redundancy of the inductive painting-similarity graph feature block."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_rich_lda_structure_importance import load_rich_frame  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402


INNER_FOLDS = 5


GROUP_ORDER = (
    "Region graphs",
    "Image statistics",
    "Space-filling curve",
    "Spectral",
    "Text",
    "Similarity graph",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)



def inductive_spread(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    n_classes: int,
    n_neighbors: int,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    model.fit(train_x, train_y)
    if len(model.classes_) != n_classes:
        raise ValueError(f"LabelSpreading saw {len(model.classes_)}/{n_classes} classes")
    train_distributions = model.label_distributions_
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(train_x)
    _, indices = neighbors.kneighbors(query_x)
    return train_distributions[indices].mean(axis=1).astype(np.float32)


def inductive_node_feature_scores(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    target_x: np.ndarray,
    n_classes: int,
    n_neighbors: int,
    alpha: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(fit_x), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        oof[inner_val] = inductive_spread(
            fit_x[inner_train],
            fit_y[inner_train],
            fit_x[inner_val],
            n_classes,
            n_neighbors,
            alpha,
            max_iter,
        )
    target_scores = inductive_spread(fit_x, fit_y, target_x, n_classes, n_neighbors, alpha, max_iter)
    return oof, target_scores


def load_text_scores(stems_path: Path, decisions_path: Path, classes_path: Path) -> pd.DataFrame:
    stems = np.load(stems_path, allow_pickle=True).astype(str)
    decisions = np.load(decisions_path).astype(np.float32)
    classes = np.load(classes_path, allow_pickle=True).astype(str)
    columns = [f"text_score_{name}" for name in classes]
    frame = pd.DataFrame(decisions, columns=columns)
    frame.insert(0, "stem", stems)
    return frame


def standardise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
    std = values.std(axis=0)
    keep = std > 1e-9
    values = values[:, keep]
    std = std[keep]
    return ((values - values.mean(axis=0)) / std).astype(np.float32), keep


def summarize_pair(left: np.ndarray, right: np.ndarray, same: bool = False) -> dict[str, float | int]:
    r2 = np.square((left.T @ right) / np.float32(len(left)), dtype=np.float32)
    if same:
        values = r2[np.triu_indices(r2.shape[0], k=1)]
    else:
        values = r2.ravel()
    return {
        "n_pairs": int(values.size),
        "mean_r2": float(values.mean()),
        "median_r2": float(np.median(values)),
        "p90_r2": float(np.quantile(values, 0.90)),
        "p95_r2": float(np.quantile(values, 0.95)),
        "max_r2": float(values.max()),
        "pairs_r2_ge_0_3": int((values >= 0.3).sum()),
        "pairs_r2_ge_0_5": int((values >= 0.5).sum()),
        "pairs_r2_ge_0_8": int((values >= 0.8).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--text-stems", type=Path, default=ROOT / "results/tfidf_caption_style_stems.npy")
    parser.add_argument("--text-decisions", type=Path, default=ROOT / "results/tfidf_caption_style_oof_decisions.npy")
    parser.add_argument("--text-classes", type=Path, default=ROOT / "results/tfidf_caption_style_classes.npy")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/similarity_graph_redundancy")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    descriptors = groups["Descriptors"]
    descriptor_values = frame[descriptors].to_numpy(dtype=np.float32)
    graph_block = np.zeros((len(frame), n_classes), dtype=np.float32)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    for fold, (train, test) in enumerate(splits, 1):
        scaler = StandardScaler().fit(descriptor_values[train])
        scaled = scaler.transform(descriptor_values)
        oof, target = inductive_node_feature_scores(
            scaled[train],
            labels[train],
            scaled[test],
            n_classes,
            args.n_neighbors,
            args.alpha,
            args.max_iter,
            SEED + fold,
        )
        graph_block[train] = oof
        graph_block[test] = target
        log(f"fold={fold}/{N_FOLDS} similarity graph scores complete")
    graph_columns = [f"similarity_graph_score_{index:02d}" for index in range(n_classes)]
    graph_frame = pd.DataFrame(graph_block, columns=graph_columns)
    graph_frame.insert(0, "stem", frame["stem"].astype(str).to_numpy())

    rich_frame, rich_features, rich_groups = load_rich_frame(args)
    rich_frame = rich_frame.merge(graph_frame, on="stem", how="inner", validate="one_to_one")
    text = load_text_scores(args.text_stems, args.text_decisions, args.text_classes)
    rich_frame = rich_frame.merge(text, on="stem", how="inner", validate="one_to_one")
    text_features = [column for column in text.columns if column != "stem"]
    log(f"{len(rich_frame):,} paintings in common")

    blocks = {}
    feature_lists = {}
    for group in GROUP_ORDER[:4]:
        columns = [feature for feature, family in zip(rich_features, rich_groups) if family == group]
        values, keep = standardise(rich_frame[columns].to_numpy(dtype=np.float32))
        kept_columns = np.asarray(columns, dtype=object)[keep]
        blocks[group] = values
        feature_lists[group] = kept_columns
    for group, columns in (("Text", text_features), ("Similarity graph", graph_columns)):
        values, keep = standardise(rich_frame[columns].to_numpy(dtype=np.float32))
        kept_columns = np.asarray(columns, dtype=object)[keep]
        blocks[group] = values
        feature_lists[group] = kept_columns

    rows = []
    source = "Similarity graph"
    for target in GROUP_ORDER:
        rows.append({
            "source": source,
            "target": target,
            "source_features": int(blocks[source].shape[1]),
            "target_features": int(blocks[target].shape[1]),
            **summarize_pair(blocks[source], blocks[target], same=(target == source)),
        })
    summary = pd.DataFrame(rows).sort_values(["median_r2", "p95_r2"], ascending=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.8f")

    partner_rows = []
    graph_values = blocks[source]
    for target in GROUP_ORDER[:-1]:
        r2 = np.square((graph_values.T @ blocks[target]) / np.float32(len(graph_values)), dtype=np.float32)
        for row, feature in enumerate(feature_lists[source]):
            best = int(np.argmax(r2[row]))
            partner_rows.append({
                "feature": feature,
                "target": target,
                "best_partner": feature_lists[target][best],
                "best_r2": float(r2[row, best]),
            })
    pd.DataFrame(partner_rows).sort_values("best_r2", ascending=False).to_csv(
        args.output_dir / "top_partners.csv",
        index=False,
        float_format="%.8f",
    )
    graph_frame.to_parquet(args.output_dir / "similarity_graph_scores.parquet", index=False)
    log(summary.round(4).to_string(index=False))
    log(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
