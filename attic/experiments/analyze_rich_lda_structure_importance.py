"""LDA structure-coefficient importance for the rich visual feature pool.

This is the rich-feature analogue of ``analyze_lda_structure_importance.py``.
It uses global rich Descriptors, rich Hilbert, rich Fourier, and the region
graph blocks used in ``plot_rich_redundancy_layout.py``.  Text and multiscale
quadrants are excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_visual_feature_importance_lda_xgb import make_lda  # noqa: E402
from fusion_curve_slic import N_FOLDS, SEED  # noqa: E402
from plot_rich_redundancy_layout import (  # noqa: E402
    FAMILY_ORDER,
    read_npy,
    read_parquet,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_rich_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    meta = pd.read_csv(args.meta).dropna(subset=["style"]).drop_duplicates("stem", keep="last")
    frame = meta[["stem", "style"]].copy()
    frame["stem"] = frame["stem"].astype(str)

    pieces = [
        ("Region graphs", read_parquet(args.slic, "slic")),
        ("Region graphs", read_parquet(args.felzenszwalb, "felz")),
        ("Region graphs", read_parquet(args.line_graph, "line")),
        ("Region graphs", read_parquet(args.line_graph_felz, "line_felz")),
        (
            "Image statistics",
            read_npy(
                args.rich_dir / "descriptors" / "desc_full_features.npy",
                args.rich_dir / "descriptors" / "desc_full_stems.npy",
                "descriptor_rich",
            ),
        ),
        (
            "Space-filling curve",
            read_npy(
                args.rich_dir / "hilbert" / "hilbert_features.npy",
                args.rich_dir / "hilbert" / "hilbert_stems.npy",
                "hilbert_rich",
            ),
        ),
        ("Spectral", read_parquet(args.rich_fourier, "fourier_rich")),
    ]

    feature_names: list[str] = []
    families: list[str] = []
    for family, piece in pieces:
        before = len(frame)
        frame = frame.merge(piece, on="stem", how="inner", validate="one_to_one")
        columns = [column for column in piece.columns if column != "stem"]
        feature_names.extend(columns)
        families.extend([family] * len(columns))
        if len(frame) != before:
            log(f"{family}: cohort changed {before:,} -> {len(frame):,}")

    counts = frame["style"].value_counts()
    frame = frame[frame["style"].isin(counts[counts >= N_FOLDS].index)].sort_values("stem").reset_index(drop=True)
    return frame, feature_names, np.asarray(families, dtype=object)


def structure_importance(model, train_values: np.ndarray) -> np.ndarray:
    standardized = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(train_values)
    ).astype(np.float64)
    scores = model.named_steps["lda"].transform(standardized)
    scores -= scores.mean(axis=0)
    score_std = scores.std(axis=0)
    valid = score_std > 1e-12
    scores[:, valid] /= score_std[valid]
    scores[:, ~valid] = 0.0
    correlations = (standardized.T @ scores) / len(standardized)
    weights = model.named_steps["lda"].explained_variance_ratio_.astype(np.float64)
    weights = weights[: correlations.shape[1]]
    weights /= weights.sum()
    return np.sqrt(np.sum(correlations ** 2 * weights[None, :], axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rich_feature_importance_lda_structure")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frame, feature_names, families = load_rich_frame(args)
    values = frame[feature_names].to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
        log(f"imputed {int(bad.sum()):,} non-finite values")
    std = values.std(axis=0)
    keep = std > 1e-9
    if (~keep).any():
        log(f"dropping {int((~keep).sum())} constant features")
    values = values[:, keep]
    feature_names = np.asarray(feature_names, dtype=object)[keep]
    families = families[keep]

    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    folds_path = args.output_dir / "folds.csv"
    rows = [] if args.restart or not folds_path.exists() else pd.read_csv(folds_path).to_dict("records")
    completed = {int(row["fold"]) for row in rows}
    log(f"{len(frame):,} images x {values.shape[1]} rich nonconstant features; {n_classes} classes")

    for fold, (train, _test) in enumerate(splits, 1):
        if fold in completed:
            log(f"fold={fold}/{N_FOLDS} already complete")
            continue
        model = make_lda(n_classes, values.shape[1])
        model.fit(values[train], labels[train])
        importance = structure_importance(model, values[train])
        normalized = importance / importance.sum()
        for index, feature in enumerate(feature_names):
            rows.append({
                "fold": fold,
                "feature": feature,
                "importance": float(importance[index]),
                "importance_normalized": float(normalized[index]),
            })
        pd.DataFrame(rows).to_csv(folds_path, index=False)
        completed.add(fold)
        log(f"fold={fold}/{N_FOLDS} complete")

    folds = pd.DataFrame(rows)
    summary = folds.groupby("feature", as_index=False).agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_normalized_mean=("importance_normalized", "mean"),
        importance_normalized_std=("importance_normalized", "std"),
    )
    meta = pd.DataFrame({
        "feature": feature_names,
        "display_name": feature_names,
        "group": families,
        "family": families,
        "subgroup": families,
    })
    summary = summary.merge(meta, on="feature", how="left", validate="one_to_one")
    summary = summary.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.8f")
    family_summary = summary.groupby("group", as_index=False).agg(
        features=("feature", "count"),
        importance_pp_sum=("importance_normalized_mean", lambda s: 100 * float(s.sum())),
        importance_pp_mean=("importance_normalized_mean", lambda s: 100 * float(s.mean())),
        top_feature=("feature", "first"),
    )
    family_summary["group"] = pd.Categorical(family_summary["group"], FAMILY_ORDER, ordered=True)
    family_summary = family_summary.sort_values("group")
    family_summary.to_csv(args.output_dir / "family_summary.csv", index=False, float_format="%.8f")
    report = {
        "method": "LDA structure coefficients weighted by discriminant explained variance",
        "features": int(values.shape[1]),
        "images": int(len(frame)),
        "elapsed_seconds": time.perf_counter() - started,
        "families": family_summary.astype({"group": str}).to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"saved: {args.output_dir}")
    log(family_summary.astype({'group': str}).to_string(index=False))


if __name__ == "__main__":
    main()
