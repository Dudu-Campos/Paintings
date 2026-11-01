"""Rich feature importance/redundancy scatter with external text scores."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_rich_lda_structure_importance import load_rich_frame, structure_importance  # noqa: E402
from analyze_visual_feature_importance_lda_xgb import make_lda  # noqa: E402
from fusion_curve_slic import N_FOLDS, SEED  # noqa: E402


COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
    "Text": "#e94149",
}
GROUP_ORDER = tuple(COLOURS)
TEXT = "#202124"
MUTED = "#65676b"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_text_scores(stems_path: Path, decisions_path: Path, classes_path: Path) -> pd.DataFrame:
    stems = np.load(stems_path, allow_pickle=True).astype(str)
    decisions = np.load(decisions_path).astype(np.float32)
    classes = np.load(classes_path, allow_pickle=True).astype(str)
    if len(stems) != len(decisions):
        raise ValueError("Text stem/score length mismatch")
    columns = [f"text_score_{name}" for name in classes]
    frame = pd.DataFrame(decisions, columns=columns)
    frame.insert(0, "stem", stems)
    if frame["stem"].duplicated().any():
        raise ValueError("Duplicate stems in text score file")
    return frame


def impute_and_standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
    means = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = values.std(axis=0, dtype=np.float64).astype(np.float32)
    keep = stds > 1e-9
    return ((values[:, keep] - means[keep]) / stds[keep]).astype(np.float32), keep


def redundancy_metrics(values: np.ndarray) -> pd.DataFrame:
    standardized, keep = impute_and_standardize(values)
    log(f"computing r2 matrix for {standardized.shape[1]:,} nonconstant features")
    correlation = (standardized.T @ standardized) / np.float32(len(standardized))
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = np.square(correlation, dtype=np.float32)
    np.fill_diagonal(r2, np.nan)

    filled = np.nan_to_num(r2, nan=-np.inf)
    top10 = np.partition(filled, -10, axis=1)[:, -10:]
    metrics = pd.DataFrame(
        {
            "p95_r2": np.nanquantile(r2, 0.95, axis=1),
            "top10_mean_r2": top10.mean(axis=1),
            "neighbors_ge_0_8": np.sum(r2 >= 0.8, axis=1),
        }
    )
    return metrics, keep


def plot_scatter(data: pd.DataFrame, output: Path, alpha: float, size: float) -> None:
    y_cut = float(data["importance_pp"].quantile(0.75))
    y_max = float(data["importance_pp"].quantile(0.997) * 1.15)
    y_max = max(y_max, float(data["importance_pp"].max() * 1.08))
    x_cut = float(data["top10_mean_r2"].quantile(0.75))
    x_max = float(data["top10_mean_r2"].quantile(0.995) * 1.10)
    x_max = min(1.0, max(x_max, float(data["top10_mean_r2"].max() * 1.04)))

    figure, axis = plt.subplots(figsize=(7.35, 5.75))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    handles = []
    for group in GROUP_ORDER:
        rows = data[data["group"].eq(group)]
        if rows.empty:
            continue
        colour = COLOURS[group]
        axis.scatter(
            rows["top10_mean_r2"],
            rows["importance_pp"],
            s=size if group != "Text" else size * 1.4,
            color=colour,
            alpha=0.78 if group == "Text" else alpha,
            linewidth=0,
            zorder=4 if group == "Text" else 3,
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=colour,
                label=f"{group}",
                markersize=7,
            )
        )

    axis.axvline(x_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    axis.axhline(y_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, y_max)
    axis.set_xlabel("Mean r² of the 10 nearest neighbors", fontsize=13)
    axis.set_ylabel("Normalized supervised importance (%)", fontsize=13)
    axis.set_title(
        "Feature importance vs. local redundancy",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
    )
    axis.tick_params(axis="both", labelsize=11.5)
    axis.legend(handles=handles, frameon=False, fontsize=10.4, loc="upper left")
    figure.tight_layout(rect=(0, 0.015, 1, 0.97))
    figure.savefig(output, dpi=230, facecolor="white")
    figure.savefig(output.with_name(output.stem + "_paper.png"), dpi=230, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_importance_redundancy_top10_with_text_en.png")
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--size", type=float, default=14.0)
    args = parser.parse_args()

    frame, visual_features, visual_groups = load_rich_frame(args)
    text = load_text_scores(args.text_stems, args.text_decisions, args.text_classes)
    before = len(frame)
    frame = frame.merge(text, on="stem", how="inner", validate="one_to_one")
    if len(frame) != before:
        log(f"text merge changed cohort {before:,} -> {len(frame):,}")

    text_features = [column for column in text.columns if column != "stem"]
    feature_names = np.asarray([*visual_features, *text_features], dtype=object)
    groups = np.asarray([*visual_groups.tolist(), *["Text"] * len(text_features)], dtype=object)
    values = frame[feature_names].to_numpy(dtype=np.float32)

    standardized, keep = impute_and_standardize(values)
    feature_names = feature_names[keep]
    groups = groups[keep]
    values = values[:, keep]
    log(f"{len(frame):,} images x {values.shape[1]:,} features including Text")

    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(dtype=str))
    splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels)
    rows = []
    for fold, (train, _test) in enumerate(splits, 1):
        model = make_lda(len(np.unique(labels)), values.shape[1])
        model.fit(values[train], labels[train])
        importance = structure_importance(model, values[train])
        normalized = importance / importance.sum()
        rows.extend(
            {
                "fold": fold,
                "feature": feature,
                "importance": float(importance[index]),
                "importance_normalized": float(normalized[index]),
            }
            for index, feature in enumerate(feature_names)
        )
        log(f"fold={fold}/{N_FOLDS} importance complete")

    importance = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_normalized_mean=("importance_normalized", "mean"),
            importance_normalized_std=("importance_normalized", "std"),
        )
    )
    metrics, keep_for_metrics = redundancy_metrics(values)
    if not np.all(keep_for_metrics):
        raise RuntimeError("Unexpected constants after initial filtering")
    data = pd.DataFrame({"feature": feature_names, "group": groups}).join(metrics)
    data = data.merge(importance, on="feature", how="inner", validate="one_to_one")
    data["importance_pp"] = 100 * data["importance_normalized_mean"]
    data["group"] = pd.Categorical(data["group"], GROUP_ORDER, ordered=True)
    data = data.sort_values(["group", "importance_pp"], ascending=[True, False]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.8f")
    plot_scatter(data, args.output, args.alpha, args.size)
    log(f"saved: {args.output}")
    log(f"saved: {args.output.with_name(args.output.stem + '_paper.png')}")
    log(
        "top Text features:\n"
        + data[data["group"].astype(str).eq("Text")]
        .sort_values("importance_pp", ascending=False)
        [["feature", "importance_pp", "top10_mean_r2"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
