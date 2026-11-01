"""Curva BAcc(k) do LDA por top-k features, banco de candidatos agora
incluindo Text e o grafo de similaridade global (alem das 4 familias
visuais ricas). Ranking recalculado a cada fold, so no treino -- mesmo
protocolo de evaluate_lda_topk_curve_by_subset.py / evaluate_p95_redundancy_filter.py.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_rich_lda_structure_importance import load_rich_frame, structure_importance  # noqa: E402
from analyze_visual_feature_importance_lda_xgb import make_lda  # noqa: E402
from fusion_curve_slic import N_FOLDS, SEED  # noqa: E402
from plot_rich_importance_redundancy_top10_with_text_en import (  # noqa: E402
    impute_and_standardize, load_text_scores,
)
from plot_rich_importance_redundancy_top10_with_text_and_graph_en import (  # noqa: E402
    load_global_graph_scores, COLOURS,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def checkpoints(n_features: int) -> list[int]:
    base = {10, 25, 50, 100, 200, 400, 700, 1000, 1400, n_features}
    return sorted(v for v in base if 1 <= v <= n_features)


def main() -> None:
    class Args:
        meta = ROOT / "data/artemis/wikiart_meta.csv"
        rich_dir = ROOT / "outputs/wikiart_rich_features"
        rich_fourier = ROOT / "results/fourier_rich_features.parquet"
        slic = ROOT / "outputs/wikiart_slic_features/combined_features.parquet"
        felzenszwalb = ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet"
        line_graph = ROOT / "results/line_graph_full_features.parquet"
        line_graph_felz = ROOT / "results/line_graph_felz_full_features.parquet"
        text_stems = ROOT / "results/tfidf_caption_style_stems.npy"
        text_decisions = ROOT / "results/tfidf_caption_style_oof_decisions.npy"
        text_classes = ROOT / "results/tfidf_caption_style_classes.npy"
        global_cache = ROOT / "results/rich_feature_redundancy_r2_with_global_graph/global_similarity_graph_extended_features.parquet"

    args = Args()
    frame, visual_features, visual_groups = load_rich_frame(args)
    text = load_text_scores(args.text_stems, args.text_decisions, args.text_classes)
    global_graph = load_global_graph_scores(args.global_cache)

    before = len(frame)
    frame = frame.merge(text, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(global_graph, on="stem", how="inner", validate="one_to_one")
    if len(frame) != before:
        log(f"merge changed cohort {before:,} -> {len(frame):,}")

    text_features = [c for c in text.columns if c != "stem"]
    graph_features = [c for c in global_graph.columns if c != "stem"]
    feature_names = np.asarray([*visual_features, *text_features, *graph_features], dtype=object)
    groups = np.asarray([*visual_groups.tolist(),
                        *["Text"] * len(text_features),
                        *["Global similarity graph"] * len(graph_features)], dtype=object)
    values = frame[feature_names].to_numpy(dtype=np.float32)

    standardized, keep = impute_and_standardize(values)
    feature_names, groups = feature_names[keep], groups[keep]
    values = values[:, keep]
    log(f"{len(frame):,} images x {values.shape[1]:,} features including Text + global graph")

    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(labels)), labels))
    ks = checkpoints(values.shape[1])
    log(f"checkpoints: {ks}")

    rows = []
    for fold, (train, test) in enumerate(splits, 1):
        model = make_lda(n_classes, values.shape[1])
        model.fit(values[train], labels[train])
        importance = structure_importance(model, values[train])
        order = np.argsort(-importance)
        for k in ks:
            selected = order[:k]
            lda = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            lda.fit(values[train][:, selected], labels[train])
            pred = lda.predict(values[test][:, selected])
            bacc = balanced_accuracy_score(labels[test], pred)
            rows.append({"fold": fold, "k": k, "bacc": bacc})
        log(f"fold={fold}/{N_FOLDS} done")

    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/topk_curve_with_text_and_graph.csv", index=False)
    summary = result.groupby("k")["bacc"].agg(["mean", "std"])
    print()
    print((summary * 100).round(2))

    figure, axis = plt.subplots(figsize=(8.0, 5.6))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.errorbar(summary.index, summary["mean"] * 100, yerr=summary["std"] * 100,
                  marker="o", markersize=5, color="#2a78d6", capsize=3, linewidth=1.4)
    axis.set_xscale("log")
    axis.set_xlabel("k (top-k features by LDA structure coefficient)", fontsize=12)
    axis.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    axis.set_title("LDA top-k curve, visual + Text + global similarity graph",
                   fontsize=13.5, fontweight="bold")
    figure.tight_layout()
    figure.savefig(ROOT / "results/topk_curve_with_text_and_graph.png", dpi=220, facecolor="white")
    log(f"saved: results/topk_curve_with_text_and_graph.png")


if __name__ == "__main__":
    main()
