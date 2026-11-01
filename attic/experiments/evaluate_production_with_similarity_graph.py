"""Production MLP with the inductive visual similarity-graph branch.

The production baseline is the current hand-designed configuration with Text:
7 global factors, L(G_felz), and native Descriptor quadrants.  The added
Similarity graph branch is built without Text, from the full visual-only
1576-dimensional space, and uses the prior-corrected inductive scores.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_felzenszwalb_line_graph import load_felz_line  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_inductive import inductive_node_feature_scores  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_prior import (  # noqa: E402
    VISUAL_FULL_MASK,
    prior_correct,
    scale_blocks,
)
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks  # noqa: E402
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402


torch.set_num_threads(int(THREADS_PER_WORKER))
OUTPUT_DIR = ROOT / "outputs/production_similarity_graph_production_space"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_summary(detail: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for classifier in sorted(detail["classifier"].unique()):
        base = detail[
            detail["classifier"].eq(classifier) & detail["condition"].eq("production")
        ].set_index("fold")["bacc"]
        for condition in sorted(
            c
            for c in detail.loc[detail["classifier"].eq(classifier), "condition"].unique()
            if c != "production"
        ):
            paired = detail[
                detail["classifier"].eq(classifier) & detail["condition"].eq(condition)
            ].set_index("fold")["bacc"]
            common = base.index.intersection(paired.index)
            delta = paired.loc[common] - base.loc[common]
            if len(delta) > 1:
                statistic, p_value = ttest_rel(paired.loc[common], base.loc[common])
            else:
                statistic, p_value = float("nan"), float("nan")
            rows.append(
                {
                    "classifier": classifier,
                    "condition": condition,
                    "folds": len(common),
                    "base_bacc_mean": 100 * float(base.loc[common].mean()),
                    "condition_bacc_mean": 100 * float(paired.loc[common].mean()),
                    "delta_bacc_pp": 100 * float(delta.mean()),
                    "delta_std_pp": 100 * float(delta.std()),
                    "t_statistic": float(statistic),
                    "p_paired": float(p_value),
                    "positive_folds": int((delta > 0).sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.6f")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument(
        "--felz-line-features",
        type=Path,
        default=ROOT / "results/line_graph_felz_full_features.parquet",
    )
    parser.add_argument(
        "--felz-features",
        type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--classifiers", nargs="+", default=["mlp"], choices=["mlp", "lda"])
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, _quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)
    visual_core, visual_core_names = selected_blocks(BASELINE_MASK & ~64, group_blocks, fixed_extra)
    production_visual_blocks = visual_core + [felzline_block] + descriptor_quadrants
    production_visual_names = visual_core_names + ["L(G_felz)", "Descriptors top_left", "Descriptors top_right", "Descriptors bottom_left", "Descriptors bottom_right"]

    log(
        f"production + similarity graph: {len(frame):,} images, {n_classes} classes; "
        f"graph space={sum(block.shape[1] for block in production_visual_blocks)} production visual dims, "
        f"graph blocks={production_visual_names}"
    )

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(row["condition"]), str(row["classifier"]), int(row["fold"])) for row in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue

        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        core, _core_names = selected_blocks(BASELINE_MASK, group_blocks, {**fixed_extra, "Text": text_block})
        production_blocks = core + [felzline_block] + descriptor_quadrants

        graph_scaled = scale_blocks(production_visual_blocks, train)
        started = time.perf_counter()
        oof_fit, target_scores = inductive_node_feature_scores(
            graph_scaled[train],
            labels[train],
            graph_scaled[test],
            n_classes,
            args.n_neighbors,
            args.alpha,
            args.max_iter,
            SEED + fold,
        )
        graph_feature = np.zeros((len(frame), n_classes), dtype=np.float32)
        graph_feature[train] = oof_fit
        graph_feature[test] = target_scores
        graph_feature = prior_correct(graph_feature, labels[train], n_classes)
        graph_elapsed = time.perf_counter() - started
        log(f"fold={fold}/{N_FOLDS} inductive prior-corrected graph calculated in {graph_elapsed:.1f}s")

        conditions = {
            "production": production_blocks,
            "production + similarity graph": production_blocks + [graph_feature],
        }
        for condition, blocks in conditions.items():
            dimensions = sum(block.shape[1] for block in blocks)
            for classifier in args.classifiers:
                if (condition, classifier, fold) in done:
                    continue
                started = time.perf_counter()
                if classifier == "mlp":
                    metrics = evaluate_mlp(blocks, labels, train, test, fold, args.epochs, args.patience)
                else:
                    metrics = evaluate_lda(blocks, labels, train, test)
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "condition": condition,
                        "classifier": classifier,
                        "fold": fold,
                        "dimensions": dimensions,
                        "n_branches": len(blocks),
                        "elapsed_seconds": elapsed + (graph_elapsed if "similarity graph" in condition else 0.0),
                        **{key: metrics[key] for key in ("bacc", "accuracy", "macro_f1")},
                    }
                )
                done.add((condition, classifier, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(
                    f"fold={fold}/{N_FOLDS} {condition:<29} | {classifier:<3} | "
                    f"{dimensions:4d}d {len(blocks):2d}br | BAcc={100 * metrics['bacc']:.2f}%"
                )

    detail = pd.DataFrame(rows)
    summary = write_summary(detail, args.output_dir)
    log(f"saved: {args.output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
