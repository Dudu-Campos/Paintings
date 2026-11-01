"""Graph-derived feature, not a replacement classifier: the per-class soft
label distribution from a k-NN LabelSpreading graph over Descriptors
similarity, concatenated as 27 extra columns onto Descriptors itself, then
fed into LDA -- same "does this extra branch help" comparison used
everywhere else in this project.

Leak-free protocol, mirroring `final_tfidf_visual_mlp.fit_text_scores`
exactly (nested inner-fold OOF for train rows, single fit-on-all-train pass
for test rows): a train row's graph feature never comes from a graph that
had that row's own label visible.
"""

from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from final_tfidf_visual_mlp import load_dataset
from fusion_curve_slic import ROOT, SEED, load_npy


N_FOLDS = 5
INNER_FOLDS = 5
SAMPLE_PATH = ROOT / "results/fourier_12k_canonical_sample.csv"
OUTPUT_DIR = ROOT / "outputs/label_spreading_feature"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def spread(
    labeled_X: np.ndarray, labeled_y: np.ndarray, unlabeled_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int,
) -> np.ndarray:
    """Fit LabelSpreading with `labeled_y` visible and `unlabeled_X` masked;
    return the unlabeled rows' soft class distributions, in global class order."""
    combined_X = np.vstack([labeled_X, unlabeled_X])
    combined_y = np.concatenate([labeled_y, -np.ones(len(unlabeled_X), dtype=int)])
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    model.fit(combined_X, combined_y)
    if len(model.classes_) != n_classes:
        raise ValueError(
            f"LabelSpreading only saw {len(model.classes_)}/{n_classes} classes among labeled rows"
        )
    return model.label_distributions_[len(labeled_X):].astype(np.float32)


def graph_feature_scores(
    fit_X: np.ndarray, fit_y: np.ndarray, target_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested-OOF graph feature for fit rows; single fit-on-all-fit pass for target rows."""
    oof = np.zeros((len(fit_X), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        oof[inner_val] = spread(
            fit_X[inner_train], fit_y[inner_train], fit_X[inner_val],
            n_classes, n_neighbors, alpha, max_iter,
        )
    target_scores = spread(fit_X, fit_y, target_X, n_classes, n_neighbors, alpha, max_iter)
    return oof, target_scores


def inductive_spread(
    train_X: np.ndarray, train_y: np.ndarray, query_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int,
) -> np.ndarray:
    """Out-of-sample extension: fit LabelSpreading on `train_X`/`train_y`
    alone (fully labeled -- just a smoothed/denoised version of the train
    label field), then score each `query_X` row from its k nearest TRAIN
    neighbors only. Query points never see each other -- this is what makes
    it inductive (safe to compare against per-instance baselines like a CNN,
    and applicable to a single new painting scored in isolation)."""
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    model.fit(train_X, train_y)
    if len(model.classes_) != n_classes:
        raise ValueError(
            f"LabelSpreading only saw {len(model.classes_)}/{n_classes} classes among labeled rows"
        )
    train_distributions = model.label_distributions_
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(train_X)
    _, indices = neighbors.kneighbors(query_X)
    return train_distributions[indices].mean(axis=1).astype(np.float32)


def inductive_graph_feature_scores(
    fit_X: np.ndarray, fit_y: np.ndarray, target_X: np.ndarray,
    n_classes: int, n_neighbors: int, alpha: float, max_iter: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same nested-OOF discipline as `graph_feature_scores`, but every score
    (fit rows via inner folds, target rows via the outer fit set) is
    produced by `inductive_spread` -- so no row's feature ever depends on
    another row from the same held-out slice (inner_val never sees
    inner_val; outer_test never sees outer_test)."""
    oof = np.zeros((len(fit_X), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), dtype=np.uint8), fit_y):
        oof[inner_val] = inductive_spread(
            fit_X[inner_train], fit_y[inner_train], fit_X[inner_val],
            n_classes, n_neighbors, alpha, max_iter,
        )
    target_scores = inductive_spread(fit_X, fit_y, target_X, n_classes, n_neighbors, alpha, max_iter)
    return oof, target_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--full", action="store_true",
        help="Use the canonical 79,207-image WikiArt cohort instead of the 12k pilot sample.",
    )
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.full:
        frame, groups = load_dataset()
        descriptor_columns = groups["Descriptors"]
        frame = frame.sort_values("stem").reset_index(drop=True)
        encoder = LabelEncoder()
        y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    else:
        sample = pd.read_csv(SAMPLE_PATH)[["stem", "style"]].drop_duplicates("stem")
        descriptors, descriptor_columns = load_npy(
            "desc_full_features.npy", "desc_full_stems.npy", "descriptor"
        )
        frame = sample.merge(descriptors, on="stem", how="inner", validate="one_to_one")
        if len(frame) != len(sample):
            raise ValueError(f"lost images merging descriptors: {len(frame):,}/{len(sample):,}")
        frame = frame.sort_values("stem").reset_index(drop=True)
        encoder = LabelEncoder()
        y = encoder.fit_transform(frame["style"].to_numpy(dtype=str))

    n_classes = len(encoder.classes_)
    X = frame[descriptor_columns].to_numpy(dtype=np.float64)
    log(f"Feature de grafo: {len(frame):,} imagens, {n_classes} estilos, k={args.n_neighbors}, alpha={args.alpha}")

    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y), dtype=np.uint8), y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        scaler = StandardScaler().fit(X[outer_train])
        X_scaled = scaler.transform(X)

        started = time.perf_counter()
        oof_fit, target_scores = graph_feature_scores(
            X_scaled[outer_train], y[outer_train], X_scaled[outer_test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
        )
        elapsed_graph = time.perf_counter() - started

        started = time.perf_counter()
        oof_fit_ind, target_scores_ind = inductive_graph_feature_scores(
            X_scaled[outer_train], y[outer_train], X_scaled[outer_test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
        )
        elapsed_graph_ind = time.perf_counter() - started

        for condition, extra_fit, extra_test, extra_elapsed in (
            ("1: Descriptors", None, None, 0.0),
            ("2: Descriptors + graph-feature (transdutivo)", oof_fit, target_scores, elapsed_graph),
            ("3: Descriptors + graph-feature (indutivo)", oof_fit_ind, target_scores_ind, elapsed_graph_ind),
        ):
            train_X = X_scaled[outer_train] if extra_fit is None else np.hstack([X_scaled[outer_train], extra_fit])
            test_X = X_scaled[outer_test] if extra_test is None else np.hstack([X_scaled[outer_test], extra_test])
            started = time.perf_counter()
            lda = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            lda.fit(train_X, y[outer_train])
            prediction = lda.predict(test_X)
            elapsed = time.perf_counter() - started
            metrics = prediction_metrics(y[outer_test], prediction)
            rows.append({
                "fold": fold, "condition": condition, "dimensions": train_X.shape[1],
                "elapsed_seconds": elapsed + extra_elapsed,
                **metrics,
            })
            log(
                f"fold={fold}/{N_FOLDS} {condition:<32} dims={train_X.shape[1]:4d} "
                f"BAcc={metrics['bacc']:.4f}"
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)

    baseline = detail[detail.condition.eq("1: Descriptors")].set_index("fold")["bacc"]
    comparison_rows = []
    for label, condition in (
        ("transdutivo", "2: Descriptors + graph-feature (transdutivo)"),
        ("indutivo", "3: Descriptors + graph-feature (indutivo)"),
    ):
        paired = detail[detail.condition.eq(condition)].set_index("fold")["bacc"]
        common_folds = baseline.index.intersection(paired.index)
        delta = paired.loc[common_folds] - baseline.loc[common_folds]
        if len(delta) > 1:
            _, p_value = ttest_rel(paired.loc[common_folds], baseline.loc[common_folds])
        else:
            p_value = float("nan")
        comparison_rows.append({
            "variant": label, "n_neighbors": args.n_neighbors, "alpha": args.alpha,
            "folds": len(common_folds),
            "delta_bacc_pp": 100 * float(delta.mean()), "delta_std_pp": 100 * float(delta.std()),
            "p_paired": float(p_value), "positive_folds": int((delta > 0).sum()),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False, float_format="%.6f")
    log(f"Saved to {args.output_dir}")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
