"""Gradient boosting (XGBoost e/ou LightGBM) contra o MLP na melhor
configuracao conhecida (1.091d, 7 modalidades + so quadrantes de
Descriptors no Multiscale -- ver `evaluate_multiscale_family_factorial.py`).

Motivacao: as features hand-designed sao tabulares (nao pixels crus), regime
classico onde gradient boosting costuma igualar ou bater MLPs, e treina
nativamente em CPU -- roda direto na margarita sem depender da GPU do
mojito. Ao contrario do LDA (StandardScaler por bloco obrigatorio) e do MLP
(idem), arvores sao invariantes a escala: os blocos entram concatenados
sem normalizacao.

Correcao de desbalanceamento (139,9x entre a maior e a menor classe):
`sample_weight` balanceado (`compute_sample_weight("balanced", y)`), mesmo
principio do `class_weight="balanced"` usado no bloco Text e dos priors
uniformes do LDA -- sem isso a BAcc penaliza pesado as classes raras.

Early stopping: mesma logica do MLP (`evaluate_mlp`): 10% do treino vira
validacao (stratify, mesmo seed), early stopping na loss de validacao.

Saida: outputs/gradient_boosting/
    fold_results.csv   BAcc por engine, condicao e fold
    summary.csv        media/std por engine
"""
from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS_PER_WORKER)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


OUTPUT_DIR = ROOT / "outputs/gradient_boosting"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def fit_xgboost(
    train_x, train_y, val_x, val_y, test_x, n_classes: int, seed: int,
    n_estimators: int, early_stopping_rounds: int,
) -> tuple[np.ndarray, int]:
    from xgboost import XGBClassifier

    weights = compute_sample_weight("balanced", train_y)
    model = XGBClassifier(
        objective="multi:softprob", num_class=n_classes,
        n_estimators=n_estimators, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", multi_strategy="multi_output_tree",
        n_jobs=int(THREADS_PER_WORKER),
        random_state=seed, early_stopping_rounds=early_stopping_rounds,
        eval_metric="mlogloss",
    )
    model.fit(
        train_x, train_y, sample_weight=weights,
        eval_set=[(val_x, val_y)], verbose=False,
    )
    prediction = model.predict(test_x)
    return prediction, int(model.best_iteration)


def fit_lightgbm(
    train_x, train_y, val_x, val_y, test_x, n_classes: int, seed: int,
    n_estimators: int, early_stopping_rounds: int,
) -> tuple[np.ndarray, int]:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    model = LGBMClassifier(
        objective="multiclass", num_class=n_classes,
        n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", n_jobs=int(THREADS_PER_WORKER),
        random_state=seed, verbosity=-1,
    )
    model.fit(
        train_x, train_y, eval_set=[(val_x, val_y)],
        callbacks=[early_stopping(early_stopping_rounds, verbose=False), log_evaluation(0)],
    )
    prediction = model.predict(test_x)
    return prediction, int(model.best_iteration_)


ENGINES = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path,
                        default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path,
                        default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-features", type=Path,
                        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path,
                        default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--engines", nargs="+", default=["xgboost", "lightgbm"],
                        choices=list(ENGINES))
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    descriptor_quadrants, quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)
    log(f"{len(frame):,} imagens, {n_classes} estilos, engines={args.engines}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["engine"]), int(r["fold"])) for r in rows}

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, base_names = selected_blocks(BASELINE_MASK, group_blocks, extra)
        best_blocks = base_blocks + descriptor_quadrants
        X = np.hstack(best_blocks).astype(np.float32, copy=False)
        dims = X.shape[1]

        fit_idx, val_idx = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
        )

        for engine in args.engines:
            if (engine, fold) in completed:
                continue
            started = time.perf_counter()
            prediction, best_iteration = ENGINES[engine](
                X[fit_idx], labels[fit_idx], X[val_idx], labels[val_idx], X[outer_test],
                n_classes, SEED + fold, args.n_estimators, args.early_stopping_rounds,
            )
            elapsed = time.perf_counter() - started
            metrics = prediction_metrics(labels[outer_test], prediction)
            rows.append({
                "fold": fold, "engine": engine, "dimensions": dims,
                "best_iteration": best_iteration, "elapsed_seconds": elapsed, **metrics,
            })
            completed.add((engine, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            log(f"fold={fold}/{N_FOLDS} {engine:<9} dims={dims:4d} "
                f"iter={best_iteration:5d} BAcc={metrics['bacc']:.4f} seconds={elapsed:.0f}")

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("engine")["bacc"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "bacc_mean", "std": "bacc_std", "count": "folds"})
        .reset_index()
    )
    summary["bacc_mean"] *= 100
    summary["bacc_std"] *= 100
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.4f")
    log("\n=== Resumo (MLP referencia: 53.63 +/- 0.82) ===")
    log(summary.to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
