"""Tabela de robustez de classificador sobre a melhor configuração (apêndice).

A afirmação central do paper -- features hand-designed batem uma CNN treinada
do zero (42,34%) -- não pode depender de ter escolhido um classificador
conveniente. Esta tabela mostra o mesmo vetor de 1.154d sob quatro
classificadores de famílias diferentes.

Benchmark de um fold (63.365 treino, 1.154 features, 27 classes):
    MLP           54,4%   ~30s   (GPU)
    LDA           45,2%    ~5s
    XGBoost       44,2%   219s
    RandomForest  34,5%    34s

LDA, XGBoost e RF são CPU pura -- rodam na `margarita` (128 threads) sem
disputar a GPU do `mojito`:

    CLF_THREADS=64 python experiments/evaluate_classifier_table.py \\
        --classifiers lda xgb rf

O MLP roda no mojito depois, com `--classifiers mlp`, e os resultados se
juntam no mesmo CSV (retomável, chaveado por classificador e fold).

Saída: outputs/classifier_table/
    fold_results.csv   BAcc, acurácia e macro F1 por classificador e fold
    summary.csv        média e desvio entre folds
"""
from __future__ import annotations

import os

THREADS = os.environ.get("CLF_THREADS", "16")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_felzenszwalb_line_graph import load_felz_line
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def metrics_of(truth, prediction) -> dict:
    return {"bacc": balanced_accuracy_score(truth, prediction),
            "accuracy": accuracy_score(truth, prediction),
            "macro_f1": f1_score(truth, prediction, average="macro")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path,
                        default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path,
                        default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-line-features", type=Path,
                        default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--felz-features", type=Path,
                        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path,
                        default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/classifier_table")
    parser.add_argument("--classifiers", nargs="+", default=["lda", "xgb", "rf"],
                        choices=["lda", "xgb", "rf", "mlp"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, _ = family_quadrant_blocks("Descriptors", group_blocks)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    log(f"{len(frame):,} imagens, {n_classes} estilos, threads={THREADS}")

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        if all((c, fold) in completed for c in args.classifiers):
            continue
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        blocks, names = selected_blocks(BASELINE_MASK, group_blocks,
                                        {**fixed_extra, "Text": text_block})
        blocks = blocks + descriptor_quadrants + [felzline_block]

        # Padronização por bloco, como no resto do pipeline.
        parts_train, parts_test = [], []
        for block in blocks:
            imputer = SimpleImputer(strategy="median").fit(block[train])
            scaler = StandardScaler().fit(imputer.transform(block[train]))
            parts_train.append(scaler.transform(imputer.transform(block[train])))
            parts_test.append(scaler.transform(imputer.transform(block[test])))
        train_x = np.hstack(parts_train).astype(np.float32)
        test_x = np.hstack(parts_test).astype(np.float32)

        # Pesos de classe: o desbalanceamento é de 139,9x e a métrica é
        # balanceada, então todo classificador recebe a mesma correção.
        counts = np.bincount(labels[train], minlength=n_classes).astype(np.float64)
        sample_weight = (1.0 / counts)[labels[train]]
        sample_weight /= sample_weight.mean()

        for classifier in args.classifiers:
            if (classifier, fold) in completed:
                continue
            started = time.perf_counter()
            if classifier == "lda":
                model = LinearDiscriminantAnalysis(
                    n_components=min(n_classes - 1, train_x.shape[1]),
                    priors=np.ones(n_classes) / n_classes)
                model.fit(train_x, labels[train])
                result = metrics_of(labels[test], model.predict(test_x))
            elif classifier == "rf":
                model = RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=2, n_jobs=int(THREADS),
                    class_weight="balanced_subsample", random_state=SEED)
                model.fit(train_x, labels[train])
                result = metrics_of(labels[test], model.predict(test_x))
            elif classifier == "xgb":
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                    objective="multi:softmax", num_class=n_classes,
                    n_jobs=int(THREADS), verbosity=0, random_state=SEED)
                model.fit(train_x, labels[train], sample_weight=sample_weight)
                result = metrics_of(labels[test], model.predict(test_x))
            else:
                from evaluate_full_factorial_ablation import evaluate_mlp
                result = evaluate_mlp(blocks, labels, train, test, fold,
                                      args.epochs, args.patience)
                result = {k: result[k] for k in ("bacc", "accuracy", "macro_f1")}
            elapsed = time.perf_counter() - started
            rows.append({"classifier": classifier, "fold": fold,
                         "dimensions": train_x.shape[1], "n_branches": len(blocks),
                         "elapsed_seconds": elapsed, **result})
            completed.add((classifier, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            log(f"  fold {fold} | {classifier:<3} | BAcc={100 * result['bacc']:.2f}% "
                f"| F1={100 * result['macro_f1']:.2f}% | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby("classifier", as_index=False)
               .agg(folds=("fold", "nunique"),
                    bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
                    accuracy_mean=("accuracy", "mean"),
                    macro_f1_mean=("macro_f1", "mean"),
                    seconds_mean=("elapsed_seconds", "mean"))
               .sort_values("bacc_mean", ascending=False))
    for column in ("bacc_mean", "bacc_std", "accuracy_mean", "macro_f1_mean"):
        summary[column] *= 100
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Tabela de classificadores (1.154d) ===")
    log(summary.round(2).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
