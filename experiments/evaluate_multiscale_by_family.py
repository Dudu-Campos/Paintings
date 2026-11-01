"""Fase 0 do plano de avaliação final: separar a contribuição de Multiscale
por família (SLIC/Descriptors/Hilbert) em vez de um único fator combinado.

A redundância global-vs-quadrante já medida é homogênea entre as três
famílias (r² mediano ~0,49-0,58 nas três) -- não aponta uma família
"culpada". Mas redundância não é o mesmo que contribuição de performance,
então este script testa isso diretamente: 5 condições pareadas (LDA+MLP,
5 folds), não um factorial completo.

  1. Baseline: pipeline sem Multiscale (mask=127, 571d).
  2. Baseline + Multiscale-SLIC only (+4x71=284d).
  3. Baseline + Multiscale-Descriptors only (+4x130=520d).
  4. Baseline + Multiscale-Hilbert only (+4x57=228d).
  5. Baseline + as três (equivalente a mask=255, 1603d) -- conferência.

Verificação de aditividade: delta(2)+delta(3)+delta(4) ~= delta(5).
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
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(int(THREADS_PER_WORKER))

from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, REGIONS, ROOT, SEED, load_dataset


BASELINE_MASK = 127  # 7 fatores originais (SLIC..Text), sem Multiscale
FAMILIES = ("SLIC", "Descriptors", "Hilbert")
CLASSIFIERS = ("lda", "mlp")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def family_quadrant_blocks(
    family: str, group_blocks: dict[str, np.ndarray]
) -> tuple[list[np.ndarray], list[str]]:
    names = [f"{family} ({region})" for region in REGIONS]
    return [group_blocks[name] for name in names], names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features"
    )
    parser.add_argument(
        "--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet"
    )
    parser.add_argument(
        "--felz-features", type=Path,
        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet",
    )
    parser.add_argument(
        "--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet"
    )
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/multiscale_by_family")
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=CLASSIFIERS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features
    )
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {
        name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()
    }
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[fourier_columns].to_numpy(dtype=np.float32),
    }

    splits = list(
        StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels), dtype=np.uint8), labels
        )
    )
    log(f"Multiscale por família: {len(frame):,} imagens, {n_classes} estilos")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {
        (str(row["classifier"]), str(row["condition"]), int(row["fold"])) for row in rows
    }

    fold_data = {
        fold: (outer_train, outer_test) for fold, (outer_train, outer_test) in enumerate(splits, start=1)
    }

    # Classificador-fora, fold-dentro: LDA (rápido) fecha os 5 folds inteiros
    # antes do MLP (lento) começar -- evita perder resultados prontos se o
    # job for interrompido no meio do MLP.
    for classifier in args.classifiers:
        for fold, (outer_train, outer_test) in fold_data.items():
            text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
            extra_blocks = {**fixed_extra_blocks, "Text": text_block}
            baseline_blocks, baseline_names = selected_blocks(BASELINE_MASK, group_blocks, extra_blocks)

            family_blocks = {}
            for family in FAMILIES:
                blocks, names = family_quadrant_blocks(family, group_blocks)
                family_blocks[family] = blocks

            conditions = {"1: baseline (sem Multiscale)": baseline_blocks}
            for family in FAMILIES:
                conditions[f"2-4: + Multiscale-{family}"] = baseline_blocks + family_blocks[family]
            all_three = baseline_blocks + [b for family in FAMILIES for b in family_blocks[family]]
            conditions["5: + Multiscale (todas as 3)"] = all_three

            for condition, condition_blocks in conditions.items():
                if (classifier, condition, fold) in completed:
                    continue
                dimensions = sum(block.shape[1] for block in condition_blocks)
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(condition_blocks, labels, outer_train, outer_test)
                else:
                    metrics = evaluate_mlp(
                        condition_blocks, labels, outer_train, outer_test, fold,
                        args.epochs, args.patience,
                    )
                elapsed = time.perf_counter() - started
                rows.append({
                    "fold": fold, "classifier": classifier, "condition": condition,
                    "dimensions": dimensions, "elapsed_seconds": elapsed, **metrics,
                })
                completed.add((classifier, condition, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(
                    f"fold={fold}/{N_FOLDS} {classifier:<4} {condition:<32} "
                    f"dims={dimensions:4d} BAcc={metrics['bacc']:.4f} seconds={elapsed:.1f}"
                )

    detail = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fold_results.csv", index=False)

    comparisons = []
    for classifier, classifier_rows in detail.groupby("classifier"):
        baseline = classifier_rows[classifier_rows.condition.eq("1: baseline (sem Multiscale)")].set_index("fold")["bacc"]
        for condition, condition_rows in classifier_rows.groupby("condition"):
            if condition == "1: baseline (sem Multiscale)":
                continue
            paired = condition_rows.set_index("fold")["bacc"]
            delta = paired - baseline
            statistic, p_value = ttest_rel(paired, baseline)
            comparisons.append({
                "classifier": classifier, "condition": condition,
                "delta_bacc_pp": 100 * float(delta.mean()), "delta_std_pp": 100 * float(delta.std()),
                "p_paired": float(p_value), "positive_folds": int((delta > 0).sum()),
            })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(args.output_dir / "comparisons.csv", index=False, float_format="%.6f")

    # Checagem de aditividade: delta(SLIC)+delta(Descriptors)+delta(Hilbert) vs delta(todas as 3)
    additivity_rows = []
    for classifier in args.classifiers:
        sub = comparisons[comparisons.classifier.eq(classifier)].set_index("condition")
        isolated_sum = sum(
            sub.loc[f"2-4: + Multiscale-{family}", "delta_bacc_pp"] for family in FAMILIES
        )
        combined = sub.loc["5: + Multiscale (todas as 3)", "delta_bacc_pp"]
        additivity_rows.append({
            "classifier": classifier, "soma_isoladas_pp": isolated_sum,
            "combinada_pp": combined, "diferenca_pp": isolated_sum - combined,
        })
    pd.DataFrame(additivity_rows).to_csv(args.output_dir / "additivity_check.csv", index=False, float_format="%.6f")

    log(f"Saved to {args.output_dir}")
    print(comparisons.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
