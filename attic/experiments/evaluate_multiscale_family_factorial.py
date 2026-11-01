"""Fatorial 2^3 completo sobre as famílias de Multiscale (SLIC/Descriptors/Hilbert).

`evaluate_multiscale_by_family.py` mediu 5 dos 8 subconjuntos possíveis (vazio,
cada família isolada, e as três juntas) e apontou que no MLP só Descriptors
contribui: só-Descriptors (1.091d, 53,63%) bate as-três (1.603d, 53,03%) por
+0,60 ± 0,18pp. Mas os três **pares** nunca foram medidos, então "Descriptors
sozinho é o ótimo" era inferência, não medição.

Este script fecha os 8 subconjuntos com o mesmo baseline (mask=127, 571d, as 7
modalidades originais sem Multiscale), 5 folds pareados, LDA e MLP. Além da
tabela completa, calcula Shapley exato entre as três famílias e o delta pareado
de cada subconjunto contra a configuração canônica (as três).

Saída: outputs/multiscale_family_factorial/
    fold_results.csv    BAcc por subconjunto, classificador e fold
    summary.csv         média/desvio por subconjunto, ordenado
    vs_canonical.csv    delta pareado de cada subconjunto contra as-três
    shapley.csv         Shapley exato por família, por fold e agregado
"""
from __future__ import annotations

import os

THREADS_PER_WORKER = "16"
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, THREADS_PER_WORKER)

import argparse
import math
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
from evaluate_multiscale_by_family import BASELINE_MASK, FAMILIES, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset

CLASSIFIERS = ("lda", "mlp")
FULL_SUBSET = (1 << len(FAMILIES)) - 1


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def subset_name(subset: int) -> str:
    active = [f for i, f in enumerate(FAMILIES) if subset & (1 << i)]
    return " + ".join(active) if active else "(nenhuma)"


def exact_shapley(values: dict[int, float]) -> dict[str, float]:
    """Mesma fórmula da ablação fatorial de 8 fatores."""
    n = len(FAMILIES)
    denominator = math.factorial(n)
    result = {}
    for index, family in enumerate(FAMILIES):
        bit = 1 << index
        contribution = 0.0
        for subset in range(1 << n):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
            contribution += weight * (values[subset | bit] - values[subset])
        result[family] = contribution
    return result


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
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/multiscale_family_factorial")
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=list(CLASSIFIERS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra_blocks = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    log(f"Fatorial 2^3 de Multiscale: {len(frame):,} imagens, {n_classes} estilos, "
        f"{1 << len(FAMILIES)} subconjuntos x {N_FOLDS} folds x "
        f"{len(args.classifiers)} classificadores")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), int(r["subset"]), int(r["fold"])) for r in rows}

    # Classificador-fora, fold-dentro: o LDA fecha os 5 folds antes do MLP
    # começar, então uma interrupção no meio do MLP não custa o LDA inteiro.
    for classifier in args.classifiers:
        for fold, (outer_train, outer_test) in enumerate(splits, start=1):
            if all((classifier, s, fold) in completed for s in range(1 << len(FAMILIES))):
                continue
            text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
            extra_blocks = {**fixed_extra_blocks, "Text": text_block}
            baseline_blocks, _ = selected_blocks(BASELINE_MASK, group_blocks, extra_blocks)
            family_blocks = {f: family_quadrant_blocks(f, group_blocks)[0] for f in FAMILIES}

            for subset in range(1 << len(FAMILIES)):
                if (classifier, subset, fold) in completed:
                    continue
                blocks = list(baseline_blocks)
                for index, family in enumerate(FAMILIES):
                    if subset & (1 << index):
                        blocks += family_blocks[family]
                dimensions = sum(block.shape[1] for block in blocks)
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, outer_train, outer_test)
                else:
                    metrics = evaluate_mlp(blocks, labels, outer_train, outer_test,
                                           fold, args.epochs, args.patience)
                elapsed = time.perf_counter() - started
                rows.append({"fold": fold, "classifier": classifier, "subset": subset,
                             "condition": subset_name(subset), "n_branches": len(blocks),
                             "dimensions": dimensions, "elapsed_seconds": elapsed,
                             **metrics})
                completed.add((classifier, subset, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  fold={fold}/{N_FOLDS} {classifier:<3} [{subset}] "
                    f"{subset_name(subset):<34} dims={dimensions:5d} "
                    f"BAcc={metrics['bacc']:.4f} {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["classifier", "subset", "condition", "dimensions"],
                              as_index=False)
               .agg(bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"))
               .sort_values(["classifier", "bacc_mean"], ascending=[True, False]))
    summary["bacc_mean"] *= 100
    summary["bacc_std"] *= 100
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")

    # Delta pareado de cada subconjunto contra a configuração canônica.
    pivot = detail.pivot_table(index=["classifier", "fold"], columns="subset", values="bacc")
    comparisons = []
    for classifier in detail["classifier"].unique():
        block = pivot.loc[classifier]
        for subset in range(1 << len(FAMILIES)):
            if subset == FULL_SUBSET:
                continue
            delta = 100 * (block[subset] - block[FULL_SUBSET])
            statistic, pvalue = ttest_rel(block[subset], block[FULL_SUBSET])
            comparisons.append({
                "classifier": classifier, "subset": subset,
                "condition": subset_name(subset),
                "delta_vs_canonical_pp": delta.mean(), "delta_std_pp": delta.std(),
                "t_statistic": statistic, "p_paired": pvalue,
                "positive_folds": int((delta > 0).sum()),
            })
    comparisons = pd.DataFrame(comparisons).sort_values(
        ["classifier", "delta_vs_canonical_pp"], ascending=[True, False])
    comparisons.to_csv(args.output_dir / "vs_canonical.csv", index=False, float_format="%.6f")

    shapley_rows = []
    for classifier in detail["classifier"].unique():
        block = pivot.loc[classifier]
        for fold in block.index:
            values = {s: float(block.loc[fold, s]) for s in range(1 << len(FAMILIES))}
            for family, contribution in exact_shapley(values).items():
                shapley_rows.append({"classifier": classifier, "fold": fold,
                                     "family": family, "shapley_pp": 100 * contribution})
    shapley = pd.DataFrame(shapley_rows)
    shapley_summary = (shapley.groupby(["classifier", "family"], as_index=False)
                       .agg(shapley_pp_mean=("shapley_pp", "mean"),
                            shapley_pp_std=("shapley_pp", "std"))
                       .sort_values(["classifier", "shapley_pp_mean"], ascending=[True, False]))
    shapley.to_csv(args.output_dir / "shapley_by_fold.csv", index=False, float_format="%.6f")
    shapley_summary.to_csv(args.output_dir / "shapley.csv", index=False, float_format="%.6f")

    log("\n=== BAcc por subconjunto ===")
    log(summary.to_string(index=False))
    log("\n=== Delta pareado contra a canônica (as três) ===")
    log(comparisons.to_string(index=False))
    log("\n=== Shapley entre famílias ===")
    log(shapley_summary.to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
