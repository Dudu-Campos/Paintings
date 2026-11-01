"""Bloco espectral do Laplaciano do RAG: sonda LDA, quatro variantes.

O extrator (`src/baselines/extract_spectral_features.py`, v2) produz 41
features do espectro do Laplaciano **normalizado** do RAG do SLIC: 19 quantis,
momentos, 10 traços de heat kernel, entropia espectral, os 4 escalares antigos
(fiedler, eig_max, spec_gap, spec_ratio) e 3 de tamanho (n_nodes, n_edges,
n_components).

Dois problemas de escala que este script isola em vez de assumir:

1. **Os traços de heat kernel não são invariantes a tamanho.** O traço de
   exp(-tL) é a soma de exp(-t·λᵢ), que é O(n). Medido nos dados:
   heat_t0.05 / n_nodes = 0,9515 e 0,9514 em duas imagens diferentes -- ou
   seja, nos t pequenos essa coluna é `n_nodes` vezes uma constante. Dividir
   por n_nodes converte o traço em valor médio e restaura a invariância.
2. **n_nodes, n_edges e n_components duplicam** features que o bloco SLIC de
   71 já carrega.

Condições, todas com a mesma base (melhor config, 1.154d):
    base            como está hoje
    + cru           as 41 features como saíram do extrator
    + normalizado   traços de heat divididos por n_nodes
    + norm. s/ tam. idem, sem n_nodes/n_edges/n_components

Saída: outputs/spectral_block/
    fold_results.csv   BAcc por condição e fold
    summary.csv        delta pareado contra a base
"""
from __future__ import annotations

import os

THREADS = "8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from evaluate_felzenszwalb_line_graph import load_felz_line
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset

HEAT = [f"spec_heat_t{t}" for t in
        ("0.05", "0.1", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "32.0")]
SIZE = ["spec_n_nodes", "spec_n_edges", "spec_n_components"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
    parser.add_argument("--spectral", type=Path,
                        default=ROOT / "outputs/wikiart_features/spectral_features_v2.parquet")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/spectral_block")
    parser.add_argument("--classifiers", nargs="+", default=["lda"], choices=["lda", "mlp"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    spectral = pd.read_parquet(args.spectral).drop_duplicates("stem")
    # O bloco SLIC já tem n_nodes/n_edges; sem prefixo o merge colide.
    spectral = spectral.rename(columns={c: f"spec_{c}" for c in spectral.columns
                                        if c != "stem"})
    before = len(frame)
    frame = frame.merge(spectral, on="stem", how="inner", validate="one_to_one")
    log(f"espectral: coorte {before:,} -> {len(frame):,} "
        f"({spectral.shape[1] - 1} features)")

    feature_columns = [c for c in spectral.columns if c != "stem"]
    raw = frame[feature_columns].to_numpy(dtype=np.float64)
    # Normalização: traço médio do heat kernel em vez do traço total.
    normalised = raw.copy()
    nodes = frame["spec_n_nodes"].to_numpy(dtype=np.float64)
    nodes[nodes <= 0] = 1.0
    for name in HEAT:
        if name in feature_columns:
            normalised[:, feature_columns.index(name)] /= nodes
    keep_no_size = [i for i, c in enumerate(feature_columns) if c not in SIZE]

    variants = {
        "+ cru": raw.astype(np.float32),
        "+ normalizado": normalised.astype(np.float32),
        "+ norm. s/ tam.": normalised[:, keep_no_size].astype(np.float32),
    }

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

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r.get("classifier", "lda")), str(r["condition"]), int(r["fold"]))
                 for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, _ = selected_blocks(BASELINE_MASK, group_blocks, extra)
        base_blocks = base_blocks + descriptor_quadrants + [felzline_block]

        conditions = {"base": base_blocks}
        for name, block in variants.items():
            conditions[name] = base_blocks + [block]

        for condition, blocks in conditions.items():
            dimensions = sum(block.shape[1] for block in blocks)
            for classifier in args.classifiers:
                if (classifier, condition, fold) in completed:
                    continue
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, train, test)
                else:
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                elapsed = time.perf_counter() - started
                rows.append({"fold": fold, "classifier": classifier, "condition": condition,
                             "n_branches": len(blocks), "dimensions": dimensions,
                             "elapsed_seconds": elapsed, **metrics})
                completed.add((classifier, condition, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  fold {fold} | {classifier:<3} | {condition:<16} | {dimensions:5d}d | "
                    f"BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    if "classifier" not in detail.columns:
        detail["classifier"] = "lda"
    summary = []
    for classifier in detail["classifier"].unique():
      pivot = detail[detail.classifier == classifier].pivot_table(
          index="fold", columns="condition", values="bacc")
      if "base" not in pivot.columns:
          continue
      for condition in pivot.columns:
        if condition == "base":
            continue
        delta = 100 * (pivot[condition] - pivot["base"])
        statistic, pvalue = ttest_rel(pivot[condition], pivot["base"])
        summary.append({"classifier": classifier, "condition": condition,
                        "bacc_mean": 100 * pivot[condition].mean(),
                        "bacc_std": 100 * pivot[condition].std(),
                        "delta_pp_mean": delta.mean(), "delta_pp_std": delta.std(),
                        "p_paired": pvalue, "positive_folds": int((delta > 0).sum())})
    summary = pd.DataFrame(summary).sort_values(["classifier", "delta_pp_mean"],
                                                ascending=[True, False])
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    for classifier in detail["classifier"].unique():
        base = detail[(detail.classifier == classifier) & (detail.condition == "base")]["bacc"]
        log(f"base {classifier}: {100 * base.mean():.3f} ± {100 * base.std():.3f}")
    log("\n=== Deltas contra a base ===")
    log(summary.round(4).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
