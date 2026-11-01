"""Melhor MLP contra melhor MLP + espectro do Laplaciano do RAG (SLIC),
coorte completa.

Bloco novo (`src/baselines/extract_spectral_features.py`, 41 features):
quantis do espectro, momentos, traco do heat kernel em multiplas escalas,
entropia espectral, mais os 4 originais legados (fiedler/eig_max/spec_gap/
spec_ratio). Le grafos ja exportados (outputs/slic_gnn_full/graphs/*.pt),
sem reler imagem, CPU pura -- ideia do mojito: "operador novo sobre
substrato existente", categoria que o fatorial de hoje mostrou render
(82% de aditividade entre quatro operadores sobre a mesma curva de
Hilbert, por analogia).

Fundido na mesma melhor configuracao conhecida (1.091d, 7 modalidades + so
quadrantes de Descriptors no Multiscale -- ver
`evaluate_multiscale_family_factorial.py`), mesmo padrao dos testes
recentes (grafo de similaridade, Hilbert pooled).

Duas condicoes pareadas, mesmos folds:
    melhor MLP           7 modalidades + quadrantes de Descriptors (1.091d, 11 branches)
    + espectro Laplaciano idem mais um branch de 41d

Saida: outputs/best_mlp_spectral/
    fold_results.csv   BAcc por classificador, condicao e fold
    summary.csv        delta pareado e t-test
"""
from __future__ import annotations

import os

THREADS_PER_WORKER = "8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS_PER_WORKER)

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
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_spectral_block(frame: pd.DataFrame, spectral_path: Path) -> np.ndarray:
    spectral = pd.read_parquet(spectral_path)
    columns = [c for c in spectral.columns if c != "stem"]
    merged = frame[["stem"]].merge(spectral, on="stem", how="left", validate="one_to_one")
    if merged[columns].isna().any().any():
        missing = merged[columns[0]].isna().sum()
        raise ValueError(f"{missing:,} imagens sem features espectrais (esperado 0)")
    return merged[columns].to_numpy(dtype=np.float32)


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
    parser.add_argument("--spectral-features", type=Path,
                        default=ROOT / "outputs/wikiart_features/spectral_features_v2.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/best_mlp_spectral")
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"],
                        choices=["mlp", "lda"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
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
    spectral_block = load_spectral_block(frame, args.spectral_features)
    log(f"{len(frame):,} imagens, {n_classes} estilos, espectral dims={spectral_block.shape[1]}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), str(r["condition"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, base_names = selected_blocks(BASELINE_MASK, group_blocks, extra)
        best_blocks = base_blocks + descriptor_quadrants
        best_names = base_names + quadrant_names

        for condition, blocks, names in (
            ("melhor MLP",              best_blocks,                     best_names),
            ("+ espectro Laplaciano",   best_blocks + [spectral_block],  best_names + ["Spectral"]),
        ):
            dims = sum(block.shape[1] for block in blocks)
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
                             "n_branches": len(blocks), "dimensions": dims,
                             "elapsed_seconds": elapsed, **metrics})
                completed.add((classifier, condition, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  fold {fold} | {classifier:<3} | {condition:<22} | {dims:5d}d "
                    f"| {len(blocks):2d} branches | BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = []
    for classifier in detail["classifier"].unique():
        sub = detail[detail.classifier == classifier]
        base = sub[sub.condition == "melhor MLP"].sort_values("fold")["bacc"].to_numpy()
        plus = sub[sub.condition == "+ espectro Laplaciano"].sort_values("fold")["bacc"].to_numpy()
        if len(base) != len(plus) or len(base) < 2:
            continue
        delta = 100 * (plus - base)
        statistic, pvalue = ttest_rel(plus, base)
        summary.append({"classifier": classifier, "folds": len(base),
                        "base_bacc_mean": 100 * base.mean(), "base_bacc_std": 100 * base.std(),
                        "spectral_bacc_mean": 100 * plus.mean(), "spectral_bacc_std": 100 * plus.std(),
                        "delta_pp_mean": delta.mean(), "delta_pp_std": delta.std(),
                        "t_statistic": statistic, "p_paired": pvalue,
                        "positive_folds": int((delta > 0).sum())})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Resumo ===")
    log(summary.round(4).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
