"""Melhor MLP contra melhor MLP + grafo de similaridade (feature indutiva).

Os testes de label spreading até agora usaram sonda LDA sobre o pipeline
canônico de 1.603d. Duas coisas mudaram:

  - O canônico não é o melhor. `evaluate_multiscale_family_factorial.py` mediu
    os 8 subconjuntos de famílias de Multiscale e o ótimo no MLP é manter só os
    quadrantes de Descriptors: **1.091d, 53,63% ± 0,82**, contra 53,03% do
    canônico de 1.603d -- mais acurácia com 512 dimensões a menos.
  - O classificador de produção é o MLP de branches, não o LDA.

Então a pergunta que decide se o grafo entra no paper é esta: sobre a melhor
configuração conhecida e com o classificador de produção, o grafo ainda soma?

Usa **apenas a variante indutiva** (`inductive_node_feature_scores`): o score de
uma imagem de teste não depende das outras imagens de teste, então o número
entra na tabela principal sem ressalva de protocolo e continua comparável às
baselines de CNN, que são indutivas.

Duas condições pareadas, mesmos folds:
    melhor MLP        7 modalidades + quadrantes de Descriptors (1.091d, 11 branches)
    + grafo           idem mais um branch de 27d com os scores propagados

Saída: outputs/best_mlp_similarity_graph/
    fold_results.csv   BAcc por condição e fold
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

torch.set_num_threads(int(THREADS_PER_WORKER))

from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_label_spreading_feature_full_pipeline import inductive_node_feature_scores
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
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
                        default=ROOT / "outputs/best_mlp_similarity_graph")
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "lda"],
                        choices=["mlp", "lda"])
    parser.add_argument("--prior-correct", action="store_true",
                        help="converte o posterior propagado em log-lift, "
                             "log p(c|x) - log p(c), usando o prior do treino do fold")
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
    # O grafo de similaridade é construído sobre Descriptors globais, como nos
    # experimentos anteriores de label spreading.
    descriptors_raw = group_blocks["Descriptors"]
    descriptor_quadrants, quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), str(r["condition"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, base_names = selected_blocks(BASELINE_MASK, group_blocks, extra)
        # Melhor configuração medida: as 7 modalidades + só os quadrantes de
        # Descriptors (ver evaluate_multiscale_family_factorial.py).
        best_blocks = base_blocks + descriptor_quadrants
        best_names = base_names + quadrant_names

        started = time.perf_counter()
        scaler = StandardScaler().fit(descriptors_raw[train])
        descriptors_scaled = scaler.transform(descriptors_raw)
        oof, target = inductive_node_feature_scores(
            descriptors_scaled[train], labels[train], descriptors_scaled[test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold)
        graph_block = np.zeros((len(frame), n_classes), dtype=np.float32)
        graph_block[train] = oof
        graph_block[test] = target
        if args.prior_correct:
            # O LabelSpreading não tem class_weight: com desbalanceamento de
            # 140x, uma vizinhança de 30 quase nunca contém as classes raras e
            # o posterior propagado herda o prior. O log-lift remove o prior,
            # que é o que `class_weight="balanced"` faz no bloco Text.
            prior = np.bincount(labels[train], minlength=n_classes).astype(np.float64)
            prior /= prior.sum()
            graph_block = (np.log(graph_block + 1e-8) - np.log(prior)).astype(np.float32)
        log(f"fold {fold}: grafo indutivo em {time.perf_counter()-started:.0f}s"
            + (" (log-lift)" if args.prior_correct else ""))

        for condition, blocks, names in (
            ("melhor MLP",        best_blocks,                 best_names),
            ("+ grafo (indutivo)", best_blocks + [graph_block], best_names + ["Similarity graph"]),
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
                log(f"  fold {fold} | {classifier:<3} | {condition:<19} | {dims:5d}d "
                    f"| {len(blocks):2d} branches | BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = []
    for classifier in detail["classifier"].unique():
        sub = detail[detail.classifier == classifier]
        base = sub[sub.condition == "melhor MLP"].sort_values("fold")["bacc"].to_numpy()
        plus = sub[sub.condition == "+ grafo (indutivo)"].sort_values("fold")["bacc"].to_numpy()
        if len(base) != len(plus) or len(base) < 2:
            continue
        delta = 100 * (plus - base)
        statistic, pvalue = ttest_rel(plus, base)
        summary.append({"classifier": classifier, "folds": len(base),
                        "base_bacc_mean": 100 * base.mean(), "base_bacc_std": 100 * base.std(),
                        "graph_bacc_mean": 100 * plus.mean(), "graph_bacc_std": 100 * plus.std(),
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
