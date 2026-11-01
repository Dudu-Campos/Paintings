"""Pipeline sem features duplicadas: 49 grupos, 68 colunas removíveis sem perda.

Auditoria não-supervisionada (`analyze_feature_redundancy_full_pipeline.py`)
encontrou 49 grupos de features com r² ≥ 0,999 entre si -- duplicatas literais,
não features parecidas. O padrão tem nome: são as estatísticas globais de cor
Lab, que **cada extrator recalculou por conta própria**:

    L_mean_w   em SLIC, Descriptors (descriptor_90), Hilbert (16), Felzenszwalb
    a_mean_w   em SLIC, Descriptors (93), Hilbert (34), Felzenszwalb
    b_mean_w   em SLIC, Descriptors (96), Hilbert (52), Felzenszwalb

Quatro branches carregam a mesma média de cor no nível global e três em cada
quadrante. Como cada branch tem seu próprio encoder de 128 dimensões, essa
medida influencia a saída por quatro caminhos enquanto uma medida única
influencia por um -- não é só desperdício de dimensão, é distorção de peso na
fusão. E o que está quadruplicado é paleta, um dos sinais mais fortes de estilo.

A remoção **não envolve rótulo nenhum**, então não há viés de seleção e não é
preciso protocolo aninhado: é a mesma decisão que remover colunas constantes.

Duas condições pareadas, mesmos folds, base = melhor config (1.154d):
    base       como está hoje
    dedup      uma cópia por grupo, ~68 colunas a menos

Verificação incluída: os grupos são recalculados **só no treino do fold 1** e
comparados com os globais. Se coincidirem, a detecção é estrutural e não
depende da amostra.

Saída: outputs/deduplicated_pipeline/
    duplicate_groups.csv  grupos encontrados, com a cópia mantida marcada
    fold_results.csv      BAcc por condição, classificador e fold
    summary.csv           delta pareado e t-test
"""
from __future__ import annotations

import os

THREADS = "16"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(int(THREADS))

from evaluate_felzenszwalb_line_graph import load_felz_line
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset

# Qual cópia sobrevive: preferência pelo bloco onde a medida é "nativa".
# Estatística de cor pertence a Descriptors, que também é o maior Shapley.
PRIORITY = ("Descriptors", "SLIC", "Felzenszwalb", "Hilbert", "Line graph",
            "Line graph (Felz)", "Fourier")
DUPLICATE_R2 = 0.999


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def rank(modality: str) -> int:
    for index, name in enumerate(PRIORITY):
        if modality.startswith(name):
            return index
    return len(PRIORITY)


def duplicate_groups(values: np.ndarray, rows: np.ndarray | None = None) -> list[list[int]]:
    """Componentes conexas de features com r² >= DUPLICATE_R2. Sem rótulos."""
    matrix = values[rows] if rows is not None else values
    deviation = matrix.std(axis=0)
    usable = deviation > 1e-12
    index_map = np.flatnonzero(usable)
    standard = (matrix[:, usable] - matrix[:, usable].mean(axis=0)) / deviation[usable]
    r2 = ((standard.T @ standard) / len(standard)) ** 2
    np.fill_diagonal(r2, 0.0)

    parent = list(range(len(index_map)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in np.argwhere(np.triu(r2, 1) >= DUPLICATE_R2):
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a
    clusters: dict[int, list[int]] = {}
    for node in range(len(index_map)):
        clusters.setdefault(find(node), []).append(int(index_map[node]))
    return [sorted(members) for members in clusters.values() if len(members) > 1]


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/deduplicated_pipeline")
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "lda"],
                        choices=["mlp", "lda"])
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
    column_names = {**groups, "Line graph": line_columns, "Felzenszwalb": felz_columns,
                    "Fourier": fourier_columns, "Line graph (Felz)": felzline_columns}
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)

    # Blocos visuais na ordem em que entram na fusão. Text fica de fora da
    # detecção de duplicatas (é dependente de fold e derivado de rótulo) e é
    # reanexado por último em cada fold.
    placeholder = {**fixed_extra,
                   "Text": np.zeros((len(frame), n_classes), dtype=np.float32)}
    _, ordered_names = selected_blocks(BASELINE_MASK, group_blocks, placeholder)
    lookup = {**group_blocks, **fixed_extra}
    visual_names, visual_blocks = [], []
    for name in ordered_names:
        if name == "Text":
            continue
        visual_names.append(name)
        visual_blocks.append(lookup[name])
    visual_names += quadrant_names + ["Line graph (Felz)"]
    visual_blocks += descriptor_quadrants + [felzline_block]

    stacked = np.concatenate(visual_blocks, axis=1).astype(np.float64)
    stacked[~np.isfinite(stacked)] = np.nan
    stacked = np.where(np.isnan(stacked), np.nanmean(stacked, axis=0), stacked)
    membership, feature_names = [], []
    for name, block in zip(visual_names, visual_blocks):
        membership += [name] * block.shape[1]
        feature_names += list(column_names[name]) if name in column_names else \
            [f"{name}_{i}" for i in range(block.shape[1])]
    membership = np.array(membership)

    clusters = duplicate_groups(stacked)
    log(f"{len(clusters)} grupos de duplicatas (r² ≥ {DUPLICATE_R2}), "
        f"{sum(len(c) for c in clusters)} features envolvidas")

    # Verificação: os grupos são estruturais ou dependem da amostra?
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    fold1_clusters = duplicate_groups(stacked, splits[0][0])
    same = ({tuple(c) for c in clusters} == {tuple(c) for c in fold1_clusters})
    log(f"grupos recalculados só no treino do fold 1 coincidem com os globais: {same}")

    drop, records = set(), []
    for cluster in clusters:
        keeper = min(cluster, key=lambda i: (rank(membership[i]), i))
        for index in cluster:
            records.append({"group_size": len(cluster), "feature": feature_names[index],
                            "modality": membership[index], "kept": index == keeper})
            if index != keeper:
                drop.add(index)
    pd.DataFrame(records).to_csv(args.output_dir / "duplicate_groups.csv", index=False)
    log(f"removendo {len(drop)} colunas de {stacked.shape[1]} visuais")

    # Reconstrói os blocos visuais sem as colunas descartadas.
    offsets, cursor = {}, 0
    for name, block in zip(visual_names, visual_blocks):
        offsets[name] = (cursor, cursor + block.shape[1])
        cursor += block.shape[1]
    dedup_blocks = {}
    for name, block in zip(visual_names, visual_blocks):
        start, end = offsets[name]
        keep = np.array([i not in drop for i in range(start, end)])
        dedup_blocks[name] = block[:, keep]
        if keep.sum() != block.shape[1]:
            log(f"  {name:<28} {block.shape[1]:4d} -> {int(keep.sum()):4d}")

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), str(r["condition"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        for condition in ("base", "dedup"):
            source = dict(zip(visual_names, visual_blocks)) if condition == "base" else dedup_blocks
            blocks = [source[name] for name in visual_names] + [text_block]
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
                log(f"  fold {fold} | {classifier:<3} | {condition:<6} | {dimensions:5d}d "
                    f"| BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = []
    for classifier in detail["classifier"].unique():
        sub = detail[detail.classifier == classifier]
        base = sub[sub.condition == "base"].sort_values("fold")["bacc"].to_numpy()
        dedup = sub[sub.condition == "dedup"].sort_values("fold")["bacc"].to_numpy()
        if len(base) != len(dedup) or len(base) < 2:
            continue
        delta = 100 * (dedup - base)
        statistic, pvalue = ttest_rel(dedup, base)
        summary.append({"classifier": classifier, "folds": len(base),
                        "base_bacc_mean": 100 * base.mean(),
                        "dedup_bacc_mean": 100 * dedup.mean(),
                        "delta_pp_mean": delta.mean(), "delta_pp_std": delta.std(),
                        "p_paired": pvalue, "positive_folds": int((delta > 0).sum())})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Resumo ===")
    log(summary.round(4).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
