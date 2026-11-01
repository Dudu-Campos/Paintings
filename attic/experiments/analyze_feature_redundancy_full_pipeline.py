"""Auditoria de redundância par-a-par do pipeline atual (todas as modalidades visuais).

Atualiza `analyze_visual_feature_redundancy_r2.py`, que é de 11/ago e cobria só
258 features (SLIC + Descriptors + Hilbert) -- anterior a Line graph,
Felzenszwalb, Fourier e aos quadrantes de Multiscale.

Nenhum rótulo entra em nenhuma estatística: para cada par de features
não-constantes calcula-se r de Pearson e r², que com intercepto é exatamente o
coeficiente de determinação da regressão linear univariada. O bloco Text é
excluído de propósito -- seus scores são derivados de rótulo e quebrariam o
caráter não-supervisionado da auditoria. Sobra o pool vision-only de 1.576d,
o mesmo usado na curva de capacidade.

Saída: results/feature_redundancy_full/
    group_summary.csv    r² dentro e entre modalidades (a tabela que interessa)
    high_pairs.csv       pares com r² acima do maior limiar
    heatmap.png          matriz r² com fronteiras de modalidade
    feature_index.csv    mapeamento coluna -> modalidade
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from evaluate_full_factorial_ablation_text import selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import ROOT, load_dataset

VISION_ONLY_MASK = 191
THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)
CONSTANT_TOLERANCE = 1e-12
# Sequencial de uma cor só, claro->escuro: r² é magnitude, nunca arco-íris.
SEQUENTIAL = "Blues"
INK, MUTED = "#1f2328", "#6b7280"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/feature_redundancy_full")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {name: frame[columns].to_numpy(dtype=np.float64)
                    for name, columns in groups.items()}
    extra_blocks = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float64),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float64),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float64),
    }
    blocks, names = selected_blocks(VISION_ONLY_MASK, group_blocks, extra_blocks)
    values = np.concatenate(blocks, axis=1)
    membership = np.concatenate([[name] * block.shape[1]
                                 for name, block in zip(names, blocks)])
    log(f"{values.shape[0]:,} imagens x {values.shape[1]} features, "
        f"{len(names)} blocos")

    # Descarta features constantes: r é indefinido para elas.
    values[~np.isfinite(values)] = np.nan
    column_means = np.nanmean(values, axis=0)
    values = np.where(np.isfinite(values), values, column_means)
    deviation = values.std(axis=0)
    keep = deviation > CONSTANT_TOLERANCE
    if (~keep).any():
        log(f"descartando {int((~keep).sum())} features constantes")
    values, membership = values[:, keep], membership[keep]

    standardized = (values - values.mean(axis=0)) / values.std(axis=0)
    log("calculando matriz de correlação...")
    correlation = (standardized.T @ standardized) / len(standardized)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    r2 = correlation ** 2
    np.fill_diagonal(r2, np.nan)

    pd.DataFrame({"feature_index": np.arange(len(membership)), "modality": membership}
                 ).to_csv(args.output_dir / "feature_index.csv", index=False)

    # Resumo por par de modalidades -- é a tabela que responde "o que sobrepõe
    # com o quê", que a versão de 11/ago só tinha para 3 blocos.
    unique = list(dict.fromkeys(membership))
    rows = []
    for index, group_a in enumerate(unique):
        for group_b in unique[index:]:
            mask_a, mask_b = membership == group_a, membership == group_b
            values_ab = r2[np.ix_(mask_a, mask_b)]
            flat = values_ab[np.isfinite(values_ab)]
            if not len(flat):
                continue
            rows.append({
                "group_a": group_a, "group_b": group_b, "n_pairs": int(flat.size),
                "mean_r2": float(flat.mean()), "median_r2": float(np.median(flat)),
                "p95_r2": float(np.percentile(flat, 95)), "max_r2": float(flat.max()),
                **{f"frac_ge_{t}": float((flat >= t).mean()) for t in THRESHOLDS},
            })
    summary = pd.DataFrame(rows).sort_values("median_r2", ascending=False)
    summary.to_csv(args.output_dir / "group_summary.csv", index=False, float_format="%.6f")

    # Pares altamente redundantes, para inspeção manual.
    high = np.argwhere(np.triu(np.nan_to_num(r2), k=1) >= THRESHOLDS[-1])
    pd.DataFrame({"feature_a": high[:, 0], "feature_b": high[:, 1],
                  "modality_a": membership[high[:, 0]], "modality_b": membership[high[:, 1]],
                  "r2": r2[high[:, 0], high[:, 1]]}
                 ).sort_values("r2", ascending=False).to_csv(
        args.output_dir / "high_pairs.csv", index=False, float_format="%.6f")

    # Heatmap: blocos de modalidade contíguos, ordem interna por clusterização.
    order, boundaries, offset = [], [], 0
    for group in unique:
        index = np.flatnonzero(membership == group)
        if len(index) > 2:
            block = np.nan_to_num(r2[np.ix_(index, index)], nan=1.0)
            distance = np.clip(1.0 - block, 0.0, None)
            np.fill_diagonal(distance, 0.0)
            inner = leaves_list(linkage(squareform(distance, checks=False), method="average"))
            index = index[inner]
        order.extend(index.tolist())
        offset += len(index)
        boundaries.append(offset)
    order = np.array(order)
    ordered = np.nan_to_num(r2[np.ix_(order, order)], nan=1.0)

    figure, axis = plt.subplots(figsize=(9.5, 8.2))
    image = axis.imshow(ordered, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    for edge in boundaries[:-1]:
        axis.axhline(edge - 0.5, color="white", linewidth=0.9)
        axis.axvline(edge - 0.5, color="white", linewidth=0.9)
    centres = [(boundaries[i] + (boundaries[i - 1] if i else 0)) / 2
               for i in range(len(boundaries))]
    axis.set_xticks(centres); axis.set_yticks(centres)
    axis.set_xticklabels(unique, rotation=45, ha="right", fontsize=7.5, color=MUTED)
    axis.set_yticklabels(unique, fontsize=7.5, color=MUTED)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    bar = figure.colorbar(image, ax=axis, fraction=0.043, pad=0.02)
    bar.set_label("pairwise r²", fontsize=9, color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_visible(False)
    axis.set_title(f"Redundancy across {len(membership)} visual features",
                   fontsize=12.5, color=INK, loc="left", pad=12)
    figure.text(0.01, 0.012,
                f"WikiArt, {len(standardized):,} images · Pearson r², no labels used · "
                "blocks ordered by internal clustering · Text excluded (label-derived)",
                fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(args.output_dir / "heatmap.png", dpi=200, facecolor="white")

    log("\n=== r² por par de modalidades (top 15 por mediana) ===")
    log(summary.head(15)[["group_a", "group_b", "median_r2", "p95_r2", "frac_ge_0.9"]]
        .round(4).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
