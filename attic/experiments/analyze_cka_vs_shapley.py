"""CKA entre modalidades contra contribuição de Shapley.

Pergunta: a similaridade representacional entre uma modalidade e o resto do
pipeline prevê quanto ela contribui? O r² par-a-par não prevê (achado 13 --
a relação chega a se inverter), mas ele só enxerga relação linear entre pares
de colunas. CKA compara espaços inteiros, é invariante a rotação ortogonal e
escala isotrópica, e é a medida que já funcionou em outro projeto do grupo.

Calcula, para cada uma das 8 modalidades do fatorial:
  - CKA com cada outra modalidade (matriz completa)
  - CKA com a concatenação de todas as outras -- a redundância que importa
e confronta com o Shapley exato já medido em `full_factorial_ablation_text/`.

Duas variantes, porque elas separam hipóteses:
  bruto       CKA linear sobre as features padronizadas. Dominado pelas
              direções de alta variância -- a crítica conhecida do CKA.
  branqueado  CKA sobre componentes de PCA normalizados por desvio, que dá
              peso igual a todas as direções. Se o CKA só prevê no branqueado,
              a explicação é que o sinal de estilo mora na cauda de baixa
              variância (consistente com k* ~ 1.330 de 1.576).

Saída: results/cka_vs_shapley/
    cka_matrix_{variante}.csv    CKA par-a-par entre modalidades
    modality_summary.csv         CKA-com-o-resto vs Shapley, por classificador
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from evaluate_full_factorial_ablation_text import (
    FACTORS, VISUAL_FACTORS, load_text_block,
)
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, REGIONS, ROOT, SEED, load_dataset


def centre(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """CKA linear na forma de features: barata quando n >> d."""
    x, y = centre(x), centre(y)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    self_x = np.linalg.norm(x.T @ x, ord="fro")
    self_y = np.linalg.norm(y.T @ y, ord="fro")
    if self_x <= 0 or self_y <= 0:
        return float("nan")
    return float(cross / (self_x * self_y))


def standardise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    matrix = np.where(np.isnan(matrix), np.nanmean(matrix, axis=0), matrix)
    deviation = matrix.std(axis=0)
    deviation[deviation < 1e-12] = 1.0
    return (matrix - matrix.mean(axis=0)) / deviation


def whiten(matrix: np.ndarray) -> np.ndarray:
    """Componentes de PCA normalizados: todas as direções com peso igual."""
    centred = centre(matrix)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    keep = singular > (singular.max() * 1e-8)
    projected = centred @ right[keep].T
    return projected / projected.std(axis=0)


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
    parser.add_argument("--shapley", type=Path,
                        default=ROOT / "outputs/full_factorial_ablation_text/shapley_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/cka_vs_shapley")
    parser.add_argument("--sample", type=int, default=20000,
                        help="subamostra para o branqueamento; 0 usa tudo")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))

    # Text é dependente de fold; usa o fold 1 e registra a escolha.
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    text_block = load_text_block(args.text_cache_dir, 1, frame, n_classes)

    modality = {
        "SLIC":         frame[groups["SLIC"]].to_numpy(np.float64),
        "Descriptors":  frame[groups["Descriptors"]].to_numpy(np.float64),
        "Hilbert":      frame[groups["Hilbert"]].to_numpy(np.float64),
        "Line graph":   frame[line_columns].to_numpy(np.float64),
        "Felzenszwalb": frame[felz_columns].to_numpy(np.float64),
        "Fourier":      frame[fourier_columns].to_numpy(np.float64),
        "Text":         text_block.astype(np.float64),
        "Multiscale":   np.concatenate(
            [frame[groups[f"{family} ({region})"]].to_numpy(np.float64)
             for family in VISUAL_FACTORS for region in REGIONS], axis=1),
    }
    modality = {name: standardise(block) for name, block in modality.items()}
    for name, block in modality.items():
        print(f"  {name:<14} {block.shape}", flush=True)

    rows_matrix, rows_summary = [], []
    for variant in ("bruto", "branqueado"):
        if variant == "branqueado":
            index = np.arange(len(labels))
            if args.sample and args.sample < len(index):
                rng = np.random.default_rng(SEED)
                index = rng.choice(index, args.sample, replace=False)
            blocks = {name: whiten(block[index]) for name, block in modality.items()}
        else:
            blocks = modality
        names = list(blocks)
        for i, name_a in enumerate(names):
            for name_b in names[i:]:
                value = linear_cka(blocks[name_a], blocks[name_b])
                rows_matrix.append({"variant": variant, "a": name_a, "b": name_b, "cka": value})
        for name in names:
            rest = np.concatenate([block for other, block in blocks.items() if other != name],
                                  axis=1)
            rows_summary.append({"variant": variant, "modality": name,
                                 "cka_with_rest": linear_cka(blocks[name], rest),
                                 "dimensions": modality[name].shape[1]})
        print(f"[{variant}] pronto", flush=True)

    matrix = pd.DataFrame(rows_matrix)
    matrix.to_csv(args.output_dir / "cka_matrix.csv", index=False, float_format="%.6f")
    summary = pd.DataFrame(rows_summary)

    shapley = pd.read_csv(args.shapley)
    shapley = shapley.rename(columns={"factor": "modality"})
    summary = summary.merge(shapley[["classifier", "modality", "shapley_pp_mean"]],
                            on="modality", how="left")
    summary.to_csv(args.output_dir / "modality_summary.csv", index=False, float_format="%.6f")

    print("\n=== CKA com o resto vs Shapley ===")
    for variant in ("bruto", "branqueado"):
        for classifier in ("lda", "mlp"):
            block = summary[(summary.variant == variant) &
                            (summary.classifier == classifier)].dropna()
            if len(block) < 3:
                continue
            r, p_r = pearsonr(block.cka_with_rest, block.shapley_pp_mean)
            rho, p_rho = spearmanr(block.cka_with_rest, block.shapley_pp_mean)
            print(f"  {variant:<11} {classifier}: pearson r={r:+.3f} (p={p_r:.3f})  "
                  f"spearman rho={rho:+.3f} (p={p_rho:.3f})  n={len(block)}")
    print()
    print(summary.pivot_table(index="modality", columns="variant",
                              values="cka_with_rest").round(4).to_string())
    print(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
