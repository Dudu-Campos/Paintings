"""Shapley exato sobre segmentadores: SLIC, Felzenszwalb e Quickshift.

`compare_segmentations_pilot.py` mede cada segmentador isolado e mostra
Felzenszwalb (31,3%) ≈ Quickshift (31,0%) > SLIC (29,7%) na coorte pareada de
12.213 imagens. O que ele nunca mede é se eles somam.

Hipótese testada aqui: a complementaridade SLIC × Felzenszwalb não vem de
"algoritmos diferentes", e sim do eixo *geometria imposta × geometria dirigida
pela imagem* — SLIC tem contagem de nós quase constante (p05→p95 = 1.272→2.022)
enquanto Felzenszwalb (699→3.670) e Quickshift (1.321→7.561) são adaptativos.
Se o eixo for esse, Quickshift está do mesmo lado que Felzenszwalb e deve somar
pouco em cima dele, apesar de empatar quando medido sozinho.

Metodologia idêntica à do resto do projeto: sonda LDA, SEED=42, 5 folds
estratificados, 2^3 = 8 subconjuntos por fold, Shapley exato por fold (v(∅) =
1/n_classes, acaso) e desvio entre folds como incerteza.

Saída: outputs/segmenter_shapley/
    subset_results.csv    BAcc de cada um dos 8 subconjuntos, por fold
    shapley_by_fold.csv   contribuição de Shapley por fator e fold
    shapley_summary.csv   média e desvio entre folds
    marginals.csv         ganhos marginais que respondem a pergunta direto
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from compare_segmentations_pilot import METHOD_PATHS, load_methods
from fusion_curve_slic import N_FOLDS, ROOT, SEED, log, make_model

FACTORS    = ("SLIC", "Felzenszwalb", "Quickshift")
OUTPUT_DIR = ROOT / "outputs/segmenter_shapley"


def exact_shapley(values: dict[int, float]) -> dict[str, float]:
    """Mesma fórmula de evaluate_full_factorial_ablation_text.py."""
    n = len(FACTORS)
    denominator = math.factorial(n)
    result = {}
    for index, factor in enumerate(FACTORS):
        bit = 1 << index
        contribution = 0.0
        for subset in range(1 << n):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
            contribution += weight * (values[subset | bit] - values[subset])
        result[factor] = contribution
    return result


def mask_name(mask: int) -> str:
    names = [f for i, f in enumerate(FACTORS) if mask & (1 << i)]
    return " + ".join(names) if names else "(vazio)"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {m: METHOD_PATHS[m] for m in FACTORS}
    frame, columns_by_method, diagnostics = load_methods(paths)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(encoder.classes_)
    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED).split(np.zeros(len(y)), y))
    log(f"{len(frame):,} imagens pareadas, {n_classes} estilos, "
        f"{len(FACTORS)} segmentadores, 71 features cada")

    rows, shapley_rows = [], []
    for fold, (train, test) in enumerate(splits, 1):
        values = {}
        for mask in range(1 << len(FACTORS)):
            if mask == 0:
                bacc = 1.0 / n_classes
            else:
                cols = [c for i, f in enumerate(FACTORS) if mask & (1 << i)
                        for c in columns_by_method[f]]
                vals = (frame[cols].replace([np.inf, -np.inf], np.nan)
                        .to_numpy(dtype=np.float64))
                model = make_model(n_classes, len(cols))
                model.fit(vals[train], y[train])
                bacc = float(balanced_accuracy_score(
                    y[test], model.predict(vals[test])))
            values[mask] = bacc
            rows.append({"fold": fold, "mask": mask, "subset": mask_name(mask),
                         "n_features": 0 if mask == 0 else
                         sum(len(columns_by_method[f]) for i, f in enumerate(FACTORS)
                             if mask & (1 << i)),
                         "bacc": bacc})
            log(f"  fold {fold} | {mask_name(mask):<38} BAcc={bacc:.4f}")

        for factor, contribution in exact_shapley(values).items():
            shapley_rows.append({"fold": fold, "factor": factor,
                                 "shapley_pp": 100 * contribution})

    results = pd.DataFrame(rows)
    shapley = pd.DataFrame(shapley_rows)
    results.to_csv(OUTPUT_DIR / "subset_results.csv", index=False, float_format="%.6f")
    shapley.to_csv(OUTPUT_DIR / "shapley_by_fold.csv", index=False, float_format="%.6f")

    summary = (shapley.groupby("factor", as_index=False)
               .agg(shapley_pp_mean=("shapley_pp", "mean"),
                    shapley_pp_std=("shapley_pp", "std"))
               .sort_values("shapley_pp_mean", ascending=False))
    summary.to_csv(OUTPUT_DIR / "shapley_summary.csv", index=False, float_format="%.6f")

    # Ganhos marginais que respondem a pergunta diretamente.
    pivot = results.pivot(index="fold", columns="mask", values="bacc")
    S, F, Q = 1, 2, 4
    marginals = []
    for label, base, added in [
        ("Quickshift em cima de SLIC+Felzenszwalb", S | F, Q),
        ("Felzenszwalb em cima de SLIC+Quickshift", S | Q, F),
        ("SLIC em cima de Felzenszwalb+Quickshift", F | Q, S),
        ("Felzenszwalb em cima de SLIC",            S,     F),
        ("Quickshift em cima de SLIC",              S,     Q),
        ("Quickshift em cima de Felzenszwalb",      F,     Q),
    ]:
        delta = 100 * (pivot[base | added] - pivot[base])
        marginals.append({"marginal": label, "base": mask_name(base),
                          "gain_pp_mean": delta.mean(), "gain_pp_std": delta.std()})
    marginals = pd.DataFrame(marginals)
    marginals.to_csv(OUTPUT_DIR / "marginals.csv", index=False, float_format="%.6f")

    log("\n=== Shapley (pp de BAcc) ===")
    log(summary.to_string(index=False))
    log("\n=== Ganhos marginais (pp de BAcc) ===")
    log(marginals.to_string(index=False))
    log("\n=== BAcc média por subconjunto ===")
    log(results.groupby(["mask", "subset", "n_features"], as_index=False)
        .agg(bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"))
        .sort_values("bacc_mean").to_string(index=False))
    log(f"\nSalvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
