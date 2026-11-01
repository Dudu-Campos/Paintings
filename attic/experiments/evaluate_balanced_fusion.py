"""Fusão não-ingênua testada num caso com gabarito: base + Hilbert pooled 1/8.

Por que este caso. O Hilbert pooled é a instância mais limpa do padrão que
apareceu quatro vezes hoje -- sinal que o LDA acha e o MLP destrói:

    LDA  +0,32 ± 0,17pp  p=0,018   5/5 folds
    MLP  −0,38 ± 0,11pp  p=0,003   0/5 folds

Os dois significativos, em direções opostas, sobre a mesma feature. Então há
gabarito: o sinal existe, e uma fusão que o aproveite deve tirar o MLP do
negativo. Testar a arquitetura nova aqui é mais informativo que testá-la só na
base, onde não se sabe qual seria o resultado certo.

Condições, todas pareadas nos mesmos folds e com a mesma base de 1.091d
(7 modalidades + quadrantes de Descriptors, 11 branches):

    base                  fusão atual, sem o pooled          (referência)
    + pooled (atual)      fusão atual                        (deve dar ~−0,38)
    + pooled (aux)        loss auxiliar por branch
    + pooled (drop)       dropout de modalidade
    + pooled (gate)       gating por amostra
    + pooled (tudo)       as três juntas

Além da acurácia, mede **dependência por branch**: zera cada branch na
inferência e registra a queda de BAcc. Perfil concentrado confirma que o head
se apoia em poucas modalidades; espalhado refuta.

Saída: outputs/balanced_fusion/
    fold_results.csv     BAcc por condição e fold
    summary.csv          deltas pareados contra base e contra "+ pooled (atual)"
    branch_dependency.csv  queda de BAcc ao desligar cada branch
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
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(int(THREADS))

from balanced_branch_mlp import train_balanced
from evaluate_full_factorial_ablation import scaled_splits
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_pooled_hilbert(frame: pd.DataFrame, pooled_dir: Path) -> np.ndarray:
    values = np.load(pooled_dir / "hilbert_features.npy")
    stems = np.load(pooled_dir / "hilbert_stems.npy", allow_pickle=True).astype(str)
    columns = [f"hilbert_pooled_{index}" for index in range(values.shape[1])]
    pooled = pd.DataFrame(values, columns=columns)
    pooled.insert(0, "stem", stems)
    merged = frame[["stem"]].merge(pooled, on="stem", how="left", validate="one_to_one")
    missing = int(merged[columns].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing:,} imagens sem Hilbert pooled")
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
    parser.add_argument("--pooled-hilbert-dir", type=Path,
                        default=ROOT / "outputs/wikiart_hilbert_pooled_eighth")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/balanced_fusion")
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--branch-drop", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    pooled_block = load_pooled_hilbert(frame, args.pooled_hilbert_dir)

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
    log(f"{len(frame):,} imagens, {n_classes} estilos, pooled={pooled_block.shape[1]}d")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    dependency_rows = []
    completed = {(str(r["condition"]), int(r["fold"])) for r in rows}

    # (nome, inclui pooled, aux_weight, branch_drop, gating)
    #
    # O gate mudou de softmax (competitivo, matava branches) para sigmoide
    # independente. As condições que o usam foram renomeadas para não misturar
    # as duas implementações no mesmo CSV; base/atual/aux/drop não mudaram e
    # reaproveitam os folds já calculados.
    CONDITIONS = [
        # nome                   pooled  aux              drop              gate   scale
        ("base",                 False, 0.0,             0.0,              False, False),
        ("+ pooled (atual)",     True,  0.0,             0.0,              False, False),
        ("+ pooled (aux)",       True,  args.aux_weight, 0.0,              False, False),
        ("+ pooled (drop)",      True,  0.0,             args.branch_drop, False, False),
        ("+ pooled (aux+drop)",  True,  args.aux_weight, args.branch_drop, False, False),
        ("+ pooled (gate sig)",  True,  0.0,             0.0,              True,  False),
        ("+ pooled (tudo sig)",  True,  args.aux_weight, args.branch_drop, True,  False),
        # Escalar por branch: ataca o fato de o BatchNorm forçar todo branch a
        # variância unitária, tenha ele sinal ou não.
        ("+ pooled (scale)",     True,  0.0,             0.0,              False, True),
        ("+ pooled (aux+scale)", True,  args.aux_weight, 0.0,              False, True),
    ]

    for fold, (outer_train, test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, base_names = selected_blocks(BASELINE_MASK, group_blocks, extra)
        base_blocks = base_blocks + descriptor_quadrants
        base_names = base_names + quadrant_names

        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train],
            random_state=SEED + fold)

        for condition, with_pooled, aux_weight, branch_drop, gating, scale in CONDITIONS:
            if (condition, fold) in completed:
                continue
            blocks = base_blocks + [pooled_block] if with_pooled else list(base_blocks)
            names = base_names + ["Hilbert pooled"] if with_pooled else list(base_names)
            train_x, validation_x, test_x = scaled_splits(blocks, fit, validation, test)
            started = time.perf_counter()
            metrics, _, dependency = train_balanced(
                train_x, labels[fit], validation_x, labels[validation], test_x, labels[test],
                [block.shape[1] for block in blocks], n_classes, SEED + fold,
                args.epochs, args.patience,
                aux_weight=aux_weight, branch_drop=branch_drop, gating=gating,
                branch_scale=scale, branch_names=names)
            elapsed = time.perf_counter() - started
            rows.append({"fold": fold, "condition": condition, "n_branches": len(blocks),
                         "dimensions": sum(b.shape[1] for b in blocks),
                         "elapsed_seconds": elapsed, **metrics})
            for entry in dependency:
                dependency_rows.append({"fold": fold, "condition": condition, **entry})
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            if dependency_rows:
                pd.DataFrame(dependency_rows).to_csv(
                    args.output_dir / "branch_dependency.csv", index=False,
                    float_format="%.6f")
            log(f"  fold {fold} | {condition:<18} | {len(blocks):2d} branches | "
                f"BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    pivot = detail.pivot_table(index="fold", columns="condition", values="bacc")
    summary = []
    for reference in ("base", "+ pooled (atual)"):
        if reference not in pivot.columns:
            continue
        for condition in pivot.columns:
            if condition == reference:
                continue
            delta = 100 * (pivot[condition] - pivot[reference])
            statistic, pvalue = ttest_rel(pivot[condition], pivot[reference])
            summary.append({"reference": reference, "condition": condition,
                            "bacc_mean": 100 * pivot[condition].mean(),
                            "bacc_std": 100 * pivot[condition].std(),
                            "delta_pp_mean": delta.mean(), "delta_pp_std": delta.std(),
                            "p_paired": pvalue,
                            "positive_folds": int((delta > 0).sum())})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")

    log("\n=== Deltas pareados ===")
    log(summary.round(4).to_string(index=False))
    if dependency_rows:
        dependency = pd.DataFrame(dependency_rows)
        profile = (dependency[dependency.condition == "+ pooled (atual)"]
                   .groupby("branch", as_index=False)
                   .agg(drop_pp=("drop_pp", "mean"))
                   .sort_values("drop_pp", ascending=False))
        log("\n=== Dependência por branch (fusão atual) ===")
        log(profile.round(3).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
