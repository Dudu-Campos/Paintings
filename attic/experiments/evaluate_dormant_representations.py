"""Fatorial 2^7 das representações já extraídas e nunca usadas no pipeline.

Sete blocos estão em escala completa (79.207) no disco e fora do pipeline --
685 dimensões prontas, custo de extração zero:

    Random walk        57d   outputs/wikiart_features/rw_features.npy
    Hybrid walk        57d   outputs/wikiart_features/hybrid_walk_features.npy
    Square graph G^2   64d   results/square_graph_full_features.parquet
    Hilbert catch22    66d   outputs/hilbert_catch22_full/
    Hilbert spectrum1d 108d  outputs/hilbert_spectrum_1d_full/
    Hilbert visibility 57d   outputs/hilbert_visibility_graph_full/
    Hilbert wavelet    276d  outputs/hilbert_wavelet_pyramid_full/

Quatro derivam da mesma curva de Hilbert, que já é bloco do pipeline, e duas
são baseadas em caminhada -- há suspeita forte de redundância interna, e o
fatorial é o que resolve isso em vez de adivinhar.

Etapa 1 (este script): 2^7 = 128 subconjuntos com sonda LDA, Shapley exato
entre os sete. ~640 ajustes, viável em menos de uma hora. A etapa 2 (MLP) roda
depois só nos subconjuntos que a etapa 1 apontar.

Base: a melhor configuração medida -- 7 modalidades + quadrantes de Descriptors
+ L(G_felz) = 1.154d, 12 branches, MLP 54,42%.

Saída: outputs/dormant_representations/
    fold_results.csv   BAcc por subconjunto e fold
    summary.csv        média por subconjunto, ordenada
    shapley.csv        Shapley exato entre os sete
    vs_base.csv        delta pareado de cada subconjunto contra a base
"""
from __future__ import annotations

import os

THREADS = "16"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import argparse
import math
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

CANDIDATES = (
    ("Random walk",        "npy",     "outputs/wikiart_features/rw_features.npy",
                                      "outputs/wikiart_features/rw_stems.npy"),
    ("Hybrid walk",        "npy",     "outputs/wikiart_features/hybrid_walk_features.npy",
                                      "outputs/wikiart_features/hybrid_walk_stems.npy"),
    ("Square graph",       "parquet", "results/square_graph_full_features.parquet", None),
    ("Hilbert catch22",    "npy",     "outputs/hilbert_catch22_full/features.npy",
                                      "outputs/hilbert_catch22_full/stems.npy"),
    ("Hilbert spectrum1d", "npy",     "outputs/hilbert_spectrum_1d_full/features.npy",
                                      "outputs/hilbert_spectrum_1d_full/stems.npy"),
    ("Hilbert visibility", "npy",     "outputs/hilbert_visibility_graph_full/features.npy",
                                      "outputs/hilbert_visibility_graph_full/stems.npy"),
    ("Hilbert wavelet",    "npy",     "outputs/hilbert_wavelet_pyramid_full/features.npy",
                                      "outputs/hilbert_wavelet_pyramid_full/stems.npy"),
)
NAMES = tuple(name for name, *_ in CANDIDATES)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_candidate(frame: pd.DataFrame, spec) -> tuple[pd.DataFrame, list[str]]:
    name, kind, path, stems_path = spec
    prefix = name.lower().replace(" ", "_")
    if kind == "npy":
        values = np.load(ROOT / path, mmap_mode="r")
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        columns = [f"{prefix}_{index}" for index in range(values.shape[1])]
        block = pd.DataFrame(np.asarray(values), columns=columns)
        block.insert(0, "stem", stems)
    else:
        raw = pd.read_parquet(ROOT / path)
        feature_columns = [c for c in raw if c not in {"stem", "style"}]
        rename = {c: f"{prefix}_{c}" for c in feature_columns}
        block = raw[["stem"] + feature_columns].rename(columns=rename)
        columns = list(rename.values())
    block = block.drop_duplicates("stem")
    before = len(frame)
    frame = frame.merge(block, on="stem", how="inner", validate="one_to_one")
    if len(frame) != before:
        log(f"  ! {name}: coorte {before:,} -> {len(frame):,}")
    return frame, columns


def exact_shapley(values: dict[int, float]) -> dict[str, float]:
    n = len(NAMES)
    denominator = math.factorial(n)
    result = {}
    for index, name in enumerate(NAMES):
        bit = 1 << index
        contribution = 0.0
        for subset in range(1 << n):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
            contribution += weight * (values[subset | bit] - values[subset])
        result[name] = contribution
    return result


def subset_name(subset: int) -> str:
    active = [n for i, n in enumerate(NAMES) if subset & (1 << i)]
    return " + ".join(active) if active else "(base)"


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dormant_representations")
    parser.add_argument("--classifiers", nargs="+", default=["lda"], choices=["lda", "mlp"])
    parser.add_argument("--only-subsets", type=int, nargs="+",
                        help="roda apenas estes subconjuntos (para a etapa 2 do MLP)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    candidate_columns = {}
    for spec in CANDIDATES:
        frame, columns = load_candidate(frame, spec)
        candidate_columns[spec[0]] = columns
    frame = frame.sort_values("stem").reset_index(drop=True)

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
    descriptor_quadrants, quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)
    candidate_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                        for name, columns in candidate_columns.items()}

    log(f"{len(frame):,} imagens, {n_classes} estilos")
    for name, block in candidate_blocks.items():
        log(f"  candidato {name:<20} {block.shape[1]:4d}d")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(r["classifier"]), int(r["subset"]), int(r["fold"])) for r in rows}
    subsets = args.only_subsets if args.only_subsets else list(range(1 << len(NAMES)))

    for classifier in args.classifiers:
        for fold, (train, test) in enumerate(splits, start=1):
            if all((classifier, s, fold) in completed for s in subsets):
                continue
            text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
            extra = {**fixed_extra, "Text": text_block}
            base_blocks, base_names = selected_blocks(BASELINE_MASK, group_blocks, extra)
            base_blocks = base_blocks + descriptor_quadrants + [felzline_block]
            base_names = base_names + quadrant_names + ["Line graph (Felz)"]

            for subset in subsets:
                if (classifier, subset, fold) in completed:
                    continue
                blocks = list(base_blocks)
                for index, name in enumerate(NAMES):
                    if subset & (1 << index):
                        blocks.append(candidate_blocks[name])
                dimensions = sum(block.shape[1] for block in blocks)
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, train, test)
                else:
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                elapsed = time.perf_counter() - started
                rows.append({"fold": fold, "classifier": classifier, "subset": subset,
                             "condition": subset_name(subset), "n_branches": len(blocks),
                             "dimensions": dimensions, "elapsed_seconds": elapsed, **metrics})
                completed.add((classifier, subset, fold))
                pd.DataFrame(rows).to_csv(result_path, index=False)
                if subset % 16 == 0 or subset < 8:
                    log(f"  fold {fold} {classifier} [{subset:3d}] {dimensions:5d}d "
                        f"BAcc={metrics['bacc']:.4f} {elapsed:.0f}s")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["classifier", "subset", "condition", "dimensions"], as_index=False)
               .agg(bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"))
               .sort_values(["classifier", "bacc_mean"], ascending=[True, False]))
    summary[["bacc_mean", "bacc_std"]] *= 100
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")

    pivot = detail.pivot_table(index=["classifier", "fold"], columns="subset", values="bacc")
    comparisons, shapley_rows = [], []
    for classifier in detail["classifier"].unique():
        block = pivot.loc[classifier]
        if 0 not in block.columns:
            continue
        for subset in block.columns:
            if subset == 0:
                continue
            delta = 100 * (block[subset] - block[0])
            statistic, pvalue = ttest_rel(block[subset], block[0])
            comparisons.append({"classifier": classifier, "subset": int(subset),
                                "condition": subset_name(int(subset)),
                                "delta_vs_base_pp": delta.mean(), "delta_std_pp": delta.std(),
                                "p_paired": pvalue, "positive_folds": int((delta > 0).sum())})
        if len(block.columns) == (1 << len(NAMES)):
            for fold in block.index:
                values = {int(s): float(block.loc[fold, s]) for s in block.columns}
                for name, contribution in exact_shapley(values).items():
                    shapley_rows.append({"classifier": classifier, "fold": fold,
                                         "representation": name,
                                         "shapley_pp": 100 * contribution})

    if comparisons:
        pd.DataFrame(comparisons).sort_values(
            ["classifier", "delta_vs_base_pp"], ascending=[True, False]).to_csv(
            args.output_dir / "vs_base.csv", index=False, float_format="%.6f")
    if shapley_rows:
        shapley = pd.DataFrame(shapley_rows)
        agg = (shapley.groupby(["classifier", "representation"], as_index=False)
               .agg(shapley_pp_mean=("shapley_pp", "mean"), shapley_pp_std=("shapley_pp", "std"))
               .sort_values(["classifier", "shapley_pp_mean"], ascending=[True, False]))
        agg.to_csv(args.output_dir / "shapley.csv", index=False, float_format="%.6f")
        log("\n=== Shapley entre as sete ===")
        log(agg.round(4).to_string(index=False))

    log("\n=== Top 12 subconjuntos ===")
    log(summary.head(12).round(3).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
