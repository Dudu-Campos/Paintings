"""Producao visual-only, mas com Descriptors/Hilbert/Fourier RICOS em vez dos
compactos, mais o grafo de similaridade indutivo (mesma metodologia validada
de evaluate_production_visual_rich_similarity_graph.py -- nested OOF,
log-lift com prior do treino, LabelSpreading kernel='knn'). Pedido do
usuario: "mlp com as rich features + grafo de similaridade", que ainda nao
tinha sido testado junto (rich features so foi usado pra analise/narrativa
ate agora, nunca entrou na base do grafo).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_felzenszwalb_line_graph import load_felz_line  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_full_factorial_ablation_text import selected_blocks  # noqa: E402
from evaluate_label_spreading_feature_visual_graph_prior import (  # noqa: E402
    prior_correct, scale_blocks,
)
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks  # noqa: E402
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from evaluate_production_visual_rich_similarity_graph import (  # noqa: E402
    INNER_FOLDS, prior_correct_rich, rich_similarity_features, write_summary,
)
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402
from plot_rich_redundancy_layout import read_npy, read_parquet  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "production_rich_plus_similarity_graph"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrant-dir", type=Path, default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-line-features", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--felz-features", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path, default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "lda"], choices=["mlp", "lda"])
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    # troca Descriptors/Hilbert/Fourier compactos por ricos, por merge em stem
    rich_desc = read_npy(args.rich_dir / "descriptors" / "desc_full_features.npy",
                         args.rich_dir / "descriptors" / "desc_full_stems.npy", "descriptor_rich")
    rich_hilbert = read_npy(args.rich_dir / "hilbert" / "hilbert_features.npy",
                            args.rich_dir / "hilbert" / "hilbert_stems.npy", "hilbert_rich")
    rich_fourier = read_parquet(args.rich_fourier, "fourier_rich")
    before = len(frame)
    frame = frame.merge(rich_desc, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(rich_hilbert, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(rich_fourier, on="stem", how="inner", validate="one_to_one")
    if len(frame) != before:
        log(f"coorte mudou ao juntar rich features: {before:,} -> {len(frame):,}")
    rich_desc_cols = [c for c in rich_desc.columns if c != "stem"]
    rich_hilbert_cols = [c for c in rich_hilbert.columns if c != "stem"]
    rich_fourier_cols = [c for c in rich_fourier.columns if c != "stem"]

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32) for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {
        "Line graph": frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier": frame[rich_fourier_cols].to_numpy(dtype=np.float32),  # rico no lugar do compacto
    }
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, _ = family_quadrant_blocks("Descriptors", group_blocks)

    # selected_blocks pegaria Descriptors/Hilbert compactos do group_blocks; troca depois
    visual_core, visual_core_names = selected_blocks(BASELINE_MASK & ~64, group_blocks, fixed_extra)
    replaced_core, replaced_names = [], []
    for block, name in zip(visual_core, visual_core_names):
        if name == "Descriptors":
            replaced_core.append(frame[rich_desc_cols].to_numpy(dtype=np.float32))
            replaced_names.append("Descriptors_rich")
        elif name == "Hilbert":
            replaced_core.append(frame[rich_hilbert_cols].to_numpy(dtype=np.float32))
            replaced_names.append("Hilbert_rich")
        else:
            replaced_core.append(block)
            replaced_names.append(name)

    production_visual_blocks = replaced_core + [felzline_block] + descriptor_quadrants
    production_visual_names = replaced_names + [
        "L(G_felz)", "Descriptors top_left", "Descriptors top_right",
        "Descriptors bottom_left", "Descriptors bottom_right",
    ]
    total_dims = sum(b.shape[1] for b in production_visual_blocks)
    log(f"{len(frame):,} imagens, {n_classes} classes; base rica+graph={total_dims}d; "
        f"blocos={production_visual_names}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(labels), dtype=np.uint8), labels))
    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(r["condition"]), str(r["classifier"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits, start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        graph_scaled = scale_blocks(production_visual_blocks, train)
        started = time.perf_counter()
        train_graph, test_graph = rich_similarity_features(
            graph_scaled[train], labels[train], graph_scaled[test],
            n_classes, args.n_neighbors, args.alpha, args.max_iter, SEED + fold,
        )
        graph_feature = np.zeros((len(frame), train_graph.shape[1]), dtype=np.float32)
        graph_feature[train] = train_graph
        graph_feature[test] = test_graph
        graph_feature = prior_correct_rich(graph_feature, labels[train], n_classes)
        graph_elapsed = time.perf_counter() - started
        log(f"fold={fold}/{N_FOLDS} rich graph {graph_feature.shape[1]}d calculated in {graph_elapsed:.1f}s")

        for condition, blocks in (
            ("base_rich", production_visual_blocks),
            ("base_rich+graph", production_visual_blocks + [graph_feature]),
        ):
            for classifier in args.classifiers:
                key = (condition, classifier, fold)
                if key in done:
                    continue
                if classifier == "mlp":
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                          args.epochs, args.patience)
                else:
                    metrics = evaluate_lda(blocks, labels, train, test)
                row = {"condition": condition, "classifier": classifier, "fold": fold, **metrics}
                rows.append(row)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  {condition:18s} {classifier:4s} fold={fold}: BAcc={metrics['bacc']*100:.2f}%")

    detail = pd.DataFrame(rows)
    summary = write_summary(detail, args.output_dir)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
