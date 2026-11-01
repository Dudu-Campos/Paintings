"""Quickshift somado ao pipeline completo de 8 modalidades (LDA + MLP).

`evaluate_segmenter_shapley.py` mostrou que Quickshift soma +1,83 ± 0,60pp em
cima de SLIC+Felzenszwalb quando os três são os únicos blocos. A pergunta aberta
é quanto disso sobrevive ao lado de Descriptors, Hilbert, Line graph, Fourier,
Text e Multiscale — as outras modalidades podem já absorver a sobreposição.

Duas configurações pareadas, mesmos folds:
    baseline   as 8 modalidades canônicas
    +Quickshift  as 8 mais um ramo de 71 features do Quickshift

Duas decisões de desenho que importam:

1. **Coorte.** Quickshift só foi extraído para 12.213 imagens, então tudo aqui
   roda nesse recorte. Os números NÃO são comparáveis com os da ablação de
   79.207; só a diferença pareada dentro da coorte é interpretável.

2. **Folds herdados, não re-estratificados.** O bloco Text vem de cache OOF
   calculado com os folds da coorte de 79.207. Re-estratificar sobre as 12.213
   colocaria no teste imagens que estavam no treino do modelo de texto daquele
   fold — vazamento que inflaria justamente o Text. Por isso cada imagem
   mantém o fold que já tinha na coorte completa.

Saída: outputs/quickshift_full_pipeline/
    fold_results.csv    BAcc por configuração, classificador e fold
    summary.csv         média, desvio, delta pareado e t-test
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_segmentations_pilot import FEATURE_COLUMNS  # noqa: E402
from evaluate_full_factorial_ablation import evaluate_lda, evaluate_mlp  # noqa: E402
from evaluate_full_factorial_ablation_text import (  # noqa: E402
    FACTORS, FULL_MASK, load_text_block, log, selected_blocks,
)
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset  # noqa: E402

QUICKSHIFT = ROOT / "outputs/wikiart_segmentation_pilot/quickshift_k3_d10/combined_features.parquet"


def load_quickshift(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Junta o bloco Quickshift, reduzindo a coorte às imagens que o têm."""
    table = pd.read_parquet(path)
    missing = set(["stem"] + FEATURE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Quickshift sem colunas: {sorted(missing)}")
    table = table[["stem"] + FEATURE_COLUMNS].drop_duplicates("stem")
    columns = [f"quickshift__{column}" for column in FEATURE_COLUMNS]
    table.columns = ["stem"] + columns
    before = len(frame)
    frame = frame.merge(table, on="stem", how="inner", validate="one_to_one")
    log(f"Quickshift: coorte {before:,} -> {len(frame):,} imagens")
    return frame, columns


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
    parser.add_argument("--quickshift-features", type=Path, default=QUICKSHIFT)
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/quickshift_full_pipeline")
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"],
                        choices=["lda", "mlp"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Coorte completa primeiro: os folds precisam ser os mesmos que geraram o
    # cache de texto, senão o bloco Text vaza.
    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    labels_full = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels_full))
    log(f"Coorte canônica: {len(frame):,} imagens, {n_classes} estilos")

    splits_full = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                       .split(np.zeros(len(labels_full), dtype=np.uint8), labels_full))
    fold_of_row = np.zeros(len(frame), dtype=np.int32)
    for fold, (_, test) in enumerate(splits_full, start=1):
        fold_of_row[test] = fold
    frame = frame.assign(canonical_fold=fold_of_row)

    # Só agora reduz para a coorte do Quickshift, carregando o fold junto.
    frame, quickshift_columns = load_quickshift(frame, args.quickshift_features)
    frame = frame.sort_values("stem").reset_index(drop=True)
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    if len(np.unique(labels)) != n_classes:
        raise ValueError("A redução de coorte perdeu classes; folds herdados não valem mais")

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    fixed_extra = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    quickshift_block = frame[quickshift_columns].to_numpy(dtype=np.float32)

    counts = frame["canonical_fold"].value_counts().sort_index()
    log(f"Imagens por fold herdado: {counts.to_dict()}")

    rows = []
    for fold in range(1, N_FOLDS + 1):
        test = np.flatnonzero(frame["canonical_fold"].to_numpy() == fold)
        train = np.flatnonzero(frame["canonical_fold"].to_numpy() != fold)
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        extra = {**fixed_extra, "Text": text_block}
        base_blocks, base_names = selected_blocks(FULL_MASK, group_blocks, extra)

        for config, blocks, names in (
            ("baseline",    base_blocks,                     base_names),
            ("+Quickshift", base_blocks + [quickshift_block], base_names + ["Quickshift"]),
        ):
            dims = sum(block.shape[1] for block in blocks)
            for classifier in args.classifiers:
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = evaluate_lda(blocks, labels, train, test)
                else:
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                elapsed = time.perf_counter() - started
                rows.append({"fold": fold, "config": config, "classifier": classifier,
                             "bacc": metrics["bacc"], "n_branches": len(blocks),
                             "dimensions": dims, "elapsed_seconds": elapsed,
                             "blocks": " | ".join(names)})
                log(f"  fold {fold} | {config:<12} | {classifier:<3} | "
                    f"{dims:5d}d | BAcc={metrics['bacc']:.4f} | {elapsed:.0f}s")
                pd.DataFrame(rows).to_csv(args.output_dir / "fold_results.csv", index=False)
                if classifier == "mlp":
                    gc_torch()

    results = pd.DataFrame(rows)
    summary = []
    for classifier in args.classifiers:
        sub = results[results.classifier == classifier]
        base = sub[sub.config == "baseline"].sort_values("fold")["bacc"].to_numpy()
        plus = sub[sub.config == "+Quickshift"].sort_values("fold")["bacc"].to_numpy()
        delta = 100 * (plus - base)
        statistic, pvalue = ttest_rel(plus, base)
        summary.append({"classifier": classifier,
                        "baseline_bacc_mean": base.mean(), "baseline_bacc_std": base.std(),
                        "quickshift_bacc_mean": plus.mean(), "quickshift_bacc_std": plus.std(),
                        "delta_pp_mean": delta.mean(), "delta_pp_std": delta.std(),
                        "t_statistic": statistic, "p_value": pvalue})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Resumo ===")
    log(summary.to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


def gc_torch() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
