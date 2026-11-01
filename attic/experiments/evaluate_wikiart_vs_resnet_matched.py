"""WikiArt contra ResNet50 congelada, com o MESMO protocolo dos externos.

O número que sustenta a comparação do paper -- empate em BAcc e vantagem
nossa em macro F1 contra features ImageNet congeladas -- foi medido antes de
dois ajustes que apareceram medindo Pandora e Rijksmuseum, e os dois apontam
para o mesmo lado:

  pico sobre k    concatenar 2.048d crus penaliza a ResNet por dimensionalidade.
                  No Rijksmuseum isso valia 10,12pp para ela e 6,70 para nós --
                  invertia o resultado.
  MLP nos dois    a sonda LDA achata os dois lados na mesma projeção linear e
                  não enxerga estrutura multi-bloco. No Pandora o MLP vale
                  +5,55pp para o lado hand-designed e −0,51 para a ResNet.

Aqui os dois braços recebem tratamento idêntico, e os blocos são exatamente os
mesmos 1.639d usados nos datasets externos (8 modalidades visuais, sem Text,
que só existe no WikiArt). Assim as três linhas da escada ficam comparáveis:

    Rijksmuseum   3,9k de treino
    Pandora      14k
    WikiArt      63k

Saída: results/wikiart_vs_resnet_matched.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "12")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from evaluate_external_modalities import fit_lda
from evaluate_felzenszwalb_line_graph import load_felz_line
from evaluate_full_factorial_ablation import evaluate_mlp
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, REGIONS, ROOT, SEED, load_dataset

FAMILIES = ("SLIC", "Descriptors", "Hilbert")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "lda"],
                        choices=["mlp", "lda"])
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/wikiart_vs_resnet_matched.csv")
    args = parser.parse_args()

    frame, groups = load_dataset(ROOT / "outputs/wikiart_quadrant_features")
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, ROOT / "results/line_graph_full_features.parquet",
        ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    frame, fourier_columns = load_fourier(
        frame, ROOT / "results/fourier_full_features.parquet")
    frame, felzline_columns = load_felz_line(
        frame, ROOT / "results/line_graph_felz_full_features.parquet")

    # ResNet congelada pelo mesmo stem, para a coorte ser idêntica nos dois braços.
    values = np.load(ROOT / "outputs/cnn_frozen_cache/resnet50_imagenet_v2_features.npy")
    stems = np.load(ROOT / "outputs/cnn_frozen_cache/stems.npy", allow_pickle=True).astype(str)
    resnet = pd.DataFrame(values, columns=[f"resnet_{i}" for i in range(values.shape[1])])
    resnet.insert(0, "stem", stems)
    frame = frame.merge(resnet.drop_duplicates("stem"), on="stem", how="inner")

    quadrant = [c for family in FAMILIES for region in REGIONS
                for c in groups.get(f"{family} ({region})", [])]
    blocks = {
        "Descriptors":  groups["Descriptors"], "Felzenszwalb": felz_columns,
        "SLIC":         groups["SLIC"],        "Line graph":   line_columns,
        "L(G_felz)":    felzline_columns,      "Hilbert":      groups["Hilbert"],
        "Fourier":      fourier_columns,       "Multiscale":   quadrant,
    }
    hand = [frame[c].to_numpy(dtype=np.float32) for c in blocks.values()]
    resnet_block = frame[[c for c in frame.columns if c.startswith("resnet_")]
                         ].to_numpy(dtype=np.float32)
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    log(f"{len(frame):,} imagens, {n_classes} estilos | "
        f"hand {sum(b.shape[1] for b in hand)}d em {len(hand)} branches | "
        f"ResNet {resnet_block.shape[1]}d")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    rows = []
    for classifier in args.classifiers:
        for name, arms in (("hand-designed", hand), ("ResNet50", [resnet_block])):
            for fold, (train, test) in enumerate(splits[:args.folds], start=1):
                started = time.perf_counter()
                if classifier == "mlp":
                    metrics = evaluate_mlp(arms, labels, train, test, fold, 150, 15)
                    metrics = {k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}
                else:
                    metrics = fit_lda(arms, labels, train, test, n_classes)
                rows.append({"classifier": classifier, "condition": name, "fold": fold,
                             "dimensions": sum(a.shape[1] for a in arms), **metrics})
                pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
                log(f"  {classifier} | {name:<13} fold {fold} | "
                    f"BAcc={100 * metrics['bacc']:.2f}% "
                    f"({time.perf_counter() - started:.0f}s)")

    detail = pd.DataFrame(rows)
    for classifier in args.classifiers:
        block = detail[detail.classifier == classifier]
        table = block.pivot_table(index="fold", columns="condition",
                                  values=["bacc", "accuracy", "macro_f1"]) * 100
        log(f"\n=== WikiArt, {classifier.upper()} ===")
        log(f"{'métrica':<10}{'hand 1639d':>12}{'ResNet 2048d':>14}"
            f"{'delta':>9}{'p':>8}{'folds':>7}")
        for label, metric in (("BAcc", "bacc"), ("Acc", "accuracy"),
                              ("macro F1", "macro_f1")):
            a, b = table[(metric, "hand-designed")], table[(metric, "ResNet50")]
            delta = a - b
            p = ttest_rel(a, b)[1] if len(a) > 2 else float("nan")
            log(f"{label:<10}{a.mean():>11.2f} {b.mean():>13.2f}"
                f"{delta.mean():>+9.2f}{p:>8.4f}{int((delta > 0).sum()):>4}/{len(delta)}")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
