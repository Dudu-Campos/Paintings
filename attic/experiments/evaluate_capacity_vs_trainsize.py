"""Como o pico de BAcc se move com o tamanho do treino (lei de escala em dados).

Motivação. As curvas BAcc(k) por número de features terminam onde o pool de
features projetadas acaba, não onde saturam -- então não permitem estimar
assíntota, e o teto de ~65% do ajuste anterior é extrapolação sem suporte.
Mas com dados finitos a curva tem um **máximo observável**: adicionar
componentes ajuda até o custo de variância superar o ganho de sinal. O máximo
é medição, não ajuste.

Este script mede esse máximo em função do tamanho do treino. O eixo x passa a
ser quantidade de dados -- que não tem limite arbitrário imposto por quantas
features alguém resolveu projetar, e que é o mesmo eixo das neural scaling
laws. Casa com o enquadramento de regime de dados do paper.

Desenho:
  - Pool **vision-only** (máscara 191: as 7 modalidades menos Text). Text é
    excluído de propósito: o cache OOF de TF-IDF foi calculado com o treino
    completo, então ao subamostrar o treino o bloco Text carregaria mais
    informação do que o tamanho de treino permitiria -- vazamento de
    quantidade de dados.
  - O **teste nunca é subamostrado**: só o treino encolhe, então BAcc é
    comparável entre tamanhos.
  - PCA ajustado só no treino subamostrado de cada fold; a projeção é feita
    uma vez com o número máximo de componentes e depois fatiada em k, o que
    troca ~360 ajustes de PCA por 30.
  - Sonda LDA, SEED=42, os mesmos 5 folds do resto do projeto.

Saída: outputs/capacity_vs_trainsize/
    fold_results.csv   BAcc por (fold, n_train, k)
    summary.csv        pico e k* por tamanho de treino
"""
from __future__ import annotations

import os

THREADS = os.environ.get("CAPACITY_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_full_factorial_ablation_text import selected_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset

VISION_ONLY_MASK = 191          # tudo menos Text
TRAIN_SIZES = (2500, 5000, 10000, 20000, 40000, 0)   # 0 = treino completo do fold


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
                        768, 1024, 1280, 1536) if k <= max_k]
    if grid[-1] != max_k:
        grid.append(max_k)
    return grid


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
                        default=ROOT / "outputs/capacity_vs_trainsize")
    parser.add_argument("--max-components", type=int, default=0,
                        help="0 = todas as dimensões do pool")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    extra_blocks = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    blocks, names = selected_blocks(VISION_ONLY_MASK, group_blocks, extra_blocks)
    values = np.concatenate(blocks, axis=1)
    values[~np.isfinite(values)] = np.nan
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    log(f"Pool vision-only: {values.shape[1]}d, {len(values):,} imagens, "
        f"{n_classes} estilos | blocos: {' | '.join(names)}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(int(r["fold"]), int(r["n_train"]), int(r["k"])) for r in rows}

    for fold, (train_all, test) in enumerate(splits, start=1):
        for requested in args.train_sizes:
            # Subamostra estratificada do treino; o teste fica intacto.
            if requested <= 0 or requested >= len(train_all):
                train = train_all
            else:
                train, _ = train_test_split(
                    train_all, train_size=requested, stratify=labels[train_all],
                    random_state=SEED + fold)
            n_train = len(train)
            max_k = min(values.shape[1], n_train - n_classes)
            if args.max_components:
                max_k = min(max_k, args.max_components)
            grid = k_grid(max_k)
            if all((fold, n_train, k) in done for k in grid):
                continue

            started = time.perf_counter()
            imputer = SimpleImputer(strategy="median").fit(values[train])
            scaler = StandardScaler().fit(imputer.transform(values[train]))
            train_x = scaler.transform(imputer.transform(values[train]))
            test_x = scaler.transform(imputer.transform(values[test]))
            pca = PCA(n_components=max_k, svd_solver="randomized",
                      random_state=SEED).fit(train_x)
            train_p, test_p = pca.transform(train_x), pca.transform(test_x)
            log(f"  fold {fold} n_train={n_train:,}: PCA {max_k} comps em "
                f"{time.perf_counter()-started:.0f}s")

            for k in grid:
                if (fold, n_train, k) in done:
                    continue
                model = LinearDiscriminantAnalysis(
                    n_components=min(n_classes - 1, k),
                    priors=np.ones(n_classes) / n_classes)
                model.fit(train_p[:, :k], labels[train])
                bacc = float(balanced_accuracy_score(
                    labels[test], model.predict(test_p[:, :k])))
                rows.append({"fold": fold, "n_train": n_train, "k": k, "bacc": bacc})
                done.add((fold, n_train, k))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            best = max(r["bacc"] for r in rows if r["fold"] == fold and r["n_train"] == n_train)
            log(f"  fold {fold} n_train={n_train:,}: pico BAcc={100*best:.2f}% "
                f"({time.perf_counter()-started:.0f}s)")

    detail = pd.DataFrame(rows)
    # Pico por (fold, n_train), depois média entre folds.
    peaks = (detail.loc[detail.groupby(["fold", "n_train"])["bacc"].idxmax()]
             .rename(columns={"k": "k_star"}))
    summary = (peaks.groupby("n_train", as_index=False)
               .agg(peak_bacc_mean=("bacc", "mean"), peak_bacc_std=("bacc", "std"),
                    k_star_median=("k_star", "median"), k_star_min=("k_star", "min"),
                    k_star_max=("k_star", "max"), folds=("fold", "nunique")))
    summary[["peak_bacc_mean", "peak_bacc_std"]] *= 100
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Pico de BAcc por tamanho de treino ===")
    log(summary.round(3).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
