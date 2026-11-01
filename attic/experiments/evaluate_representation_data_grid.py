"""Grade representação × dados: uma curva de escala por número de modalidades.

A lei de escala em dados sozinha não é contribuição -- Kaplan, Hoffmann e
Hestness já a estabeleceram. O eixo do paper é **representação**, e a figura
que interessa cruza os dois: para cada número de modalidades ativas, uma curva
de erro contra tamanho de treino.

Três desfechos possíveis, três conclusões distintas:

  curvas paralelas      representação e dado são intercambiáveis, e a distância
                        vertical entre curvas dá a **taxa de câmbio** ("somar
                        Felzenszwalb equivale a multiplicar os dados por X")
  expoente cresce com m representações também tornam o modelo mais eficiente em
                        dados, não só mais preciso
  curvas convergem      representação extra só ajuda no regime de pouco dado

Desenho:
  - **7 modalidades visuais**, aninhadas em ordem de Shapley do LDA (achado 1):
    Descriptors → Felzenszwalb → SLIC → Line graph → Hilbert → Fourier →
    Multiscale. Text fica de fora de propósito: seu cache OOF foi calculado com
    o treino completo, então subamostrar o treino faria o bloco carregar
    informação de 63 mil imagens e contaminar exatamente o eixo medido.
  - **Pico sobre k** (componentes de PCA), como em
    `evaluate_capacity_vs_trainsize.py`. Sem isso os subconjuntos grandes
    ficariam artificialmente ruins com pouco dado por penalidade de
    dimensionalidade, e a figura mediria isso em vez de conteúdo
    representacional.
  - **O fold de teste nunca encolhe**; só o treino é subamostrado.

Ordenação: a ordem aninhada é gulosa (Shapley), o que é adequado aqui porque
cada curva é uma medição válida do conjunto que ela nomeia -- não se está
inferindo saturação a partir da ordem. Uma versão livre de ordem exigiria
amostrar máscaras do 2^7, que é ~6× mais caro.

Custo: 7 subconjuntos × 6 tamanhos × 5 folds. Só CPU (PCA + LDA), não toca na
GPU. Feito para rodar na `margarita`, que tem 128 threads:

    OMP_NUM_THREADS=64 python experiments/evaluate_representation_data_grid.py

Saída: outputs/representation_data_grid/
    fold_results.csv   BAcc por (subconjunto, n_train, k, fold)
    peaks.csv          pico sobre k por (subconjunto, n_train, fold)
    summary.csv        pico médio e expoente ajustado por subconjunto
"""
from __future__ import annotations

import os

THREADS = os.environ.get("GRID_THREADS", "16")
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

from evaluate_full_factorial_ablation import evaluate_mlp

from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, REGIONS, ROOT, SEED, load_dataset

# Ordem de Shapley do LDA na ablação de 8 fatores (achado 1), sem Text.
ORDER = ("Descriptors", "Felzenszwalb", "SLIC", "Line graph", "Hilbert",
         "Fourier", "Multiscale")
TRAIN_SIZES = (2500, 5000, 10000, 20000, 40000, 0)   # 0 = treino completo


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
                        768, 1024, 1280, 1536) if k <= max_k]
    if not grid or grid[-1] != max_k:
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
                        default=ROOT / "outputs/representation_data_grid")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    parser.add_argument("--max-modalities", type=int, default=len(ORDER))
    parser.add_argument("--classifier", choices=["lda", "mlp", "xgb", "rf"], default="lda",
                        help="mlp usa os blocos como branches em dimensão nativa "
                             "(forma de produção), sem PCA nem varredura de k")
    parser.add_argument("--multiscale-pca", type=int, default=0, metavar="N",
                        help="comprime Multiscale a N componentes antes de "
                             "concatenar (PCA ajustado só no treino de cada "
                             "fold). 91 = média das outras seis modalidades. "
                             "0 desliga, mantendo os 1.032d nativos.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)

    # Multiscale = todos os blocos de quadrante, como na ablação de 8 fatores.
    quadrant_columns = [column for family in ("SLIC", "Descriptors", "Hilbert")
                        for region in REGIONS
                        for column in groups[f"{family} ({region})"]]
    modality_columns = {
        "Descriptors":  groups["Descriptors"],
        "Felzenszwalb": felz_columns,
        "SLIC":         groups["SLIC"],
        "Line graph":   line_columns,
        "Hilbert":      groups["Hilbert"],
        "Fourier":      fourier_columns,
        "Multiscale":   quadrant_columns,
    }
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    log(f"{len(frame):,} imagens, {n_classes} estilos, threads={THREADS}")
    for name in ORDER:
        log(f"  {name:<14} {len(modality_columns[name]):5d}d")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(int(r["n_modalities"]), int(r["fold"]), int(r["n_train"]), int(r["k"]))
            for r in rows}

    for count in range(1, args.max_modalities + 1):
        names = list(ORDER[:count])
        # Com --multiscale-pca, Multiscale sai do bloco concatenado e volta
        # comprimido dentro do fold: o PCA precisa ver só o treino.
        squeezed = args.multiscale_pca > 0 and "Multiscale" in names
        plain = [name for name in names if not (squeezed and name == "Multiscale")]
        columns = [c for name in plain for c in modality_columns[name]]
        values = np.array(frame[columns].to_numpy(dtype=np.float32), copy=True)
        values[~np.isfinite(values)] = np.nan
        raw_multiscale, n_dims = None, values.shape[1]
        if squeezed:
            raw_multiscale = np.array(
                frame[modality_columns["Multiscale"]].to_numpy(dtype=np.float32),
                copy=True)
            raw_multiscale[~np.isfinite(raw_multiscale)] = np.nan
            n_dims += args.multiscale_pca
        log(f"=== {count} modalidade(s): {' + '.join(names)} ({n_dims}d"
            + (f", Multiscale {raw_multiscale.shape[1]}→{args.multiscale_pca}"
               if squeezed else "") + ")")

        for fold, (train_all, test) in enumerate(splits, start=1):
            for requested in args.train_sizes:
                if requested <= 0 or requested >= len(train_all):
                    train = train_all
                else:
                    train, _ = train_test_split(
                        train_all, train_size=requested, stratify=labels[train_all],
                        random_state=SEED + fold)
                n_train = len(train)
                max_k = min(n_dims, n_train - n_classes)
                grid = k_grid(max_k)
                if args.classifier == "lda":
                    if all((count, fold, n_train, k) in done for k in grid):
                        continue
                elif (count, fold, n_train, 0) in done:
                    continue

                # Compressão de Multiscale, ajustada só no treino deste fold.
                fold_values, multiscale_block = values, None
                if squeezed:
                    imputer = SimpleImputer(strategy="median").fit(raw_multiscale[train])
                    filled = imputer.transform(raw_multiscale)
                    scaler = StandardScaler().fit(filled[train])
                    scaled = scaler.transform(filled)
                    reducer = PCA(n_components=args.multiscale_pca,
                                  svd_solver="randomized", random_state=SEED)
                    reducer.fit(scaled[train])
                    multiscale_block = reducer.transform(scaled).astype(np.float32)
                    fold_values = np.hstack([values, multiscale_block])

                if args.classifier in ("xgb", "rf"):
                    # Árvores veem todas as colunas de uma vez; sem PCA, sem
                    # branches. Padronização mantida para paridade com o resto.
                    started = time.perf_counter()
                    imp = SimpleImputer(strategy="median").fit(fold_values[train])
                    sc = StandardScaler().fit(imp.transform(fold_values[train]))
                    tx = sc.transform(imp.transform(fold_values[train])).astype(np.float32)
                    ex = sc.transform(imp.transform(fold_values[test])).astype(np.float32)
                    cnt = np.bincount(labels[train], minlength=n_classes).astype(np.float64)
                    sw = (1.0 / cnt)[labels[train]]; sw /= sw.mean()
                    if args.classifier == "rf":
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(
                            n_estimators=300, min_samples_leaf=2, n_jobs=int(THREADS),
                            class_weight="balanced_subsample", random_state=SEED)
                        model.fit(tx, labels[train])
                    else:
                        from xgboost import XGBClassifier
                        model = XGBClassifier(
                            n_estimators=100, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                            objective="multi:softmax", num_class=n_classes,
                            n_jobs=int(THREADS), verbosity=0, random_state=SEED)
                        model.fit(tx, labels[train], sample_weight=sw)
                    bacc = float(balanced_accuracy_score(labels[test], model.predict(ex)))
                    rows.append({"n_modalities": count, "modalities": " + ".join(names),
                                 "dimensions": n_dims, "fold": fold,
                                 "n_train": n_train, "k": 0, "bacc": bacc})
                    done.add((count, fold, n_train, 0))
                    pd.DataFrame(rows).to_csv(result_path, index=False)
                    log(f"  m={count} fold={fold} n={n_train:>6,} BAcc={100*bacc:.2f}% "
                        f"({time.perf_counter() - started:.0f}s)")
                    continue

                if args.classifier == "mlp":
                    # Um branch por modalidade, dimensão nativa (exceto
                    # Multiscale sob --multiscale-pca). Sem PCA conjunto: o
                    # ponto é medir o classificador de produção.
                    started = time.perf_counter()
                    blocks = [frame[modality_columns[name]].to_numpy(dtype=np.float32)
                              for name in plain]
                    if squeezed:
                        blocks.append(multiscale_block)
                    metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                           args.epochs, args.patience)
                    rows.append({"n_modalities": count, "modalities": " + ".join(names),
                                 "dimensions": n_dims, "fold": fold,
                                 "n_train": n_train, "k": 0, "bacc": metrics["bacc"]})
                    done.add((count, fold, n_train, 0))
                    pd.DataFrame(rows).to_csv(result_path, index=False)
                    log(f"  m={count} fold={fold} n={n_train:>6,} "
                        f"BAcc={100 * metrics['bacc']:.2f}% "
                        f"({time.perf_counter() - started:.0f}s)")
                    continue

                started = time.perf_counter()
                imputer = SimpleImputer(strategy="median").fit(fold_values[train])
                scaler = StandardScaler().fit(imputer.transform(fold_values[train]))
                train_x = scaler.transform(imputer.transform(fold_values[train]))
                test_x = scaler.transform(imputer.transform(fold_values[test]))
                pca = PCA(n_components=max_k, svd_solver="randomized",
                          random_state=SEED).fit(train_x)
                train_p, test_p = pca.transform(train_x), pca.transform(test_x)

                best = -1.0
                for k in grid:
                    if (count, fold, n_train, k) in done:
                        continue
                    model = LinearDiscriminantAnalysis(
                        n_components=min(n_classes - 1, k),
                        priors=np.ones(n_classes) / n_classes)
                    model.fit(train_p[:, :k], labels[train])
                    bacc = float(balanced_accuracy_score(
                        labels[test], model.predict(test_p[:, :k])))
                    rows.append({"n_modalities": count, "modalities": " + ".join(names),
                                 "dimensions": n_dims, "fold": fold,
                                 "n_train": n_train, "k": k, "bacc": bacc})
                    done.add((count, fold, n_train, k))
                    best = max(best, bacc)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                log(f"  m={count} fold={fold} n={n_train:>6,} pico={100 * best:.2f}% "
                    f"({time.perf_counter() - started:.0f}s)")

    detail = pd.DataFrame(rows)
    peaks = detail.loc[detail.groupby(["n_modalities", "fold", "n_train"])["bacc"].idxmax()]
    peaks.to_csv(args.output_dir / "peaks.csv", index=False, float_format="%.6f")

    summary = []
    for (count, modalities), block in peaks.groupby(["n_modalities", "modalities"]):
        curve = block.groupby("n_train")["bacc"].mean() * 100
        if len(curve) < 3:
            continue
        # Expoente na forma padrão das scaling laws: erro, não acurácia.
        x = np.log(curve.index.to_numpy(dtype=float))
        y = np.log(100 - curve.to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (intercept + slope * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        summary.append({"n_modalities": count, "modalities": modalities,
                        "error_exponent": slope, "r2": r2,
                        **{f"bacc_n{int(n)}": v for n, v in curve.items()}})
    summary = pd.DataFrame(summary)
    if summary.empty:
        log("resumo vazio (poucos tamanhos de treino para ajustar expoente)")
        return
    summary = summary.sort_values("n_modalities")
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Expoente do erro por número de modalidades ===")
    log(summary[["n_modalities", "error_exponent", "r2"]].round(4).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
