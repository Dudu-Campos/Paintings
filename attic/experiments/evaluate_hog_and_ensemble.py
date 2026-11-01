"""Duas coisas que o pipeline atual deixa na mesa, medidas sobre a config de 1.154d.

**1. HOG-PCA128.** O único número de MLP que existe para ele é de 12/ago
(`results/hog_pca128_mlp_results.csv`): −2,24pp. Mas foi medido sobre uma
configuração de **520d com baseline de 48,31%** -- anterior ao achado 11 (só
quadrantes de Descriptors pagam o branch), anterior ao L(G_felz) do achado 14,
e com a fusão ingênua que o achado 16 mostrou desperdiçar 81% do recuperável.
Na sonda LDA o mesmo bloco **ajuda** (achado 6). É o sexto caso do padrão
"LDA extrai, MLP destrói", e o único nunca reavaliado.

Aqui ele é medido sobre a config atual, com o MLP normal **e** com loss
auxiliar por branch -- a única das 8 variantes do achado 16 que funcionou
(levou o caso-gabarito de −0,38 para −0,07), e que nunca foi aplicada a
produção.

**2. Ensemble LDA + MLP.** Nunca tentado, e há evidência direta de que deveria
pagar: o achado 15 mediu as mesmas cinco representações valendo **+1,43pp no
LDA e −2,10pp no MLP**. Sinais opostos sobre as mesmas features é a condição
em que os erros são decorrelacionados. Combinação por média de probabilidade
com peso varrido, escolhido no treino de cada fold -- nunca no teste.

Saída: outputs/hog_and_ensemble/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from balanced_branch_mlp import train_balanced
from evaluate_felzenszwalb_line_graph import load_felz_line
from evaluate_full_factorial_ablation_text import load_text_block, selected_blocks
from evaluate_multiscale_by_family import BASELINE_MASK, family_quadrant_blocks
from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, ROOT, SEED, load_dataset

HOG_COMPONENTS = 128


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def metrics_of(truth, prediction) -> dict:
    return {"bacc": balanced_accuracy_score(truth, prediction),
            "accuracy": accuracy_score(truth, prediction),
            "macro_f1": f1_score(truth, prediction, average="macro")}


def standardize(blocks, train):
    """Padroniza bloco a bloco e devolve a matriz densa mais os tamanhos."""
    parts, sizes = [], []
    for block in blocks:
        imputer = SimpleImputer(strategy="median").fit(block[train])
        scaler = StandardScaler().fit(imputer.transform(block[train]))
        parts.append(scaler.transform(imputer.transform(block)).astype(np.float32))
        sizes.append(block.shape[1])
    return np.hstack(parts), sizes


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
    parser.add_argument("--hog-dir", type=Path,
                        default=ROOT / "outputs/haralick_hog_features")
    parser.add_argument("--text-cache-dir", type=Path, default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/hog_and_ensemble")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    hog_values = np.load(args.hog_dir / "features.npy")
    hog_stems = np.load(args.hog_dir / "stems.npy", allow_pickle=True).astype(str)
    hog_frame = pd.DataFrame(hog_values, columns=[f"hog_{i}" for i in range(hog_values.shape[1])])
    hog_frame.insert(0, "stem", hog_stems)
    frame = frame.merge(hog_frame.drop_duplicates("stem"), on="stem", how="inner")
    hog_block = frame[[c for c in frame.columns if c.startswith("hog_")]].to_numpy(np.float32)

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    fixed_extra = {"Line graph": frame[line_columns].to_numpy(np.float32),
                   "Felzenszwalb": frame[felz_columns].to_numpy(np.float32),
                   "Fourier": frame[fourier_columns].to_numpy(np.float32)}
    felzline_block = frame[felzline_columns].to_numpy(np.float32)
    descriptor_quadrants, _ = family_quadrant_blocks("Descriptors", group_blocks)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    log(f"{len(frame):,} imagens, {n_classes} estilos | HOG cru {hog_block.shape[1]}d")

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(r["condition"]), int(r["fold"])) for r in rows}

    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        base, _ = selected_blocks(BASELINE_MASK, group_blocks,
                                  {**fixed_extra, "Text": text_block})
        base = base + descriptor_quadrants + [felzline_block]

        # HOG comprimido no fold, como no achado 6: fit só no treino.
        imputer = SimpleImputer(strategy="median").fit(hog_block[train])
        filled = imputer.transform(hog_block)
        scaler = StandardScaler().fit(filled[train])
        reducer = PCA(n_components=HOG_COMPONENTS, svd_solver="randomized",
                      random_state=SEED).fit(scaler.transform(filled)[train])
        hog_pca = reducer.transform(scaler.transform(filled)).astype(np.float32)

        conditions = [("base", base, 0.0), ("base + HOG-PCA128", base + [hog_pca], 0.0),
                      ("base (aux loss)", base, 0.3),
                      ("base + HOG-PCA128 (aux loss)", base + [hog_pca], 0.3)]
        fold_probabilities = {}
        for name, blocks, aux in conditions:
            if (name, fold) in done and (f"LDA {name}", fold) in done:
                continue
            values, sizes = standardize(blocks, train)
            inner_train, validation = train_test_split(
                train, test_size=0.10, stratify=labels[train], random_state=SEED + fold)

            started = time.perf_counter()
            metrics, prediction, _ = train_balanced(
                values[inner_train], labels[inner_train], values[validation],
                labels[validation], values[test], labels[test], sizes, n_classes,
                SEED + fold, args.epochs, args.patience, aux_weight=aux)
            rows.append({"condition": name, "classifier": "mlp", "fold": fold,
                         "dimensions": values.shape[1], "n_branches": len(sizes),
                         "aux_weight": aux, "elapsed_seconds": time.perf_counter() - started,
                         **{k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}})
            log(f"  fold {fold} | {name:<30} MLP  BAcc={100 * metrics['bacc']:.2f}%")

            if aux == 0.0:      # LDA e ensemble só nas condições sem intervenção
                model = LinearDiscriminantAnalysis(
                    n_components=min(n_classes - 1, values.shape[1]),
                    priors=np.ones(n_classes) / n_classes).fit(values[train], labels[train])
                lda_probability = model.predict_proba(values[test])
                rows.append({"condition": f"LDA {name}", "classifier": "lda", "fold": fold,
                             "dimensions": values.shape[1], "n_branches": len(sizes),
                             "aux_weight": 0.0, "elapsed_seconds": 0.0,
                             **metrics_of(labels[test], lda_probability.argmax(1))})
                # `prediction` do MLP são rótulos; para combinar preciso da
                # distribuição, então reajusto um LDA sobre os logits não é
                # possível -- uso a predição dura do MLP como one-hot suave.
                fold_probabilities[name] = (lda_probability, prediction)

        # Ensemble. O peso NÃO é escolhido: registro a varredura inteira e o
        # número que vale é o de w=0,5, que não olha o teste para nada. O
        # máximo da varredura entra como teto-oráculo, rotulado como tal --
        # escolher w pelo BAcc do teste seria seleção no conjunto de avaliação,
        # exatamente o viés que a varredura de k já obriga a declarar.
        for name, (lda_probability, mlp_prediction) in fold_probabilities.items():
            if (f"ensemble {name} (w=0.5)", fold) in done:
                continue
            one_hot = np.eye(n_classes, dtype=np.float64)[mlp_prediction]
            sweep = []
            for weight in np.linspace(0.0, 1.0, 21):
                blended = weight * lda_probability + (1 - weight) * one_hot
                sweep.append((weight, balanced_accuracy_score(labels[test],
                                                              blended.argmax(1))))
            pd.DataFrame(sweep, columns=["weight", "bacc"]).assign(
                condition=name, fold=fold).to_csv(
                args.output_dir / f"ensemble_sweep_fold{fold}_{name.replace(' ', '_')}.csv",
                index=False, float_format="%.6f")
            half = 0.5 * lda_probability + 0.5 * one_hot
            rows.append({"condition": f"ensemble {name} (w=0.5)", "classifier": "ensemble",
                         "fold": fold, "dimensions": 0, "n_branches": 0,
                         "aux_weight": 0.5, "elapsed_seconds": 0.0,
                         **metrics_of(labels[test], half.argmax(1))})
            best_weight, best_score = max(sweep, key=lambda item: item[1])
            blended = best_weight * lda_probability + (1 - best_weight) * one_hot
            rows.append({"condition": f"ensemble {name} (oráculo)", "classifier": "ensemble",
                         "fold": fold, "dimensions": 0, "n_branches": 0,
                         "aux_weight": best_weight, "elapsed_seconds": 0.0,
                         **metrics_of(labels[test], blended.argmax(1))})
            log(f"  fold {fold} | ensemble {name:<21} "
                f"w=0.5: {100 * rows[-2]['bacc']:.2f}%  "
                f"oráculo w={best_weight:.2f}: {100 * best_score:.2f}%")
        pd.DataFrame(rows).to_csv(result_path, index=False)

    detail = pd.DataFrame(rows)
    table = detail.pivot_table(index="fold", columns="condition", values="bacc") * 100
    log("\n=== resultados (BAcc, %) ===")
    log(f"{'condição':<34}{'BAcc':>8}{'±':>7}{'vs base':>9}{'p':>8}{'folds':>7}")
    for name in table.columns:
        delta = (table[name] - table["base"]).dropna()
        p = ttest_rel(table[name], table["base"])[1] if name != "base" else float("nan")
        log(f"{name:<34}{table[name].mean():>8.2f}{table[name].std():>7.2f}"
            f"{delta.mean():>+9.2f}{p:>8.4f}{int((delta > 0).sum()):>4}/{len(delta)}")
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
