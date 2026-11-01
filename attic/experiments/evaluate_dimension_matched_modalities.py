"""Dimensão fixa, número de modalidades variável: views reestruturam informação?

A pergunta que as nossas curvas até agora NÃO respondem. Quando a curva de
saturação por modalidade sobe, não se sabe se subiu por **informação nova** ou
só por **mais dimensões** -- as modalidades têm tamanhos diferentes, então os
dois efeitos estão colados em toda medição anterior.

Este script separa: mantém o **orçamento total de dimensões fixo** e varia o
número de modalidades. Cada uma das m modalidades é comprimida para ⌊D/m⌋
componentes de PCA (ajustado só no treino de cada fold) e concatenada. Se a
acurácia sobe com m a D constante, a informação foi *reestruturada*, não
acrescentada -- e isso não é previsível de antemão.

Braços de referência em dimensão nativa, sem compressão:
    Descriptors 130d, Fourier 152d, **Multiscale 1.032d** e as 7 juntas 1.576d.
O de Multiscale é literalmente o caso "1.000 features de uma modalidade só",
que é o contraste retórico da comparação.

Viabilidade: para m modalidades cada uma precisa ter pelo menos ⌊D/m⌋ colunas.
A menor é Hilbert com 57d e ela entra em m=5, então orçamentos grandes só
admitem m alto. O script pula automaticamente as combinações inviáveis.

Ressalva registrada: PCA preserva **variância**, não discriminabilidade -- e
nós medimos que neste pipeline o sinal mora na cauda de baixa variância
(500 componentes = 97% da variância, mas k* ≈ 1.330). Então a compressão pode
penalizar de forma desigual modalidades cuja variância e discriminabilidade se
alinham mal.

Duas checagens de robustez cobrem os dois lados desse viés, e a conclusão só
vale se sobreviver às três:

    --sampling pca          preserva variância, ignora rótulo (o padrão)
    --sampling random       ignora os dois; colunas cruas sorteadas, várias
                            sementes -- piso sem viés nenhum
    --sampling supervised   usa rótulo: top-`width` colunas por coeficiente de
                            estrutura da LDA, ajustado só no treino do fold.
                            É o teto do que a seleção por modalidade alcança,
                            e o critério é o mesmo das curvas top-k do projeto.

Se "mais modalidades ganha de mais dimensões" só aparecesse sob PCA, seria
artefato de compressão. Sob os três regimes de seleção, é a redistribuição
entre modalidades que está fazendo o trabalho.

Saída: outputs/dimension_matched/
    fold_results.csv   BAcc por (orçamento, m, fold)
    summary.csv        média por condição, ordenada
"""
from __future__ import annotations

import os

THREADS = os.environ.get("DIM_THREADS", "12")
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluate_multiscale_graph_fourier import load_fourier
from evaluate_multiscale_graph_fusion import load_graph_blocks
from final_tfidf_visual_mlp import N_FOLDS, REGIONS, ROOT, SEED, load_dataset

ORDER = ("Descriptors", "Felzenszwalb", "SLIC", "Line graph", "Hilbert",
         "Fourier", "Multiscale")
BUDGETS = (63, 126, 189, 252, 315, 399)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = np.array(frame[columns].to_numpy(dtype=np.float64), copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def fit_lda(train_x, train_y, test_x, test_y, n_classes) -> float:
    model = LinearDiscriminantAnalysis(
        n_components=min(n_classes - 1, train_x.shape[1]),
        priors=np.ones(n_classes) / n_classes)
    model.fit(train_x, train_y)
    return float(balanced_accuracy_score(test_y, model.predict(test_x)))


def structure_rank(train_x, train_y, n_classes):
    """Ordena colunas por coeficiente de estrutura da LDA (ajuste só no treino).

    Mesmo critério das curvas top-k do projeto: correlação de cada coluna com
    os discriminantes, ponderada pela variância explicada de cada um. É
    supervisionado, então não herda o viés de variância do PCA -- que é
    exatamente a ressalva registrada no cabeçalho.
    """
    model = LinearDiscriminantAnalysis(
        priors=np.ones(n_classes) / n_classes).fit(train_x, train_y)
    scores = model.transform(train_x).astype(np.float64)
    scores -= scores.mean(axis=0)
    deviation = scores.std(axis=0)
    valid = deviation > 1e-12
    scores[:, valid] /= deviation[valid]
    scores[:, ~valid] = 0.0
    correlations = (train_x.T @ scores) / len(train_x)
    weights = model.explained_variance_ratio_.astype(np.float64)[:correlations.shape[1]]
    weights /= weights.sum()
    importance = np.sqrt(np.sum(correlations ** 2 * weights[None, :], axis=1))
    return np.argsort(-importance)


def project(values, train, test, width, sampling, rng,
            train_y=None, n_classes=0, cache=None, key=None):
    """Comprime um bloco para `width` colunas, ajustando só no treino."""
    imputer = SimpleImputer(strategy="median").fit(values[train])
    scaler = StandardScaler().fit(imputer.transform(values[train]))
    train_x = scaler.transform(imputer.transform(values[train]))
    test_x = scaler.transform(imputer.transform(values[test]))
    if width >= values.shape[1]:
        return train_x, test_x
    if sampling == "random":
        # Sem viés de variância: sorteia colunas cruas.
        picked = rng.choice(values.shape[1], width, replace=False)
        return train_x[:, picked], test_x[:, picked]
    if sampling == "supervised":
        # O ranking depende de (modalidade, fold), não do orçamento: cachear
        # evita reajustar a mesma LDA uma vez por orçamento.
        order = None if cache is None else cache.get(key)
        if order is None:
            order = structure_rank(train_x, train_y, n_classes)
            if cache is not None:
                cache[key] = order
        picked = order[:width]
        return train_x[:, picked], test_x[:, picked]
    pca = PCA(n_components=width, svd_solver="randomized", random_state=SEED).fit(train_x)
    return pca.transform(train_x), pca.transform(test_x)


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dimension_matched")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--sampling", choices=["pca", "random", "supervised"],
                        default="pca")
    parser.add_argument("--seeds", type=int, default=1,
                        help="repetições (só faz sentido com --sampling random)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    quadrant_columns = [c for family in ("SLIC", "Descriptors", "Hilbert")
                        for region in REGIONS for c in groups[f"{family} ({region})"]]
    modality = {
        "Descriptors":  prepare(frame, groups["Descriptors"]),
        "Felzenszwalb": prepare(frame, felz_columns),
        "SLIC":         prepare(frame, groups["SLIC"]),
        "Line graph":   prepare(frame, line_columns),
        "Hilbert":      prepare(frame, groups["Hilbert"]),
        "Fourier":      prepare(frame, fourier_columns),
        "Multiscale":   prepare(frame, quadrant_columns),
    }
    labels = LabelEncoder().fit_transform(frame.label.to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    log(f"{len(frame):,} imagens, {n_classes} estilos, amostragem={args.sampling}")
    for name in ORDER:
        log(f"  {name:<14} {modality[name].shape[1]:5d}d")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(r["condition"]), int(r["fold"]), int(r["seed"])) for r in rows}

    # Braços de referência em dimensão nativa.
    references = [("nativo: Descriptors (130d)", ["Descriptors"]),
                  ("nativo: Fourier (152d)", ["Fourier"]),
                  ("nativo: Multiscale (1032d)", ["Multiscale"]),
                  ("nativo: as 7 (1576d)", list(ORDER))]

    rank_cache: dict = {}

    def run(condition, names, width, fold, train, test, seed):
        if (condition, fold, seed) in done:
            return
        rng = np.random.default_rng(SEED + 1000 * seed + fold)
        parts_train, parts_test = [], []
        for name in names:
            a, b = project(modality[name], train, test,
                           width or modality[name].shape[1], args.sampling, rng,
                           labels[train], n_classes, rank_cache, (name, fold))
            parts_train.append(a); parts_test.append(b)
        train_x = np.hstack(parts_train); test_x = np.hstack(parts_test)
        started = time.perf_counter()
        bacc = fit_lda(train_x, labels[train], test_x, labels[test], n_classes)
        rows.append({"condition": condition, "n_modalities": len(names),
                     "budget": train_x.shape[1], "per_modality": width or 0,
                     "fold": fold, "seed": seed, "bacc": bacc,
                     "elapsed_seconds": time.perf_counter() - started})
        done.add((condition, fold, seed))
        pd.DataFrame(rows).to_csv(result_path, index=False)
        log(f"  {condition:<34} fold={fold} s={seed} "
            f"{train_x.shape[1]:5d}d BAcc={bacc:.4f}")

    for fold, (train, test) in enumerate(splits, start=1):
        for condition, names in references:
            run(condition, names, None, fold, train, test, 0)
        for budget in args.budgets:
            for count in range(1, len(ORDER) + 1):
                names = list(ORDER[:count])
                width = budget // count
                if width < 1 or any(modality[n].shape[1] < width for n in names):
                    continue      # inviável: alguma modalidade é menor que a fatia
                condition = f"D={budget}, m={count} ({width}/mod)"
                for seed in range(args.seeds):
                    run(condition, names, width, fold, train, test, seed)

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["condition", "n_modalities", "budget", "per_modality"],
                              as_index=False)
               .agg(bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
                    n=("bacc", "size")))
    summary[["bacc_mean", "bacc_std"]] *= 100
    summary = summary.sort_values(["budget", "n_modalities"])
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.6f")
    log("\n=== BAcc por orçamento e número de modalidades ===")
    log(summary.round(3).to_string(index=False))
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
