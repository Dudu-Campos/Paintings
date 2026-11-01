"""Pergunta: dá pra usar p95_r2 (achado: correlaciona forte com importância,
rho=0,717) como filtro barato e não-supervisionado de feature pro LDA?

Compara três ordens de seleção top-k no mesmo banco de 1.642 features "rich"
(Region graphs + Image statistics + Space-filling curve + Spectral):
  - p95: ranking fixo por p95_r2 (não-supervisionado, calculado uma vez no
    dataset inteiro -- mesmo cache usado no gráfico de dispersão. Ressalva:
    isso é transdutivo, igual ao aviso já registrado no achado 29 do
    CLAUDE.md sobre o agrupamento espectral -- não usa rótulo, mas viu as
    imagens de teste.)
  - supervised: coeficiente de estrutura do LDA, recalculado a cada fold
    *só* no treino (sem vazamento), é o método já estabelecido do projeto
    (evaluate_lda_topk_curve_by_subset.py).
  - random: permutação fixa, piso de comparação.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_rich_lda_structure_importance import load_rich_frame  # noqa: E402

SEED = 42
N_FOLDS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_lda(n_classes: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, n_features),
            priors=np.ones(n_classes) / n_classes,
        )),
    ])


def structure_order(values: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    model = make_lda(n_classes, values.shape[1])
    model.fit(values, labels)
    standardized = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(values)
    ).astype(np.float64)
    scores = model.named_steps["lda"].transform(standardized)
    scores -= scores.mean(axis=0)
    score_std = scores.std(axis=0)
    valid = score_std > 1e-12
    scores[:, valid] /= score_std[valid]
    scores[:, ~valid] = 0.0
    correlations = (standardized.T @ scores) / len(standardized)
    weights = model.named_steps["lda"].explained_variance_ratio_.astype(np.float64)
    weights = weights[: correlations.shape[1]]
    weights /= weights.sum()
    importance = np.sqrt(np.sum(correlations ** 2 * weights[None, :], axis=1))
    return np.argsort(-importance)


def checkpoints(n_features: int) -> list[int]:
    base = {100, 200, 400, 600, 800, 1000, 1200, 1400, n_features}
    return sorted(v for v in base if 1 <= v <= n_features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--redundancy-csv", type=Path,
                        default=ROOT / "results/rich_importance_redundancy_metric_scatter_pt.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/p95_redundancy_filter_curve.csv")
    args = parser.parse_args()

    frame, feature_names, families = load_rich_frame(args)
    redundancy = pd.read_csv(args.redundancy_csv).set_index("feature")["p95_r2"]
    missing = [name for name in feature_names if name not in redundancy.index]
    if missing:
        log(f"{len(missing)} features sem p95_r2 no CSV (provavelmente degeneradas, "
            f"ja excluidas na analise original) -- restringindo a intersecao. "
            f"ex: {missing[:5]}")
        feature_names = [name for name in feature_names if name in redundancy.index]

    values = frame[feature_names].to_numpy(dtype=np.float32)
    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    log(f"{values.shape[0]:,} imagens, {values.shape[1]}d, {n_classes} classes")

    p95 = redundancy.loc[feature_names].to_numpy(dtype=np.float64)
    p95_order = np.argsort(-p95)

    rng = np.random.default_rng(SEED)
    random_order = rng.permutation(values.shape[1])

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(values, labels))
    ks = checkpoints(values.shape[1])
    log(f"checkpoints: {ks}")

    rows = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        sup_order = structure_order(values[train_idx], labels[train_idx], n_classes)
        for k in ks:
            for method, order in (("p95", p95_order), ("supervised", sup_order), ("random", random_order)):
                columns = order[:k]
                model = make_lda(n_classes, k)
                model.fit(values[train_idx][:, columns], labels[train_idx])
                pred = model.predict(values[test_idx][:, columns])
                bacc = balanced_accuracy_score(labels[test_idx], pred)
                rows.append({"fold": fold_index, "k": k, "method": method, "balanced_accuracy": bacc})
        log(f"fold {fold_index} done ({time.strftime('%H:%M:%S')})")

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = result.groupby(["method", "k"])["balanced_accuracy"].agg(["mean", "std"]).reset_index()
    full_baseline = summary[summary["k"] == values.shape[1]]
    print()
    print("=== resumo (BAcc %, media +/- std entre 5 folds) ===")
    for method in ("p95", "supervised", "random"):
        sub = summary[summary["method"] == method].sort_values("k")
        line = "  ".join(f"k={row.k}:{row['mean']*100:.2f}" for _, row in sub.iterrows())
        print(f"{method:11s} {line}")
    print()
    print(f"baseline (todas as {values.shape[1]}d): "
          f"{full_baseline['mean'].mean()*100:.3f}%")


if __name__ == "__main__":
    main()
