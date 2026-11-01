"""Curva PCA -> LDA no banco de 1.642 features "rich" (mesmo carregador do
experimento de filtro p95_r2), pra estimar o teto de verdade -- metodologia
do achado 7 do CLAUDE.md (PCA decorrelaciona, ao contrario do top-k
supervisionado que fica "reto" em log-log por mais tempo e nao identifica
teto num banco finito).
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
from sklearn.decomposition import PCA
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


def checkpoints(n_features: int) -> list[int]:
    base = {20, 40, 80, 160, 320, 640, 1000, 1400, n_features}
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
    parser.add_argument("--output", type=Path, default=ROOT / "results/rich_pca_ceiling_curve.csv")
    args = parser.parse_args()

    frame, feature_names, families = load_rich_frame(args)
    values = frame[feature_names].to_numpy(dtype=np.float32)
    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    log(f"{values.shape[0]:,} imagens, {values.shape[1]}d, {n_classes} classes")

    ks = checkpoints(values.shape[1])
    log(f"checkpoints: {ks}")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(values, labels))
    rows = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, 1):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(imputer.fit_transform(values[train_idx]))
        Xte = scaler.transform(imputer.transform(values[test_idx]))

        pca = PCA(n_components=max(ks), random_state=SEED)
        Ztr_full = pca.fit_transform(Xtr)
        Zte_full = pca.transform(Xte)

        for k in ks:
            lda = LinearDiscriminantAnalysis(
                n_components=min(n_classes - 1, k),
                priors=np.ones(n_classes) / n_classes,
            )
            lda.fit(Ztr_full[:, :k], labels[train_idx])
            pred = lda.predict(Zte_full[:, :k])
            bacc = balanced_accuracy_score(labels[test_idx], pred)
            rows.append({"fold": fold_index, "k": k, "balanced_accuracy": bacc})
        log(f"fold {fold_index} done")

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result.groupby("k")["balanced_accuracy"].agg(["mean", "std"])
    print()
    print(summary.assign(**{"mean%": summary["mean"] * 100}))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
