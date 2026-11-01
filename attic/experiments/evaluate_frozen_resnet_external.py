"""Baseline de ResNet50 congelada nos datasets externos (Pandora, Rijksmuseum).

Estabelece o **alvo** antes de o lado hand-designed existir nesses datasets.
Hoje, no WikiArt, o placar é praticamente empate -- 54,42% nosso contra
53,0-55,5% da ResNet congelada, com vantagem nossa de 4,45pp em macro F1 --
e é esse o confronto que interessa, não mais a CNN treinada do zero (que em
18 mil imagens seria um baseline artificialmente fraco, já que precisaria
aprender a representação a partir do alvo; a congelada traz a dela pronta do
ImageNet e treina só a cabeça, nas mesmas amostras que nós).

Os dois datasets testam coisas diferentes de propósito:

    Pandora       mesma tarefa (estilo, 18 classes), mas **balanceado**
                  (719-1.276 por classe contra 139,9x de desbalanceamento no
                  WikiArt). Testa se os ganhos sobrevivem sem as correções de
                  desbalanceamento que o pipeline carrega.
    Rijksmuseum   tarefa **diferente**: período de meio-século, não estilo. Se
                  a complementaridade aparecer aqui também, ela deixa de ser
                  propriedade do rótulo do WikiArt.

Saída: results/frozen_resnet_external.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED, N_FOLDS = 42, 5
DATASETS = {
    "pandora": {
        "features": ROOT / "outputs/pandora_cnn_frozen",
        "meta": ROOT / "data/pandora_meta.csv",
        "label": "style",
    },
    "rijksmuseum": {
        "features": ROOT / "outputs/rijksmuseum_cnn_frozen",
        "meta": ROOT / "data/rijksmuseum_labels.csv",
        "label": "period",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load(name: str, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    spec = DATASETS[name]
    values = np.load(spec["features"] / "cnn_full_features.npy")
    stems = np.load(spec["features"] / "cnn_full_stems.npy", allow_pickle=True).astype(str)
    meta = pd.read_csv(spec["meta"]).dropna(subset=[spec["label"]])
    mapping = dict(zip(meta["stem"].astype(str), meta[spec["label"]].astype(str)))
    keep = np.array([s in mapping for s in stems])
    values, stems = values[keep], stems[keep]
    labels = np.array([mapping[s] for s in stems])
    # Classes minúsculas viram ruído em acurácia balanceada: uma classe com 3
    # exemplos contribui 1/K do score e é decidida por sorte no split.
    counts = pd.Series(labels).value_counts()
    frequent = counts[counts >= minimum].index
    keep = np.isin(labels, frequent)
    log(f"{name}: {keep.sum():,} imagens, {len(frequent)} classes "
        f"(descartadas {len(counts) - len(frequent)} com <{minimum} exemplos)")
    return values[keep], labels[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/frozen_resnet_external.csv")
    args = parser.parse_args()

    rows = []
    for name in args.datasets:
        if not (DATASETS[name]["features"] / "cnn_full_features.npy").exists():
            log(f"{name}: features ainda não extraídas, pulando")
            continue
        values, raw_labels = load(name, args.min_per_class)
        labels = LabelEncoder().fit_transform(raw_labels)
        n_classes = len(np.unique(labels))
        splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (train, test) in enumerate(splits.split(values, labels), start=1):
            scaler = StandardScaler().fit(values[train])
            train_x = scaler.transform(values[train])
            test_x = scaler.transform(values[test])
            model = LinearDiscriminantAnalysis(
                n_components=min(n_classes - 1, train_x.shape[1]),
                priors=np.ones(n_classes) / n_classes)
            model.fit(train_x, labels[train])
            predicted = model.predict(test_x)
            rows.append({
                "dataset": name, "classifier": "lda", "fold": fold,
                "n_images": len(labels), "n_classes": n_classes,
                "bacc": balanced_accuracy_score(labels[test], predicted),
                "accuracy": accuracy_score(labels[test], predicted),
                "macro_f1": f1_score(labels[test], predicted, average="macro")})
            log(f"  {name} fold {fold} | BAcc={100 * rows[-1]['bacc']:.2f}% "
                f"| F1={100 * rows[-1]['macro_f1']:.2f}%")

    if not rows:
        return
    detail = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, float_format="%.6f")
    summary = (detail.groupby(["dataset", "classifier"], as_index=False)
               .agg(n_images=("n_images", "first"), n_classes=("n_classes", "first"),
                    bacc=("bacc", "mean"), bacc_std=("bacc", "std"),
                    accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean")))
    for column in ("bacc", "bacc_std", "accuracy", "macro_f1"):
        summary[column] *= 100
    log("\n=== ResNet50 congelada (2048d) + LDA ===")
    log(summary.round(2).to_string(index=False))
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
