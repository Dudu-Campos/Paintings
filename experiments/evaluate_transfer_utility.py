"""Quanto o pré-treino do ImageNet vale em cada domínio — o eixo que faltava.

O paper justifica o mérito de features projetadas dizendo que existem domínios
onde transfer learning não ajuda e onde há pouco dado. A segunda metade está
medida (escada de escala, cruzamento em ~97 mil imagens); a **primeira nunca
foi**. Todos os datasets testados até aqui são de arte, e em todos eles a
ResNet do ImageNet vai muito bem -- ou seja, escolhemos o terreno onde a
premissa é falsa e medimos lá.

Este script produz a coordenada que falta. A **utilidade do transfer** num
dataset é definida como

    U = BAcc(ResNet50 ImageNet, congelada) − BAcc(ResNet50 aleatória, congelada)

com arquitetura, pré-processamento, GAP, sonda e folds idênticos nos dois
braços. A única diferença é o conteúdo dos pesos, então U isola o valor do
**conhecimento transferido** -- que é exatamente a variável da premissa.

Por que a âncora aleatória e não a CNN treinada do zero: a CNN do zero difere
da congelada em pesos, otimização e capacidade de adaptar a representação ao
alvo, três coisas ao mesmo tempo. Ela responde "vale a pena treinar aqui?",
pergunta diferente e já respondida em outro lugar do paper. Rede convolucional
aleatória, além disso, não é um piso trivial: a arquitetura sozinha (filtros
aleatórios, hierarquia, pooling) já extrai textura e cor, e é justamente esse
piso que precisa ser descontado para não creditar ao ImageNet o que é da
convolução.

Protocolo: sonda LDA com pico sobre k componentes de PCA. O pico é obrigatório
porque as condições podem ter dimensão efetiva diferente, e concatenação crua
penaliza o braço mais largo -- no Rijksmuseum isso já valeu 10,12pp e inverteu
o vencedor uma vez. O k é escolhido pelo BAcc do próprio fold de teste,
identicamente em todas as condições: é otimista e **precisa estar na legenda**.

Saída: results/transfer_utility.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED, N_FOLDS = 42, 5

DATASETS = {
    "wikiart": {
        "meta": ROOT / "data/artemis/wikiart_meta.csv",
        "label": "style", "key": "stem",
        "imagenet": ROOT / "outputs/wikiart_features",
        "random": ROOT / "outputs/cnn_random_wikiart",
    },
    "pandora": {
        "meta": ROOT / "data/pandora_meta.csv",
        "label": "style", "key": "stem",
        "imagenet": ROOT / "outputs/pandora_cnn_frozen",
        "random": ROOT / "outputs/pandora_cnn_random",
    },
    "rijksmuseum": {
        "meta": ROOT / "data/rijksmuseum_labels.csv",
        "label": "period", "key": "stem",
        "imagenet": ROOT / "outputs/rijksmuseum_cnn_frozen",
        "random": ROOT / "outputs/rijksmuseum_cnn_random",
    },
    "rijksdraw": {
        "meta": ROOT / "data/rijksdraw_labels.csv",
        "label": "period", "key": "stem",
        "imagenet": ROOT / "outputs/rijksdraw_cnn_frozen",
        "random": ROOT / "outputs/rijksdraw_cnn_random",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
                        768, 1024, 1280, 1536, 2048) if k <= max_k]
    if not grid or grid[-1] != max_k:
        grid.append(max_k)
    return grid


def load_arm(directory: Path, spec: dict, minimum: int):
    values = np.load(directory / "cnn_full_features.npy")
    stems = np.load(directory / "cnn_full_stems.npy", allow_pickle=True).astype(str)
    meta = pd.read_csv(spec["meta"]).dropna(subset=[spec["label"]])
    keys = (meta[spec["key"]].astype(str).map(lambda v: Path(v).stem)
            if spec["key"] == "filename" else meta[spec["key"]].astype(str))
    mapping = dict(zip(keys, meta[spec["label"]].astype(str)))
    keep = np.array([s in mapping for s in stems])
    values, stems = values[keep], stems[keep]
    labels = np.array([mapping[s] for s in stems])
    counts = pd.Series(labels).value_counts()
    frequent = counts[counts >= minimum].index
    keep = np.isin(labels, frequent)
    return values[keep], labels[keep], stems[keep]


def probe(values, labels, n_classes, train, test):
    """LDA sobre PCA, devolvendo o melhor k e suas três métricas."""
    scaler = StandardScaler().fit(values[train])
    train_x, test_x = scaler.transform(values[train]), scaler.transform(values[test])
    limit = min(train_x.shape[1], len(train) - 1)
    pca = PCA(n_components=limit, random_state=SEED).fit(train_x)
    train_p, test_p = pca.transform(train_x), pca.transform(test_x)
    best = None
    for k in k_grid(limit):
        model = LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, k),
            priors=np.ones(n_classes) / n_classes)
        model.fit(train_p[:, :k], labels[train])
        predicted = model.predict(test_p[:, :k])
        score = balanced_accuracy_score(labels[test], predicted)
        if best is None or score > best["bacc"]:
            best = {"k": k, "bacc": score,
                    "accuracy": accuracy_score(labels[test], predicted),
                    "macro_f1": f1_score(labels[test], predicted, average="macro")}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/transfer_utility.csv")
    args = parser.parse_args()

    rows = []
    for name in args.datasets:
        spec = DATASETS[name]
        available = {arm: spec[arm] for arm in ("imagenet", "random")
                     if (spec[arm] / "cnn_full_features.npy").exists()}
        if len(available) < 2:
            log(f"{name}: faltam braços ({set(('imagenet', 'random')) - set(available)}), pulando")
            continue

        # Os dois braços precisam da MESMA coorte: se um extraiu uma imagem que
        # o outro falhou, a diferença deixa de ser só os pesos.
        arms = {arm: load_arm(path, spec, args.min_per_class)
                for arm, path in available.items()}
        shared = sorted(set.intersection(*[set(stems) for _, _, stems in arms.values()]))
        log(f"{name}: {len(shared):,} imagens em comum aos dois braços")

        reference = None
        for arm, (values, labels, stems) in arms.items():
            order = {s: i for i, s in enumerate(stems)}
            index = np.array([order[s] for s in shared])
            arms[arm] = (values[index], labels[index])
            reference = labels[index] if reference is None else reference
        assert all(np.array_equal(reference, lab) for _, lab in arms.values())

        encoded = LabelEncoder().fit_transform(reference)
        n_classes = len(np.unique(encoded))
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(encoded), dtype=np.uint8), encoded))
        for arm, (values, _) in arms.items():
            for fold, (train, test) in enumerate(splits, start=1):
                best = probe(values, encoded, n_classes, train, test)
                rows.append({"dataset": name, "arm": arm, "fold": fold,
                             "n_images": len(encoded), "n_classes": n_classes,
                             "best_k": best["k"], "bacc": best["bacc"],
                             "accuracy": best["accuracy"],
                             "macro_f1": best["macro_f1"]})
                log(f"  {name}/{arm} fold {fold} | k={best['k']:>4} "
                    f"| BAcc={100 * best['bacc']:.2f}%")

    if not rows:
        log("nada avaliado")
        return
    detail = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, float_format="%.6f")

    log("\n=== BAcc por braço (LDA sobre PCA, pico em k) ===")
    summary = (detail.groupby(["dataset", "arm"], as_index=False)
               .agg(n=("n_images", "first"), classes=("n_classes", "first"),
                    bacc=("bacc", "mean"), std=("bacc", "std")))
    summary[["bacc", "std"]] *= 100
    log(summary.round(2).to_string(index=False))

    log("\n=== Utilidade do transfer  U = ImageNet − aleatória (pp) ===")
    from scipy.stats import ttest_rel
    lines = []
    for name, group in detail.groupby("dataset"):
        wide = group.pivot(index="fold", columns="arm", values="bacc") * 100
        delta = wide["imagenet"] - wide["random"]
        lines.append({"dataset": name, "n": int(group["n_images"].iloc[0]),
                      "imagenet": wide["imagenet"].mean(),
                      "random": wide["random"].mean(),
                      "U": delta.mean(), "std": delta.std(),
                      "p": ttest_rel(wide["imagenet"], wide["random"])[1],
                      "folds": f"{int((delta > 0).sum())}/{len(delta)}"})
    log(pd.DataFrame(lines).round(3).to_string(index=False))
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
