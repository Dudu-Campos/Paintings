"""Sonda os embeddings de 2.048d congelados, todos no mesmo fold e protocolo.

Compara, no **fold 1 canônico**, quatro backbones que produzem 2.048d depois do
global average pooling:

    v1        ResNet-50 pré-treinada a regredir 607d hand-designed, sem máscara
    v2        idem, com máscara de 50% na entrada e Descriptors comprimido por
              PCA-95% (130d -> 48d), alvo de 525d
    imagenet  ResNet-50 do ImageNet, congelada -- o alvo a bater
    random    ResNet-50 de pesos aleatórios, congelada -- âncora de transfer
              zero, mede o que a arquitetura entrega sozinha

Fold 1 e só o fold 1, porque é nele que os backbones v1/v2 foram pré-treinados:
usar outro fold colocaria imagens do treino deles no teste da sonda. Os braços
`imagenet` e `random` não têm essa restrição, mas rodam no mesmo fold para a
comparação ser pareada.

Protocolo: LDA sobre PCA com **pico sobre k**, obrigatório porque a dimensão
*efetiva* difere muito entre os braços mesmo com 2.048 colunas nominais em
todos (pesos aleatórios geram features bem mais correlacionadas, e concatenação
crua penalizaria justamente eles). O k é escolhido pelo BAcc do próprio fold de
teste, identicamente em todas as condições -- é otimista e precisa estar na
legenda. O MLP de um branch entra junto porque a sonda linear achata
sistematicamente o que uma cabeça não-linear extrai.

Saída: results/pretrained_embedding_probe.csv
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_branch_architecture import (BranchMLP, log, seed_everything,  # noqa: E402
                                          train_arm)
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATES = {
    "v1 (sem máscara, alvo 607d)":
        (ROOT / "outputs/cnn_pretrained_handdesigned/features_fold1.npy",
         ROOT / "outputs/cnn_pretrained_handdesigned/stems.npy"),
    "v2 (máscara 50%, Descriptors PCA)":
        (ROOT / "outputs/cnn_pretrained_handdesigned_v2/features_fold1.npy",
         ROOT / "outputs/cnn_pretrained_handdesigned_v2/stems.npy"),
    "ImageNet congelada":
        (ROOT / "outputs/wikiart_features/cnn_full_features.npy",
         ROOT / "outputs/wikiart_features/cnn_full_stems.npy"),
    "aleatória congelada":
        (ROOT / "outputs/cnn_random_wikiart/cnn_full_features.npy",
         ROOT / "outputs/cnn_random_wikiart/cnn_full_stems.npy"),
}


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
                        768, 1024, 1280, 1536, 2048) if k <= max_k]
    if not grid or grid[-1] != max_k:
        grid.append(max_k)
    return grid


def probe_lda(values, labels, train, test, n_classes):
    scaler = StandardScaler().fit(values[train])
    train_x, test_x = scaler.transform(values[train]), scaler.transform(values[test])
    limit = min(train_x.shape[1], len(train) - 1)
    pca = PCA(n_components=limit, random_state=SEED).fit(train_x)
    train_p, test_p = pca.transform(train_x), pca.transform(test_x)
    best = None
    for k in k_grid(limit):
        model = LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, k), priors=np.ones(n_classes) / n_classes)
        model.fit(train_p[:, :k], labels[train])
        predicted = model.predict(test_p[:, :k])
        score = balanced_accuracy_score(labels[test], predicted)
        if best is None or score > best["bacc"]:
            best = {"k": k, "bacc": score,
                    "accuracy": accuracy_score(labels[test], predicted),
                    "macro_f1": f1_score(labels[test], predicted, average="macro")}
    return best


def probe_mlp(values, labels, train, test, n_classes, epochs, patience):
    fit, validation = train_test_split(
        train, test_size=0.1, stratify=labels[train], random_state=SEED + 1)
    scaler = StandardScaler().fit(values[fit])
    pieces = [scaler.transform(values[index]).astype(np.float32)
              for index in (fit, validation, test)]
    seed_everything(SEED + 1)
    model = BranchMLP([values.shape[1]], n_classes).to(DEVICE)
    metrics = train_arm(model, pieces[0], labels[fit], pieces[1], labels[validation],
                        pieces[2], labels[test], n_classes, SEED + 1, epochs, patience)
    return {"k": values.shape[1], **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--classifiers", nargs="+", default=["lda", "mlp"],
                        choices=["lda", "mlp"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/pretrained_embedding_probe.csv")
    args = parser.parse_args()

    frame, _ = load_dataset()
    labels_text = frame["label"].to_numpy(dtype=str)
    labels = LabelEncoder().fit_transform(labels_text)
    n_classes = len(np.unique(labels))
    stems = frame["stem"].to_numpy(dtype=str)
    position = {stem: index for index, stem in enumerate(stems)}

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    train, test = splits[args.fold - 1]
    log(f"coorte {len(stems):,} | fold {args.fold}: "
        f"{len(train):,} treino / {len(test):,} teste | {n_classes} classes")

    rows = []
    for name, (feature_path, stem_path) in CANDIDATES.items():
        if not feature_path.exists():
            log(f"{name}: ausente ({feature_path.name}), pulando")
            continue
        raw = np.load(feature_path)
        raw_stems = np.load(stem_path, allow_pickle=True).astype(str)
        # Realinha para a ordem da coorte; um braço pode ter sido extraído numa
        # ordem ou num superconjunto diferente.
        index = np.full(len(stems), -1, dtype=np.int64)
        for source, stem in enumerate(raw_stems):
            target = position.get(stem)
            if target is not None:
                index[target] = source
        if (index < 0).any():
            log(f"{name}: {int((index < 0).sum()):,} stems ausentes, pulando")
            continue
        values = np.ascontiguousarray(raw[index]).astype(np.float32)

        for classifier in args.classifiers:
            started = time.perf_counter()
            if classifier == "lda":
                result = probe_lda(values, labels, train, test, n_classes)
            else:
                result = probe_mlp(values, labels, train, test, n_classes,
                                   args.epochs, args.patience)
            rows.append({"embedding": name, "classifier": classifier,
                         "dims": values.shape[1], "fold": args.fold,
                         "selected_k": result["k"], "bacc": result["bacc"],
                         "accuracy": result["accuracy"],
                         "macro_f1": result["macro_f1"],
                         "elapsed": time.perf_counter() - started})
            log(f"  {name:<36} {classifier:<4} k={result['k']:>4} "
                f"BAcc={100 * result['bacc']:5.2f}%  F1={100 * result['macro_f1']:5.2f}%")

    if not rows:
        log("nada avaliado")
        return
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")
    display = table.copy()
    display[["bacc", "accuracy", "macro_f1"]] *= 100
    log("\n=== sonda dos embeddings de 2.048d (fold 1) ===")
    log(display[["embedding", "classifier", "selected_k", "bacc", "accuracy",
                 "macro_f1"]].round(2).to_string(index=False))
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
