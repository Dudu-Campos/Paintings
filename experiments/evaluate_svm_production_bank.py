"""RBF-SVM sobre o banco de produção: quanto custa tirar a rede neural?

Motivação de argumentação, não de acurácia. A objeção que a fusão com backbone
convida é "o deep learning só mudou de lugar: em vez de fine-tunar o ViT, vocês
treinam um MLP junto com ele". Com uma máquina de kernel no lugar do MLP:

* nas linhas visual-only o pipeline fica **sem nenhuma rede neural** --
  operadores determinísticos mais um classificador convexo;
* na linha de fusão o ViT continua lá, mas a fusão passa a ser concatenação
  seguida de kernel, **sem nenhum parâmetro de fusão aprendido por gradiente**.

Preço já medido em outro recorte (WikiArt, 79.207 imagens, 258 features
"Visual GIST", mesmos 5 folds): MLP 45,22 contra RBF-SVM 43,18, ou seja
**−2,04pp**. Random Forest (28,67) e kNN (28,09) desabam 16pp no mesmo vetor --
não é "qualquer clássico serve", é o SVM especificamente.

O que este script mede é se aqueles 2pp se mantêm sobre o banco **multi-bloco**
de produção. Motivo de dúvida: o §6 registra que LDA e MLP concordam em ≤0,35pp
em vetores densos e divergem ~9pp em multi-bloco, e um RBF único sobre blocos
concatenados trata todas as dimensões com a mesma escala -- a mesma patologia.
(Contra-evidência de 25/ago: otimizar 8 pesos por bloco na métrica do kNN rende
só +0,43pp de pureza, então peso igual não é tão ruim quanto parece.)

Contrapartida honesta a registrar: o SVM é **O(n²)** e adiciona uma segunda peça
de custo global ao pipeline, ao lado do grafo. O §5.4 vende custo por imagem e
limitado em resolução; isso precisa ir junto do número.

Hiperparâmetros iguais aos da medição do WikiArt (C=10, gamma="scale",
class_weight="balanced") de propósito: ajustar aqui e não lá tornaria os dois
números incomparáveis.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean  # noqa: E402

HAND = VISUAL + ["Gabor"]
# Referência do MLP nos mesmos folds (results/multitask_vit_fusion.csv).
MLP_REFERENCE = {"hand-8": 49.49, "ViT-B/16": 50.98, "hand-8 + ViT": 56.59,
                 "hand-8 + ResNet50": 54.50}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--conditions", nargs="+",
                        default=["hand-8", "ViT-B/16", "hand-8 + ViT"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/svm_production_bank.csv")
    args = parser.parse_args()

    names = HAND + ["ResNet50", "ViT-B/16"]
    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in names:
        block = load_block(args.dataset, name)
        if block is None:
            raise FileNotFoundError(name)
        frame = frame.merge(block, on="stem", how="inner")
    frame = frame.reset_index(drop=True)
    if args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in frame["style"].to_numpy(str)])
        frame = frame[mask].reset_index(drop=True)
    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(str))
    blocks = {n: clean(frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
                       .to_numpy(np.float32)) for n in names}
    log(f"{len(labels):,} imagens, {len(np.unique(labels))} classes")

    conditions = {
        "hand-8": [blocks[n] for n in HAND],
        "ViT-B/16": [blocks["ViT-B/16"]],
        "hand-8 + ViT": [blocks[n] for n in HAND] + [blocks["ViT-B/16"]],
        "hand-8 + ResNet50": [blocks[n] for n in HAND] + [blocks["ResNet50"]],
    }
    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold))
                for r in pd.read_csv(args.output).itertuples()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        for name in args.conditions:
            if (name, fold) in done:
                continue
            chosen = conditions[name]
            started = time.perf_counter()
            # Padronização por bloco ajustada só no treino do fold, como no MLP.
            scaled = np.hstack([StandardScaler().fit(b[train]).transform(b).astype(np.float32)
                                for b in chosen])
            model = SVC(C=args.c, kernel="rbf", gamma="scale",
                        class_weight="balanced", cache_size=12_000)
            model.fit(scaled[train], labels[train])
            predicted = model.predict(scaled[test])
            truth = labels[test]
            elapsed = time.perf_counter() - started
            row = {"condition": name, "fold": fold,
                   "dimensions": scaled.shape[1], "n_support": int(model.n_support_.sum()),
                   "bacc": balanced_accuracy_score(truth, predicted),
                   "accuracy": accuracy_score(truth, predicted),
                   "macro_f1": f1_score(truth, predicted, average="macro"),
                   "seconds": elapsed}
            pd.DataFrame([row]).to_csv(args.output, mode="a", index=False,
                                       header=not args.output.exists(), float_format="%.6f")
            log(f"  f{fold} {name:<18} BAcc={100 * row['bacc']:5.2f} "
                f"({scaled.shape[1]}d, {row['n_support']:,} SV, {elapsed / 60:.1f} min)")
            del scaled

    detail = pd.read_csv(args.output)
    print()
    print(f"{'condição':<20} {'dims':>6} {'SVM':>13} {'MLP':>7} {'Δ':>7} {'min/fold':>9}")
    for name in conditions:
        piece = detail[detail.condition.eq(name)]
        if piece.empty:
            continue
        values = 100 * piece["bacc"].to_numpy()
        reference = MLP_REFERENCE.get(name)
        delta = f"{values.mean() - reference:+7.2f}" if reference else "      —"
        print(f"{name:<20} {int(piece.dimensions.iloc[0]):>6} "
              f"{values.mean():6.2f} ± {values.std(ddof=1) if len(values) > 1 else 0:4.2f} "
              f"{reference if reference else float('nan'):>7.2f} {delta} "
              f"{piece['seconds'].mean() / 60:>9.1f}")


if __name__ == "__main__":
    main()
