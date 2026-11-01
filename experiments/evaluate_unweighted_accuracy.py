"""Acurácia não balanceada, para comparação com a literatura do benchmark.

O projeto reporta **acurácia balanceada** por decisão metodológica: o corpus tem
26,8x de desbalanceamento e queremos medir estilo, não frequência. Mas a
literatura do MultitaskPainting100k ("Wikiart25/125") reporta **acurácia
simples**, e os métodos não estão disponíveis para reexecução -- então a única
via de comparação é adotar a métrica deles.

Duas mudanças em relação ao pipeline de produção, e as duas importam:

1. **Sem `class_weight`.** Treinar ponderado otimiza BAcc e *sacrifica* acurácia
   de propósito, sobre-predizendo classes raras. É por isso que a nossa
   acurácia sai MENOR que a nossa BAcc (43,40 contra 49,49 no hand-8), o
   contrário do que se espera num corpus desbalanceado.
2. **Early stopping por acurácia**, não por BAcc. Selecionar modelo por uma
   métrica e reportar outra é o mesmo erro na etapa seguinte.

Referências publicadas (acurácia), para leitura junto:

    Wikiart25   Gram_L 70,99 · Wang 69,53 · Transformer 72,25 · MSA-Net 74,37
    Wikiart125  Bianco 57,20 · Wang 56,78 · Transformer 57,79 · MSA-Net 59,55

Acurácia trivial (classe majoritária, Impressionism): 12,71% em 25 classes,
10,54% em 125.

Ressalva que o texto do artigo precisa carregar: esses métodos são redes
treinadas ponta a ponta no alvo, e as nossas condições aqui usam features
congeladas mais um MLP pequeno. A comparação de acurácia é legítima como
posicionamento, mas o degrau de supervisão é diferente -- ver §4.1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_full_factorial_ablation import scaled_splits  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean  # noqa: E402
from evaluate_robustness_mlp_aug import BranchMLP, DEVICE, seed_everything  # noqa: E402

HAND = VISUAL + ["Gabor"]
PUBLISHED = {"25": {"Gram_L": 70.99, "Wang": 69.53, "Transformer": 72.25, "MSA-Net": 74.37},
             "125": {"Bianco": 57.20, "Wang": 56.78, "Transformer": 57.79, "MSA-Net": 59.55}}
TRIVIAL = {"25": 12.71, "125": 10.54}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def fit_plain(blocks, labels, train, test, n_classes, fold, epochs=150, patience=15):
    """MLP de produção, mas SEM ponderação de classe e com seleção por acurácia."""
    fit, validation = train_test_split(train, test_size=0.1, stratify=labels[train],
                                       random_state=SEED + fold)
    X, V, T = scaled_splits(blocks, fit, validation, test)
    seed_everything(SEED + fold)
    model = BranchMLP([b.shape[1] for b in blocks], n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()          # sem weight: este é o ponto
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(X),
                                      torch.from_numpy(labels[fit]).long()),
                        batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED + fold))
    vt, tt = torch.from_numpy(V).to(DEVICE), torch.from_numpy(T).to(DEVICE)
    best, state, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predicted = model(vt).argmax(1).cpu().numpy()
        score = accuracy_score(labels[validation], predicted)   # seleção por acurácia
        if score > best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        predicted = model(tt).argmax(1).cpu().numpy()
    truth = labels[test]
    return {"accuracy": accuracy_score(truth, predicted),
            "bacc": balanced_accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", default=["25", "125"])
    parser.add_argument("--conditions", nargs="+",
                        default=["hand-8", "ResNet50", "ViT-B/16", "ResNet50 + ViT",
                                 "hand-8 + ViT", "hand-8 + ResNet50 + ViT"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/unweighted_accuracy_multitask.csv")
    args = parser.parse_args()

    names = HAND + ["ResNet50", "ViT-B/16"]
    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in names:
        block = load_block("multitask", name)
        if block is None:
            raise FileNotFoundError(name)
        frame = frame.merge(block, on="stem", how="inner")
    frame = frame.reset_index(drop=True)
    all_labels = frame["style"].to_numpy(str)
    all_blocks = {n: clean(frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
                           .to_numpy(np.float32)) for n in names}

    done = set()
    if args.output.exists():
        done = {(str(r.cut), str(r.condition), int(r.fold))
                for r in pd.read_csv(args.output, dtype={"cut": str}).itertuples()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for cut in args.cuts:
        if cut == "25":
            wanted = {key(l) for l in (ROOT / "data/multitask_25class_benchmark.txt")
                      .read_text().splitlines() if l.strip()}
            mask = np.array([key(v) in wanted for v in all_labels])
        else:
            mask = np.ones(len(all_labels), dtype=bool)
        blocks = {n: b[mask] for n, b in all_blocks.items()}
        labels = LabelEncoder().fit_transform(all_labels[mask])
        n_classes = len(np.unique(labels))
        log(f"=== {cut} classes: {len(labels):,} imagens, {n_classes} classes")
        conditions = {
            "hand-8": [blocks[n] for n in HAND],
            "ResNet50": [blocks["ResNet50"]],
            "ViT-B/16": [blocks["ViT-B/16"]],
            "ResNet50 + ViT": [blocks["ResNet50"], blocks["ViT-B/16"]],
            "hand-8 + ViT": [blocks[n] for n in HAND] + [blocks["ViT-B/16"]],
            "hand-8 + ResNet50 + ViT": [blocks[n] for n in HAND]
                                       + [blocks["ResNet50"], blocks["ViT-B/16"]],
        }
        for fold, (train, test) in enumerate(StratifiedKFold(
                5, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels), start=1):
            for name in args.conditions:
                if (cut, name, fold) in done:
                    continue
                chosen = conditions[name]
                started = time.perf_counter()
                metrics = fit_plain(chosen, labels, train, test, n_classes, fold)
                pd.DataFrame([{"cut": cut, "condition": name, "fold": fold,
                               "dimensions": sum(b.shape[1] for b in chosen),
                               "seconds": time.perf_counter() - started, **metrics}]).to_csv(
                    args.output, mode="a", index=False,
                    header=not args.output.exists(), float_format="%.6f")
                log(f"  {cut:>3}cl f{fold} {name:<24} Acc={100 * metrics['accuracy']:5.2f} "
                    f"(BAcc={100 * metrics['bacc']:5.2f}, {time.perf_counter() - started:.0f}s)")

    detail = pd.read_csv(args.output, dtype={"cut": str})
    for cut in args.cuts:
        block = detail[detail.cut.eq(cut)]
        if block.empty:
            continue
        print()
        print(f"=== Wikiart{cut} (acurácia; trivial = {TRIVIAL[cut]:.2f}%) ===")
        for name, value in PUBLISHED[cut].items():
            print(f"  {'[publicado] ' + name:<32} {value:6.2f}")
        print("  " + "-" * 40)
        for name in args.conditions:
            piece = block[block.condition.eq(name)]
            if piece.empty:
                continue
            values = 100 * piece["accuracy"].to_numpy()
            print(f"  {name:<32} {values.mean():6.2f} ± "
                  f"{values.std(ddof=1) if len(values) > 1 else 0:4.2f} "
                  f"({len(values)} folds, {int(piece.dimensions.iloc[0])}d)")


if __name__ == "__main__":
    main()
