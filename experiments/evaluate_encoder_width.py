"""O gargalo de 128d por branch está handicapando os backbones?

O `BranchMLP` de produção projeta **todo** bloco para 128 dimensões, qualquer
que seja a entrada: o Gabor de 32d é expandido, o ViT de 768d e a ResNet de
2048d são comprimidos. Ou seja, damos ao backbone a mesma largura de canal que
damos à menor das nossas modalidades, e descartamos 640 das 768 dimensões do ViT
antes da cabeça.

Isso é ingênuo por construção -- e essa ingenuidade é a defesa do §6 contra
"afinaram a favor do próprio método". Mas ela corta contra o adversário, não
contra nós: se o ViT sobe quando recebe largura adequada, parte do nosso ganho
de fusão medido (+5,60pp) pode ser artefato de tê-lo estrangulado. Numa
comparação com SOTA isso não é aceitável.

Dois esquemas, tudo o mais igual:

* `uniforme128` -- o de produção, `Linear(d, 128)` para todo bloco.
* `sem-compressao` -- `Linear(d, min(d, 512))`: nenhum bloco é comprimido abaixo
  da própria dimensão, e os grandes são limitados em 512 para a concatenação
  não explodir. Descriptors 130->130, Gabor 32->32, ViT 768->512,
  ResNet 2048->512.

O número que decide é o **delta da fusão**, não o nível: se `hand-8 + ViT menos
ViT` sobreviver ao ViT bem servido, o argumento está blindado; se encolher
muito, precisamos saber antes do revisor.

Métrica: acurácia não balanceada e sem `class_weight`, que é a da literatura do
benchmark (ver `evaluate_unweighted_accuracy.py`). BAcc sai junto.
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_full_factorial_ablation import scaled_splits  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402

HAND = VISUAL + ["Gabor"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


class WidthBranchMLP(nn.Module):
    """BranchMLP com largura de encoder por bloco em vez de 128 fixo."""

    def __init__(self, sizes, widths, n_classes):
        super().__init__()
        self.sizes = sizes
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(s, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(0.25))
            for s, w in zip(sizes, widths)])
        previous, layers = sum(widths), []
        for width in (512, 256, 128):
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width),
                       nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        parts = torch.split(x, self.sizes, dim=1)
        return self.head(torch.cat([e(p) for e, p in zip(self.encoders, parts)], 1))


def widths_for(sizes, scheme):
    if scheme == "uniforme128":
        return [128] * len(sizes)
    return [min(s, 512) for s in sizes]


def fit(blocks, labels, train, test, n_classes, fold, scheme, epochs=150, patience=15):
    fit_idx, val_idx = train_test_split(train, test_size=0.1, stratify=labels[train],
                                        random_state=SEED + fold)
    X, V, T = scaled_splits(blocks, fit_idx, val_idx, test)
    sizes = [b.shape[1] for b in blocks]
    seed_everything(SEED + fold)
    model = WidthBranchMLP(sizes, widths_for(sizes, scheme), n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(X),
                                      torch.from_numpy(labels[fit_idx]).long()),
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
        score = accuracy_score(labels[val_idx], predicted)
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
            "macro_f1": f1_score(truth, predicted, average="macro"),
            "encoder_dims": sum(widths_for(sizes, scheme))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", nargs="+", default=["uniforme128", "sem-compressao"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/encoder_width_multitask.csv")
    args = parser.parse_args()

    names = HAND + ["ResNet50", "ViT-B/16"]
    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in names:
        frame = frame.merge(load_block("multitask", name), on="stem", how="inner")
    frame = frame.reset_index(drop=True)
    wanted = {key(l) for l in (ROOT / "data/multitask_25class_benchmark.txt")
              .read_text().splitlines() if l.strip()}
    frame = frame[[key(v) in wanted for v in frame["style"].to_numpy(str)]].reset_index(drop=True)
    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(str))
    n_classes = len(np.unique(labels))
    blocks = {n: clean(frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
                       .to_numpy(np.float32)) for n in names}
    log(f"{len(labels):,} imagens, {n_classes} classes")

    conditions = {
        "hand-8": [blocks[n] for n in HAND],
        "ViT-B/16": [blocks["ViT-B/16"]],
        "ResNet50": [blocks["ResNet50"]],
        "hand-8 + ViT": [blocks[n] for n in HAND] + [blocks["ViT-B/16"]],
        "ResNet50 + ViT": [blocks["ResNet50"], blocks["ViT-B/16"]],
        "hand-8 + ResNet50 + ViT": [blocks[n] for n in HAND]
                                   + [blocks["ResNet50"], blocks["ViT-B/16"]],
    }
    done = set()
    if args.output.exists():
        done = {(str(r.scheme), str(r.condition), int(r.fold))
                for r in pd.read_csv(args.output).itertuples()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for fold, (train, test) in enumerate(StratifiedKFold(
            5, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels), start=1):
        if fold > args.folds:
            break
        for scheme in args.schemes:
            for name, chosen in conditions.items():
                if (scheme, name, fold) in done:
                    continue
                started = time.perf_counter()
                metrics = fit(chosen, labels, train, test, n_classes, fold, scheme)
                pd.DataFrame([{"scheme": scheme, "condition": name, "fold": fold,
                               "input_dims": sum(b.shape[1] for b in chosen),
                               "seconds": time.perf_counter() - started, **metrics}]).to_csv(
                    args.output, mode="a", index=False,
                    header=not args.output.exists(), float_format="%.6f")
                log(f"  f{fold} {scheme:<14} {name:<24} Acc={100 * metrics['accuracy']:5.2f} "
                    f"(enc {metrics['encoder_dims']}d, {time.perf_counter() - started:.0f}s)")

    detail = pd.read_csv(args.output)
    print()
    pivot = detail.pivot_table(index="condition", columns="scheme", values="accuracy") * 100
    print(f"{'condição':<26} {'uniforme128':>12} {'sem-compressão':>15} {'Δ':>7}")
    for name in conditions:
        if name not in pivot.index:
            continue
        row = pivot.loc[name]
        a, b = row.get("uniforme128", np.nan), row.get("sem-compressao", np.nan)
        print(f"{name:<26} {a:>12.2f} {b:>15.2f} {b - a:>+7.2f}")
    print()
    for scheme in args.schemes:
        p = detail[detail.scheme.eq(scheme)].pivot_table(
            index="fold", columns="condition", values="accuracy") * 100
        if not {"hand-8 + ViT", "ViT-B/16"} <= set(p.columns):
            continue
        d = (p["hand-8 + ViT"] - p["ViT-B/16"]).to_numpy()
        print(f"delta da fusão (hand-8 sobre o ViT), {scheme}: "
              f"{d.mean():+.2f} ± {d.std(ddof=1):.2f}pp")


if __name__ == "__main__":
    main()
