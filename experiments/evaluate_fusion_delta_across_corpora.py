"""Delta absoluto da fusão (ResNet50 + hand-designed) nos cinco corpora.

Motivação: as três condições já existiam espalhadas em `results/`, mas medidas
com coortes, conjuntos de blocos (8 vs 9) e arquiteturas de MLP diferentes --
o WikiArt/Multitask/OilPainting saíram do MLP estreito (enc 128, head
512-256-128) e o Food-101 do corrigido em largura. Comparar **delta absoluto**
entre corpora exige que a única coisa a variar seja o corpus.

Aqui: mesmas 8 modalidades visuais, mesmo MLP, três condições por corpus.

* `hand`   -- as 8 modalidades visuais
* `resnet` -- ResNet50 congelada (2048d) sozinha
* `fusion` -- as duas juntas, um branch por bloco

A largura do MLP é a corrigida (enc 512, head 2048-1024-1024). O gargalo do
softmax (Yang et al., ICLR 2018) limita o posto da matriz de logits à largura
da penúltima camada; com 128 e 365 classes o Places-LT era estruturalmente
incapaz de representar as distribuições, e a arquitetura estreita custava
5,5pp lá contra ~0 nos corpora de arte. Uma comparação de deltas absolutos
entre corpora com número de classes de 17 a 365 não pode carregar esse viés.

Protocolo por corpus: os de arte usam StratifiedKFold(5, seed=42); Food-101 e
Places-LT usam a partição oficial publicada, que é o único caminho para número
comparável com a literatura. A validação interna é sempre 10% do treino.

Saída incremental e idempotente (a absinto mata jobs longos por SIGKILL).
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
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402

SEED, N_FOLDS = 42, 5
# As 7 modalidades presentes nos CINCO corpora. Gabor fica de fora: nunca foi
# extraído para o WikiArt e as imagens daquele corpus não estão mais em disco,
# então incluí-lo custaria a comparabilidade que é o ponto desta medição. São
# 32d de 671 e a escada do Food-101 mediu +0,89pp para ele.
from corpora import VISUAL, clean  # noqa: E402,F401

# corpus -> (protocolo, argumento). "split" usa a coluna `split` do metadado.
CORPORA = {
    "wikiart": ("kfold", None),
    "multitask": ("kfold", "data/multitask_25class_benchmark.txt"),
    "oilpainting": ("kfold", None),
    "food101": ("split", None),
    "places_lt": ("split", None),
}
# `descriptors` é o ponto de partida da agregação: cor + LBP + Haralick sobre a
# imagem inteira, isto é, o que se teria sem nenhuma das outras views. O ganho
# `hand − descriptors` é o que a agregação de representações compra, e é essa a
# grandeza a comparar entre corpora -- não o delta da fusão com a CNN.
CONDITIONS = {"descriptors": ["Descriptors"], "hand": VISUAL,
              "resnet": ["ResNet50"], "fusion": VISUAL + ["ResNet50"]}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class WideBranchMLP(nn.Module):
    """Um encoder por modalidade, concatenados numa cabeça compartilhada.

    Larguras parametrizadas: o BranchMLP de produção fixa enc=128 e cabeça
    512-256-128, o que estrangula corpora com muitas classes (ver docstring do
    módulo).
    """

    def __init__(self, sizes, n_classes, enc=512, head=(2048, 1024, 1024)):
        super().__init__()
        self.sizes = sizes
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(s, enc), nn.BatchNorm1d(enc), nn.GELU(), nn.Dropout(0.25))
            for s in sizes])
        layers, previous = [], enc * len(sizes)
        for width in head:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width),
                       nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        parts = torch.split(x, self.sizes, dim=1)
        return self.head(torch.cat([e(p) for e, p in zip(self.encoders, parts)], dim=1))


def load_corpus(dataset: str, names: list[str]) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Blocos alinhados por `stem`, na interseção de todos eles."""
    if dataset == "wikiart":
        from registry import load_blocks
        cohort = load_blocks("wikiart", names, allow_external=True)
        return ({n: cohort.blocks[n].astype(np.float32) for n in names},
                np.asarray(cohort.labels), pd.DataFrame({"stem": cohort.stems}))

    meta_path, label_column = {
        "multitask": ("data/multitask_meta.csv", "style"),
        "oilpainting": ("data/oilpainting_meta.csv", "style"),
        "food101": ("data/food101_meta.csv", "style"),
        "places_lt": ("data/places_lt_meta.csv", "style"),
    }[dataset]
    keep = ["stem", label_column] + (["split"] if CORPORA[dataset][0] == "split" else [])
    frame = pd.read_csv(ROOT / meta_path, dtype={"stem": str})[keep]
    for name in names:
        block = load_block(dataset, name)
        if block is None:
            raise FileNotFoundError(f"{dataset}: bloco {name} ausente")
        frame = frame.merge(block, on="stem", how="inner")
    frame = frame.reset_index(drop=True)
    blocks = {n: frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
              .to_numpy(np.float32) for n in names}
    return blocks, frame[label_column].to_numpy(str), frame


def fit_mlp(blocks, labels, fit_idx, val_idx, test_idx, n_classes, seed,
            epochs=150, patience=15) -> dict:
    X, V, T = scaled_splits(blocks, fit_idx, val_idx, test_idx)
    seed_everything(seed)
    model = WideBranchMLP([b.shape[1] for b in blocks], n_classes).to(DEVICE)
    counts = np.bincount(labels[fit_idx], minlength=n_classes).astype(np.float32)
    weight = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(X),
                                      torch.from_numpy(labels[fit_idx]).long()),
                        batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_tensor = torch.from_numpy(V).to(DEVICE)
    test_tensor = torch.from_numpy(T).to(DEVICE)
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
            predicted = model(val_tensor).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(labels[val_idx], predicted)
        if score > best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        predicted = model(test_tensor).argmax(1).cpu().numpy()
    truth = labels[test_idx]
    return {"bacc": balanced_accuracy_score(truth, predicted),
            "accuracy": accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro"),
            "dimensions": sum(b.shape[1] for b in blocks)}


def splits_for(dataset: str, labels: np.ndarray, frame: pd.DataFrame):
    """(fold, train, test) pelo protocolo do corpus."""
    if CORPORA[dataset][0] == "split":
        train = np.flatnonzero(frame["split"].eq("train").to_numpy())
        test = np.flatnonzero(frame["split"].eq("test").to_numpy())
        return [(0, train, test)]
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    return [(i, tr, te) for i, (tr, te) in
            enumerate(folds.split(np.zeros(len(labels)), labels), start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="+", default=list(CORPORA))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/fusion_delta_across_corpora.csv")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.dataset), str(r.condition), int(r.fold)) for r in previous.itertuples()}

    for dataset in args.corpora:
        wanted = sorted({n for c in args.conditions for n in CONDITIONS[c]},
                        key=(VISUAL + ["ResNet50"]).index)
        try:
            blocks, raw_labels, frame = load_corpus(dataset, wanted)
        except Exception as error:
            log(f"{dataset}: PULADO ({type(error).__name__}: {error})")
            continue

        classes_file = CORPORA[dataset][1]
        if classes_file:
            # Três divergências entre o arquivo do benchmark e o metadado, e
            # nenhuma delas dá erro por conta própria -- a falha é silenciosa e
            # o script rodaria com o corpus errado:
            #   "Abstract Art"            vs "Abstract_Art"      (separador)
            #   "Analytical\u00a0Realism"    vs espaço comum        (NBSP)
            #   "Naive Art (Primitivism)" vs "Naïve_Art_(...)"   (trema)
            # Sem normalizar as três casam 13 das 25 classes; sem a terceira,
            # 24 -- e 24 de 25 é exatamente o erro que passa batido.
            def key(value: str) -> str:
                folded = unicodedata.normalize("NFKD", value)
                folded = "".join(c for c in folded if not unicodedata.combining(c))
                return re.sub(r"[\s_]+", "_", folded.strip()).lower()
            keep_classes = {key(l) for l in (ROOT / classes_file).read_text().splitlines()
                            if l.strip()}
            mask = np.array([key(v) in keep_classes for v in raw_labels])
            if mask.sum() == 0 or len(np.unique(raw_labels[mask])) != len(keep_classes):
                raise ValueError(f"{dataset}: casaram "
                                 f"{len(np.unique(raw_labels[mask]))} de "
                                 f"{len(keep_classes)} classes do benchmark")
            blocks = {n: b[mask] for n, b in blocks.items()}
            raw_labels, frame = raw_labels[mask], frame[mask].reset_index(drop=True)

        blocks = {n: clean(b) for n, b in blocks.items()}
        labels = LabelEncoder().fit_transform(raw_labels)
        n_classes = len(np.unique(labels))
        log(f"{dataset}: {len(labels):,} imagens, {n_classes} classes, "
            f"{sum(b.shape[1] for b in blocks.values())}d em {len(blocks)} blocos")

        for fold, train, test in splits_for(dataset, labels, frame):
            fit_idx, val_idx = train_test_split(
                train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
            for condition in args.conditions:
                if (dataset, condition, fold) in done:
                    continue
                chosen = [blocks[n] for n in CONDITIONS[condition]]
                started = time.perf_counter()
                metrics = fit_mlp(chosen, labels, fit_idx, val_idx, test,
                                  n_classes, seed=SEED + fold)
                pd.DataFrame([{"dataset": dataset, "condition": condition, "fold": fold,
                               "n_train": len(train), "n_test": len(test),
                               "n_classes": n_classes, "seconds": time.perf_counter() - started,
                               **metrics}]).to_csv(
                    args.output, mode="a", index=False,
                    header=not args.output.exists(), float_format="%.6f")
                log(f"  {dataset}/{condition}/f{fold}: BAcc {100 * metrics['bacc']:.2f} "
                    f"({metrics['dimensions']}d, {time.perf_counter() - started:.0f}s)")

    summarize(args.output)


def summarize(path: Path) -> None:
    if not path.exists():
        return
    detail = pd.read_csv(path)
    print()
    print(f"{'corpus':<13} {'classes':>7} {'treino':>8} "
          f"{'hand':>13} {'ResNet50':>13} {'fusão':>13} {'Δ fusão':>9}")
    for dataset in CORPORA:
        block = detail[detail.dataset.eq(dataset)]
        if block.empty:
            continue
        pivot = block.pivot_table(index="fold", columns="condition", values="bacc")
        cells = {}
        for condition in ("hand", "resnet", "fusion"):
            if condition not in pivot:
                cells[condition] = "     —      "
                continue
            values = 100 * pivot[condition].to_numpy()
            cells[condition] = (f"{values.mean():6.2f}±{values.std(ddof=1):4.2f}"
                                if len(values) > 1 else f"{values.mean():6.2f}      ")
        delta = ""
        if {"fusion", "resnet"} <= set(pivot.columns):
            paired = 100 * (pivot["fusion"] - pivot["resnet"]).to_numpy()
            delta = f"{paired.mean():+6.2f}"
        row = block.iloc[0]
        print(f"{dataset:<13} {int(row.n_classes):>7} {int(row.n_train):>8,} "
              f"{cells['hand']:>13} {cells['resnet']:>13} {cells['fusion']:>13} {delta:>9}")


if __name__ == "__main__":
    main()
