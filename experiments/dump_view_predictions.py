"""Predições OOF por imagem, uma coluna por view, para análise de conjuntos.

O IoU de acertos entre views e o UpSet precisam de predições por imagem, e a
ablação de desempenho não as guarda. Aqui cada fold prediz o próprio teste, e a
concatenação dos cinco cobre a coorte uma vez -- OOF por construção, sem
nenhuma imagem predita por um modelo que a viu.

Um classificador só (MLP, o de produção sem branches) porque o objeto de estudo
é a **view**, não o estimador: o §ablação já mede que os cinco classificadores
ordenam as views igual (Spearman 0,984), então a estrutura de acerto/erro que
interessa é a que sobrevive a essa escolha.

O grafo de similaridade entra com a mesma construção por fold da ablação (OOF
aninhado sobre o espaço visual), porque pré-computá-lo vazaria o teste.
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
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_redundancy_selection_curve import PlainMLP  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from evaluate_standalone_by_classifier import EXTERNAL_SUPERVISION, PER_FOLD  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def predict(train_x, train_y, val_x, val_y, test_x, n_classes, seed,
            epochs=120, patience=12):
    """Igual ao `fit` da curva de seleção, mas devolve o rótulo predito em vez
    da métrica -- a métrica agregada perde exatamente a informação de qual
    imagem cada view acerta."""
    seed_everything(seed)
    model = PlainMLP(train_x.shape[1], n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weight = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tensors = [torch.tensor(a, dtype=torch.float32, device=DEVICE)
               for a in (train_x, val_x, test_x)]
    target = torch.tensor(train_y, dtype=torch.long, device=DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensors[0], target),
        batch_size=512, shuffle=True, drop_last=True)
    from sklearn.metrics import balanced_accuracy_score
    best, best_state, waited = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            criterion(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = balanced_accuracy_score(
                val_y, model(tensors[1]).argmax(1).cpu().numpy())
        if score > best:
            best, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(tensors[2]).argmax(1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/view_predictions_multitask.npz")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    if args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
    else:
        # `--classes-file none` roda a coorte inteira (multitask125). O recorte
        # de 25 é o benchmark do §7.11; sem ele o LDA deixa de ficar preso em
        # 24 componentes, que é a ressalva do §5.1. `wanted` vira o conjunto
        # completo porque ele é reusado adiante para recortar o metadado.
        wanted = {key(v) for v in raw_labels}
        mask = np.ones(len(raw_labels), dtype=bool)
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    n_classes = len(np.unique(labels))

    meta = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})
    reference = meta[[key(str(v)) in wanted for v in meta["style"]]].stem.to_numpy()
    for name, (values_path, stems_path) in {**EXTERNAL, **EXTERNAL_SUPERVISION}.items():
        values = np.load(ROOT / values_path)
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        blocks[name] = clean(values[index.to_numpy(dtype=int)])

    visual = [n for n in VISUAL + ["Gabor"] if n in blocks]
    order = visual + list(EXTERNAL) + list(EXTERNAL_SUPERVISION) + sorted(PER_FOLD)
    log(f"{len(labels):,} imagens, {n_classes} classes, {len(order)} views")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    saved = {"labels": labels, "stems": reference}
    if args.output.exists():
        saved.update({k: v for k, v in np.load(args.output, allow_pickle=True).items()})

    for name in order:
        if name in saved:
            continue
        predictions = np.full(len(labels), -1, dtype=np.int16)
        for fold, (train, test) in enumerate(splits, start=1):
            if name in PER_FOLD:
                from evaluate_production_visual_extended_similarity_graph import (
                    prior_correct_rich, rich_similarity_features)
                space = np.hstack([StandardScaler().fit(blocks[n][train])
                                   .transform(blocks[n]).astype(np.float32) for n in visual])
                oof, target = rich_similarity_features(
                    space[train], labels[train], space[test], n_classes,
                    30, 0.4, 100, SEED + fold)
                values = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
                values[train] = prior_correct_rich(oof, labels[train], n_classes)
                values[test] = prior_correct_rich(target, labels[train], n_classes)
                if "class scores" in name:
                    values = values[:, :2 * n_classes]
            else:
                values = blocks[name]
            scaled = StandardScaler().fit(values[train]).transform(values).astype(np.float32)
            fit_idx, val_idx = train_test_split(
                train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
            predictions[test] = predict(scaled[fit_idx], labels[fit_idx],
                                        scaled[val_idx], labels[val_idx],
                                        scaled[test], n_classes, SEED + fold)
        covered = predictions >= 0
        log(f"  {name:<32} acurácia {100 * (predictions[covered] == labels[covered]).mean():5.2f}"
            f"  ({covered.mean() * 100:.0f}% da coorte)")
        saved[name] = predictions
        np.savez_compressed(args.output, **saved)
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
