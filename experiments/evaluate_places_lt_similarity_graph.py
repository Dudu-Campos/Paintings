"""O grafo de similaridade no Places-LT, com breakdown many/medium/few.

Por que este script existe em vez de `evaluate_multitask_similarity_graph.py`:

1. **Aquele usa o BranchMLP estreito** (enc 128, cabeça 512-256-128). Com 365
   classes o posto da matriz de logits fica limitado pela penúltima camada de
   128 (gargalo do softmax, Yang et al., ICLR 2018) e a arquitetura estreita
   custa ~5,5pp aqui. Uma medição de "o grafo paga?" sob um teto estrutural
   mede o teto, não o grafo.

2. **Falta o breakdown por faixa**, que é a pergunta. O breakdown de 25/ago
   mostrou que o ganho da agregação **inverte de sinal na cauda**: +9,02pp em
   `many`, +6,53 em `medium` e **−3,06 em `few`**. São 804 imagens de treino
   para 71 classes (11 por classe), e um encoder multi-branch de 607d não tem o
   que ajustar com isso.

   Isso é a previsão do achado 5.1 levada ao limite: se mais representações
   aumentam o expoente do erro contra o tamanho do treino, então no regime de
   pouquíssimo dado elas têm de ajudar menos, e em algum ponto prejudicar.

O grafo é um operador de pares -- kNN sobre as modalidades concatenadas, com os
rótulos dos vizinhos agregados por log-lift. A hipótese a testar é que ele
sobrevive onde o encoder não sobrevive, porque não precisa ajustar parâmetro
por classe: 11 exemplos viram 11 x n comparações.

Ressalva medida antes de rodar: o bloco sai com 2*n_classes + 30 = **760d** nas
365 classes, maior que as 607d de todo o banco visual. Se a cauda quebrar por
capacidade, um bloco desse tamanho pode piorar em vez de salvar -- e essa é
justamente a distinção entre "o grafo resolve" e "é preciso a formulação por
pares de verdade".
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import (  # noqa: E402
    CONDITIONS, SEED, VISUAL, WideBranchMLP, clean, load_corpus)
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features)

# Convenção do protocolo LT (Liu et al., CVPR 2019).
BANDS = (("many", 100, np.inf), ("medium", 20, 100), ("few", 0, 20))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def band_of(frequency: np.ndarray) -> np.ndarray:
    out = np.empty(len(frequency), dtype=object)
    for name, low, high in BANDS:
        out[(frequency > low) & (frequency <= high)] = name
    out[frequency > 100] = "many"
    return out


def fit_and_recall(blocks, labels, fit_idx, val_idx, test_idx, n_classes, seed):
    """Recall por classe no teste, para o MLP largo. Devolve vetor (n_classes,)."""
    import copy

    import torch
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score
    from torch.utils.data import DataLoader, TensorDataset

    from evaluate_full_factorial_ablation import scaled_splits
    from evaluate_robustness_mlp_aug import DEVICE, seed_everything

    X, V, T = scaled_splits(blocks, fit_idx, val_idx, test_idx)
    seed_everything(seed)
    model = WideBranchMLP([b.shape[1] for b in blocks], n_classes).to(DEVICE)
    counts = np.bincount(labels[fit_idx], minlength=n_classes).astype(np.float32)
    weight = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    loader = DataLoader(TensorDataset(torch.from_numpy(X),
                                      torch.from_numpy(labels[fit_idx]).long()),
                        batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_tensor = torch.from_numpy(V).to(DEVICE)
    test_tensor = torch.from_numpy(T).to(DEVICE)
    best, state, stale = -1.0, None, 0
    for _ in range(150):
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
        if stale >= 15:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        predicted = model(test_tensor).argmax(1).cpu().numpy()
    truth = labels[test_idx]
    return np.array([(predicted[truth == c] == c).mean() if (truth == c).any() else np.nan
                     for c in range(n_classes)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/places_lt_similarity_graph")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.output_dir / "graph_block.npy"
    detail_path = args.output_dir / "band_recall.csv"

    names = VISUAL + ["ResNet50"]
    blocks, raw_labels, frame = load_corpus("places_lt", names)
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    test = np.flatnonzero(frame["split"].eq("test").to_numpy())
    fit_idx, val_idx = train_test_split(train, test_size=0.1,
                                        stratify=labels[train], random_state=SEED)
    frequency = np.bincount(labels[train], minlength=n_classes)
    band = band_of(frequency)
    log(f"{len(labels):,} imagens, {n_classes} classes, "
        f"train={len(train):,} test={len(test):,}")
    log("  " + " | ".join(f"{b}: {int((band == b).sum())} classes, "
                          f"{int(frequency[band == b].sum()):,} img"
                          for b, _, _ in BANDS))

    # Espaço do grafo: as 7 visuais, sem a ResNet -- é a convenção de produção,
    # e misturar a CNN aqui contaminaria o grafo com supervisão externa.
    if graph_path.exists():
        graph_block = np.load(graph_path)
        log(f"grafo em cache: {graph_block.shape[1]}d")
    else:
        started = time.perf_counter()
        scaled = [StandardScaler().fit(blocks[n][train]).transform(blocks[n]).astype(np.float32)
                  for n in VISUAL]
        graph_space = np.hstack(scaled)
        del scaled
        log(f"espaço do grafo: {graph_space.shape[1]}d, k={args.n_neighbors}, "
            f"alpha={args.alpha}, nested OOF + log-lift")
        oof, target = rich_similarity_features(
            graph_space[train], labels[train], graph_space[test], n_classes,
            args.n_neighbors, args.alpha, args.max_iter, SEED)
        graph_block = np.zeros((len(labels), oof.shape[1]), dtype=np.float32)
        graph_block[train] = prior_correct_rich(oof, labels[train], n_classes)
        graph_block[test] = prior_correct_rich(target, labels[train], n_classes)
        del graph_space, oof, target
        np.save(graph_path, graph_block)
        log(f"grafo pronto: {graph_block.shape[1]}d em "
            f"{(time.perf_counter() - started) / 60:.1f} min")

    conditions = {
        "descriptors": [blocks["Descriptors"]],
        "hand": [blocks[n] for n in VISUAL],
        "grafo só": [graph_block],
        "descriptors+grafo": [blocks["Descriptors"], graph_block],
        "hand+grafo": [blocks[n] for n in VISUAL] + [graph_block],
        "resnet": [blocks["ResNet50"]],
    }
    done = set()
    if detail_path.exists():
        done = set(pd.read_csv(detail_path)["condition"])

    for name, chosen in conditions.items():
        if name in done:
            continue
        started = time.perf_counter()
        recall = fit_and_recall(chosen, labels, fit_idx, val_idx, test,
                                n_classes, seed=SEED)
        row = {"condition": name, "dimensions": sum(b.shape[1] for b in chosen),
               "seconds": time.perf_counter() - started,
               "overall": np.nanmean(recall)}
        row.update({b: np.nanmean(recall[band == b]) for b, _, _ in BANDS})
        pd.DataFrame([row]).to_csv(detail_path, mode="a", index=False,
                                   header=not detail_path.exists(), float_format="%.6f")
        np.save(args.output_dir / f"recall_{name.replace(' ', '_').replace('+', '_')}.npy", recall)
        log(f"  {name:<18} {row['dimensions']:>5}d  total {100 * row['overall']:5.2f} | " +
            " ".join(f"{b} {100 * row[b]:5.2f}" for b, _, _ in BANDS))

    detail = pd.read_csv(detail_path).drop_duplicates("condition", keep="last")
    print()
    print(f"{'condição':<20} {'dims':>6} {'total':>7} {'many':>7} {'medium':>7} {'few':>7}")
    for row in detail.itertuples():
        print(f"{row.condition:<20} {int(row.dimensions):>6} {100 * row.overall:>7.2f} "
              f"{100 * row.many:>7.2f} {100 * row.medium:>7.2f} {100 * row.few:>7.2f}")


if __name__ == "__main__":
    main()
