"""Redundância linear prevê contribuição? Curvas de seleção incremental.

O §6 registra que **redundância não prevê contribuição**, e o instrumento
principal foi o Shapley exato sobre 2^9 máscaras -- calculado com **LDA**
(`evaluate_factorial_shapley_multitask.py`, `LinearDiscriminantAnalysis`). Isso
é mais forte do que parece: num classificador linear uma feature linearmente
dependente contribui *exatamente* zero, então o r² falhou justamente onde tinha
mais chance de acertar. A explicação não é a não-linearidade -- é que R² mede
redundância de **variância**, não de **direção discriminante**: Hilbert tem
R²=0,919 contra o resto, e os 8,1% residuais podem estar quase inteiramente
alinhados com uma fronteira de classe.

Este experimento testa a mesma pergunta um nível abaixo -- na **feature**, não
no bloco -- com seleção incremental controlada e um classificador não-linear.
Quatro ordens de seleção, mesma arquitetura, mesmos folds:

  `random`      ordem uniforme (a linha de base honesta)
  `min-r2`      guloso: a próxima feature minimiza o r² médio contra as já
                escolhidas -- diversidade máxima
  `max-r2`      guloso: maximiza -- redundância máxima
  `max-r2-strat` idem, mas obrigado a rodízio entre as 8 modalidades

O quarto braço existe por um confundidor que invalidaria os outros três: as
features mutuamente mais redundantes do banco são os bins de Fourier entre si e
os quadrantes de Descriptors, então `max-r2` puro tende a colapsar num bloco só.
Sem o braço estratificado, uma diferença entre as curvas mediria "um bloco
contra oito" (§4.3 outra vez) e não redundância. A composição por bloco do
conjunto selecionado sai no CSV justamente para que essa leitura seja
verificável em vez de suposta.

MLP **sem branches** de propósito: a estrutura por bloco imporia identidade de
modalidade, e a pergunta aqui é sobre features.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class PlainMLP(nn.Module):
    """Sem encoder por bloco: a entrada é o vetor selecionado, ponto.

    `first_width` existe para o controle que separa as duas explicações do ganho
    das branches: o BranchMLP expande 639 -> 1024 (8 encoders x 128) enquanto o
    denso vai 639 -> 512, então parte do +1,96pp pode ser largura e não estrutura
    bloco-diagonal. Com 1024 aqui, as duas redes têm a mesma largura na primeira
    camada e o denso ainda leva 4x mais parâmetros nela.
    """

    def __init__(self, n_features: int, n_classes: int, first_width: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, first_width), nn.BatchNorm1d(first_width),
            nn.GELU(), nn.Dropout(0.35),
            nn.Linear(first_width, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def r2_matrix(values: np.ndarray) -> np.ndarray:
    """r² par-a-par. É o instrumento nº 1 do §6, recalculado por fold sobre as
    linhas de treino -- não supervisionado, mas manter fora do teste evita que a
    ordem de seleção veja o conjunto de avaliação."""
    standard = values - values.mean(axis=0)
    deviation = standard.std(axis=0)
    standard /= np.where(deviation > 0, deviation, 1.0)
    correlation = (standard.T @ standard) / len(standard)
    return np.square(np.nan_to_num(correlation, nan=0.0)).astype(np.float32)


def select(order: str, r2: np.ndarray, block_of: np.ndarray, budget: int,
           rng: np.random.Generator) -> list[int]:
    n_features = r2.shape[0]
    if order == "random":
        return list(rng.permutation(n_features)[:budget])

    chosen = [int(rng.integers(n_features))]
    # Soma corrente do r² contra os escolhidos: evita recomputar a média a cada
    # passo, o que tornaria o guloso O(d^2 * budget).
    affinity = r2[chosen[0]].copy()
    taken = np.zeros(n_features, dtype=bool)
    taken[chosen[0]] = True
    sign = -1.0 if order == "min-r2" else 1.0
    while len(chosen) < budget:
        score = sign * affinity
        score[taken] = -np.inf
        if order == "max-r2-strat":
            # Rodízio: só concorrem as features do bloco da vez, o que impede o
            # guloso de colapsar num bloco só sem mudar o critério dentro dele.
            wanted = len(chosen) % (int(block_of.max()) + 1)
            eligible = (block_of == wanted) & ~taken
            if eligible.any():
                score = np.where(eligible, score, -np.inf)
        pick = int(np.argmax(score))
        chosen.append(pick)
        taken[pick] = True
        affinity += r2[pick]
    return chosen


def fit(train_x, train_y, val_x, val_y, test_x, test_y, n_classes, seed,
        epochs=120, patience=12, first_width=512):
    seed_everything(seed)
    model = PlainMLP(train_x.shape[1], n_classes, first_width).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weight = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tensors = [torch.tensor(a, dtype=torch.float32, device=DEVICE)
               for a in (train_x, val_x, test_x)]
    targets = [torch.tensor(a, dtype=torch.long, device=DEVICE)
               for a in (train_y, val_y, test_y)]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensors[0], targets[0]),
        batch_size=512, shuffle=True, drop_last=True)

    best, best_state, waited = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            criterion(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(tensors[1]).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(val_y, prediction)
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
        prediction = model(tensors[2]).argmax(1).cpu().numpy()
    return {"bacc": 100 * balanced_accuracy_score(test_y, prediction),
            "accuracy": 100 * accuracy_score(test_y, prediction),
            "macro_f1": 100 * f1_score(test_y, prediction, average="macro"),
            "validation_bacc": 100 * best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256, 512, 639])
    parser.add_argument("--orders", nargs="+",
                        default=["random", "min-r2", "max-r2", "max-r2-strat"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-width", type=int, default=512)
    parser.add_argument("--pool", choices=["bank", "descriptors"], default="bank",
                        help="`bank` sorteia das 8 modalidades; `descriptors` "
                             "restringe às 130 de Descriptors, estratificando "
                             "pelos SUB-grupos dela (Lab hist L/a/b, Lab stats, "
                             "LBP, Haralick). O braço restrito tem de usar o "
                             "mesmo critério, senão a diferença mistura pool "
                             "com critério de seleção.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/redundancy_selection_curve.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus(args.dataset, VISUAL + ["Gabor"])
    if args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("casamento de classes do benchmark incompleto")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    order_names = [n for n in VISUAL + ["Gabor"] if n in blocks]
    features = np.hstack([blocks[n] for n in order_names])
    block_of = np.concatenate([np.full(blocks[n].shape[1], i)
                               for i, n in enumerate(order_names)])
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))

    if args.pool == "descriptors":
        # Fronteiras conferidas em results/feature_importance_vs_redundancy_pt.csv:
        # Lab histogram L/a/b 30 cada, Lab statistics 9, LBP 18, Haralick/GLCM 13.
        bounds = [(0, 30), (30, 60), (60, 90), (90, 99), (99, 117), (117, 130)]
        if blocks[order_names[0]].shape[1] != 130 or order_names[0] != "Descriptors":
            raise ValueError("pool=descriptors espera Descriptors (130d) como primeiro bloco")
        features = features[:, :130]
        block_of = np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(bounds)])
        order_names = [f"Descriptors[{a}:{b}]" for a, b in bounds]

    log(f"{args.dataset}: {features.shape[0]:,} imagens, {features.shape[1]} features, "
        f"{n_classes} classes, pool={args.pool}, {len(order_names)} grupos")

    rows = []
    done = set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(int(r["fold"]), str(r["order"]), int(r["n_features"]), int(r["seed"]))
                for r in rows}

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    for fold, (train_index, test_index) in enumerate(splits[:args.folds], start=1):
        fit_idx, val_idx = train_test_split(
            train_index, test_size=0.1, stratify=labels[train_index],
            random_state=SEED + fold)
        scaler = StandardScaler().fit(features[train_index])
        scaled = scaler.transform(features).astype(np.float32)
        r2 = r2_matrix(scaled[train_index])
        log(f"fold {fold}: matriz r² {r2.shape}, média fora da diagonal "
            f"{r2[~np.eye(len(r2), dtype=bool)].mean():.4f}")

        for order in args.orders:
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + fold)
                picked = select(order, r2, block_of, max(args.sizes), rng)
                for size in args.sizes:
                    if (fold, order, size, seed) in done:
                        continue
                    columns = np.array(picked[:size])
                    submatrix = r2[np.ix_(columns, columns)]
                    off = ~np.eye(len(columns), dtype=bool)
                    shares = np.bincount(block_of[columns],
                                         minlength=len(order_names)) / len(columns)
                    started = time.perf_counter()
                    metrics = fit(scaled[fit_idx][:, columns], labels[fit_idx],
                                  scaled[val_idx][:, columns], labels[val_idx],
                                  scaled[test_index][:, columns], labels[test_index],
                                  n_classes, seed=SEED + fold,
                                  first_width=args.first_width)
                    rows.append({
                        "fold": fold, "order": order, "n_features": size, "seed": seed,
                        "mean_r2": float(submatrix[off].mean()) if size > 1 else 0.0,
                        "blocks_used": int((shares > 0).sum()),
                        "block_entropy": float(-(shares[shares > 0]
                                                 * np.log(shares[shares > 0])).sum()),
                        "seconds": time.perf_counter() - started, **metrics})
                    pd.DataFrame(rows).to_csv(args.output, index=False)
                    log(f"  f{fold} {order:<13} k={size:>3} s={seed}  "
                        f"BAcc {metrics['bacc']:5.2f}  r̄²={rows[-1]['mean_r2']:.3f}  "
                        f"blocos={rows[-1]['blocks_used']}")

    frame = pd.DataFrame(rows)
    log("\n" + frame.pivot_table(index="n_features", columns="order",
                                 values="bacc").round(2).to_string())
    log("\nr² médio do conjunto selecionado:")
    log(frame.pivot_table(index="n_features", columns="order",
                          values="mean_r2").round(3).to_string())
    log("\nblocos representados:")
    log(frame.pivot_table(index="n_features", columns="order",
                          values="blocks_used").round(1).to_string())


if __name__ == "__main__":
    main()
