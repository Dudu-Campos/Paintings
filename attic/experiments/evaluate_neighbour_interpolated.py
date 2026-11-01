"""Tronco compartilhado + termo de vizinhos com porta aprendida (kNN-LM).

Os dois modelos anteriores usam as mesmas features para **um** papel cada:

* atenção (§7.10, 42,18): as features só definem quem é vizinho -- a posição
  absoluta da consulta nunca entra na predição.
* branch MLP (48,24): as features só discriminam -- a vizinhança não existe.

Aqui o mesmo vetor cumpre os dois papéis num modelo só:

    h   = Tronco(x)                      # 607 -> 512 -> 256, UM só, não sete
    p_d = softmax(Direto(h))             # caminho discriminativo
    w   = softmax(<h_q, h_j> / sqrt(d) * T)      # atenção sobre os 30 vizinhos
    p_v = sum_j w_j * [y_j = c]                  # já é uma distribuição
    g   = sigmoid(Porta(h)) in [0,1]             # por imagem
    p   = (1-g) * p_d + g * p_v

Interpolação em espaço de probabilidade, como no kNN-LM (Khandelwal et al.,
2020), não em espaço de logits: `p_v` pode ser exatamente zero na classe certa
quando nenhum vizinho a carrega, e somar `log(0)` a logits explodiria. Assim o
termo é limitado e `g` é diretamente legível como "quanto confiar na vizinhança
para esta imagem".

Isso é o `logit-add` do Places-LT (§6) com **λ dependente da entrada** em vez de
escalar global -- lá o λ único serviu 365 classes e ficou em +0,030.

Por que esperar mais que a produção (50,28): lá o grafo é **pré-computado e
fixo**, e não sabe o que o caminho direto já acerta. Treinados juntos, o termo
de vizinhos pode se especializar no resíduo.

Três condições, e as duas ablações respondem de graça a pergunta de projeto:

* `direto só`      -- porta forçada a 0. **Um tronco único sobre os 607d
                      achatados.** Se bater 48,24, a promessa de "sem fusão
                      complexa" se sustenta; se ficar abaixo, o preço dos sete
                      branches é real (o §6 registra que vetor multi-bloco
                      achatado é onde LDA e MLP divergem ~9pp).
* `vizinhos só`    -- porta forçada a 1. A atenção com tronco compartilhado.
* `interpolado`    -- porta aprendida.

Vazamento: vizinhos de consulta de treino vêm sempre de folds internos
disjuntos, com `assert` (ver `evaluate_neighbour_attention.py`).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import INNER_FOLDS, key, neighbours_for  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402

MODES = ("direto só", "vizinhos só", "interpolado")
FLOORS = (("atenção (projeção separada)", 42.18), ("branch MLP (7 encoders)", 48.24),
          ("produção (hand-8 + grafo)", 50.28))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class InterpolatedNeighbourModel(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 512, latent: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(hidden, latent), nn.BatchNorm1d(latent), nn.GELU(), nn.Dropout(0.25))
        self.direct = nn.Linear(latent, n_classes)
        self.gate = nn.Linear(latent, 1)
        self.temperature = nn.Parameter(torch.tensor(3.0))
        self.latent, self.n_classes = latent, n_classes

    def forward(self, query, neighbours, neighbour_labels, mode):
        h = self.trunk(query)
        direct = torch.softmax(self.direct(h), dim=1)
        if mode == "direto só":
            return torch.log(direct.clamp_min(1e-9)), torch.zeros(len(h), device=h.device)

        batch, k, dim = neighbours.shape
        hn = self.trunk(neighbours.reshape(batch * k, dim)).reshape(batch, k, -1)
        scores = (h.unsqueeze(1) * hn).sum(-1) / math.sqrt(self.latent)
        weights = torch.softmax(scores * self.temperature.clamp(0.1, 50.0), dim=1)
        neighbour_probabilities = torch.zeros(batch, self.n_classes, device=h.device)
        neighbour_probabilities.scatter_add_(1, neighbour_labels, weights)

        if mode == "vizinhos só":
            g = torch.ones(batch, device=h.device)
        else:
            g = torch.sigmoid(self.gate(h)).squeeze(1)
        mixed = (1 - g).unsqueeze(1) * direct + g.unsqueeze(1) * neighbour_probabilities
        return torch.log(mixed.clamp_min(1e-9)), g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    # A configuração de produção tem 8 modalidades (VISUAL + Gabor); comparar o
    # interpolado de 7 contra ela é comparação torta.
    parser.add_argument("--extra-blocks", nargs="*", default=[])
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/neighbour_interpolated_multitask.csv")
    args = parser.parse_args()

    names = VISUAL + list(args.extra_blocks)
    blocks, raw_labels, _ = load_corpus(args.dataset, names)
    if args.classes_file and args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("casamento de classes do benchmark incompleto")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"{len(names)} blocos: {names}")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.mode), int(r.fold)) for r in previous.itertuples()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        pending = [m for m in args.modes if (m, fold) not in done]
        if not pending:
            continue
        space = np.hstack([StandardScaler().fit(blocks[n][train]).transform(blocks[n])
                           .astype(np.float32) for n in names])
        needs_neighbours = any(m != "direto só" for m in pending)
        train_neighbours = test_neighbours = None
        if needs_neighbours:
            started = time.perf_counter()
            inner = np.zeros(len(train), dtype=np.int64)
            for i, (_, part) in enumerate(StratifiedKFold(
                    INNER_FOLDS, shuffle=True, random_state=SEED + fold)
                    .split(np.zeros(len(train)), labels[train])):
                inner[part] = i
            train_neighbours = np.zeros((len(train), args.k), dtype=np.int64)
            for i in range(INNER_FOLDS):
                own = np.flatnonzero(inner == i)
                train_neighbours[own] = neighbours_for(
                    space, labels, train, train[own], args.k, exclude_self_fold=(inner, i))
            test_neighbours = neighbours_for(space, labels, train, test, args.k)
            assert not (train_neighbours == train[:, None]).any(), "auto-referência!"
            log(f"fold {fold}: vizinhos em {time.perf_counter() - started:.0f}s")

        fit_local, val_local = train_test_split(
            np.arange(len(train)), test_size=0.1,
            stratify=labels[train], random_state=SEED + fold)
        features = torch.from_numpy(space).to(DEVICE)
        target = torch.from_numpy(labels).long().to(DEVICE)

        for mode in pending:
            started = time.perf_counter()
            seed_everything(SEED + fold)
            model = InterpolatedNeighbourModel(space.shape[1], n_classes).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            counts = np.bincount(labels[train][fit_local], minlength=n_classes).astype(np.float32)
            weight = torch.tensor((1 / np.maximum(counts, 1)) / (1 / np.maximum(counts, 1)).mean(),
                                  device=DEVICE)
            criterion = nn.NLLLoss(weight=weight)

            def run(rows, neighbour_rows, training):
                order = (np.random.permutation(len(rows)) if training
                         else np.arange(len(rows)))
                predictions = np.zeros(len(rows), dtype=np.int64)
                gates = []
                for start in range(0, len(order), args.batch_size):
                    chunk = order[start:start + args.batch_size]
                    if training and len(chunk) < 2:
                        continue
                    q = torch.from_numpy(rows[chunk]).to(DEVICE)
                    nb = (torch.from_numpy(neighbour_rows[chunk]).to(DEVICE)
                          if neighbour_rows is not None else None)
                    logp, g = model(features[q],
                                    features[nb] if nb is not None else None,
                                    target[nb] if nb is not None else None, mode)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        criterion(logp, target[q]).backward()
                        optimizer.step()
                    else:
                        predictions[chunk] = logp.argmax(1).cpu().numpy()
                        gates.append(g.detach().cpu().numpy())
                return predictions, (np.concatenate(gates) if gates else np.zeros(1))

            best, state, stale, epoch = -1.0, None, 0, 0
            for epoch in range(args.epochs):
                model.train()
                run(train[fit_local], None if mode == "direto só" else train_neighbours[fit_local], True)
                scheduler.step()
                model.eval()
                with torch.no_grad():
                    predicted, _ = run(train[val_local],
                                       None if mode == "direto só" else train_neighbours[val_local],
                                       False)
                score = balanced_accuracy_score(labels[train][val_local], predicted)
                if score > best:
                    best, state, stale = score, copy.deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                predicted, gates = run(test, None if mode == "direto só" else test_neighbours, False)
            bacc = balanced_accuracy_score(labels[test], predicted)
            pd.DataFrame([{"dataset": args.dataset, "mode": mode + args.tag,
                           "fold": fold, "k": args.k, "n_blocks": len(names),
                           "epochs_run": epoch + 1, "val_bacc": best, "bacc": bacc,
                           "gate_mean": float(gates.mean()), "gate_std": float(gates.std()),
                           "temperature": float(model.temperature.detach().cpu()),
                           "seconds": time.perf_counter() - started}]).to_csv(
                args.output, mode="a", index=False,
                header=not args.output.exists(), float_format="%.6f")
            log(f"  f{fold} {mode:<12} BAcc={100 * bacc:5.2f} (val {100 * best:5.2f}, "
                f"porta={gates.mean():.3f}±{gates.std():.3f}, {epoch + 1} ép., "
                f"{time.perf_counter() - started:.0f}s)")
        del features, target, space
        torch.cuda.empty_cache()

    detail = pd.read_csv(args.output)
    print()
    for mode in MODES:
        piece = detail[detail["mode"].eq(mode)]
        if piece.empty:
            continue
        values = 100 * piece["bacc"].to_numpy()
        print(f"{mode:<14} {values.mean():6.2f} ± "
              f"{values.std(ddof=1) if len(values) > 1 else 0:4.2f}  "
              f"(porta média {piece['gate_mean'].mean():.3f}, {len(values)} folds)")
    best = detail[detail["mode"].eq("interpolado")]
    if not best.empty:
        mean = 100 * best["bacc"].mean()
        print()
        for name, floor in FLOORS:
            print(f"  interpolado vs {name:<28} {mean - floor:+6.2f}")


if __name__ == "__main__":
    main()
