"""Matching Networks sobre o nosso kNN: a métrica aprendida como a fusão.

Proposta autônoma, **não** mais uma modalidade. O argumento é a simetria do eixo
de supervisão (§4.1), que hoje tem três degraus do lado da CNN e um do nosso:
do zero / congelada / fine-tuned contra apenas o banco estático. Atenção
aprendida é o nosso degrau de adaptação ao alvo -- e usa só os rótulos do
corpus, enquanto o fine-tuning usa os do corpus **mais** 1,28M do ImageNet.

O 5.2 não é afetado: ele fala do operador **sem parâmetros**, que continua sendo
o degrau de baixo.

Desenho (Vinyals et al., 2016, com a métrica aprendida):

    z = Proj(x)                        # 607d -> 128d, ESTA é a fusão aprendida
    a_j = <z_q, z_j> / sqrt(P) * T     # escore por vizinho, agnóstico a classe
    p(c) = sum_j softmax(a)_j * [y_j = c]

O escore é função só de (consulta, vizinho) -- **nenhuma identidade de classe
entra nele**. Isso é obrigatório: o §5.5 mostra o que acontece com parâmetros
ajustados sobre 11 exemplos por classe, e um escore agnóstico transfere da
cabeça para a cauda porque não tem parâmetro por classe.

**Vazamento é o risco número um.** Os vizinhos de um ponto de treino são pontos
de treino com rótulo conhecido, e um modelo com parâmetros acha o atalho
"atenda ao vizinho cujo rótulo é o meu" -- foi assim que o SNF caiu para 3,4%
(§6). Aqui os vizinhos de uma consulta de treino vêm **sempre de folds internos
disjuntos**, nunca do próprio fold. A consulta de teste recupera do treino
inteiro, que é a mesma assimetria que o grafo já tem por OOF aninhado.

Pisos a bater (Multitask 25 classes, §5.7/§5.8):
    voto uniforme sobre 30 vizinhos   36,89
    grafo, agregação hand-designed    39,84   <- o piso real
    teto do oráculo                   91,89
    alvo (configuração de produção)   50,28
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402

INNER_FOLDS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


class NeighbourAttention(nn.Module):
    """Projeção aprendida + atenção por produto interno sobre os k vizinhos.

    A projeção é a fusão: ela é quem decide quanto cada uma das 607 dimensões
    (e portanto cada bloco) pesa na distância. Ler os pesos dela por bloco e
    comparar com os marginais já medidos (Shapley, permutation importance) é o
    laço de validação que só existe porque o resto do projeto foi feito.
    """

    def __init__(self, dim: int, projection: int = 128, hidden: int = 512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden, projection))
        self.temperature = nn.Parameter(torch.tensor(3.0))
        self.projection = projection

    def forward(self, query, neighbours, neighbour_labels, n_classes):
        batch, k, dim = neighbours.shape
        zq = self.project(query)
        zn = self.project(neighbours.reshape(batch * k, dim)).reshape(batch, k, -1)
        scores = (zq.unsqueeze(1) * zn).sum(-1) / math.sqrt(self.projection)
        weights = torch.softmax(scores * self.temperature.clamp(0.1, 50.0), dim=1)
        probabilities = torch.zeros(batch, n_classes, device=query.device)
        probabilities.scatter_add_(1, neighbour_labels, weights)
        return torch.log(probabilities.clamp_min(1e-9))


def neighbours_for(space, labels, train, query, k, exclude_self_fold=None):
    """Índices dos k vizinhos de `query` dentro de `train`.

    `exclude_self_fold` recebe (atribuição de fold interno de train, fold da
    consulta) e retira do candidato o fold a que a consulta pertence -- é o que
    impede o atalho de auto-referência.
    """
    if exclude_self_fold is None:
        pool = train
    else:
        assignment, own = exclude_self_fold
        pool = train[assignment != own]
    finder = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(space[pool])
    _, local = finder.kneighbors(space[query])
    return pool[local]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--projection", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/neighbour_attention_multitask.csv")
    args = parser.parse_args()

    blocks, raw_labels, frame = load_corpus(args.dataset, VISUAL)
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
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    done = set()
    if args.output.exists():
        done = {int(f) for f in pd.read_csv(args.output)["fold"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        if fold in done:
            log(f"fold {fold}: já feito, pulando")
            continue
        started = time.perf_counter()
        space = np.hstack([StandardScaler().fit(blocks[n][train]).transform(blocks[n])
                           .astype(np.float32) for n in VISUAL])

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
        # sanidade: nenhuma consulta de treino pode ter a si mesma como vizinha
        assert not (train_neighbours == train[:, None]).any(), "auto-referência!"
        log(f"fold {fold}: vizinhos em {time.perf_counter() - started:.0f}s")

        fit_local, val_local = train_test_split(
            np.arange(len(train)), test_size=0.1,
            stratify=labels[train], random_state=SEED + fold)

        features = torch.from_numpy(space).to(DEVICE)
        target = torch.from_numpy(labels).long().to(DEVICE)
        seed_everything(SEED + fold)
        model = NeighbourAttention(space.shape[1], args.projection).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        counts = np.bincount(labels[train][fit_local], minlength=n_classes).astype(np.float32)
        weight = torch.tensor((1 / np.maximum(counts, 1)) / (1 / np.maximum(counts, 1)).mean(),
                              device=DEVICE)
        criterion = nn.NLLLoss(weight=weight)

        def run(query_rows, neighbour_rows, training: bool):
            order = np.random.permutation(len(query_rows)) if training else np.arange(len(query_rows))
            predictions = np.zeros(len(query_rows), dtype=np.int64)
            for start in range(0, len(order), args.batch_size):
                chunk = order[start:start + args.batch_size]
                if training and len(chunk) < 2:
                    continue
                q = torch.from_numpy(query_rows[chunk]).to(DEVICE)
                nb = torch.from_numpy(neighbour_rows[chunk]).to(DEVICE)
                logp = model(features[q], features[nb], target[nb], n_classes)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    criterion(logp, target[q]).backward()
                    optimizer.step()
                else:
                    predictions[chunk] = logp.argmax(1).cpu().numpy()
            return predictions

        best, state, stale = -1.0, None, 0
        for epoch in range(args.epochs):
            model.train()
            run(train[fit_local], train_neighbours[fit_local], True)
            scheduler.step()
            model.eval()
            with torch.no_grad():
                predicted = run(train[val_local], train_neighbours[val_local], False)
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
            predicted = run(test, test_neighbours, False)
        bacc = balanced_accuracy_score(labels[test], predicted)

        pd.DataFrame([{"dataset": args.dataset, "fold": fold, "k": args.k,
                       "projection": args.projection, "epochs_run": epoch + 1,
                       "val_bacc": best, "bacc": bacc,
                       "temperature": float(model.temperature.detach().cpu()),
                       "seconds": time.perf_counter() - started}]).to_csv(
            args.output, mode="a", index=False,
            header=not args.output.exists(), float_format="%.6f")
        log(f"fold {fold}: BAcc teste {100 * bacc:.2f} (val {100 * best:.2f}, "
            f"{epoch + 1} épocas, T={float(model.temperature.detach().cpu()):.2f}, "
            f"{(time.perf_counter() - started) / 60:.1f} min)")
        del features, target
        torch.cuda.empty_cache()

    detail = pd.read_csv(args.output)
    values = 100 * detail["bacc"].to_numpy()
    print()
    print(f"atenção sobre {args.k} vizinhos: {values.mean():.2f} ± "
          f"{values.std(ddof=1) if len(values) > 1 else 0:.2f}  ({len(values)} folds)")
    for name, floor in (("voto uniforme", 36.89), ("grafo (hand-designed)", 39.84),
                        ("alvo: produção", 50.28), ("teto do oráculo", 91.89)):
        print(f"  vs {name:<24} {values.mean() - floor:+6.2f}")


if __name__ == "__main__":
    main()
