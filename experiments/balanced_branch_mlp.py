"""MLP de branches com fusão não-ingênua: loss auxiliar, dropout de modalidade, gating.

Motivação medida, não teórica. Em quatro casos independentes hoje o LDA extraiu
valor que o MLP de branches destruiu:

    quadrantes de Hilbert          LDA +0,23  MLP −0,52
    Multiscale (3 famílias)        LDA prefere as três, MLP prefere só Descriptors
    grafo de similaridade          LDA +0,83  MLP −0,22
    Hilbert pooled 1/8             LDA +0,20  MLP −0,38

O sinal existe -- uma projeção linear o encontra -- e a fusão atual o perde.
A hipótese é que o head aprende a se apoiar nos branches fortes e trata os
fracos como ruído, em vez de extrair o pouco que eles têm.

Três intervenções, independentes e ablatáveis:

  aux_weight    cabeça de classificação por branch, com loss própria. Força
                cada branch a ser discriminativo sozinho em vez de pegar
                carona no head compartilhado.
  branch_drop   zera embeddings de branches inteiros durante o treino (por
                amostra, com reescala). Impede que o head dependa de um
                subconjunto fixo de branches.
  gating        pesos por amostra sobre os branches. Uma pintura pontilhista
                e uma minimalista não precisam das mesmas modalidades.

A arquitetura base é idêntica à de produção (`BranchMLP` em
`evaluate_robustness_mlp_aug.py`): cada branch projeta para 128 dimensões,
o que já equaliza branches de 27d e de 1.032d na fusão.

`branch_dependency()` mede o problema em vez de supor: zera cada branch na
inferência e devolve a queda de BAcc. Perfil concentrado confirma a hipótese
de dependência; perfil espalhado a refuta.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

BRANCH_WIDTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BalancedBranchMLP(nn.Module):
    def __init__(self, branch_sizes: list[int], n_classes: int, *,
                 aux_heads: bool = False, gating: bool = False,
                 branch_scale: bool = False) -> None:
        super().__init__()
        self.branch_sizes = branch_sizes
        self.n_branches = len(branch_sizes)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, BRANCH_WIDTH), nn.BatchNorm1d(BRANCH_WIDTH),
                          nn.GELU(), nn.Dropout(0.25))
            for size in branch_sizes
        ])
        width = BRANCH_WIDTH * self.n_branches
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(128, n_classes),
        )
        self.aux = nn.ModuleList(
            [nn.Linear(BRANCH_WIDTH, n_classes) for _ in branch_sizes]) if aux_heads else None
        # Gate a partir dos embeddings destacados: aprende a ponderar, não a
        # reescrever, e não deixa o gradiente do gate reentrar nos encoders.
        #
        # Sigmoide independente por branch, NÃO softmax. Com softmax os branches
        # competem por massa fixa: se o gate concentra num deles, os outros vão
        # a peso ~0, seus encoders param de receber gradiente e morrem -- medido
        # na prática, −2,19pp num fold. A sigmoide deixa cada branch ser
        # atenuado ou reforçado sem tirar de ninguém.
        self.gate = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Linear(64, self.n_branches)) if gating else None
        if self.gate is not None:
            # Inicializa em peso 1.0 para todo branch: no primeiro passo o gate
            # é a identidade e só se afasta dela se isso reduzir a loss.
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

        # Escalar aprendível por branch, depois do BatchNorm.
        #
        # O BN normaliza a saída de todo branch para variância unitária,
        # independentemente de ele carregar sinal ou ruído -- um branch inútil
        # entrega 128 dimensões com a mesma amplitude de um branch bom, e o
        # head precisa gastar capacidade descontando isso. Um escalar por
        # branch, inicializado em 1, deixa o próprio treino atenuar até ~0 o
        # que não paga. Um parâmetro por branch, sem competição e sem depender
        # da amostra.
        self.scale = nn.Parameter(torch.ones(self.n_branches)) if branch_scale else None

    def encode(self, values: torch.Tensor) -> list[torch.Tensor]:
        blocks = torch.split(values, self.branch_sizes, dim=1)
        return [encoder(block) for encoder, block in zip(self.encoders, blocks)]

    def forward(self, values: torch.Tensor, *, branch_drop: float = 0.0,
                keep: np.ndarray | None = None):
        embeddings = self.encode(values)

        if keep is not None:                       # diagnóstico: desliga branches
            embeddings = [e if keep[i] else torch.zeros_like(e)
                          for i, e in enumerate(embeddings)]

        if self.training and branch_drop > 0.0:
            # Máscara por amostra, com reescala para preservar a esperança.
            batch = embeddings[0].shape[0]
            mask = (torch.rand(batch, self.n_branches, device=embeddings[0].device)
                    >= branch_drop).float()
            # Nunca zera todos os branches de uma amostra.
            empty = mask.sum(dim=1) == 0
            if empty.any():
                mask[empty, torch.randint(0, self.n_branches, (int(empty.sum()),),
                                          device=mask.device)] = 1.0
            mask = mask / (1.0 - branch_drop)
            embeddings = [e * mask[:, i:i + 1] for i, e in enumerate(embeddings)]

        if self.scale is not None:
            embeddings = [e * self.scale[i] for i, e in enumerate(embeddings)]

        if self.gate is not None:
            detached = torch.cat([e.detach() for e in embeddings], dim=1)
            weights = 2.0 * torch.sigmoid(self.gate(detached))   # 1.0 na inicialização
            embeddings = [e * weights[:, i:i + 1] for i, e in enumerate(embeddings)]

        logits = self.head(torch.cat(embeddings, dim=1))
        aux_logits = ([head(e) for head, e in zip(self.aux, embeddings)]
                      if self.aux is not None else None)
        return logits, aux_logits


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _predict(model: BalancedBranchMLP, tensor: torch.Tensor,
             keep: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    logits, _ = model(tensor, keep=keep)
    return logits.argmax(1).cpu().numpy()


def branch_dependency(model, tensor, truth, branch_names) -> list[dict]:
    """Queda de BAcc ao zerar cada branch na inferência."""
    full = balanced_accuracy_score(truth, _predict(model, tensor))
    rows = []
    for index, name in enumerate(branch_names):
        keep = np.ones(len(branch_names), dtype=bool)
        keep[index] = False
        score = balanced_accuracy_score(truth, _predict(model, tensor, keep=keep))
        rows.append({"branch": name, "bacc_without": 100 * score,
                     "drop_pp": 100 * (full - score)})
    return rows


def train_balanced(
    train_x, train_y, validation_x, validation_y, test_x, test_y,
    branch_sizes, n_classes, seed, max_epochs, patience, *,
    aux_weight: float = 0.0, branch_drop: float = 0.0, gating: bool = False,
    branch_scale: bool = False, branch_names: list[str] | None = None,
) -> tuple[dict, np.ndarray, list[dict]]:
    """Mesmo protocolo de `train_one`, com as intervenções opcionais."""
    seed_everything(seed)
    model = BalancedBranchMLP(branch_sizes, n_classes, aux_heads=aux_weight > 0.0,
                              gating=gating, branch_scale=branch_scale).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed))
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE, non_blocking=True)
    test_tensor = torch.from_numpy(test_x).to(DEVICE, non_blocking=True)

    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, aux_logits = model(xb, branch_drop=branch_drop)
            loss = criterion(logits, yb)
            if aux_logits is not None:
                aux = torch.stack([criterion(a, yb) for a in aux_logits]).mean()
                loss = loss + aux_weight * aux
            loss.backward()
            optimizer.step()
        scheduler.step()
        score = balanced_accuracy_score(validation_y, _predict(model, validation_tensor))
        if score > best_score:
            best_score, best_epoch, stale = float(score), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    prediction = _predict(model, test_tensor)
    metrics = {
        "bacc": balanced_accuracy_score(test_y, prediction),
        "accuracy": accuracy_score(test_y, prediction),
        "macro_f1": f1_score(test_y, prediction, average="macro"),
        "validation_bacc": best_score, "best_epoch": best_epoch,
    }
    dependency = (branch_dependency(model, test_tensor, test_y, branch_names)
                  if branch_names else [])
    if model.scale is not None and branch_names:
        # O escalar aprendido é o diagnóstico direto: um branch que o treino
        # levou perto de zero é um branch que o head preferiu não ouvir.
        learned = model.scale.detach().cpu().numpy()
        for entry, value in zip(dependency, learned):
            entry["learned_scale"] = float(value)
    del model, validation_tensor, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, prediction, dependency
