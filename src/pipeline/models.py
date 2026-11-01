"""Os classificadores de produção do projeto.

**Por que este arquivo existe.** Até 06/set o `BranchMLP` -- o classificador que
produz todos os números de produção do §3 em diante -- morava dentro de
`experiments/evaluate_robustness_mlp_aug.py`, um teste exploratório de
aumentação por modalidade, e era importado de lá por **48 scripts**. O
`WideBranchMLP` morava em `evaluate_minc2500_official.py`. Quem quisesse saber o
que é "o nosso método" tinha de abrir um script de ablação e adivinhar qual parte
era o método e qual era o experimento.

Os arquivos de origem continuam existindo como experimentos legítimos; eles
agora importam daqui e re-exportam os nomes, então nenhum consumidor quebrou.

**Nada aqui pode ser "melhorado" sem refazer as tabelas.** O §5.6 registra o
caso em que reimplementar o laço de treino com `label_smoothing` omitido e
semente `SEED+fold` em vez de `SEED` custou **0,47pp em silêncio**, e o braço de
controle deixou de reproduzir o valor conhecido. Um braço de controle que não
reproduz o valor conhecido não é controle.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GUARDA CONTRA O MODO DE FALHA DO §8, e ela e o conserto do mecanismo exato.
# `DEVICE` e constante de MODULO, fixada no import: se a GPU ja estiver caida
# quando o processo nasce, `is_available()` da falso, o device vira `cpu` EM
# SILENCIO e o job roda ~14x mais lento produzindo numeros validos -- foi assim
# que 67 celulas do multitask sairam na CPU em 12/out sem nada no CSV
# denunciando. Com `PRISM_EXIGIR_GPU=1` o processo ABORTA em vez de degradar.
#
# Desligada por padrao DE PROPOSITO: ligar por default quebraria qualquer
# execucao legitima em maquina sem GPU, e nenhum numero publicado muda com ela
# desligada. O lancador e que decide -- ver experiments/gpu_watchdog.sh.
if os.environ.get("PRISM_EXIGIR_GPU") and DEVICE.type != "cuda":
    raise SystemExit(
        "PRISM_EXIGIR_GPU=1 e torch.cuda.is_available() e False: a GPU caiu ou "
        "o driver nao carregou. Abortando em vez de rodar na CPU em silencio "
        "(§8). Confira `nvidia-smi` e logs/gpu_watchdog/.")


class BranchMLP(nn.Module):
    """O MLP estreito de produção: um encoder por bloco, cabeça 512-256-128.

    Adotado em 25/ago (§7.8) sobre o largo, por decisão explícita de manter
    todas as tabelas válidas e de não afinar a arquitetura a favor do próprio
    método. Cada bloco vai para **128 dims** independentemente do tamanho de
    entrada -- é por isso que a fração do vetor concatenado cru NÃO é o que a
    cabeça vê, correção registrada no §5.6 contra a explicação de "diluição
    dimensional" que estava errada.

    O §5.16 mede o que a primeira camada bloco-diagonal vale: +0,93pp contra o
    denso de mesma largura (p=0,057, 4/5), com **8x menos pesos**.
    """

    def __init__(self, branch_sizes: list[int], n_classes: int) -> None:
        super().__init__()
        self.branch_sizes = branch_sizes
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25)
            ) for size in branch_sizes
        ])
        width = 128 * len(branch_sizes)
        self.head = nn.Sequential(
            nn.Linear(width, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(128, n_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = [
            encoder(block) for encoder, block
            in zip(self.encoders, torch.split(values, self.branch_sizes, dim=1))
        ]
        return self.head(torch.cat(encoded, dim=1))


class LinearProbe(nn.Module):
    """Classificador LINEAR pelo mesmo protocolo dos MLPs.

    Existe para o braco linear da comparacao de bancos hand-designed. O
    protocolo publicado do Fisher vector (Sanchez et al., IJCV 2013) usa **SVM
    linear**, e o `LinearSVC` do sklearn nao escala aqui: medido em 08/out, ele
    nao terminou em 600 s nem no banco de 617 dims do food101 (75.750 x 101
    classes), e o FV tem 32.768 dims e exigiria float64 (~20 GB) com 101
    problemas um-contra-resto.

    Entao o braco linear e um softmax linear treinado pelo MESMO caminho do
    braco de MLP -- mesma padronizacao, mesma ponderacao de classe, mesmas
    epocas e mesma parada --, o que torna a comparacao entre as duas sondas
    limpa: a UNICA coisa que muda e a capacidade. **Desvio declarado:** dobradica
    virou entropia cruzada com `label_smoothing`. Usar `LinearSVC` so nos bancos
    pequenos e outra coisa no FV nao seria comparacao.

    Nota de protocolo: o `standardise_blocks` padroniza por coluna, o que
    desfaz a normalizacao L2 que faz parte do FV improved. E o preco de ter UM
    pipeline para todos os bancos (§5.19); o braco sem padronizacao fica como
    controle de justica a rodar se o FV perder.
    """

    def __init__(self, branch_sizes, n_classes) -> None:
        super().__init__()
        self.net = nn.Linear(int(sum(branch_sizes)), n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class WideBranchMLP(nn.Module):
    """A variante larga (enc=512, cabeça 2048-1024-1024).

    Usada nos corpora de split oficial e na tabela de sensibilidade do §7.8, que
    é obrigatória ao reportar o estreito: no Places-LT, com 365 classes, o
    estreito limita o posto da matriz de logits a 128 (gargalo do softmax, Yang
    et al. ICLR 2018) e **penaliza a ResNet mais que a nós** -- o gap vai de 14,6
    para 17,9. Publicar o número estreito sem essa tabela é o tipo de coisa que
    um revisor encontra sozinho.

    O §4.1e mede que alargar o ramo do backbone não compra nada (+0,15pp n.s.) e
    que dar a ele a fatia proporcional às dimensões brutas **piora** 0,87pp
    (p=0,011, 0/5): o encoder de 2048->512 não estrangula, regulariza.
    """

    def __init__(self, sizes, n_classes, enc=512, head=(2048, 1024, 1024)):
        super().__init__()
        self.sizes = list(sizes)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, enc), nn.BatchNorm1d(enc), nn.GELU(), nn.Dropout(0.25))
            for size in sizes
        ])
        layers, previous = [], enc * len(sizes)
        for width in head:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, values):
        return self.head(torch.cat([
            encoder(part) for encoder, part
            in zip(self.encoders, torch.split(values, self.sizes, dim=1))
        ], dim=1))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(
    train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
    validation_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
    branch_sizes: list[int], n_classes: int, seed: int, max_epochs: int, patience: int,
    track_history: bool = False, label_smoothing: float = 0.0, weight_decay: float = 1e-4,
    return_probs: bool = False, model_factory=None,
) -> tuple[dict, np.ndarray]:
    """Treina um `BranchMLP` e escolhe a época pela BAcc de VALIDAÇÃO.

    A seleção por validação é o que mantém honesto qualquer número reportado no
    teste. `class_weight` está embutido via `weights` -- ver §7.12: treinar
    ponderado otimiza BAcc e **sacrifica acurácia de propósito** (7,30pp no
    hand-8), então toda tabela tem de declarar qual das quatro células reporta.

    `model_factory` é aditivo pela mesma razão e com o mesmo contrato: `None`
    constrói o `BranchMLP`, então todo chamador existente é bit-a-bit idêntico.
    Existe para trocar SÓ a arquitetura mantendo laço, otimizador, agendamento,
    `class_weight` e seleção de época -- é o que o §5.19 precisa para comparar
    bancos sob MLP sem branch sem reimplementar nada. A fábrica recebe
    `(branch_sizes, n_classes)`, a mesma assinatura do `BranchMLP`.

    `return_probs=True` é aditivo e não muda nada do caminho default: guarda
    `softmax(logits)` do teste em `metrics["test_probs"]`, sem tocar na
    assinatura de retorno (ainda `tuple[dict, np.ndarray]`) nem no `prediction`
    que os 51+ chamadores já esperam. Existe para permitir decisão agregada por
    grupo de linhas fora desta função (ex.: média de posteriores por tile antes
    do argmax), sem reimplementar o laço -- a lição do §5.6 é não reimplementar,
    só estender.
    """
    seed_everything(seed)
    model = (model_factory or BranchMLP)(branch_sizes, n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE, non_blocking=True)
    validation_target = torch.from_numpy(validation_y).long().to(DEVICE, non_blocking=True)
    test_tensor = torch.from_numpy(test_x).to(DEVICE, non_blocking=True)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = [] if track_history else None
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            if track_history:
                batch_losses.append(float(loss.item()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor)
            prediction = validation_logits.argmax(1).cpu().numpy()
            if track_history:
                validation_loss = float(criterion(validation_logits, validation_target).item())
        score = balanced_accuracy_score(validation_y, prediction)
        if track_history:
            history.append({
                "epoch": epoch, "train_loss": float(np.mean(batch_losses)),
                "validation_loss": validation_loss, "validation_bacc": score,
            })
        if score > best_score:
            best_score, best_state, best_epoch, stale = (
                float(score), copy.deepcopy(model.state_dict()), epoch, 0)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_tensor)
        prediction = test_logits.argmax(1).cpu().numpy()
    metrics = {
        "bacc": balanced_accuracy_score(test_y, prediction),
        "accuracy": accuracy_score(test_y, prediction),
        "macro_f1": f1_score(test_y, prediction, average="macro"),
        "validation_bacc": best_score, "best_epoch": best_epoch,
    }
    if track_history:
        metrics["history"] = history
    if return_probs:
        metrics["test_probs"] = torch.softmax(test_logits, dim=1).cpu().numpy()
    del model, validation_tensor, validation_target, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, prediction
