"""logit-add: o grafo somado aos LOGITS, com zero dimensões de entrada.

Terceira tentativa de fazer o grafo de similaridade pagar em 125 classes, e a
primeira motivada pelo mecanismo que a sessão de 31/ago mediu em vez do que se
supunha.

O §5.6 diz que o grafo perde em 125 classes por evidência vazia. Duas curas
foram testadas e falharam: mais vizinhos (k=30/60/100, sem gradiente) e redução
por classe (`dmin`/`dmean`, nulos). O §5.7 então mediu que **a informação está
lá**: pureza 18,57% contra 5,03% de acaso (3,7x, MELHOR que os 3,1x de 25
classes), e o voto puro do kNN dá 32,44 contra 38,53 do hand-8 inteiro.

Logo o que falha não é a informação, é a **conversão**. E o custo dela apareceu
como variância: somar 125-250 dimensões ao vetor de 617d levou o desvio entre
folds de 0,45 para 1,23 e 3,10. O bloco disputa capacidade do `BranchMLP`.

`logit-add` é a única codificação que ataca isso: em vez de concatenar o
log-lift como features, soma `lambda * log-lift` diretamente aos logits. **Zero
dimensões de entrada, zero disputa de capacidade.** No Places-LT (§6) ela deu o
melhor `many` de todas as codificações (+2,23pp) com zero dimensões; nunca foi
testada em 125 classes -- e, apesar do §6 descrevê-la como "pronta", não havia
implementação no repositório.

Dois braços, e o segundo é o que a medição por faixa do §5.7 motiva:

    lambda global      um escalar. A versão mínima.
    lambda por faixa   tres escalares (raras/médias/frequentes), com lambda_c
                       determinado pela faixa da classe c. O voto converte 34,60
                       de um teto de 88,39 nas frequentes e apenas 8,52 de 52,93
                       nas médias -- um escalar único tem de conciliar regimes
                       que não se parecem, e é ali que os 44pp estão parados.

O log-lift é calculado aqui, autocontido, e não importado dos avaliadores da
sessão paralela (que estão sob edição). É o operador sem parâmetros do §5.2:
kNN, similaridade exponencial com escala robusta do treino, histograma ponderado
e correção por prior. **OOF aninhado** é obrigatório -- sem ele as linhas de
treino veem o próprio rótulo pelo vizinho, que é o mecanismo do SNF (§6).

Saída: results/multitask_logit_add.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_robustness_mlp_aug import BranchMLP, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS, INNER_FOLDS = 42, 5, 5
BRANCH_WIDTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)


def log_lift(train_x, train_y, query_x, n_classes, k):
    """`log p(c|x) − log p(c)` a partir da vizinhança. Sem parâmetro ajustado."""
    finder = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(train_x)
    distances, indices = finder.kneighbors(query_x)
    # Escala robusta tirada do TREINO: sem isso a densidade local do teste
    # entraria na normalização e o bloco deixaria de ser indutivo.
    train_distances, _ = finder.kneighbors(train_x)
    scale = max(float(np.median(train_distances[:, 1:])), 1e-6)
    similarity = np.exp(-distances / scale).astype(np.float32)

    labels = train_y[indices]
    posterior = np.zeros((len(query_x), n_classes), dtype=np.float32)
    for column in range(k):
        np.add.at(posterior, (np.arange(len(query_x)), labels[:, column]),
                  similarity[:, column])
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-8)
    prior = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    prior /= prior.sum()
    return (np.log(posterior + 1e-6) - np.log(prior + 1e-6)).astype(np.float32)


def build_oof(fit_x, fit_y, target_x, n_classes, k):
    oof = np.zeros((len(fit_x), n_classes), dtype=np.float32)
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
    for inner_train, inner_val in splitter.split(np.zeros(len(fit_y), np.uint8), fit_y):
        oof[inner_val] = log_lift(fit_x[inner_train], fit_y[inner_train],
                                  fit_x[inner_val], n_classes, k)
    return oof, log_lift(fit_x, fit_y, target_x, n_classes, k)


class LogitAdd(nn.Module):
    """O `BranchMLP` canônico, com um caminho aditivo opcional nos logits.

    Envolve a classe importada em vez de reimplementá-la: assim o braço base é
    identico bit a bit ao de `base.fit`, e não "equivalente na minha leitura" --
    a primeira versão deste script reescreveu o laço e deu 37,73 no fold 1
    contra 38,20 do mesmo fold via `base.fit`, por omitir `label_smoothing` e
    trocar a convenção de semente.

    `lambda_index` mapeia classe -> escalar: um único valor é o lambda global,
    tres são as faixas de frequência. O lambda nasce em ZERO, então na
    inicialização o modelo é exatamente a base e qualquer ganho é atribuível ao
    caminho aditivo, não a um ponto de partida diferente.
    """

    def __init__(self, sizes, n_classes, lambda_index=None):
        super().__init__()
        self.inner = BranchMLP(sizes, n_classes)
        if lambda_index is None:
            self.lambda_index, self.lam = None, None
        else:
            self.register_buffer("lambda_index", torch.as_tensor(lambda_index).long())
            self.lam = nn.Parameter(torch.zeros(int(lambda_index.max()) + 1))

    def forward(self, values, lift=None):
        logits = self.inner(values)
        if self.lam is not None and lift is not None:
            logits = logits + self.lam[self.lambda_index].unsqueeze(0) * lift
        return logits


def train_arm(blocks, lift, labels, fit, validation, test, n_classes,
              lambda_index, seed, epochs, patience):
    seed_everything(seed)
    parts = ([], [], [])
    for block in blocks:
        scaler = StandardScaler().fit(block[fit])
        for target, index in zip(parts, (fit, validation, test)):
            target.append(scaler.transform(block[index]).astype(np.float32))
    train_x, val_x, test_x = (np.hstack(p).astype(np.float32) for p in parts)
    sizes = [b.shape[1] for b in blocks]

    tensors = {}
    for name, index, values in (("fit", fit, train_x), ("val", validation, val_x),
                                ("test", test, test_x)):
        tensors[name] = (torch.from_numpy(values),
                         torch.from_numpy(lift[index]) if lift is not None else None)

    model = LogitAdd(sizes, n_classes, lambda_index).to(DEVICE)
    counts = np.bincount(labels[fit], minlength=n_classes).astype(np.float32)
    inverse = 1.0 / np.maximum(counts, 1)
    # `label_smoothing=0.05` não é detalhe: é o que `base.fit` usa, e omiti-lo
    # custou ~0,5pp no fold 1 da primeira versão deste script.
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(inverse / inverse.mean(), device=DEVICE),
        label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = TensorDataset(tensors["fit"][0], torch.from_numpy(labels[fit]).long(),
                            tensors["fit"][1] if lift is not None
                            else torch.zeros(len(fit), 1))
    loader = DataLoader(dataset, batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    def predict(name):
        model.eval()
        with torch.no_grad():
            values, held = tensors[name]
            held = held.to(DEVICE) if held is not None else None
            return model(values.to(DEVICE), held).argmax(1).cpu().numpy()

    best, state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, lb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            lb = lb.to(DEVICE) if lift is not None else None
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb, lb), yb).backward()
            optimizer.step()
        scheduler.step()
        score = balanced_accuracy_score(labels[validation], predict("val"))
        if score > best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    lam = (model.lam.detach().cpu().numpy().tolist() if model.lam is not None else [])
    # Acc e F1 saem de graca do mesmo vetor de predicao, e a matriz do artigo
    # precisa das tres colunas. BAcc sozinha nao permite comparar com o
    # benchmark do §7.11, que e acuracia simples.
    prediction = predict("test")
    truth = labels[test]
    metrics = {"bacc": 100 * balanced_accuracy_score(truth, prediction),
               "accuracy": 100 * accuracy_score(truth, prediction),
               "macro_f1": 100 * f1_score(truth, prediction, average="macro"),
               # `val_bacc` e a BAcc da epoca escolhida, no conjunto de
               # validacao. Ela ja era calculada e descartada; quem faz SELECAO
               # entre configuracoes precisa dela, senao escolhe no teste e a
               # curva sai otimista por construcao.
               "val_bacc": 100 * best}
    return metrics, lam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    # Recorte de classes, para o 2x2 controlado do §5.6: as MESMAS imagens em 25
    # e 125 classes isolam a granularidade de rotulo de todo o resto (corpus,
    # tamanho, features, pipeline).
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multitask_logit_add.csv")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    raw = np.asarray(cohort.labels, dtype=str)
    mask = np.ones(len(raw), dtype=bool)
    if args.classes_file is not None:
        # Os rotulos do benchmark divergem do metadado em tres eixos e NENHUM da
        # erro sozinho (§8): separador, espaco nao-quebravel e trema. Sem
        # normalizar os tres casam 13 das 25; sem o terceiro casam **24**, que e
        # o erro que passa batido. Dai o `key()` e a checagem de contagem.
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw])
        found = {key(v) for v in raw[mask]}
        if found != wanted:
            raise SystemExit(f"casamento incompleto: {len(found)}/{len(wanted)} classes")
    raw = raw[mask]
    labels = LabelEncoder().fit_transform(raw)
    n_classes = len(np.unique(labels))
    blocks = [cohort.blocks[name][mask] for name in cohort.blocks]
    space = np.hstack(blocks)
    frequency = np.bincount(labels, minlength=n_classes)
    # Mesmas faixas do §5.5/§5.7: cauda, meio e cabeça por frequência.
    band = np.digitize(frequency, [np.percentile(frequency, 33),
                                   np.percentile(frequency, 66)])
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"hand-8 {space.shape[1]}d, k={args.k}")
    log(f"  faixas: {np.bincount(band).tolist()} classes (raras/médias/frequentes)")

    arms = [("hand-8", None, None),
            ("+ logit-add λ global", np.zeros(n_classes, dtype=int), "global"),
            ("+ logit-add λ por faixa", band.astype(int), "faixa")]
    rows = []
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        log(f"retomando {len(rows)} linhas")
    done = {(r["condition"], r["fold"]) for r in rows}

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), np.uint8), labels))
    for fold, (outer, test) in enumerate(splits, start=1):
        fit, validation = train_test_split(
            outer, test_size=0.10, stratify=labels[outer], random_state=SEED)
        lift = None
        if any((name, fold) not in done for name, index, _ in arms if index is not None):
            started = time.perf_counter()
            scaler = StandardScaler().fit(space[outer])
            scaled = scaler.transform(space).astype(np.float32)
            oof, held = build_oof(scaled[outer], labels[outer], scaled[test],
                                  n_classes, args.k)
            lift = np.zeros((len(labels), n_classes), dtype=np.float32)
            lift[outer], lift[test] = oof, held
            log(f"  fold {fold}: log-lift em {time.perf_counter() - started:.0f}s")
        for name, index, kind in arms:
            if (name, fold) in done:
                continue
            started = time.perf_counter()
            metrics, lam = train_arm(blocks, None if index is None else lift, labels,
                                      fit, validation, test, n_classes, index,
                                      SEED, args.epochs, args.patience)
            score = metrics["bacc"]
            rows.append({"condition": name, "fold": fold, **metrics,
                         "lambda": ";".join(f"{v:.4f}" for v in lam),
                         "elapsed_seconds": time.perf_counter() - started})
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
            log(f"  {name:<26} fold {fold} | BAcc {score:5.2f} | "
                f"λ=[{rows[-1]['lambda']}] | {rows[-1]['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    reference = table[table.condition == "hand-8"].set_index("fold")["bacc"]
    log("\n=== marginal sobre hand-8 (pareado) ===")
    for name, _, _ in arms:
        current = table[table.condition == name].set_index("fold")["bacc"]
        line = f"  {name:<26} {current.mean():6.2f} ± {current.std(ddof=1):4.2f}"
        if name != "hand-8" and len(current) == len(reference) == N_FOLDS:
            delta = (current - reference).dropna()
            line += (f"   Δ {delta.mean():+5.2f} ± {delta.std(ddof=1):4.2f}pp"
                     f"  p={stats.ttest_rel(current, reference).pvalue:.4f}"
                     f"  {int((delta > 0).sum())}/{len(delta)}")
        log(line)
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
