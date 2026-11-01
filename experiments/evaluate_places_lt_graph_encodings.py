"""Como entregar o grafo ao MLP no Places-LT, sem que ele dilua o encoder.

Medido em 25/ago: o bloco do grafo inteiro (762d, dos quais 730 são posterior
de classe) **dobra o recall da cauda** (2,31 -> 5,59 em `few`) e conserta a
inversão de sinal do ganho de agregação, mas custa −2,91pp em `many` e −3,50 em
`medium`, dando −1,97 no total. Num concat de 1.369d o bloco domina a cabeça
compartilhada e dilui o que o encoder aprendeu onde havia dado.

Cinco formas de entregar o mesmo grafo, da mais cara à mais barata em dimensão:

* `completo`     -- as 762d, o que já foi medido. Controle.
* `stats`        -- só as 32 estatísticas (margem, entropia, densidade,
                    concordância). Sem identidade de classe nenhuma.
* `top3-esparso` -- o posterior zerado fora das 3 maiores entradas. Continua
                    365d, mas 3-esparso.
* `top3-embed`   -- o mesmo vetor esparso por um `Linear(365->16, bias=False)`,
                    que é **algebricamente idêntico** a uma soma ponderada de
                    embeddings de classe: sum_k valor_k * W[:, classe_k]. É a
                    forma correta de dar um ID nominal a uma rede; passar o
                    índice como escalar imporia uma ordem inexistente entre
                    classes.
* `logit-add`    -- o log-lift NÃO entra como feature: entra somado aos logits,
                    `logits = MLP(visual) + lambda * log-lift`, com lambda
                    escalar aprendido. Custa um parâmetro em vez de 730x512,
                    preserva as 365 identidades na posição certa por construção
                    e **não pode diluir o encoder**, porque não toca no branch.
                    É o decoupling de Kang et al. (ICLR 2020) explícito, e a
                    mesma receita de kNN-LM e de logit adjustment.

O que cada resultado significa: se `stats` sozinho já entrega o ganho da cauda,
o sinal útil é confiança de vizinhança, não identidade. Se só as variantes com
identidade entregam, é identidade mesmo. E se `logit-add` mantiver a cauda SEM
o custo na cabeça, o problema era dominância do bloco -- não uma troca inerente
entre cabeça e cauda.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402

BANDS = ("many", "medium", "few")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class GraphMLP(nn.Module):
    """BranchMLP largo, com duas extensões para este experimento.

    `encoder_widths` permite um branch estreito para o bloco esparso do grafo
    (o embedding de 16d), em vez dos 512 uniformes.

    `logit_bias` soma um vetor por classe à saída, com peso escalar aprendido.
    O lambda nasce em zero: no primeiro passo o modelo é exatamente o baseline
    sem grafo, e qualquer ganho é atribuível ao termo.
    """

    def __init__(self, sizes, n_classes, encoder_widths=None,
                 head=(2048, 1024, 1024), logit_bias=False):
        super().__init__()
        self.sizes = sizes
        widths = encoder_widths or [512] * len(sizes)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(s, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(0.25))
            for s, w in zip(sizes, widths)])
        layers, previous = [], sum(widths)
        for width in head:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width),
                       nn.GELU(), nn.Dropout(0.35)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.head = nn.Sequential(*layers)
        self.log_lambda = nn.Parameter(torch.zeros(1)) if logit_bias else None

    def forward(self, x, bias=None):
        parts = torch.split(x, self.sizes, dim=1)
        logits = self.head(torch.cat([e(p) for e, p in zip(self.encoders, parts)], dim=1))
        if self.log_lambda is not None and bias is not None:
            logits = logits + self.log_lambda * bias
        return logits


def top_k_sparse(posterior: np.ndarray, k: int) -> np.ndarray:
    """Zera tudo fora das k maiores entradas de cada linha.

    Mantém as 365 posições: a identidade da classe fica codificada pela
    POSIÇÃO, não por um valor. Passar o índice como número imporia uma ordem
    entre classes que não existe.
    """
    out = np.zeros_like(posterior)
    index = np.argpartition(-posterior, k, axis=1)[:, :k]
    rows = np.arange(len(posterior))[:, None]
    out[rows, index] = posterior[rows, index]
    return out


def fit(blocks, labels, fit_idx, val_idx, test_idx, n_classes, *,
        encoder_widths=None, bias_block=None, seed=SEED, epochs=150, patience=15):
    scalers = [StandardScaler().fit(b[fit_idx]) for b in blocks]

    def stack(index):
        return np.hstack([s.transform(b[index]) for s, b in zip(scalers, blocks)]).astype(np.float32)

    X, V, T = stack(fit_idx), stack(val_idx), stack(test_idx)
    bias = {"fit": None, "val": None, "test": None}
    if bias_block is not None:
        bias = {"fit": torch.from_numpy(bias_block[fit_idx]),
                "val": torch.from_numpy(bias_block[val_idx]).to(DEVICE),
                "test": torch.from_numpy(bias_block[test_idx]).to(DEVICE)}

    seed_everything(seed)
    model = GraphMLP([b.shape[1] for b in blocks], n_classes,
                     encoder_widths=encoder_widths,
                     logit_bias=bias_block is not None).to(DEVICE)
    counts = np.bincount(labels[fit_idx], minlength=n_classes).astype(np.float32)
    weight = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    tensors = [torch.from_numpy(X), torch.from_numpy(labels[fit_idx]).long()]
    if bias["fit"] is not None:
        tensors.append(bias["fit"])
    loader = DataLoader(TensorDataset(*tensors), batch_size=4096, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_tensor, test_tensor = torch.from_numpy(V).to(DEVICE), torch.from_numpy(T).to(DEVICE)

    best, state, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            bb = batch[2].to(DEVICE) if len(batch) > 2 else None
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb, bb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predicted = model(val_tensor, bias["val"]).argmax(1).cpu().numpy()
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
        predicted = model(test_tensor, bias["test"]).argmax(1).cpu().numpy()
    truth = labels[test_idx]
    recall = np.array([(predicted[truth == c] == c).mean() if (truth == c).any() else np.nan
                       for c in range(n_classes)])
    lam = float(model.log_lambda.detach().cpu()) if model.log_lambda is not None else float("nan")
    return recall, lam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--graph", type=Path,
                        default=ROOT / "outputs/places_lt_similarity_graph/graph_block.npy")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/places_lt_graph_encodings.csv")
    args = parser.parse_args()

    blocks_by_name, raw_labels, frame = load_corpus("places_lt", VISUAL)
    blocks_by_name = {n: clean(b) for n, b in blocks_by_name.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    test = np.flatnonzero(frame["split"].eq("test").to_numpy())
    fit_idx, val_idx = train_test_split(train, test_size=0.1,
                                        stratify=labels[train], random_state=SEED)
    frequency = np.bincount(labels[train], minlength=n_classes)
    # Convenção de Liu et al., CVPR 2019: many >100, medium 20-100, few <20.
    band = np.where(frequency > 100, "many", np.where(frequency >= 20, "medium", "few"))

    graph = np.load(args.graph)
    log_lift, stats = graph[:, :n_classes], graph[:, 2 * n_classes:]
    # O top-k opera sobre o log-lift deslocado para não-negativo: zerar entradas
    # de um vetor que vai a -14,6 faria "zero" significar uma classe provável.
    sparse = top_k_sparse(log_lift - log_lift.min(axis=1, keepdims=True), args.top_k)
    hand = [blocks_by_name[n] for n in VISUAL]
    log(f"{len(labels):,} imagens, {n_classes} classes, train={len(train):,} test={len(test):,}")
    log(f"grafo {graph.shape[1]}d -> log-lift {log_lift.shape[1]}d, stats {stats.shape[1]}d; "
        f"top-{args.top_k} com {int((sparse != 0).sum(1).mean())} não-zeros/linha")

    variants = {
        "hand (sem grafo)": dict(blocks=hand),
        "completo":         dict(blocks=hand + [graph]),
        "stats":            dict(blocks=hand + [stats]),
        "top3-esparso":     dict(blocks=hand + [sparse]),
        "top3-embed":       dict(blocks=hand + [sparse],
                                 encoder_widths=[512] * len(hand) + [args.embed_dim]),
        "logit-add":        dict(blocks=hand, bias_block=log_lift.astype(np.float32)),
    }

    done = set()
    if args.output.exists():
        done = set(pd.read_csv(args.output)["variant"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name, kwargs in variants.items():
        if name in done:
            continue
        started = time.perf_counter()
        recall, lam = fit(labels=labels, fit_idx=fit_idx, val_idx=val_idx,
                          test_idx=test, n_classes=n_classes, **kwargs)
        extra = sum(b.shape[1] for b in kwargs["blocks"]) - sum(b.shape[1] for b in hand)
        row = {"variant": name, "extra_dims": extra, "lambda": lam,
               "seconds": time.perf_counter() - started,
               "overall": float(np.nanmean(recall))}
        row.update({b: float(np.nanmean(recall[band == b])) for b in BANDS})
        pd.DataFrame([row]).to_csv(args.output, mode="a", index=False,
                                   header=not args.output.exists(), float_format="%.6f")
        np.save(args.output.parent / f"places_lt_recall_{name.split()[0].replace('-', '_')}.npy",
                recall)
        log(f"  {name:<18} +{extra:>4}d  total {100 * row['overall']:5.2f} | " +
            " ".join(f"{b} {100 * row[b]:5.2f}" for b in BANDS) +
            ("" if np.isnan(lam) else f"   lambda={lam:+.3f}"))

    detail = pd.read_csv(args.output).drop_duplicates("variant", keep="last")
    base = detail[detail.variant.eq("hand (sem grafo)")]
    print()
    print(f"{'variante':<18} {'+dims':>6} {'total':>7} {'many':>7} {'medium':>7} {'few':>7}"
          f" {'Δ total':>9} {'Δ few':>8}")
    for row in detail.itertuples():
        delta = ""
        if not base.empty:
            delta = (f"{100 * (row.overall - base.overall.iloc[0]):>+9.2f} "
                     f"{100 * (row.few - base.few.iloc[0]):>+8.2f}")
        print(f"{row.variant:<18} {int(row.extra_dims):>6} {100 * row.overall:>7.2f} "
              f"{100 * row.many:>7.2f} {100 * row.medium:>7.2f} {100 * row.few:>7.2f}{delta}")


if __name__ == "__main__":
    main()
