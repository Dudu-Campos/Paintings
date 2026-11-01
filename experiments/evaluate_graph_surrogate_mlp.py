"""Surrogate dos blocos de grafo: dá para prevê-los a partir dos blocos baratos?

Os quatro blocos de segmentação (Felz 71, SLIC 71, L(G_felz) 63, L(G_slic) 63 =
268d dos 617) são ~98% do custo de extração: 2.100 ms/img contra 37 ms dos
outros quatro (Descriptors 130, Hilbert 57, Fourier 134, Gabor 32 = 349d).

Aqui um MLP aprende `[349d baratos] -> [268d de grafo]`, sem ver rótulo nenhum, e
o pipeline roda com os valores PREVISTOS no lugar dos reais.

Os quatro braços são obrigatórios (§9.4), senão o número não significa nada:

| braço            | o que mede                                              |
|------------------|---------------------------------------------------------|
| hand-8 real      | teto                                                    |
| hand-4 barato    | piso -- quanto os blocos de grafo valem de fato          |
| hand-4 + previsto| a proposta                                              |
| hand-4 + placebo | surrogate com pesos ALEATÓRIOS, contra "ganhou por somar |
|                  | 268 dimensões quaisquer"                                 |

**OOF interno no surrogate.** Se o surrogate for treinado no fold de treino e
aplicado a tudo, as linhas de treino recebem predição *in-sample* (boa) e as de
teste *out-of-sample* (pior) -- a mesma assimetria que quebrou o SNF (§6) e que o
§5.6 investigou no grafo. Aqui cada fold externo refaz o surrogate em 5 folds
internos, então treino e teste veem predições igualmente out-of-sample.

**Avaliar por BAcc a jusante, não por r².** O §5.17 mediu ordenamento isolado
anticorrelacionado com o do pipeline; o r² sai no CSV como diagnóstico, não como
critério.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

import evaluate_preprocess_effect_oilpainting as base  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS, INNER_FOLDS = 42, 5, 5
CHEAP = ("Descriptors", "Hilbert", "Fourier", "Gabor")
GRAPH = ("Felzenszwalb", "SLIC", "Line graph", "L(G_felz)")


class Surrogate(nn.Module):
    """349 -> 268. Largo o suficiente para não ser o gargalo do teste."""

    def __init__(self, n_in: int, n_out: int, width: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(width, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_surrogate(x_fit, y_fit, x_val, y_val, seed, epochs, patience, random_weights=False):
    seed_everything(seed)
    model = Surrogate(x_fit.shape[1], y_fit.shape[1]).to(DEVICE)
    if random_weights:            # placebo: nunca treina
        model.eval()
        return model, float("nan")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
                        batch_size=1024, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(x_val).to(DEVICE)
    yv = torch.from_numpy(y_val).to(DEVICE)
    best, state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            loss = float(nn.functional.mse_loss(model(xv), yv))
        if loss < best - 1e-5:
            best, state, stale = loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    # r2 = 1 - MSE/var, com alvo padronizado var=1
    return model, 1.0 - best


def predict(model, x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 8192):
            out.append(model(torch.from_numpy(x[i:i + 8192]).to(DEVICE)).cpu().numpy())
    return np.vstack(out)


def surrogate_block(cheap, target, train_idx, test_idx, seed, epochs, patience, placebo):
    """Predição out-of-sample para TODAS as linhas: treino por OOF interno,
    teste pelo modelo ajustado no treino externo inteiro."""
    x_scaler = StandardScaler().fit(cheap[train_idx])
    y_scaler = StandardScaler().fit(target[train_idx])
    x = x_scaler.transform(cheap).astype(np.float32)
    y = y_scaler.transform(target).astype(np.float32)

    out = np.zeros_like(y)
    scores = []
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    dummy = np.zeros(len(train_idx), dtype=np.uint8)
    for rel_fit, rel_val in inner.split(dummy, dummy):
        fit_i, val_i = train_idx[rel_fit], train_idx[rel_val]
        model, r2 = train_surrogate(x[fit_i], y[fit_i], x[val_i], y[val_i],
                                    seed, epochs, patience, placebo)
        out[val_i] = predict(model, x[val_i])
        scores.append(r2)
    fit_i, val_i = train_test_split(train_idx, test_size=0.10, random_state=seed)
    model, r2 = train_surrogate(x[fit_i], y[fit_i], x[val_i], y[val_i],
                                seed, epochs, patience, placebo)
    out[test_idx] = predict(model, x[test_idx])
    return y_scaler.inverse_transform(out).astype(np.float32), float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--mlp-epochs", type=int, default=150)
    ap.add_argument("--mlp-patience", type=int, default=15)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multitask_graph_surrogate.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = LabelEncoder().fit_transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(np.unique(labels))
    cheap = np.hstack([cohort.blocks[n] for n in CHEAP]).astype(np.float32)
    sizes = [cohort.blocks[n].shape[1] for n in GRAPH]
    target = np.hstack([cohort.blocks[n] for n in GRAPH]).astype(np.float32)
    log(f"multitask: {len(stems):,} imagens, {n_classes} classes | "
        f"barato {cheap.shape[1]}d -> grafo {target.shape[1]}d")

    done = set()
    if args.output.exists():
        done = {(str(r.condition), int(r.fold)) for r in pd.read_csv(args.output).itertuples()}
        log(f"retomando {len(done)} linhas")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))
    cheap_blocks = [cohort.blocks[n] for n in CHEAP]
    real_blocks = [cohort.blocks[n] for n in HAND8]

    for number, (train_idx, test_idx) in enumerate(folds, 1):
        plans = [("hand-8 real", real_blocks, None),
                 ("hand-4 barato (sem grafo)", cheap_blocks, None),
                 ("hand-4 + grafo PREVISTO", None, False),
                 ("hand-4 + grafo PLACEBO", None, True)]
        for name, blocks, placebo in plans:
            if (name, number) in done:
                continue
            started = time.perf_counter()
            r2 = float("nan")
            if blocks is None:
                pred, r2 = surrogate_block(cheap, target, train_idx, test_idx,
                                           SEED, args.epochs, args.patience, placebo)
                parts, offset = [], 0
                for size in sizes:
                    parts.append(pred[:, offset:offset + size])
                    offset += size
                blocks = cheap_blocks + parts
            metrics = base.fit(blocks, labels, train_idx, test_idx, n_classes, SEED,
                               args.mlp_epochs, args.mlp_patience, "narrow", None)
            row = {"condition": name, "fold": number, "r2_surrogate": r2,
                   "seconds": time.perf_counter() - started, **metrics}
            frame = pd.DataFrame([row])
            if args.output.exists():
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{name:28s} fold {number} | {row['dimensions']:4d}d | "
                f"BAcc {metrics['bacc']:5.2f} | r2 {r2:6.3f} | {row['seconds']:.0f}s")

    frame = pd.read_csv(args.output)
    pivot = frame.pivot_table(index="fold", columns="condition", values="bacc")
    if pivot.notna().all().all():
        from scipy import stats
        b = pivot["hand-8 real"].to_numpy()
        print(f"\n{'condição':28s} {'BAcc':>6} {'Δ vs real':>10} {'p':>8} {'folds':>8}")
        for c in ["hand-8 real", "hand-4 barato (sem grafo)",
                  "hand-4 + grafo PREVISTO", "hand-4 + grafo PLACEBO"]:
            x = pivot[c].to_numpy(); d = x - b
            if c == "hand-8 real":
                print(f"{c:28s} {x.mean():6.2f}")
                continue
            _, p = stats.ttest_rel(x, b)
            print(f"{c:28s} {x.mean():6.2f} {d.mean():+10.2f} {p:8.4f} {(d < 0).sum():4d}/5 piores")
        r2 = frame[frame.condition.eq("hand-4 + grafo PREVISTO")].r2_surrogate.mean()
        print(f"\nr2 medio do surrogate (validacao interna, alvo padronizado): {r2:.3f}")


if __name__ == "__main__":
    main()
