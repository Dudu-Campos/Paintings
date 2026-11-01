"""Quais features o BranchMLP de produção não está usando -- e a poda ajuda?

Motivação: a curva de seleção mostra k=512 (48,09) batendo k=639 (46,99), ou
seja **remover features melhora**. Isto localiza quais.

Duas etapas, e a segunda só faz sentido com o controle:

**Etapa 1, permutation importance.** Treina o BranchMLP uma vez por fold e,
sem retreinar, permuta cada coluna do conjunto de **validação** medindo a queda
de BAcc. 639 features x R repetições x folds, só forward -- minutos.

Na validação, **não** no teste: ranquear por queda no teste e depois avaliar o
modelo podado no mesmo teste é circular, e com ruído de ~0,5pp entre folds o
viés de seleção explicaria o efeito inteiro. O teste é tocado uma vez por
condição, e nunca entra na escolha das features.

**Etapa 2, poda e retreino.** Remove as N piores e retreina. **Sempre pareado
com remover N features ALEATÓRIAS**, porque sem esse controle não se distingue
"removi as inúteis" de "reduzir dimensionalidade ajuda qualquer que seja o
corte" -- e a curva de seleção já mostra que o segundo efeito existe.

**Viés declarado.** Permutation importance subestima features redundantes: ao
permutar uma, a gêmea continua entregando o sinal e o modelo não perde nada. O
§5.9 mediu que neste banco redundância é *positivamente* associada a uso
(r=+0,60 parcial), então o fundo do ranking deve vir enriquecido em features
redundantes -- exatamente as que não se quer remover. Por isso a saída cruza a
importância com p95(r²): se a correlação for negativa, o ranking está
contaminado e a poda por ele é a decisão errada.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_robustness_mlp_aug import BranchMLP, DEVICE, seed_everything  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def train(x, y, xv, yv, sizes, n_classes, seed, epochs=150, patience=15):
    seed_everything(seed)
    model = BranchMLP(sizes, n_classes).to(DEVICE)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weight = (1.0 / np.maximum(counts, 1)) / (1.0 / np.maximum(counts, 1)).mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tx = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor(y, dtype=torch.long, device=DEVICE)
    tv = torch.tensor(xv, dtype=torch.float32, device=DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tx, ty), batch_size=512, shuffle=True, drop_last=True)
    best, state, waited = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = balanced_accuracy_score(yv, model(tv).argmax(1).cpu().numpy())
        if score > best:
            best, waited = score, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def score(model, x, y):
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        return 100 * balanced_accuracy_score(y, model(t).argmax(1).cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prune", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    blocks, raw, _ = load_corpus(args.dataset, VISUAL + ["Gabor"])
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    mask = np.array([key(v) in wanted for v in raw])
    if len(np.unique(raw[mask])) != len(wanted):
        raise ValueError("casamento de classes do benchmark incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    raw = raw[mask]
    names = [n for n in VISUAL + ["Gabor"] if n in blocks]
    features = np.hstack([blocks[n] for n in names])
    block_of = np.concatenate([np.full(blocks[n].shape[1], i) for i, n in enumerate(names)])
    labels = LabelEncoder().fit_transform(raw)
    n_classes = len(np.unique(labels))
    log(f"{features.shape[1]} features em {len(names)} blocos, {n_classes} classes")

    importance = np.zeros((args.folds, features.shape[1]), dtype=np.float32)
    rows = []
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    for fold, (train_index, test_index) in enumerate(splits[:args.folds], start=1):
        fit_idx, val_idx = train_test_split(
            train_index, test_size=0.1, stratify=labels[train_index], random_state=SEED + fold)
        scaler = StandardScaler().fit(features[train_index])
        scaled = scaler.transform(features).astype(np.float32)
        sizes = [blocks[n].shape[1] for n in names]

        model = train(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx],
                      sizes, n_classes, SEED + fold)
        base_test = score(model, scaled[test_index], labels[test_index])
        # A permutação roda na VALIDAÇÃO, não no teste. Medir a importância no
        # teste, ranquear por ela e depois avaliar o modelo podado no MESMO
        # teste é circular: as features saem por não atrapalharem justamente
        # aquele conjunto. Com ruído de ~0,5pp entre folds, esse viés explicaria
        # o efeito inteiro.
        base_val = score(model, scaled[val_idx], labels[val_idx])
        log(f"fold {fold}: teste {base_test:.2f}  validação {base_val:.2f}")

        rng = np.random.default_rng(SEED + fold)
        probe_x = scaled[val_idx].copy()
        probe_y = labels[val_idx]
        for column in range(features.shape[1]):
            keep = probe_x[:, column].copy()
            drops = []
            for _ in range(args.repeats):
                probe_x[:, column] = keep[rng.permutation(len(keep))]
                drops.append(base_val - score(model, probe_x, probe_y))
            probe_x[:, column] = keep
            importance[fold - 1, column] = float(np.mean(drops))
            if column % 128 == 0:
                log(f"  permutação {column}/{features.shape[1]}")

        rows.append({"fold": fold, "condition": "completo", "dimensions": features.shape[1],
                     "bacc": base_test})

        # --- etapa 2: poda pelas piores, contra poda ALEATÓRIA do mesmo tamanho
        ranking = np.argsort(importance[fold - 1])
        for n_drop in args.prune:
            for label, dropped in (("poda pior", ranking[:n_drop]),
                                   ("poda aleatória", rng.choice(features.shape[1],
                                                                 n_drop, replace=False))):
                keep_mask = np.ones(features.shape[1], dtype=bool)
                keep_mask[dropped] = False
                new_sizes = [int((block_of[keep_mask] == i).sum()) for i in range(len(names))]
                if min(new_sizes) == 0:
                    log(f"  n={n_drop} {label}: bloco esvaziado, pulado")
                    continue
                pruned = scaled[:, keep_mask]
                m = train(pruned[fit_idx], labels[fit_idx], pruned[val_idx], labels[val_idx],
                          new_sizes, n_classes, SEED + fold)
                value = score(m, pruned[test_index], labels[test_index])
                rows.append({"fold": fold, "condition": f"{label} -{n_drop}",
                             "dimensions": int(keep_mask.sum()), "bacc": value})
                log(f"  -{n_drop:>3} {label:<16} {int(keep_mask.sum()):>3}d  {value:6.2f}")
        pd.DataFrame(rows).to_csv(args.output_dir / "branch_feature_pruning.csv", index=False)
        np.save(args.output_dir / "branch_permutation_importance.npy", importance)

    frame = pd.DataFrame(rows)
    log("\n" + frame.groupby("condition")[["dimensions", "bacc"]]
        .agg({"dimensions": "first", "bacc": ["mean", "std"]}).round(2).to_string())


if __name__ == "__main__":
    main()
