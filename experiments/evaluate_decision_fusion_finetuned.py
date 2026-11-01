"""Nossas features ainda somam sobre uma ResNet50 FINE-TUNED?

A objeção que a fusão com backbone convida é "o deep learning só mudou de
lugar: em vez de fine-tunar, vocês treinam um MLP junto com o congelado". O
adversário correto de uma fusão treinada é um backbone que recebeu todo o treino
possível no alvo -- e no WikiArt ele existe: `outputs/cnn_imagenet_finetuned`,
ResNet50 ImageNet fine-tunada, 64,71 BAcc em 5 folds.

**As imagens do WikiArt não estão mais em disco**, então não dá para passar os
checkpoints e extrair a penúltima camada: a fusão em nível de *features* está
bloqueada nesse corpus. O que existe são os **posteriors OOF por fold**
(`fold_0N_posteriors.npz`, com `stems`), cobrindo as 79.207 imagens uma vez
cada. Isso permite fusão em nível de *decisão*, que é mais fraca mas
**inteiramente livre de vazamento**: cada posterior vem de um modelo que não viu
aquela imagem, e a regra de combinação aqui não tem parâmetro ajustado.

Três números:

* `oráculo`  -- fração em que **pelo menos um** dos dois acerta. É o teto de
                qualquer combinação dos dois, e mede complementaridade bruta.
* `média`    -- média das log-probabilidades, sem nenhum parâmetro. É o piso
                honesto de uma combinação.
* `peso w`   -- melhor `w` fixo em `w*log p_nosso + (1-w)*log p_finetuned`,
                varrido em grade. Um único escalar, ajustado no próprio teste --
                é otimista de propósito, serve de teto de regra linear.

Se o oráculo for muito acima do fine-tuned sozinho, existe informação nossa que
o fine-tuning não capturou, e a objeção do "deslocamento" perde força. Se a
média já superar o fine-tuned, a evidência é direta e sem ajuste nenhum.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

SEED, N_FOLDS = 42, 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def our_oof_posteriors(names, epochs, patience):
    """Posteriors OOF do MLP de produção sobre o banco visual do WikiArt."""
    import copy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from registry import load_blocks
    from evaluate_full_factorial_ablation import scaled_splits
    from evaluate_robustness_mlp_aug import DEVICE, BranchMLP, seed_everything

    cohort = load_blocks("wikiart", names, allow_external=False)
    blocks = [cohort.blocks[n].astype(np.float32) for n in names if n in cohort.blocks]
    stems = np.asarray(cohort.stems).astype(str)
    labels = LabelEncoder().fit(np.asarray(cohort.labels))
    y = labels.transform(np.asarray(cohort.labels))
    n_classes = len(labels.classes_)
    log(f"nosso banco: {len(y):,} imagens, {sum(b.shape[1] for b in blocks)}d, "
        f"{n_classes} classes")

    posteriors = np.zeros((len(y), n_classes), dtype=np.float32)
    for fold, (train, test) in enumerate(StratifiedKFold(
            N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y), start=1):
        fit, val = train_test_split(train, test_size=0.1, stratify=y[train],
                                    random_state=SEED + fold)
        X, V, T = scaled_splits(blocks, fit, val, test)
        seed_everything(SEED + fold)
        model = BranchMLP([b.shape[1] for b in blocks], n_classes).to(DEVICE)
        counts = np.bincount(y[fit], minlength=n_classes).astype(np.float32)
        weight = (1 / np.maximum(counts, 1)) / (1 / np.maximum(counts, 1)).mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y[fit]).long()),
                            batch_size=4096, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + fold))
        vt, tt = torch.from_numpy(V).to(DEVICE), torch.from_numpy(T).to(DEVICE)
        best, state, stale = -1.0, None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                score = balanced_accuracy_score(y[val], model(vt).argmax(1).cpu().numpy())
            if score > best:
                best, state, stale = score, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= patience:
                break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            posteriors[test] = torch.softmax(model(tt), 1).cpu().numpy()
        log(f"  fold {fold}: BAcc {100 * balanced_accuracy_score(y[test], posteriors[test].argmax(1)):.2f}")
    return stems, y, posteriors, list(labels.classes_)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned-dir", type=Path,
                        default=ROOT / "outputs/cnn_imagenet_finetuned")
    parser.add_argument("--blocks", nargs="+",
                        default=["Descriptors", "Felzenszwalb", "SLIC", "Line graph",
                                 "L(G_felz)", "Hilbert", "Fourier"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "outputs/wikiart_our_oof_posteriors.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/decision_fusion_finetuned.csv")
    args = parser.parse_args()

    if args.cache.exists():
        blob = np.load(args.cache, allow_pickle=True)
        stems, y, ours, classes = (blob["stems"].astype(str), blob["y"],
                                   blob["probabilities"], list(blob["classes"]))
        log(f"posteriors nossos em cache: {ours.shape}")
    else:
        stems, y, ours, classes = our_oof_posteriors(args.blocks, args.epochs, args.patience)
        with open(args.cache, "wb") as handle:
            np.savez_compressed(handle, stems=stems, y=y, probabilities=ours,
                                classes=np.array(classes, dtype=object))

    # Fine-tuned: juntar os 5 folds e alinhar classes pelo classes.json.
    ft_classes = json.loads((args.finetuned_dir / "classes.json").read_text())
    if isinstance(ft_classes, dict):
        ft_classes = [c for c, _ in sorted(ft_classes.items(), key=lambda kv: kv[1])]
    ft_stems, ft_probabilities = [], []
    for fold in range(1, N_FOLDS + 1):
        blob = np.load(args.finetuned_dir / f"fold_{fold:02d}_posteriors.npz", allow_pickle=True)
        ft_stems.append(blob["stems"].astype(str))
        ft_probabilities.append(blob["probabilities"])
    ft_stems = np.concatenate(ft_stems)
    ft_probabilities = np.concatenate(ft_probabilities).astype(np.float32)
    log(f"fine-tuned: {ft_probabilities.shape}, {len(set(ft_stems))} stems únicos")

    order = {c: i for i, c in enumerate(ft_classes)}
    if list(classes) != list(ft_classes):
        missing = set(classes) - set(order)
        if missing:
            raise ValueError(f"classes sem correspondência: {sorted(missing)[:5]}")
        ft_probabilities = ft_probabilities[:, [order[c] for c in classes]]
        log("colunas do fine-tuned reordenadas para a nossa codificação")

    position = {s: i for i, s in enumerate(ft_stems)}
    rows = np.array([position.get(s, -1) for s in stems])
    keep = rows >= 0
    log(f"interseção: {int(keep.sum()):,} de {len(stems):,}")
    ours, truth = ours[keep], y[keep]
    theirs = ft_probabilities[rows[keep]]

    ours_log = np.log(np.clip(ours, 1e-9, None))
    theirs_log = np.log(np.clip(theirs, 1e-9, None))
    ours_ok = ours.argmax(1) == truth
    theirs_ok = theirs.argmax(1) == truth

    results = [
        ("nosso (hand-designed)", balanced_accuracy_score(truth, ours.argmax(1))),
        ("ResNet50 fine-tuned", balanced_accuracy_score(truth, theirs.argmax(1))),
        ("média das log-probs", balanced_accuracy_score(
            truth, (ours_log + theirs_log).argmax(1))),
    ]
    best_w, best_score = None, -1.0
    for w in np.linspace(0, 1, 21):
        score = balanced_accuracy_score(truth, (w * ours_log + (1 - w) * theirs_log).argmax(1))
        if score > best_score:
            best_w, best_score = w, score
    results.append((f"melhor peso fixo (w={best_w:.2f})", best_score))

    print()
    for name, score in results:
        print(f"{name:<28} BAcc {100 * score:6.2f}")
    print()
    print(f"{'ambos acertam':<28} {100 * (ours_ok & theirs_ok).mean():6.2f}%")
    print(f"{'só o fine-tuned acerta':<28} {100 * (~ours_ok & theirs_ok).mean():6.2f}%")
    print(f"{'SÓ O NOSSO acerta':<28} {100 * (ours_ok & ~theirs_ok).mean():6.2f}%")
    print(f"{'nenhum acerta':<28} {100 * (~ours_ok & ~theirs_ok).mean():6.2f}%")
    print(f"{'ORÁCULO (algum acerta)':<28} {100 * (ours_ok | theirs_ok).mean():6.2f}%")
    print()
    print("Leitura: 'só o nosso acerta' é informação que o fine-tuning NÃO capturou.")
    print("Se for substancial, a objeção do 'deep learning deslocado' perde força.")

    pd.DataFrame([{"condition": n, "bacc": s} for n, s in results] + [
        {"condition": "oráculo (algum acerta)", "bacc": float((ours_ok | theirs_ok).mean())},
        {"condition": "só o nosso acerta", "bacc": float((ours_ok & ~theirs_ok).mean())},
        {"condition": "só o fine-tuned acerta", "bacc": float((~ours_ok & theirs_ok).mean())},
    ]).to_csv(args.output, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
