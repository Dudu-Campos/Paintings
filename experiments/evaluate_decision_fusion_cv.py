"""Fusao de decisao em corpus de validacao cruzada (os tres artisticos).

Complemento de `evaluate_decision_fusion_official.py`, que so atende split
oficial. Aqui os dois lados sao 5-fold CV sobre a MESMA coorte, entao cada
imagem aparece exatamente uma vez como teste em cada lado -- basta montar o
vetor OOF de corpus inteiro dos dois e alinhar por `stem`. **As fronteiras de
fold NAO precisam coincidir**, o que evita depender de o StratifiedKFold do
trainer casar com o nosso.

Nosso lado: configuracao de producao do §3 -- hand-8 + grafo de similaridade
com OOF aninhado, BranchMLP ESTREITO (§7.8). Cinco sementes, emparelhadas seed
a seed com as do backbone.

A fusao e a media das log-probabilidades, zero parametros -- o unico instrumento
valido contra backbone fine-tuned (§4.1c).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_decision_fusion_official import decompose, log_average, scores  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features)
from evaluate_robustness_mlp_aug import BranchMLP, DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
GRID = ROOT / "outputs/final_finetune_grid"
GRAPH = dict(n_neighbors=30, alpha=0.4, max_iter=100)


def our_oof(blocks, labels, folds, n_classes, seed, epochs, patience, use_graph,
            text_fn=None):
    """Posterior OOF de corpus inteiro: cada linha prevista quando estava fora."""
    posteriors = np.zeros((len(labels), n_classes), dtype=np.float32)
    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        fit_idx, val_idx = train_test_split(train_idx, test_size=0.10, random_state=seed,
                                            stratify=labels[train_idx])
        parts, sizes = [[], [], []], []
        for block in blocks:
            scaler = StandardScaler().fit(block[fit_idx])
            for target, index in zip(parts, (fit_idx, val_idx, test_idx)):
                target.append(scaler.transform(block[index]).astype(np.float32))
            sizes.append(block.shape[1])
        if use_graph:
            source = np.hstack(blocks)
            scaler = StandardScaler().fit(source[fit_idx])
            space = scaler.transform(source).astype(np.float32)
            oof, held = rich_similarity_features(
                space[fit_idx], labels[fit_idx],
                np.vstack([space[val_idx], space[test_idx]]),
                n_classes, seed=seed, **GRAPH)
            oof = prior_correct_rich(oof, labels[fit_idx], n_classes)
            held = prior_correct_rich(held, labels[fit_idx], n_classes)
            for target, values in zip(parts, (oof, held[:len(val_idx)], held[len(val_idx):])):
                target.append(values.astype(np.float32))
            sizes.append(oof.shape[1])
        # O texto entra DEPOIS do grafo, de proposito: o kNN do grafo roda sobre
        # `np.hstack(blocks)`, que e o espaco hand-8 da configuracao publicada.
        # Acrescentar o texto antes mudaria a vizinhanca e o experimento deixaria
        # de ser "a mesma fusao, mais um bloco".
        if text_fn is not None:
            text = text_fn(fold_no, train_idx, test_idx)
            scaler = StandardScaler().fit(text[fit_idx])
            for target, index in zip(parts, (fit_idx, val_idx, test_idx)):
                target.append(scaler.transform(text[index]).astype(np.float32))
            sizes.append(text.shape[1])
        train_x, val_x, test_x = (np.hstack(p) for p in parts)

        seed_everything(seed)
        model = BranchMLP(sizes, n_classes).to(DEVICE)
        counts = np.bincount(labels[fit_idx], minlength=n_classes).astype(np.float32)
        inverse = 1.0 / np.maximum(counts, 1)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(inverse / inverse.mean(), device=DEVICE),
                                        label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = DataLoader(TensorDataset(torch.from_numpy(train_x),
                                          torch.from_numpy(labels[fit_idx]).long()),
                            batch_size=4096, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        val_tensor = torch.from_numpy(val_x).to(DEVICE)
        best, state, stale = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                score = balanced_accuracy_score(labels[val_idx],
                                                model(val_tensor).argmax(1).cpu().numpy())
            if score > best:
                best, state, stale = score, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= patience:
                break
        if state is not None:
            model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            posteriors[test_idx] = torch.softmax(
                model(torch.from_numpy(test_x).to(DEVICE)), dim=1).cpu().numpy()
    return posteriors


def backbone_oof(grid: Path, model: str, seed: int, stems: np.ndarray, n_classes: int):
    """Junta os 5 folds da grade num posterior de corpus inteiro."""
    values = np.zeros((len(stems), n_classes), dtype=np.float32)
    seen = np.zeros(len(stems), dtype=bool)
    index = pd.Index(stems)
    for path in sorted((grid / model / f"seed_{seed}").glob("fold_*/fold_*_posteriors.npz")):
        blob = np.load(path, allow_pickle=True)
        order = index.get_indexer(blob["stems"].astype(str))
        if (order < 0).any():
            raise ValueError(f"{path}: {int((order < 0).sum())} stems fora da coorte")
        values[order] = blob["probabilities"]
        seen[order] = True
    if not seen.all():
        raise ValueError(f"{model} seed {seed}: {int((~seen).sum())} linhas sem posterior")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="oilpainting")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"])
    parser.add_argument("--no-graph", action="store_true", help="nosso lado sem o grafo")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--posteriors", type=Path, default=None)
    parser.add_argument("--with-text", action="store_true",
                        help="acrescenta o canal de texto (OOF aninhado por fold)")
    parser.add_argument("--captions", default="estudante")
    # O nome do dataset no registry e o do diretorio da grade divergem no
    # multitask: as features estao sob `multitask` e os posteriors do backbone
    # sob `multitask125`. `build_artistic_matrix.py` ja carrega essa excecao.
    parser.add_argument("--grid-name", default=None,
                        help="subdiretorio de outputs/final_finetune_grid, se "
                             "diferente de --dataset")
    args = parser.parse_args()
    sufixo = "_text" if args.with_text else ""
    args.output = args.output or ROOT / f"results/{args.dataset}_decision_fusion_cv{sufixo}.csv"
    args.posteriors = args.posteriors or ROOT / f"outputs/{args.dataset}_our_oof_cv{sufixo}.npz"
    grid = GRID / (args.grid_name or args.dataset)

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    encoder = LabelEncoder().fit(np.asarray(cohort.labels, dtype=str))
    labels = encoder.transform(np.asarray(cohort.labels, dtype=str))
    n_classes = len(encoder.classes_)
    blocks = [cohort.blocks[n] for n in HAND8]
    dimensions = int(sum(b.shape[1] for b in blocks)) + (0 if args.no_graph else 2 * n_classes + 32)
    log(f"{args.dataset}: {len(stems):,} imagens, {n_classes} classes, "
        f"{dimensions}d {'(sem grafo)' if args.no_graph else '(hand-8 + grafo)'}")

    reference = json.loads((grid / args.models[0] / f"seed_{args.seeds[0]}/fold_1/classes.json").read_text())
    if list(encoder.classes_) != list(reference):
        raise ValueError(f"ordem de classe divergente:\n  nossa={list(encoder.classes_)}\n  grade={reference}")

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(stems, labels))

    text_fn = None
    if args.with_text:
        from text_fusion_block import load_texts, make_text_fn
        classes = np.asarray(encoder.classes_, dtype=str)
        labels_str = np.asarray(cohort.labels, dtype=str)
        texts = load_texts(args.dataset, stems, captions=args.captions)
        # O cache e por fold e NAO por semente: os folds externos sao fixos em
        # SEED=42, entao o bloco de texto e o mesmo para as cinco sementes. E o
        # que torna o custo do canal linear em corpora, nao em unidades.
        text_fn = make_text_fn(args.dataset, stems, labels_str, classes,
                               ROOT / "outputs/text_nested" / args.dataset, texts)
        log(f"canal de texto ligado: {len(texts):,} legendas, "
            f"OOF aninhado por fold em outputs/text_nested/{args.dataset}")

    ours = {}
    if args.posteriors.exists():
        cached = np.load(args.posteriors, allow_pickle=True)
        if list(cached["stems"]) == list(stems):
            ours = {s: cached[f"seed_{s}"] for s in args.seeds if f"seed_{s}" in cached}
            log(f"reaproveitando {len(ours)} posteriors de {args.posteriors}")
    for seed in args.seeds:
        if seed in ours:
            continue
        started = time.perf_counter()
        ours[seed] = our_oof(blocks, labels, folds, n_classes, seed,
                             args.epochs, args.patience, not args.no_graph,
                             text_fn=text_fn)
        log(f"nosso seed {seed}: BAcc {scores(labels, ours[seed])['bacc']:.2f} "
            f"({time.perf_counter() - started:.0f}s)")
    args.posteriors.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.posteriors, stems=stems, y_true=labels,
                        **{f"seed_{s}": v for s, v in ours.items()})

    rows = []
    for model in args.models:
        for seed in args.seeds:
            backbone = backbone_oof(grid, model, seed, stems, n_classes)
            side = ours[seed]
            row = {"dataset": args.dataset, "model": model, "seed": seed, "dimensions": dimensions}
            row |= {f"ours_{k}": v for k, v in scores(labels, side).items()}
            row |= {f"backbone_{k}": v for k, v in scores(labels, backbone).items()}
            row |= {f"fusion_{k}": v for k, v in scores(labels, log_average(side, backbone)).items()}
            row |= decompose(labels, side, backbone)
            rows.append(row)
            log(f"{model:9s} seed {seed}: nosso {row['ours_bacc']:.2f} | backbone "
                f"{row['backbone_bacc']:.2f} | fusao {row['fusion_bacc']:.2f} "
                f"(Delta {row['fusion_bacc'] - row['backbone_bacc']:+.2f}) | "
                f"SO NOSSO {row['only_ours']:.2f}%")

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output, index=False, float_format="%.4f")
    log(f"\nsalvo: {args.output}")
    for model, block in detail.groupby("model"):
        delta = block.fusion_bacc - block.backbone_bacc
        capture = 100 * block.only_ours / (block.only_ours + block.neither)
        log(f"\n=== {args.dataset} / {model} ({len(block)} seeds) ===")
        for field in ["ours_bacc", "backbone_bacc", "fusion_bacc"]:
            log(f"  {field:16s} {block[field].mean():6.2f} +- {block[field].std():.2f}")
        log(f"  Delta            {delta.mean():+6.2f} +- {delta.std():.2f}  "
            f"({int((delta > 0).sum())}/{len(block)})")
        log(f"  so nosso {block.only_ours.mean():.2f}% | erro do backbone "
            f"{(block.only_ours + block.neither).mean():.2f}% | CAPTURA {capture.mean():.1f}%")


if __name__ == "__main__":
    main()
