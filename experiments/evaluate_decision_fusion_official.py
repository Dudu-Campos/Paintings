"""Fusao de decisao: banco hand-designed + backbone fine-tuned, split oficial.

Media das log-probabilidades, ZERO parametros ajustados -- o instrumento valido
contra backbone fine-tuned, porque fusao de FEATURES ali colapsa (§4.1c: a
cabeca ignora as nossas dimensoes quando o vizinho acerta 96,65% no treino).

Dois formatos de corpus, com o mesmo caminho de codigo depois da preparacao:

  minc2500  split oficial na coluna `split` do metadado (train/val/test)
  nct_crc   coorte de treino e coorte de teste SEPARADAS no registry
            (`nct_crc` = NCT-CRC-HE-100K, `nct_crc_val` = CRC-VAL-HE-7K),
            sem `val` oficial -- recorte estratificado de 10% do treino,
            que e o protocolo do §7.14.

Emparelha seed a seed com os posteriors do backbone.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_minc2500_official import HAND8, WideBranchMLP  # noqa: E402
from evaluate_robustness_mlp_aug import DEVICE, seed_everything  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SPLIT_SEED = 42
GRID = ROOT / "outputs/final_finetune_grid"
# `test_cohort` None significa que o split esta na coluna `split` do metadado.
DATASETS = {
    "minc2500": {"meta": "data/minc2500_meta.csv", "test_cohort": None},
    "nct_crc": {"meta": "data/nct_crc_meta.csv", "test_cohort": "nct_crc_val"},
    # Places-LT: o split LT publicado esta na coluna `split` (62.500 treino /
    # 7.300 val / 36.500 teste). NAO trocar por CV -- o desbalanceamento de 996x
    # so existe dentro do treino do LT, e 5 folds sobre a coorte inteira dao 41x
    # e perdem o regime que motiva o corpus (§7.5).
    "places_lt": {"meta": "data/places_lt_meta.csv", "test_cohort": None},
    # Food-101: split oficial na coluna `split` (75.750 treino / 25.250 teste),
    # perfeitamente balanceado (1.000 por classe), entao BAcc = Acc.
    "food101": {"meta": "data/food101_meta.csv", "test_cohort": None},
}


def scale(train_blocks, val_blocks, test_blocks):
    """Um StandardScaler por bloco, ajustado SO no treino."""
    parts = [[], [], []]
    for train, val, test in zip(train_blocks, val_blocks, test_blocks):
        scaler = StandardScaler().fit(train)
        for target, values in zip(parts, (train, val, test)):
            target.append(scaler.transform(values).astype(np.float32))
    return tuple(np.hstack(part) for part in parts)


def our_posteriors(train_blocks, val_blocks, test_blocks, y_train, y_val,
                   n_classes, seed, epochs, patience):
    train_x, val_x, test_x = scale(train_blocks, val_blocks, test_blocks)
    seed_everything(seed)
    model = WideBranchMLP([b.shape[1] for b in train_blocks], n_classes).to(DEVICE)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    inverse = 1.0 / np.maximum(counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(inverse / inverse.mean(), device=DEVICE),
                                    label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_train).long()),
        batch_size=4096, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_tensor = torch.from_numpy(val_x).to(DEVICE)
    test_tensor = torch.from_numpy(test_x).to(DEVICE)

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
            score = balanced_accuracy_score(y_val, model(val_tensor).argmax(1).cpu().numpy())
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
        probabilities = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
        # O posterior de VAL sai da MESMA epoca restaurada. Ele e otimista por
        # construcao -- foi esse `val` que escolheu a epoca --, e a mesma
        # ressalva vale para o backbone (§9.1). Emitir permite MEDIR se o vies e
        # simetrico entre os dois lados, em vez de argumentar sobre ele.
        val_probabilities = torch.softmax(model(val_tensor), dim=1).cpu().numpy()
    return (probabilities.astype(np.float32),
            val_probabilities.astype(np.float32), epoch)


def scores(truth, probabilities):
    pred = probabilities.argmax(1)
    return {"bacc": 100 * balanced_accuracy_score(truth, pred),
            "accuracy": 100 * accuracy_score(truth, pred),
            "macro_f1": 100 * f1_score(truth, pred, average="macro")}


def log_average(left, right, weight=0.5):
    blend = weight * np.log(np.maximum(left, 1e-12)) + (1 - weight) * np.log(np.maximum(right, 1e-12))
    blend -= blend.max(axis=1, keepdims=True)
    exponentiated = np.exp(blend)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def decompose(truth, ours, backbone):
    hit_ours = ours.argmax(1) == truth
    hit_backbone = backbone.argmax(1) == truth
    return {"both": 100 * float((hit_ours & hit_backbone).mean()),
            "only_backbone": 100 * float((~hit_ours & hit_backbone).mean()),
            "only_ours": 100 * float((hit_ours & ~hit_backbone).mean()),
            "neither": 100 * float((~hit_ours & ~hit_backbone).mean()),
            "oracle": 100 * float((hit_ours | hit_backbone).mean())}


def prepare(dataset: str, condition: tuple[str, ...]):
    """Blocos, rotulos, stems de teste, classes e os stems de CADA split.

    O ultimo item existe para quem precisa recortar um bloco externo (congelado,
    Haralick/HOG) exatamente nas mesmas linhas -- sem ele o chamador reimplementa
    a logica de split e as duas versoes divergem em silencio.
    """
    spec = DATASETS[dataset]
    cohort = load_blocks(dataset, condition, allow_external=True)

    if spec["test_cohort"] is None:
        meta = pd.read_csv(ROOT / spec["meta"], dtype={"stem": str})[["stem", "split"]]
        frame = pd.DataFrame({"stem": cohort.stems, "label": cohort.labels}).merge(
            meta, on="stem", how="left", validate="one_to_one")
        if frame["split"].isna().any():
            raise ValueError(f"some cohort stems are absent from {spec['meta']}")
        encoder = LabelEncoder().fit(frame["label"].to_numpy(str))
        labels = encoder.transform(frame["label"].to_numpy(str))
        where = frame["split"].to_numpy(str)
        train_idx = np.flatnonzero(where == "train")
        val_idx = np.flatnonzero(where == "val")
        test_idx = np.flatnonzero(where == "test")
        if len(test_idx) == 0:
            raise ValueError(f"{dataset}: no rows with split == 'test' in the cohort")
        if len(val_idx) == 0:
            train_idx, val_idx = train_test_split(train_idx, test_size=0.10,
                                                  random_state=SPLIT_SEED, stratify=labels[train_idx])
        blocks = [cohort.blocks[name] for name in condition]
        keys = np.asarray(cohort.stems, dtype=str)
        return (dict(train=[b[train_idx] for b in blocks], val=[b[val_idx] for b in blocks],
                     test=[b[test_idx] for b in blocks]),
                dict(train=labels[train_idx], val=labels[val_idx], test=labels[test_idx]),
                keys[test_idx], list(encoder.classes_),
                dict(train=keys[train_idx], val=keys[val_idx], test=keys[test_idx]))

    held = load_blocks(spec["test_cohort"], condition, allow_external=True)
    encoder = LabelEncoder().fit(np.asarray(cohort.labels, dtype=str))
    unknown = sorted(set(np.asarray(held.labels, dtype=str)) - set(encoder.classes_))
    if unknown:
        raise ValueError(f"{spec['test_cohort']} labels absent from {dataset}: {unknown}")
    y_all = encoder.transform(np.asarray(cohort.labels, dtype=str))
    y_test = encoder.transform(np.asarray(held.labels, dtype=str))
    train_idx, val_idx = train_test_split(np.arange(len(y_all)), test_size=0.10,
                                          random_state=SPLIT_SEED, stratify=y_all)
    log(f"coorte de teste separada ({spec['test_cohort']}): {len(y_test):,} linhas; "
        f"val = 10% estratificado do treino (seed {SPLIT_SEED})")
    keys = np.asarray(cohort.stems, dtype=str)
    held_keys = np.asarray(held.stems, dtype=str)
    return (dict(train=[cohort.blocks[n][train_idx] for n in condition],
                 val=[cohort.blocks[n][val_idx] for n in condition],
                 test=[held.blocks[n] for n in condition]),
            dict(train=y_all[train_idx], val=y_all[val_idx], test=y_test),
            held_keys, list(encoder.classes_),
            dict(train=keys[train_idx], val=keys[val_idx], test=held_keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="minc2500", choices=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"])
    parser.add_argument("--condition", nargs="+", default=list(HAND8))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--posteriors", type=Path, default=None)
    parser.add_argument("--with-text", action="store_true",
                        help="acrescenta o canal de texto (OOF aninhado)")
    parser.add_argument("--captions", default="estudante")
    parser.add_argument("--emit-val", action="store_true",
                        help="grava tambem o posterior do `val` OFICIAL na melhor epoca, "
                             "para escolher o peso da fusao fora do teste (§4.1f)")
    args = parser.parse_args()
    sufixo = "_text" if args.with_text else ""
    args.output = args.output or ROOT / f"results/{args.dataset}_decision_fusion{sufixo}.csv"
    args.posteriors = args.posteriors or ROOT / f"outputs/{args.dataset}_our_posteriors{sufixo}.npz"
    args.val_posteriors = ROOT / f"outputs/{args.dataset}_our_val_posteriors{sufixo}.npz"
    grid = GRID / args.dataset

    blocks, labels, test_stems, classes, split_stems = prepare(args.dataset, tuple(args.condition))
    n_classes = len(classes)
    truth = labels["test"]

    if args.with_text:
        from text_fusion_block import (CAPTIONS as CAPTION_TEMPLATE, TEST_COHORT,
                                       captioned_stems, load_texts, make_text_fn)
        # Ordem do bloco: treino, validacao, teste -- a mesma em que os splits
        # entram em `our_posteriors`. Os stems saem de `prepare`, nunca
        # reconstruidos do metadado: o §5.18b registra -2,92pp por reconstruir e
        # obter os MESMOS stems em ORDEM DIFERENTE, sem nenhum erro na saida.
        classes_arr = np.asarray(classes, dtype=str)
        # Nem todo corpus tem legenda para a coorte inteira: no places_lt o
        # arquivo cobre 100.252 das 106.300 linhas -- faltam os 528 do bug de
        # rotulo do §5.18d (sem imagem local) e 5.520 do `val` oficial, que
        # quase nunca foi legendado. O TESTE esta completo nos quatro corpora, e
        # e ele que funde com o backbone; entao as linhas sem legenda saem de
        # TREINO e VALIDACAO e o teste fica intocado. Se faltasse no teste, a
        # fusao deixaria de ser comparavel com o braco sem texto e o certo
        # seria abortar -- por isso a guarda abaixo.
        have = captioned_stems(ROOT / CAPTION_TEMPLATE[args.captions].format(
            dataset=TEST_COHORT.get(args.dataset, args.dataset)))
        if args.dataset in TEST_COHORT:
            have_fit = captioned_stems(
                ROOT / CAPTION_TEMPLATE[args.captions].format(dataset=args.dataset))
        else:
            have_fit = have
        gap_test = int((~np.isin(split_stems["test"], list(have))).sum())
        if gap_test:
            raise SystemExit(
                f"{args.dataset}: {gap_test:,} linhas de TESTE sem legenda -- a fusao "
                "deixaria de ser comparavel com o braco sem texto")
        keep = {k: np.isin(split_stems[k], list(have_fit)) for k in ("train", "val")}
        for k in ("train", "val"):
            dropped = int((~keep[k]).sum())
            if dropped:
                log(f"  {k}: {dropped:,} de {len(keep[k]):,} linhas sem legenda, "
                    f"descartadas (teste intocado)")
                blocks[k] = [b[keep[k]] for b in blocks[k]]
                labels[k] = labels[k][keep[k]]
                split_stems[k] = split_stems[k][keep[k]]
        per_class = len(labels["val"]) / max(n_classes, 1)
        log(f"  apos o corte: train={len(labels['train']):,} val={len(labels['val']):,} "
            f"({per_class:.1f} por classe) test={len(labels['test']):,}")

        stems_all = np.concatenate([split_stems["train"], split_stems["val"],
                                    split_stems["test"]])
        labels_all = np.concatenate([labels["train"], labels["val"], labels["test"]])
        labels_str = classes_arr[labels_all]
        if args.dataset in TEST_COHORT:
            texts = load_texts(args.dataset,
                               np.concatenate([split_stems["train"], split_stems["val"]]),
                               held_stems=split_stems["test"], captions=args.captions)
        else:
            texts = load_texts(args.dataset, stems_all, captions=args.captions)
        n_fit = len(split_stems["train"]) + len(split_stems["val"])
        # `val` entra em train_idx: ele escolhe a epoca, entao precisa de escore
        # de OOF interno tambem -- dar a ele o modelo do treino inteiro vazaria
        # para a selecao de epoca.
        text_fn = make_text_fn(args.dataset, stems_all, labels_str, classes_arr,
                               ROOT / "outputs/text_nested" / f"{args.dataset}_official",
                               texts)
        block = text_fn(0, np.arange(n_fit), np.arange(n_fit, len(stems_all)))
        n_tr = len(split_stems["train"])
        for key, piece in (("train", block[:n_tr]), ("val", block[n_tr:n_fit]),
                           ("test", block[n_fit:])):
            blocks[key].append(piece.astype(np.float32))
        log(f"canal de texto ligado: {len(texts):,} legendas, bloco de {block.shape[1]}d")
    dimensions = int(sum(b.shape[1] for b in blocks["train"]))
    log(f"{args.dataset}: train={len(labels['train']):,} val={len(labels['val']):,} "
        f"test={len(truth):,} classes={n_classes} dims={dimensions}")

    reference = json.loads((grid / args.models[0] / f"seed_{args.seeds[0]}/fold_1/classes.json").read_text())
    if classes != list(reference):
        raise ValueError(f"class order mismatch:\n  ours={classes}\n  grid={reference}")

    ours = {}
    ours_val: dict[int, np.ndarray] = {}
    if args.posteriors.exists():
        cached = np.load(args.posteriors, allow_pickle=True)
        if list(cached["stems"]) == list(test_stems):
            ours = {seed: cached[f"seed_{seed}"] for seed in args.seeds if f"seed_{seed}" in cached}
            log(f"reusing {len(ours)} cached posteriors from {args.posteriors}")
    if args.emit_val:
        # O cache de teste sozinho nao serve: sem o val em disco a semente tem de
        # ser re-treinada, senao o arquivo de val sai com buraco em silencio.
        have_val = set()
        if args.val_posteriors.exists():
            blob = np.load(args.val_posteriors, allow_pickle=True)
            if list(blob["stems"]) == list(split_stems["val"]):
                have_val = {int(k.split("_")[1]) for k in blob.files if k.startswith("seed_")}
                ours_val = {s_: blob[f"seed_{s_}"] for s_ in have_val}
        dropped = [s_ for s_ in ours if s_ not in have_val]
        for s_ in dropped:
            ours.pop(s_)
        if dropped:
            log(f"--emit-val: {len(dropped)} sementes re-treinadas (tinham teste, nao val)")
    for seed in args.seeds:
        if seed in ours:
            continue
        started = time.perf_counter()
        ours[seed], ours_val[seed], epochs_run = our_posteriors(
            blocks["train"], blocks["val"], blocks["test"], labels["train"], labels["val"],
            n_classes, seed, args.epochs, args.patience)
        marks = scores(truth, ours[seed])
        log(f"ours seed {seed}: BAcc {marks['bacc']:.2f} Acc {marks['accuracy']:.2f} "
            f"({epochs_run} epochs, {time.perf_counter() - started:.0f}s)")
    args.posteriors.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.posteriors, stems=test_stems, y_true=truth,
                        **{f"seed_{seed}": value for seed, value in ours.items()})
    if args.emit_val:
        with args.val_posteriors.open("wb") as handle:   # §8: savez acrescenta .npz
            np.savez_compressed(handle, stems=split_stems["val"], y_true=labels["val"],
                                **{f"seed_{s_}": v for s_, v in ours_val.items()})
        log(f"val posteriors -> {args.val_posteriors} ({len(ours_val)} sementes, "
            f"{len(split_stems['val']):,} linhas)")

    rows = []
    for model in args.models:
        for seed in args.seeds:
            blob = np.load(grid / model / f"seed_{seed}/fold_1/fold_01_posteriors.npz", allow_pickle=True)
            order = pd.Index(list(blob["stems"])).get_indexer(test_stems)
            if (order < 0).any():
                raise ValueError(f"{model} seed {seed}: {int((order < 0).sum())} stems missing")
            backbone = blob["probabilities"][order]
            if not np.array_equal(blob["y_true"][order], truth):
                raise ValueError(f"{model} seed {seed}: label mismatch after alignment")

            side = ours[seed]
            row = {"dataset": args.dataset, "model": model, "seed": seed, "dimensions": dimensions}
            row |= {f"ours_{k}": v for k, v in scores(truth, side).items()}
            row |= {f"backbone_{k}": v for k, v in scores(truth, backbone).items()}
            row |= {f"fusion_{k}": v for k, v in scores(truth, log_average(side, backbone)).items()}
            row |= decompose(truth, side, backbone)
            sweep = {w: balanced_accuracy_score(truth, log_average(side, backbone, w).argmax(1))
                     for w in np.round(np.arange(0.05, 1.0, 0.05), 2)}
            row["best_weight"] = float(max(sweep, key=sweep.get))
            row["fusion_bacc_best_weight"] = 100 * max(sweep.values())
            rows.append(row)
            log(f"{model:9s} seed {seed}: ours {row['ours_bacc']:.2f} | backbone "
                f"{row['backbone_bacc']:.2f} | fusion {row['fusion_bacc']:.2f} "
                f"(Delta {row['fusion_bacc'] - row['backbone_bacc']:+.2f}) | "
                f"SO NOSSO {row['only_ours']:.2f}%")

    detail = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, float_format="%.4f")
    log(f"\nsaved: {args.output}")
    for model, block in detail.groupby("model"):
        log(f"\n=== {args.dataset} / {model} (media +- desvio sobre {len(block)} seeds) ===")
        for field in ["ours_bacc", "backbone_bacc", "fusion_bacc", "fusion_bacc_best_weight"]:
            log(f"  {field:24s} {block[field].mean():6.2f} +- {block[field].std():.2f}")
        log(f"  Delta fusao vs backbone  {(block.fusion_bacc - block.backbone_bacc).mean():+6.2f} "
            f"+- {(block.fusion_bacc - block.backbone_bacc).std():.2f}  "
            f"({int((block.fusion_bacc > block.backbone_bacc).sum())}/{len(block)})")
        for field in ["both", "only_backbone", "only_ours", "neither", "oracle"]:
            log(f"  {field:24s} {block[field].mean():6.2f}%")


if __name__ == "__main__":
    main()
