"""§7.17: quanto o Letterbox custa ao backbone CONGELADO.

A extracao congelada usa `Letterbox` com barras cinzas (`extract_cnn_full.py`),
enquanto os backbones fine-tuned da grade avaliam com `Resize(256)+CenterCrop`
(`--augment strong`). O §8 mediu o par valendo 4,91pp nos mesmos pesos de um ViT
fine-tunado; aqui a pergunta e se o congelado tambem sofre -- e ele e o
adversario do §4.1b, entao um handicap ali corre CONTRA o artigo.

Mesmas imagens, mesmos pesos, mesma coorte, mesmos folds: muda so a
transformacao. Reproduzir os 49,33 do §4.3 no braco Letterbox valida o caminho.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_minc2500_official import HAND8, WideBranchMLP  # noqa: E402
from evaluate_production_visual_extended_similarity_graph import (  # noqa: E402
    prior_correct_rich, rich_similarity_features)
from evaluate_robustness_mlp_aug import DEVICE, BranchMLP, seed_everything  # noqa: E402
from registry import LABELS, load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
# §7.8: o estreito e a versao final; o largo fica como nota de sensibilidade.
HEADS = {"narrow": BranchMLP, "wide": WideBranchMLP}
# Hiperparametros de producao do grafo (§3): 30 vizinhos, alpha 0,4, 100 iters.
GRAPH = dict(n_neighbors=30, alpha=0.4, max_iter=100)
BACKBONES = {
    "oilpainting": {
        "ResNet50 letterbox": "outputs/oilpainting_cnn_frozen",
        "ResNet50 resize+crop": "outputs/oilpainting_resnet50_frozen_rc",
        "ViT letterbox": "outputs/oilpainting_vit_frozen",
        "ViT resize+crop": "outputs/oilpainting_vit_b_16_frozen_rc",
    },
    # resize+crop extraido em 02/set para os dois, que so tinham Letterbox.
    # Cuidado ao ler o custo: o caminho `letterbox` faz pre-downscale de imagens
    # grandes e o `resize_crop` decodifica em resolucao plena, entao o segundo
    # rodou a 51 img/s contra 360 do primeiro SOBRE OS MESMOS ORIGINAIS. Nao e
    # diferenca de fonte; e o `resize_crop` fazendo a coisa canonica e cara.
    "rijksdraw": {
        "ResNet50 letterbox": "outputs/rijksdraw_cnn_frozen",
        "ResNet50 resize+crop": "outputs/rijksdraw_resnet50_frozen_rc",
        "ViT letterbox": "outputs/rijksdraw_vit_frozen",
        "ViT resize+crop": "outputs/rijksdraw_vit_b_16_frozen_rc",
    },
    "multitask": {
        "ResNet50 letterbox": "outputs/multitask_cnn_frozen",
        "ResNet50 resize+crop": "outputs/multitask_resnet50_frozen_rc",
        "ViT letterbox": "outputs/multitask_vit_frozen",
        "ViT resize+crop": "outputs/multitask_vit_b_16_frozen_rc",
    },
    # Externos a arte: split oficial, e o ViT ja nasce com resize+crop (§7.17).
    "food101": {"ResNet50 letterbox": "outputs/food101_cnn_frozen",
                "ViT resize+crop": "outputs/food101_vit_frozen"},
    "nct_crc": {"ResNet50 letterbox": "outputs/nct_crc_cnn_frozen",
                "ViT resize+crop": "outputs/nct_crc_vit_frozen"},
    "minc2500": {"ResNet50 letterbox": "outputs/minc2500_cnn_frozen",
                 "ViT resize+crop": "outputs/minc2500_vit_frozen"},
    "places_lt": {"ResNet50 letterbox": "outputs/places_lt_cnn_frozen",
                  "ViT resize+crop": "outputs/places_lt_vit_frozen"},
}
# Datasets com split oficial: uma particao so, em vez de 5 folds de CV.
OFFICIAL = {"food101", "nct_crc", "minc2500", "places_lt"}
# Corpora cujo TESTE e uma coorte separada no registry, nao uma
# fatia da coluna `split`.
TEST_COHORT = {"nct_crc": "nct_crc_val"}


def load_classical(dataset: str, stems: np.ndarray) -> dict[str, np.ndarray]:
    """Haralick (13d) e HOG (1764d) de `extract_hog_features.py`, na ordem
    `(texture, gradients)` -- a mesma do extrator do PRISM, para que os numeros
    sejam comparaveis aos do WikiArt ja publicados no §6."""
    path = ROOT / f"outputs/{dataset}_haralick_hog"
    if not path.exists():
        return {}
    values = np.load(path / "features.npy")
    order = pd.Index(np.load(path / "stems.npy", allow_pickle=True).astype(str)).get_indexer(stems)
    if (order < 0).any():
        raise ValueError(f"{path}: {int((order < 0).sum())} stems ausentes")
    values = values[order].astype(np.float32)
    return {"Haralick": values[:, :13], "HOG": values[:, 13:]}


def load_backbone(path: Path, stems: np.ndarray) -> np.ndarray:
    values = np.load(path / "cnn_full_features.npy")
    order = pd.Index(np.load(path / "cnn_full_stems.npy", allow_pickle=True).astype(str)).get_indexer(stems)
    if (order < 0).any():
        raise ValueError(f"{path}: {int((order < 0).sum())} stems ausentes")
    return values[order].astype(np.float32)


def load_clip(dataset: str, stems: np.ndarray) -> np.ndarray:
    """CLIP ViT-B/32 congelado, ALINHADO POR STEM. O cache e um .npy sem stems,
    gravado na ordem de `results/<ds>_student_captions.csv` e NAO na da coorte do
    registry -- casar por posicao deu BAcc 0,0017 no `evaluate_banks_dense_mlp`
    com o tamanho batendo exatamente. Mesmo contrato do `load_backbone`: aborta
    com ValueError se faltar stem, para o ramo de coorte externa funcionar."""
    values = np.load(ROOT / f"outputs/{dataset}_clip_vitb32.npy")
    order = pd.read_csv(ROOT / f"results/{dataset}_student_captions.csv",
                        dtype={"stem": str})["stem"].to_numpy(str)
    if len(order) != len(values):
        raise ValueError(f"{dataset}: cache CLIP {len(values)} x legendas {len(order)}")
    idx = pd.Index(order).get_indexer(stems)
    if (idx < 0).any():
        raise ValueError(f"{dataset}: {int((idx < 0).sum())} stems fora do cache CLIP")
    return values[idx].astype(np.float32)


def clip_stems(dataset: str) -> set[str]:
    return set(pd.read_csv(ROOT / f"results/{dataset}_student_captions.csv",
                           dtype={"stem": str})["stem"])


def fit(blocks, labels, train_idx, test_idx, n_classes, seed, epochs, patience,
        head="narrow", graph_source=None):
    fit_idx, val_idx = train_test_split(train_idx, test_size=0.10, random_state=seed,
                                        stratify=labels[train_idx])
    parts = [[], [], []]
    sizes = []
    for block in blocks:
        scaler = StandardScaler().fit(block[fit_idx])
        for target, index in zip(parts, (fit_idx, val_idx, test_idx)):
            target.append(scaler.transform(block[index]).astype(np.float32))
        sizes.append(block.shape[1])

    if graph_source is not None:
        # OOF aninhado: as linhas de treino recebem posterior de folds internos
        # disjuntos, val e teste recebem o ajuste sobre o treino inteiro. Sem
        # isso o bloco vira auto-referencia e o LDA desaba (§6, SNF).
        scaler = StandardScaler().fit(graph_source[fit_idx])
        space = scaler.transform(graph_source).astype(np.float32)
        oof, held = rich_similarity_features(
            space[fit_idx], labels[fit_idx],
            np.vstack([space[val_idx], space[test_idx]]),
            n_classes, seed=seed, **GRAPH)
        oof = prior_correct_rich(oof, labels[fit_idx], n_classes)
        held = prior_correct_rich(held, labels[fit_idx], n_classes)
        for target, values in zip(parts, (oof, held[:len(val_idx)], held[len(val_idx):])):
            target.append(values.astype(np.float32))
        sizes.append(oof.shape[1])

    train_x, val_x, test_x = (np.hstack(p) for p in parts)
    seed_everything(seed)
    model = HEADS[head](sizes, n_classes).to(DEVICE)
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
            score = balanced_accuracy_score(labels[val_idx], model(val_tensor).argmax(1).cpu().numpy())
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
        pred = model(torch.from_numpy(test_x).to(DEVICE)).argmax(1).cpu().numpy()
    truth = labels[test_idx]
    return {"bacc": 100 * balanced_accuracy_score(truth, pred),
            "accuracy": 100 * accuracy_score(truth, pred),
            "macro_f1": 100 * f1_score(truth, pred, average="macro"),
            # o bloco do grafo nasce dentro do fit, entao a dimensao real so e
            # conhecida aqui -- somar `blocks` fora deixaria a coluna mentindo
            "dimensions": int(sum(sizes))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    # SEMENTES, acrescentado em 03/set. Ate aqui as condicoes de features rodavam
    # com semente unica e o desvio reportado era ENTRE FOLDS, enquanto os
    # backbones fine-tuned ja tinham 5 sementes -- duas unidades de replicacao
    # diferentes na mesma tabela. Pior: o §5.6 mediu que o `hand-8` reproduz em
    # 38,53 / 38,71 / 38,74 entre execucoes com a MESMA semente
    # (nao-determinismo de GPU), e que um "+0,91pp com 5/5 folds" virou "+0,68
    # com 3/5" ao reexecutar. A conclusao registrada la foi que o conserto e
    # semente, nao fold. Isto e esse conserto.
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                        help="a 42 e a que ja esta em disco; passar 42 43 44 45 46")
    parser.add_argument("--dataset", default="oilpainting", choices=sorted(BACKBONES))
    parser.add_argument("--heads", nargs="+", default=["narrow"], choices=list(HEADS))
    parser.add_argument("--output", type=Path,
                        default=None)
    # CLIP CONGELADO (30/out): o revisor estranharia a fusao com o backbone
    # moderno faltar. Entra como mais um backbone congelado, pelo MESMO caminho
    # das linhas `hand-8 + ResNet/ViT congelado` -- e nao no
    # `evaluate_banks_dense_mlp`, cuja fusao e concatenacao num bloco unico
    # sem branches, que nao e a arquitetura do PRISM.
    parser.add_argument("--clip", action="store_true",
                        help="acrescenta CLIP ViT-B/32 congelado como backbone")
    parser.add_argument("--only", nargs="+", default=None,
                        help="roda so estas condicoes (as outras ja estao em disco)")
    parser.add_argument("--restrict-to-clip", action="store_true",
                        help="corta a coorte aos stems com CLIP (places_lt: 528 "
                             "ausentes, §5.18d) e as classes com <2 linhas de treino; "
                             "exige --output separado, porque muda a coorte")
    args = parser.parse_args()
    args.output = args.output or ROOT / f"results/{args.dataset}_final_table.csv"
    if args.restrict_to_clip and args.output.name.endswith("_final_table.csv"):
        raise SystemExit("--restrict-to-clip muda a coorte: passe --output separado")

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    hand = [cohort.blocks[n] for n in HAND8]
    if args.restrict_to_clip:
        keep = np.isin(stems, list(clip_stems(args.dataset)))
        log(f"--restrict-to-clip: {int((~keep).sum()):,} de {len(stems):,} stems sem CLIP")
        if args.dataset in OFFICIAL:
            # classe com <2 linhas de treino sai de treino E de teste: mante-la
            # so no teste daria recall zero por contagem, nao por desempenho
            # (o caso do `d_desert_road`, §5.18e)
            meta = pd.read_csv(ROOT / LABELS[args.dataset][0], dtype={"stem": str})
            split = meta.set_index("stem")["split"].reindex(stems).to_numpy(str)
            train_counts = pd.Series(labels_str[keep & (split == "train")]).value_counts()
            weak = set(train_counts[train_counts < 2].index) | (
                set(labels_str[keep]) - set(train_counts.index))
            keep &= ~np.isin(labels_str, list(weak))
            log(f"  classes cortadas (<2 de treino): {sorted(weak)}")
        stems, labels_str = stems[keep], labels_str[keep]
        hand = [b[keep] for b in hand]
        log(f"  coorte restrita: {len(stems):,} stems, {len(set(labels_str))} classes")
    external = TEST_COHORT.get(args.dataset)
    n_train = len(stems)
    if external:
        # NCT-CRC: o teste e uma COORTE SEPARADA (pacientes diferentes), nao uma
        # fatia da coluna `split` -- e e essa separacao que derruba o numero de
        # ~99% interno para 87,6% externo (§7.14). Empilha as duas e indexa por
        # posicao; nada de `split`.
        held = load_blocks(external, HAND8, allow_external=True)
        hand = [np.vstack([a, held.blocks[n]]).astype(np.float32)
                for a, n in zip(hand, HAND8)]
        stems = np.concatenate([stems, np.asarray(held.stems, dtype=str)])
        labels_str = np.concatenate([labels_str, np.asarray(held.labels, dtype=str)])
        log(f"coorte de teste SEPARADA: {n_train:,} treino ({args.dataset}) + "
            f"{len(held.stems):,} teste ({external})")
    labels = LabelEncoder().fit_transform(labels_str)
    n_classes = len(np.unique(labels))

    backbones = {}
    for key, path in BACKBONES[args.dataset].items():
        # com --only, carregar so o backbone que alguma condicao pedida usa --
        # senao uma copia local sem os caches de ResNet/ViT aborta por nada
        if args.only and not any(c == key or c.endswith(f" + {key}") for c in args.only):
            continue
        if not external:
            backbones[key] = load_backbone(ROOT / path, stems)
            continue
        # O bloco do backbone tem de existir nas DUAS coortes. O ViT congelado
        # nunca foi extraido para `nct_crc_val` e as imagens do corpus nao estao
        # mais em disco, entao a condicao e PULADA com aviso -- nunca preenchida
        # com zeros nem com o treino repetido, que passaria despercebido.
        # O criterio e COBERTURA DE STEMS, nao existencia de diretorio: em
        # 05/set eu pulei o ViT e o Haralick/HOG do NCT-CRC porque nao havia
        # `outputs/nct_crc_val_*`, quando na verdade `outputs/nct_crc_*` ja
        # cobria as 107.180 linhas das DUAS coortes. Presenca de diretorio nao
        # e o mesmo que presenca das linhas.
        try:
            backbones[key] = load_backbone(ROOT / path, stems)
            continue
        except ValueError:
            pass
        other = ROOT / path.replace(args.dataset, external, 1)
        if not other.exists():
            log(f"  [pulado] {key}: sem features para as linhas de teste")
            continue
        backbones[key] = np.vstack([load_backbone(ROOT / path, stems[:n_train]),
                                    load_backbone(other, stems[n_train:])]).astype(np.float32)
    if args.clip:
        if external:   # NCT-CRC: cache por coorte, `nct_crc` treino + `nct_crc_val` teste
            backbones["CLIP ViT-B/32"] = np.vstack([
                load_clip(args.dataset, stems[:n_train]),
                load_clip(external, stems[n_train:])])
        else:
            backbones["CLIP ViT-B/32"] = load_clip(args.dataset, stems)
    log(f"{args.dataset}: {len(stems):,} imagens, {n_classes} classes, hand-8 {sum(b.shape[1] for b in hand)}d")

    # graph_source = espaco sobre o qual o kNN roda; None desliga o bloco.
    conditions = {}
    try:                       # o caminho comum: um arquivo so cobre tudo
        classical = load_classical(args.dataset, stems)
    except ValueError:
        classical = load_classical(args.dataset, stems[:n_train])
        other = load_classical(external, stems[n_train:])
        for name in list(classical):
            if name not in other:
                log(f"  [pulado] {name}: sem features para as linhas de teste")
                classical.pop(name)
            else:
                classical[name] = np.vstack([classical[name], other[name]]).astype(np.float32)
    for name, values in classical.items():
        conditions[name] = ([values], None)
    conditions["hand-8"] = (hand, None)
    conditions["hand-8 + grafo"] = (hand, np.hstack(hand))
    for name, values in backbones.items():
        conditions[name] = ([values], None)
        conditions[f"hand-8 + {name}"] = (hand + [values], None)
        # `nos` no artigo e hand-8 + grafo (§3, sem Text), entao a linha de
        # fusao com congelado tem de ser a configuracao de producao tambem.
        conditions[f"hand-8 + grafo + {name}"] = (hand + [values], np.hstack(hand))
    if args.only:
        faltam = [c for c in args.only if c not in conditions]
        if faltam:
            raise SystemExit(f"--only: {faltam} fora de {list(conditions)}")
        conditions = {c: conditions[c] for c in args.only}

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        # Linhas antigas nao tem coluna `seed`: sao todas da 42. Tratar como tal
        # evita recomputar 136 min de resultado ja valido.
        if "seed" not in previous.columns:
            previous["seed"] = SEED
        done = {(str(r.head), str(r.condition), int(r.fold), int(r.seed))
                for r in previous.itertuples()}
        log(f"retomando {len(done)} linhas")

    # BUG CORRIGIDO em 05/set: `OFFICIAL` estava DEFINIDO e nunca usado, entao os
    # quatro corpora externos rodavam 5-fold CV sobre a coorte inteira e o split
    # publicado era ignorado. Para o Places-LT isso e o erro que o §7.5 proibe --
    # CV sobre as 106.300 derruba o desbalanceamento de 996x para 41x e apaga o
    # regime long-tail que motiva o corpus; no NCT-CRC apaga a separacao por
    # paciente do teste externo. Toda tabela externa gerada antes desta data e
    # suspeita e tem de ser refeita.
    if external:
        folds = [(np.arange(n_train), np.arange(n_train, len(labels)))]
        log(f"split EXTERNO: {n_train:,} treino / {len(labels) - n_train:,} teste "
            f"-- 1 particao x {len(args.seeds)} sementes")
    elif args.dataset in OFFICIAL:
        meta = pd.read_csv(ROOT / LABELS[args.dataset][0], dtype={"stem": str})
        if "split" not in meta.columns:
            raise SystemExit(f"{args.dataset}: metadado sem coluna `split`")
        where = (pd.Series(meta["split"].to_numpy(), index=meta["stem"].to_numpy())
                 .reindex(stems))
        if where.isna().any():
            raise SystemExit(f"{args.dataset}: {int(where.isna().sum())} stems sem split")
        where = where.to_numpy(str)
        train_idx = np.flatnonzero(where != "test")   # `val` oficial entra no treino;
        test_idx = np.flatnonzero(where == "test")    # o `fit` separa 10% por dentro.
        if not len(test_idx):
            raise SystemExit(f"{args.dataset}: nenhuma linha com split == 'test'")
        folds = [(train_idx, test_idx)]
        log(f"split OFICIAL: {len(train_idx):,} treino / {len(test_idx):,} teste "
            f"-- 1 particao x {len(args.seeds)} sementes")
    else:
        folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                     .split(stems, labels))
        log(f"CV {N_FOLDS} folds x {len(args.seeds)} sementes")
    for head in args.heads:
      for name, (blocks, graph_source) in conditions.items():
        for seed in args.seeds:
          for number, (train_idx, test_idx) in enumerate(folds, 1):
            if (head, name, number, seed) in done:
                continue
            started = time.perf_counter()
            metrics = fit(blocks, labels, train_idx, test_idx, n_classes, seed,
                          args.epochs, args.patience, head, graph_source)
            row = {"head": head, "condition": name, "fold": number, "seed": seed,
                   "seconds": time.perf_counter() - started, **metrics}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame([row])
            if args.output.exists():
                # Append com esquema mudado grava os valores novos sob o
                # cabecalho antigo, em silencio -- foi assim que `dimensions`
                # vazou para a coluna de F1. Reindexar amarra a ordem.
                frame = frame.reindex(columns=pd.read_csv(args.output, nrows=0).columns)
            frame.to_csv(args.output, mode="a", index=False,
                         header=not args.output.exists(), float_format="%.4f")
            log(f"{head:6s} | {name:32s} fold {number} | {row['dimensions']:5d}d | "
                f"BAcc {metrics['bacc']:5.2f} | {row['seconds']:.0f}s")

    from scipy import stats
    detail = pd.read_csv(args.output)
    for head, block in detail.groupby("head"):
        summary = block.groupby("condition").agg(dims=("dimensions", "first"), bacc=("bacc", "mean"),
                                                 sd=("bacc", "std"), acc=("accuracy", "mean"))
        log(f"\n=== {args.dataset}, 5 folds, MLP {head} ===")
        for name, r in summary.sort_values("bacc").iterrows():
            log(f"  {name:34s} {int(r.dims):5d}d  BAcc {r.bacc:5.2f} +- {r.sd:4.2f}  Acc {r.acc:5.2f}")

        def paired(left: str, right: str, tag: str) -> None:
            a = block[block.condition == left].sort_values("fold").bacc.values
            b = block[block.condition == right].sort_values("fold").bacc.values
            if len(a) != N_FOLDS or len(b) != N_FOLDS:
                return
            log(f"  {tag:44s} {(b - a).mean():+5.2f}pp  p={stats.ttest_rel(b, a).pvalue:.4f}  "
                f"{int((b > a).sum())}/{N_FOLDS}")

        log(f"  --- deltas ({head}) ---")
        paired("hand-8", "hand-8 + grafo", "grafo sobre hand-8")
        paired("hand-8 + ViT resize+crop", "hand-8 + grafo + ViT resize+crop",
               "grafo sobre hand-8 + ViT")
        for base in ["ResNet50", "ViT"]:
            for prefix in ["", "hand-8 + "]:
                paired(f"{prefix}{base} letterbox", f"{prefix}{base} resize+crop",
                       f"{prefix}{base}: resize+crop − letterbox")


if __name__ == "__main__":
    main()
