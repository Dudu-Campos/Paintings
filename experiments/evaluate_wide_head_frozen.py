"""As congeladas do §9.2 sob a cabeca LARGA: o teto de compressao esta mordendo?

**O achado que motivou.** O CLIP da 60,09 sob `LinearSVC` e **54,39** sob o
`BranchMLP` estreito de producao. O mecanismo e contavel: o `BranchMLP` manda
TODO bloco para 128 dims, entao um bloco DENSO UNICO chega a cabeca com 128
dimensoes, enquanto o hand-8 chega com 8 x 128 = 1.024. A compressao e de 4x
para o CLIP (512), 6x para o ViT (768) e **16x para a ResNet50 (2.048)** --
contra ~0,5x a 1x para os blocos do hand-8, varios dos quais sao MENORES que 128
e portanto sao EXPANDIDOS.

Isso poe em duvida a manchete do §9.2 -- "batemos as duas congeladas por +2,6 a
+3,0pp" --, porque as duas congeladas sao exatamente blocos densos unicos.

O `WideBranchMLP` (enc=512, cabeca 2048-1024-1024) e o instrumento que o §7.8 ja
declara obrigatorio como tabela de sensibilidade e que o §4.1e usou para medir
alocacao por bloco. Com enc=512 a ResNet comprime 4x em vez de 16x e o hand-8
sobe para 8 x 512 = 4.096 -- nao iguala a alocacao, mas move o teto na direcao
que a objecao aponta. Se as congeladas subirem mais que o hand-8, a alegacao do
§9.2 encolhe.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/pipeline"))
from corpora import HAND8                                    # noqa: E402
from registry import load_blocks                             # noqa: E402
from protocol import N_FOLDS, SEED, scaled_splits            # noqa: E402
from models import WideBranchMLP, seed_everything, DEVICE    # noqa: E402

import copy                                                  # noqa: E402
import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset       # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def train_wide(train_x, train_y, val_x, val_y, test_x, test_y, sizes, n_classes,
               seed, max_epochs, patience, label_smoothing=0.05):
    """Copia fiel de `models.train_one`, trocando SO a classe do modelo.

    O §5.6 proibe reimplementar o laco de treino (uma reimplementacao omitiu
    `label_smoothing` e trocou a semente, custando 0,47pp em silencio). Aqui a
    unica diferenca deliberada e `WideBranchMLP` no lugar de `BranchMLP`; tudo
    mais -- otimizador, agendador, batch, pesos de classe, selecao de epoca por
    BAcc de validacao -- e identico, linha a linha.
    """
    seed_everything(seed)
    model = WideBranchMLP(sizes, n_classes).to(DEVICE)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE),
        label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed))
    val_t = torch.from_numpy(val_x).to(DEVICE)
    test_t = torch.from_numpy(test_x).to(DEVICE)
    best, best_state, stale = -1.0, None, 0
    for _ in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            score = balanced_accuracy_score(val_y, model(val_t).argmax(1).cpu().numpy())
        if score > best:
            best, best_state, stale = float(score), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(test_t).argmax(1).cpu().numpy()
    out = {"bacc": balanced_accuracy_score(test_y, pred),
           "accuracy": accuracy_score(test_y, pred),
           "macro_f1": f1_score(test_y, pred, average="macro")}
    del model, val_t, test_t
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def by_stem(path, stem_path, stems, name):
    feats = np.load(path)
    owned = np.load(stem_path, allow_pickle=True).astype(str)
    position = {s: i for i, s in enumerate(owned)}
    faltam = [s for s in stems if s not in position]
    if faltam:
        raise SystemExit(f"{name}: {len(faltam):,} stems sem feature, ex.: {faltam[:3]}")
    return feats[np.array([position[s] for s in stems], dtype=np.int64)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--only", default=None, help="subconjunto de arms, separado por virgula")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ds = args.dataset
    args.output = args.output or ROOT / f"results/{ds}_wide_head_frozen.csv"

    cohort = load_blocks(ds, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    labels = np.array([index_of[c] for c in labels_str], dtype=np.int64)
    hand = [cohort.blocks[n] for n in HAND8]

    frame = pd.read_csv(ROOT / f"results/{ds}_student_captions.csv",
                        dtype={"stem": str, "style": str})
    pos = {s: i for i, s in enumerate(frame["stem"].to_numpy(str))}
    order = np.array([pos[s] for s in stems], dtype=np.int64)
    clip = np.load(ROOT / f"outputs/{ds}_clip_vitb32.npy")[order].astype(np.float32)
    if (frame["style"].to_numpy(str)[order] != labels_str).sum():
        raise SystemExit("rotulos divergem apos alinhar CLIP por stem")
    rn50 = by_stem(ROOT / f"outputs/{ds}_cnn_frozen/cnn_full_features.npy",
                   ROOT / f"outputs/{ds}_cnn_frozen/cnn_full_stems.npy", stems, "ResNet50")
    vit = by_stem(ROOT / f"outputs/{ds}_vit_frozen/cnn_full_features.npy",
                  ROOT / f"outputs/{ds}_vit_frozen/cnn_full_stems.npy", stems, "ViT")
    log(f"{ds}: {len(stems):,} imagens | {len(classes)} classes | alinhados por stem")

    arms = {
        "hand8": hand,
        "clip": [clip],
        "resnet50_frozen": [rn50],
        "vit_frozen": [vit],
        "hand8+clip": hand + [clip],
        # Controle que decide a leitura: se a complementaridade sobre as
        # congeladas SOBREVIVE a cabeca larga, entao o -0,94 contra o CLIP e
        # sobre o CLIP ser forte; se ela tambem colapsar, a alegacao inteira do
        # §9.2/§4.1b e dependente de cabeca. Sao afirmacoes muito diferentes.
        "hand8+resnet50": hand + [rn50],
        "hand8+vit": hand + [vit],
    }
    if args.only:
        escolhidos = set(args.only.split(","))
        desconhecidos = escolhidos - set(arms)
        if desconhecidos:
            raise SystemExit(f"arms desconhecidos: {sorted(desconhecidos)}")
        arms = {k: v for k, v in arms.items() if k in escolhidos}
    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(stems, labels))
    rows = []
    for name, blocks in arms.items():
        dims = sum(b.shape[1] for b in blocks)
        for fold, (train, test) in enumerate(splits, 1):
            fit, val = train_test_split(train, test_size=0.1, stratify=labels[train],
                                        random_state=SEED + fold)
            train_x, val_x, test_x = scaled_splits(blocks, fit, val, test)
            m = train_wide(train_x, labels[fit], val_x, labels[val], test_x, labels[test],
                           [b.shape[1] for b in blocks], len(classes),
                           SEED + fold, args.epochs, args.patience)
            rows.append({"arm": name, "dims": dims, "fold": fold, "bacc": 100 * m["bacc"],
                         "accuracy": 100 * m["accuracy"], "macro_f1": 100 * m["macro_f1"]})
            pd.DataFrame(rows).to_csv(args.output, index=False)
            log(f"  {name:16s} fold {fold}  BAcc {rows[-1]['bacc']:5.2f}")
        v = [r["bacc"] for r in rows if r["arm"] == name]
        log(f"  {name:16s} ({dims:4d}d)  BAcc {np.mean(v):5.2f} +- {np.std(v):.2f}")

    log("\n=== resumo (WideBranchMLP, enc=512) ===")
    table = pd.DataFrame(rows)
    for name in arms:
        v = table.loc[table["arm"] == name, "bacc"].to_numpy()
        log(f"  {name:16s} {v.mean():5.2f} +- {v.std():.2f}")
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
