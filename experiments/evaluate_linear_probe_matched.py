"""Probe linear com estimador PAREADO nos quatro espacos densos + hand-8.

**O que motivou.** O CLIP da **60,09** BAcc sob LinearSVC e **54,39** sob o
`BranchMLP` de producao -- a sonda linear crua bate a nossa cabeca por 5,7pp num
bloco denso. O mecanismo e arquitetural e contavel: o `BranchMLP` manda TODO
bloco para 128 dims, entao com **um** bloco a cabeca recebe 128 dimensoes,
enquanto com o hand-8 ela recebe 8 x 128 = 1.024.

Isso poe uma duvida nas baselines do §9.2: `ResNet50 congelada` (2.048d -> 128) e
`ViT congelado` (768d -> 128) foram medidas com essa mesma compressao, e o
numero de manchete da secao e "batemos as duas congeladas por +2,6 a +3,0pp". Se
a compressao as penaliza, a alegacao encolhe ou inverte -- e e o tipo de coisa
que um revisor encontra sozinho.

O §4.1e mediu que a compressao **regulariza** em vez de estrangular, mas naquele
experimento o encoder era `enc=512` (`WideBranchMLP`, MINC) e o backbone entrava
com 2048->512, compressao de 4x. Aqui o estreito de producao comprime 16x e o
bloco e unico. Sao regimes diferentes e o §4.1e declara esse escopo.

Preprocessamento: `StandardScaler` por bloco ajustado so no treino, que e a
convencao do §2. O CLIP roda tambem em L2 (como no `evaluate_clip_text_arms`)
para amarrar com o 60,09 ja medido e mostrar que a diferenca nao e de
preprocessamento.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/pipeline"))
from corpora import HAND8                       # noqa: E402
from registry import load_blocks                # noqa: E402
from protocol import N_FOLDS, SEED              # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def by_stem(path: Path, stem_path: Path, stems: np.ndarray, name: str) -> np.ndarray:
    """Reindexa features externas pela ordem da coorte. NUNCA por posicao (§4.1f)."""
    feats = np.load(path)
    owned = np.load(stem_path, allow_pickle=True).astype(str)
    position = {s: i for i, s in enumerate(owned)}
    faltam = [s for s in stems if s not in position]
    if faltam:
        raise SystemExit(f"{name}: {len(faltam):,} stems da coorte sem feature, ex.: {faltam[:3]}")
    return feats[np.array([position[s] for s in stems], dtype=np.int64)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ds = args.dataset
    args.output = args.output or ROOT / f"results/{ds}_linear_probe_matched.csv"

    cohort = load_blocks(ds, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels = np.asarray(cohort.labels, dtype=str)
    hand = np.hstack([cohort.blocks[n] for n in HAND8]).astype(np.float32)
    log(f"{ds}: {len(stems):,} imagens | {len(np.unique(labels))} classes | hand-8 {hand.shape[1]}d")

    frame = pd.read_csv(ROOT / f"results/{ds}_student_captions.csv", dtype={"stem": str, "style": str})
    clip_raw = np.load(ROOT / f"outputs/{ds}_clip_vitb32.npy")
    pos = {s: i for i, s in enumerate(frame["stem"].to_numpy(str))}
    order = np.array([pos[s] for s in stems], dtype=np.int64)
    clip = clip_raw[order].astype(np.float32)
    if (frame["style"].to_numpy(str)[order] != labels).sum():
        raise SystemExit("rotulos divergem apos alinhar CLIP por stem")

    rn50 = by_stem(ROOT / f"outputs/{ds}_cnn_frozen/cnn_full_features.npy",
                   ROOT / f"outputs/{ds}_cnn_frozen/cnn_full_stems.npy", stems, "ResNet50")
    vit = by_stem(ROOT / f"outputs/{ds}_vit_frozen/cnn_full_features.npy",
                  ROOT / f"outputs/{ds}_vit_frozen/cnn_full_stems.npy", stems, "ViT")
    log(f"  alinhados por stem: CLIP {clip.shape[1]}d | ResNet50 {rn50.shape[1]}d | ViT {vit.shape[1]}d")

    l2 = clip / np.linalg.norm(clip, axis=1, keepdims=True)
    arms = {
        "hand8": (hand, "scaler"),
        "clip": (clip, "scaler"),
        "clip_l2": (l2, "none"),          # amarra com o 60,09 do evaluate_clip_text_arms
        "resnet50_frozen": (rn50, "scaler"),
        "vit_frozen": (vit, "scaler"),
        "hand8+clip": (np.hstack([hand, clip]), "scaler"),
    }

    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(stems, labels))
    rows = []
    for name, (matrix, prep) in arms.items():
        started = time.time()
        per_fold = []
        for fold, (train, test) in enumerate(splits, 1):
            if prep == "scaler":
                scaler = StandardScaler().fit(matrix[train])
                train_x = scaler.transform(matrix[train]).astype(np.float32)
                test_x = scaler.transform(matrix[test]).astype(np.float32)
            else:
                train_x, test_x = matrix[train], matrix[test]
            probe = LinearSVC(class_weight="balanced", random_state=SEED)
            probe.fit(train_x, labels[train])
            score = balanced_accuracy_score(labels[test], probe.predict(test_x))
            per_fold.append(100 * score)
            rows.append({"arm": name, "dims": matrix.shape[1], "fold": fold, "bacc": 100 * score})
            pd.DataFrame(rows).to_csv(args.output, index=False)
            log(f"  {name:16s} fold {fold}  BAcc {100*score:5.2f}")
        log(f"  {name:16s} ({matrix.shape[1]:4d}d)  BAcc {np.mean(per_fold):5.2f} "
            f"+- {np.std(per_fold):.2f}   [{(time.time()-started)/60:.1f} min]")

    log("\n=== resumo (LinearSVC, estimador pareado) ===")
    table = pd.DataFrame(rows)
    for name in arms:
        v = table.loc[table["arm"] == name, "bacc"].to_numpy()
        log(f"  {name:16s} {v.mean():5.2f} +- {v.std():.2f}")
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
