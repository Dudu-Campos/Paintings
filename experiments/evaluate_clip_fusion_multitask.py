"""O CLIP congelado como terceiro backbone da matriz do §9.2, e a fusao com hand-8.

**Por que este script existe.** O probe linear do CLIP ViT-B/32 da **60,09** BAcc
na coorte de 99.816 e 125 classes -- 24pp acima das duas congeladas do §9.2 e
**10pp acima da nossa ResNet50 fine-tunada**. Um revisor com o cache do CLIP em
disco chega a esse numero em quinze minutos, entao ele tem de estar no artigo
medido por nos.

A pergunta que decide se ele machuca a alegacao central e a do §4.1b: **o hand-8
complementa o CLIP?** Fusao de FEATURES e limpa aqui porque o CLIP e congelado --
ele nunca viu rotulo deste corpus, entao nao ha o vazamento de 29,33pp que o
§4.1c mediu contra backbone fine-tuned.

E ha previsao registrada a testar. O §4.1f mede que o Delta da fusao de peso
igual cruza zero em gap ~22,4pp. Aqui o gap e **60,09 - 38,74 = 21,35pp**, em
cima do cruzamento -- teste fora da amostra, com um backbone de regime de
supervisao (400M pares imagem-texto) que nao existe entre os sete que ajustaram
aquela reta. (Aquela alegacao e sobre fusao de DECISAO; esta mede a de features,
que e a celula comparavel com o §9.2. Sao complementares, nao a mesma coisa.)
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/pipeline"))
from corpora import HAND8                       # noqa: E402
from registry import load_blocks                # noqa: E402
from protocol import N_FOLDS, SEED, evaluate_mlp  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ds = args.dataset
    args.output = args.output or ROOT / f"results/{ds}_clip_fusion.csv"

    cohort = load_blocks(ds, HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    labels = np.array([index_of[c] for c in labels_str], dtype=np.int64)
    hand = [cohort.blocks[name] for name in HAND8]
    log(f"{ds}: {len(stems):,} imagens | {len(classes)} classes | "
        f"hand-8 {sum(b.shape[1] for b in hand)}d")

    # --- CLIP, alinhado POR STEM ----------------------------------------
    # O §4.1f registra o bug que este bloco evita: um diagnostico pareou nossos
    # posteriors com os do backbone POR POSICAO, e o MINC guarda os stems em
    # ordem diferente nos dois arquivos. Deu 562 melhoram / 137 pioram com Delta
    # negativo -- so a contradicao aritmetica denunciou. Alinhar por stem,
    # sempre, e abortar se faltar algum.
    frame = pd.read_csv(ROOT / f"results/{ds}_student_captions.csv",
                        dtype={"stem": str, "style": str})
    feats = np.load(ROOT / f"outputs/{ds}_clip_vitb32.npy")
    if len(feats) != len(frame):
        raise SystemExit(f"desalinhamento cache x csv: {len(feats):,} x {len(frame):,}")
    position = {stem: i for i, stem in enumerate(frame["stem"].to_numpy(str))}
    faltam = [s for s in stems if s not in position]
    if faltam:
        raise SystemExit(f"{len(faltam):,} stems da coorte sem feature CLIP, "
                         f"ex.: {faltam[:3]}")
    order = np.array([position[s] for s in stems], dtype=np.int64)
    clip = feats[order].astype(np.float32)
    # Conferir o ROTULO depois de alinhar -- o teste que o §4.1f prescreve. Se a
    # reindexacao estivesse errada os estilos discordariam.
    checagem = frame["style"].to_numpy(str)[order]
    divergem = int((checagem != labels_str).sum())
    if divergem:
        raise SystemExit(f"{divergem:,} rotulos divergem apos alinhar por stem")
    log(f"  CLIP alinhado por stem: {clip.shape[1]}d, rotulos conferem em "
        f"{len(stems):,}/{len(stems):,}")

    arms = {
        "hand8": hand,                 # reproduz 38,74 do §9.2 -> valida o caminho
        "clip": [clip],
        "hand8+clip": hand + [clip],
    }

    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
                  .split(stems, labels))
    rows = []
    for name, blocks in arms.items():
        dims = sum(b.shape[1] for b in blocks)
        for fold, (train, test) in enumerate(splits, 1):
            metrics = evaluate_mlp(blocks, labels, train, test, fold,
                                   args.epochs, args.patience, label_smoothing=0.05)
            rows.append({"arm": name, "dims": dims, "fold": fold,
                         "bacc": 100 * metrics["bacc"],
                         "accuracy": 100 * metrics["accuracy"],
                         "macro_f1": 100 * metrics["macro_f1"]})
            log(f"  {name:12s} fold {fold}  BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False)
        vals = [r["bacc"] for r in rows if r["arm"] == name]
        log(f"  {name:12s} ({dims}d)  BAcc {np.mean(vals):5.2f} +- {np.std(vals):.2f}")

    table = pd.DataFrame(rows)
    log("\n=== resumo ===")
    for name in arms:
        v = table.loc[table["arm"] == name, "bacc"].to_numpy()
        log(f"  {name:12s} {v.mean():5.2f} +- {v.std():.2f}")
    base = table.loc[table["arm"] == "clip", "bacc"].to_numpy()
    fus = table.loc[table["arm"] == "hand8+clip", "bacc"].to_numpy()
    if len(base) == len(fus) == args.folds:
        delta = fus - base
        log(f"  hand-8 SOBRE o CLIP: {delta.mean():+.2f} +- {delta.std():.2f}pp "
            f"({int((delta > 0).sum())}/{args.folds} folds)")
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
