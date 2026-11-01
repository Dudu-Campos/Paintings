#!/usr/bin/env python3
"""Lista de stems de TREINO de cada fold, para o controle INDUTIVO do §5.20.

O Fisher vector ajusta PCA e GMM numa amostra de imagens. Nos corpora de CV o
metadado nao tem coluna `split`, entao o default ajusta na COORTE INTEIRA --
transdutivo (as duas sao nao-supervisionadas, nao ha vazamento de rotulo, mas o
teste esta no pool). Medido no rijksdraw: a transducao vale **+0,87pp sob sonda
linear** (p=0,0076, 5/5) e +0,38 n.s. sob denso-128.

A particao vem de `StratifiedKFold(5, shuffle=True, random_state=SEED)` com SEED
CONSTANTE, que e o que o `evaluate_banks_dense_mlp.py` usa -- NAO a semente do
laco, que ali so muda a inicializacao do MLP. Conferido no codigo em 12/out. E
por isso que 1 fold x 5 sementes e teste pareado: as cinco sementes compartilham
a particao.

Uso:
    python experiments/build_fold_train_stems.py --dataset rijksdraw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from corpora import HAND8          # noqa: E402
from registry import load_blocks   # noqa: E402

SEED, N_FOLDS = 42, 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    c = load_blocks(args.dataset, list(HAND8))
    stems = [str(s) for s in c.stems]
    y = LabelEncoder().fit_transform(np.asarray(c.labels, dtype=str))
    partes = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(y)), y))
    for k, (tr, te) in enumerate(partes, 1):
        alvo = args.out_dir / f"{args.dataset}_fold{k}_train_stems.txt"
        alvo.write_text("\n".join(stems[i] for i in tr) + "\n")
        print(f"  fold {k}: treino {len(tr):,}  teste {len(te):,}  -> {alvo.name}")


if __name__ == "__main__":
    main()
