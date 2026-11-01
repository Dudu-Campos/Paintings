"""O peso da fusao escolhido no `val` OFICIAL -- a rota limpa do §4.1f.

O §4.1f reporta o peso por CROSS-FIT dentro do teste, com a ressalva de que ele
"remove a circularidade da selecao, nao o uso da distribuicao do teste". A rota
sem essa ressalva escolhe o peso num conjunto que nenhum dos dois lados viu no
gradiente -- e ela existe em UM corpus so:

* `minc2500`: `val` oficial de 2.875 linhas, fora do gradiente dos dois lados.
* `places_lt`: teria 7.300, mas so **1.780 imagens estao no NFS** (o teste esta
  completo). 4,9 por classe em 365 e mais ruidoso que o cross-fit, entao nao
  serve como alternativa limpa.
* os outros cinco: o "val" e fatia do proprio treino, JA vista pelo gradiente do
  backbone -- o §4.1c mede 96,65% de acerto nas linhas de treino contra 67,32%
  no teste.

RESSALVA QUE SOBREVIVE, e agora esta medida em vez de argumentada: o `val` foi
usado para escolher a EPOCA dos dois lados, entao os posteriors dele sao
otimistas. O script imprime `val - teste` de cada lado para que o leitor veja o
tamanho do vies, em vez de aceitar a palavra de que ele e pequeno.

Alinhamento SEMPRE por stem, nunca por posicao (§4.1f: o MINC guarda os stems em
ordem diferente nos dois arquivos, e parear por posicao ja produziu contradicao).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = np.round(np.arange(0.0, 1.001, 0.05), 2)


def log(message: str) -> None:
    print(message, flush=True)


def blend(ours: np.ndarray, backbone: np.ndarray, w: float) -> np.ndarray:
    return w * np.log(np.maximum(ours, 1e-12)) + (1 - w) * np.log(np.maximum(backbone, 1e-12))


def bacc(y: np.ndarray, probabilities: np.ndarray) -> float:
    return 100 * balanced_accuracy_score(y, probabilities.argmax(1))


def align(source_stems, target_stems, values, label=""):
    order = pd.Index(list(source_stems)).get_indexer(list(target_stems))
    if (order < 0).any():
        raise ValueError(f"{label}: {int((order < 0).sum())} stems ausentes")
    return values[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="minc2500")
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_validation.csv")
    args = ap.parse_args()

    ours_test = np.load(ROOT / f"outputs/{args.dataset}_our_posteriors.npz", allow_pickle=True)
    ours_val = np.load(ROOT / f"outputs/{args.dataset}_our_val_posteriors.npz", allow_pickle=True)
    test_stems, y_test = ours_test["stems"].astype(str), ours_test["y_true"]
    val_stems, y_val = ours_val["stems"].astype(str), ours_val["y_true"]
    log(f"{args.dataset}: val={len(val_stems):,}  test={len(test_stems):,}  "
        f"classes={int(y_test.max()) + 1}")

    grid = ROOT / "outputs/final_finetune_grid" / args.dataset
    rows = []
    for model in args.models:
        for seed in args.seeds:
            base = grid / model / f"seed_{seed}"
            tb = np.load(next(base.glob("fold_*/fold_*_posteriors.npz")), allow_pickle=True)
            vb = np.load(next(base.glob("fold_*/val_posteriors.npz")), allow_pickle=True)
            B_test = align(tb["stems"].astype(str), test_stems, tb["probabilities"],
                           f"{model}/{seed} teste")
            B_val = align(vb["stems"].astype(str), val_stems, vb["probabilities"],
                          f"{model}/{seed} val")
            if not (align(tb["stems"].astype(str), test_stems, tb["y_true"]) == y_test).all():
                raise ValueError(f"{model}/{seed}: rotulos de teste divergem apos alinhar")
            if not (align(vb["stems"].astype(str), val_stems, vb["y_true"]) == y_val).all():
                raise ValueError(f"{model}/{seed}: rotulos de val divergem apos alinhar")
            P_test, P_val = ours_test[f"seed_{seed}"], ours_val[f"seed_{seed}"]

            # (1) peso escolhido no VAL, aplicado ao teste -- a rota limpa.
            w_val = float(max(WEIGHTS, key=lambda w: bacc(y_val, blend(P_val, B_val, w))))
            # (2) cross-fit DENTRO do teste -- o estimador do §4.1f.
            pred_cross = np.empty_like(y_test)
            w_cross = []
            for tr, te in StratifiedKFold(args.folds, shuffle=True,
                                          random_state=42).split(np.zeros(len(y_test)), y_test):
                w = max(WEIGHTS, key=lambda w: bacc(y_test[tr], blend(P_test[tr], B_test[tr], w)))
                w_cross.append(w)
                pred_cross[te] = blend(P_test[te], B_test[te], w).argmax(1)
            # (3) peso varrido NO TESTE -- espia o desfecho, so como teto.
            w_peek = float(max(WEIGHTS, key=lambda w: bacc(y_test, blend(P_test, B_test, w))))

            rows.append(dict(
                dataset=args.dataset, model=model, seed=seed,
                ours_val=bacc(y_val, P_val), ours_test=bacc(y_test, P_test),
                backbone_val=bacc(y_val, B_val), backbone_test=bacc(y_test, B_test),
                w_half=bacc(y_test, blend(P_test, B_test, 0.5)),
                w_validation=bacc(y_test, blend(P_test, B_test, w_val)),
                w_crossfit=100 * balanced_accuracy_score(y_test, pred_cross),
                w_peek=bacc(y_test, blend(P_test, B_test, w_peek)),
                weight_val=w_val, weight_cross=float(np.mean(w_cross)), weight_peek=w_peek))
            r = rows[-1]
            log(f"  {model}/seed {seed}: w_val={w_val:.2f} w_cross={r['weight_cross']:.2f} "
                f"w_peek={w_peek:.2f} | backbone {r['backbone_test']:.2f} -> "
                f"val {r['w_validation']:.2f}  cross {r['w_crossfit']:.2f}  "
                f"half {r['w_half']:.2f}")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False, float_format="%.4f")
    log(f"\nFIM: {args.output}")


if __name__ == "__main__":
    main()
