"""Matriz de Q de Yule entre TODOS os modelos que tem posterior no mesmo fold.

POR QUE ELA EXISTE. O Q de Yule entre o nosso banco e uma CNN da ~+0,6 nas 14
celulas do `fusion_weight_crossfit_yule.csv`. Mas +0,6 so e informativo contra
um NULO, e o nulo certo nao e a independencia -- e **quanto dao duas CNNs entre
si**. Se dois backbones quaisquer tambem derem +0,6, entao o nosso numero nao
diz nada sobre features hand-designed: e o nivel basal de diversidade entre
modelos diferentes. Se derem muito mais, o nosso +0,6 significa alguma coisa.

E o mesmo raciocinio do §4.1f item 4, que mediu o teto do oraculo contra o nulo
de independencia e descobriu que o teto nao era merito nenhum. Aqui o objeto
auditado e o proprio Q.

Reporta Q MACRO (media por classe) como principal: o Q agrupado infla com o
numero de classes -- medido, rho=+0,656 (p=0,011) entre `Q_agrupado - Q_macro`
e `n_classes` nas 14 celulas --, pelo mesmo mecanismo que faz a Acc divergir da
BAcc sob desbalanceamento (§7.12).

Alinhamento POR STEM, nunca por posicao, e rotulos conferidos depois (§4.1f).

Uso:
    python experiments/evaluate_pairwise_yule.py --dataset oilpainting --fold 1
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]


def yule_q(ok_a, ok_b):
    a = float((ok_a & ok_b).sum()); b = float((ok_a & ~ok_b).sum())
    c = float((~ok_a & ok_b).sum()); d = float((~ok_a & ~ok_b).sum())
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    num, den = a * d - b * c, a * d + b * c
    return (float(num / den) if den else np.nan,
            float(a * d / (b * c)) if b * c else np.inf)


def yule_q_macro(y, ok_a, ok_b) -> float:
    qs = [yule_q(ok_a[y == c], ok_b[y == c])[0]
          for c in np.unique(y) if (y == c).sum() >= 2]
    return float(np.nanmean(qs)) if qs else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="oilpainting")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-name", default=None,
                    help="subdiretorio das grades, se diferir do dataset "
                         "(os posteriors do multitask vivem sob `multitask125`)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or ROOT / f"results/{args.dataset}_pairwise_yule.csv"

    fname = f"fold_{args.fold:02d}_posteriors.npz"
    grid_name = args.grid_name or args.dataset
    fontes = {
        "finetuned": ROOT / "outputs/final_finetune_grid" / grid_name,
        "do zero": ROOT / "outputs/scratch_grid" / grid_name,
    }
    models: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    stems_ref = y_ref = None
    for rotulo, raiz in fontes.items():
        if not raiz.exists():
            continue
        for arch_dir in sorted(raiz.iterdir()):
            p = arch_dir / f"seed_{args.seed}" / f"fold_{args.fold}" / fname
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            stems = np.array([str(s) for s in z["stems"]])
            if stems_ref is None:
                stems_ref, y_ref = stems, np.asarray(z["y_true"], np.int64)
            elif not np.array_equal(stems, stems_ref):
                raise SystemExit(f"{p}: stems divergem do primeiro modelo")
            models[f"{arch_dir.name} {rotulo}"] = (
                np.asarray(z["probabilities"], np.float64),
                np.asarray(z["y_true"], np.int64),
                "ImageNet + alvo" if rotulo == "finetuned" else "zero externa")

    if stems_ref is None:
        raise SystemExit(f"nenhum posterior em {list(fontes.values())}")

    our = ROOT / f"outputs/{args.dataset}_our_oof_cv.npz"
    if our.exists():
        z = np.load(our, allow_pickle=True)
        at = {str(s): i for i, s in enumerate(z["stems"])}
        idx = np.array([at[s] for s in stems_ref])
        if not (np.asarray(z["y_true"])[idx] == y_ref).all():
            raise SystemExit("rotulos do nosso OOF divergem DEPOIS de alinhar")
        models["hand-8 + grafo"] = (np.asarray(z[f"seed_{args.seed}"], np.float64)[idx],
                                    y_ref, "zero externa")

    y = y_ref
    ok = {k: (v[0].argmax(1) == y) for k, v in models.items()}
    print(f"{args.dataset}, fold {args.fold}, {len(y):,} imagens de teste, "
          f"{len(np.unique(y))} classes\n")
    print(f"{'modelo':28s} {'supervisao':18s} {'BAcc':>6s}")
    for k, v in models.items():
        print(f"{k:28s} {v[2]:18s} {100*balanced_accuracy_score(y, v[0].argmax(1)):6.2f}")

    rows = []
    for a, b in combinations(models, 2):
        q, orr = yule_q(ok[a], ok[b])
        rows.append({"dataset": args.dataset, "fold": args.fold,
                     "model_a": a, "model_b": b,
                     "sup_a": models[a][2], "sup_b": models[b][2],
                     "par": ("nosso x CNN" if "hand-8" in (a + b)
                             else "CNN x CNN"),
                     "yule_q": q, "odds_ratio": orr,
                     "yule_q_macro": yule_q_macro(y, ok[a], ok[b]),
                     "both": 100 * float((ok[a] & ok[b]).mean()),
                     "only_a": 100 * float((ok[a] & ~ok[b]).mean()),
                     "only_b": 100 * float((~ok[a] & ok[b]).mean()),
                     "neither": 100 * float((~ok[a] & ~ok[b]).mean())})
    t = pd.DataFrame(rows).sort_values("yule_q_macro")
    t.to_csv(out, index=False, float_format="%.4f")

    print(f"\n{'par':58s} {'Q macro':>8s} {'Q':>7s} {'OR':>6s}  tipo")
    for _, r in t.iterrows():
        print(f"{r['model_a'] + '  x  ' + r['model_b']:58s} "
              f"{r['yule_q_macro']:+8.3f} {r['yule_q']:+7.3f} {r['odds_ratio']:6.2f}  {r['par']}")

    print("\nO TESTE: o nosso banco e mais distinto de uma CNN do que duas CNNs sao entre si?")
    for tipo, sub in t.groupby("par"):
        print(f"  {tipo:12s} n={len(sub)}  Q macro {sub.yule_q_macro.mean():+.3f} "
              f"(min {sub.yule_q_macro.min():+.3f}, max {sub.yule_q_macro.max():+.3f})  "
              f"OR medio {sub.odds_ratio.mean():.2f}")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
