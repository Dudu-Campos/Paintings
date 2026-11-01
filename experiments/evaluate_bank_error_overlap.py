"""hand-8 x WND-CHARM: eles ERRAM NAS MESMAS IMAGENS? (Q de Yule)

Pergunta do usuario (07/out): o hand-8 ganha do WND-CHARM por +5,9 a +9,6pp em
cinco corpora e por **+0,06pp no rijksdraw**. Tres mecanismos candidatos, e a
sobreposicao de erro separa dois deles:

  (a) COR colapsa    -- desenho e tinta sobre papel; medido em outro script
  (b) TETO baixo     -- os dois extraem pouco e convergem perto do teto. Se for
                        isso, erram em imagens DIFERENTES com a mesma taxa, e o
                        Q fica comparavel ao dos outros corpora.
  (c) REDUNDANCIA    -- o banco de Shamir cobre o que o nosso cobre ali. Se for
                        isso, o Q do rijksdraw e MAIOR que o dos outros.

Q = (ad-bc)/(ad+bc) sobre a tabela 2x2 de acerto/erro. Funcao da razao de
chances, logo invariante a escala de linha e coluna -- e por isso compara
corpora com acuracias diferentes, que e exatamente o que se precisa aqui.

**O Q POOLADO INFLA COM O NUMERO DE CLASSES** (medido neste projeto:
rho=+0,656 contra n_classes), entao o `q_macro` -- media dos Q por classe
verdadeira -- e o analogo da BAcc e e ele que compara entre corpora.

Precisa das predicoes em disco:
    python experiments/evaluate_handcrafted_banks_lda.py --dataset <ds> --pred-dir outputs/banks_lda_pred

Uso:
    python experiments/evaluate_bank_error_overlap.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ["rijksdraw", "oilpainting", "multitask", "nct_crc", "minc2500", "food101"]
PAR = ("hand-8", "WND-CHARM")


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def carregar(pred_dir: Path, ds: str, banco: str, var: str, fold: int):
    slug = re.sub(r"[^0-9A-Za-z]+", "_", banco).strip("_")
    p = pred_dir / f"{ds}__{slug}__{var}__s42_f{fold}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["pred"], z["truth"], z["stems"].astype(str)


def yule(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Q e a razao de chances sobre o 2x2 de (acerto de A, acerto de B)."""
    n11 = float(np.sum(a & b)); n00 = float(np.sum(~a & ~b))
    n10 = float(np.sum(a & ~b)); n01 = float(np.sum(~a & b))
    den = n11 * n00 + n10 * n01
    if den == 0:
        return np.nan, np.nan
    q = (n11 * n00 - n10 * n01) / den
    orr = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
    return q, orr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/banks_lda_pred")
    ap.add_argument("--variante", default="shrinkage", choices=["svd", "shrinkage"])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/bank_error_overlap.csv")
    args = ap.parse_args()

    linhas = []
    for ds in CORPORA:
        for fold in args.folds:
            got = [carregar(args.pred_dir, ds, b, args.variante, fold) for b in PAR]
            if any(g is None for g in got):
                continue
            (pa, ta, sa), (pb, tb, sb) = got
            if set(sa) != set(sb):
                raise SystemExit(f"{ds} f{fold}: stems divergem entre os bancos")
            ordem = {s: i for i, s in enumerate(sa)}
            inv = np.empty(len(sb), dtype=int)
            inv[np.array([ordem[s] for s in sb])] = np.arange(len(sb))
            pb, tb = pb[inv], tb[inv]
            if not np.array_equal(ta, tb):
                raise SystemExit(f"{ds} f{fold}: rotulos divergem apos alinhar")
            acerto_a, acerto_b = (pa == ta), (pb == ta)

            q, orr = yule(acerto_a, acerto_b)
            # macro: um Q por classe VERDADEIRA, depois media -- o analogo da BAcc
            qs = []
            for c in np.unique(ta):
                m = ta == c
                qc, _ = yule(acerto_a[m], acerto_b[m])
                if np.isfinite(qc):
                    qs.append(qc)
            linhas.append({
                "dataset": ds, "variante": args.variante, "fold": fold,
                "n": len(ta), "classes": int(len(np.unique(ta))),
                "acc_hand8": 100 * acerto_a.mean(),
                "acc_wnd": 100 * acerto_b.mean(),
                "ambos": 100 * np.mean(acerto_a & acerto_b),
                "so_hand8": 100 * np.mean(acerto_a & ~acerto_b),
                "so_wnd": 100 * np.mean(~acerto_a & acerto_b),
                "nenhum": 100 * np.mean(~acerto_a & ~acerto_b),
                "q_pooled": q, "q_macro": float(np.mean(qs)),
                "classes_com_q": len(qs), "odds_ratio": orr})

    if not linhas:
        raise SystemExit("sem predicoes -- rodar o produtor com --pred-dir")
    t = pd.DataFrame(linhas)
    t.to_csv(args.output, index=False, float_format="%.4f")

    g = t.groupby("dataset").agg(
        classes=("classes", "first"), folds=("fold", "size"),
        acc_hand8=("acc_hand8", "mean"), acc_wnd=("acc_wnd", "mean"),
        so_hand8=("so_hand8", "mean"), so_wnd=("so_wnd", "mean"),
        nenhum=("nenhum", "mean"),
        q_pooled=("q_pooled", "mean"), q_macro=("q_macro", "mean"))
    log("\n=== hand-8 x WND-CHARM: sobreposicao de ACERTO ===")
    log(g.sort_values("q_macro", ascending=False).round(3).to_string())
    log("\nse o rijksdraw tiver o MAIOR q_macro, o empate e REDUNDANCIA;")
    log("se o q_macro dele estiver na faixa dos outros, e TETO.")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
