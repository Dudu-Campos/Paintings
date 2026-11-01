"""Quanto do hand-8 e COR, e quanto isso explica o empate do rijksdraw.

Pergunta do usuario (07/out): o ganho do hand-8 sobre o WND-CHARM e de +5,9 a
+9,6pp em cinco corpora e **+0,06pp no rijksdraw**. Hipotese: desenho e tinta
sobre papel (saturacao relativa medida em 0,144 contra 0,385 do oilpainting),
e o banco de Shamir e desenhado para microscopia em TONS DE CINZA -- textura,
morfologia e transformadas --, enquanto ~26% do hand-8 e cromatico. Se a cor
colapsa, nos perdemos um pedaco e o WND-CHARM nao perde nada.

O conjunto de cor (149 de 617, 24,1% -- contado pelo script, nao estimado):
  Descriptors[0:99]   hist Lab (90) + estatisticas globais Lab (9)   -- §10.2
  Felzenszwalb / SLIC          20 cada  (color_dist_*, {L,a,b}_*_w)
  Line graph / L(G_felz)       11 cada  (lg_color_*)
  Hilbert / Fourier / Gabor     0       -- operam sobre intensidade

Controle: o braco `hand-8` completo tem de reproduzir a tabela dos quatro
bancos. Reproduz nos tres (45,56 / 39,38 / 33,10 sob shrinkage) -- braco de
controle que nao reproduz valor conhecido nao e controle (§5.6).

RESULTADO (07/out): a hipotese esta na direcao certa e NAO e o mecanismo.
Remover a cor custa +0,67 no rijksdraw contra +1,04 no oilpainting e +1,32 no
multitask -- menos ali, como previsto, mas 0,37pp de diferenca contra 6,33pp de
vao a explicar, ou seja ~6% dele. E o braco `so cor` REFUTA a premissa: cor
sozinha entrega 77% do banco no rijksdraw (35,19 de 45,56) contra 66% nos
outros dois. Cor nao esta morta em desenho -- tom de papel e de tinta datam a
obra. Ela e MAIS redundante com o resto do banco ali, nao menos informativa.

Uso:
    python experiments/evaluate_colour_share_banks.py --datasets rijksdraw oilpainting multitask
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                                   # noqa: E402
from registry import load_blocks                            # noqa: E402
from evaluate_handcrafted_banks_lda import fit_probe        # noqa: E402

SEED, N_FOLDS = 42, 5
# cor dentro do Descriptors: hist Lab + estatisticas globais Lab (§10.2)
DESCRIPTORS_COR = 99
PADRAO_COR = re.compile(r"color|(?:^|_)(?:L|a|b)_(?:mean|std|min|max|gini|p\d+)", re.I)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mascara_cor(columns: dict[str, list[str]]) -> np.ndarray:
    """True nas colunas CROMATICAS, na ordem do hstack dos blocos."""
    partes = []
    for bloco, cols in columns.items():
        if bloco == "Descriptors":
            m = np.zeros(len(cols), dtype=bool)
            m[:DESCRIPTORS_COR] = True          # por indice: o bloco nao tem nome
        else:
            m = np.array([bool(PADRAO_COR.search(c)) for c in cols])
        partes.append(m)
    return np.concatenate(partes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["rijksdraw", "oilpainting", "multitask"])
    ap.add_argument("--variante", default="shrinkage", choices=["svd", "shrinkage"])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/colour_share_banks.csv")
    args = ap.parse_args()

    rows = pd.read_csv(args.output).to_dict("records") if args.output.exists() else []
    done = {(r["dataset"], r["condicao"], int(r["fold"])) for r in rows}

    for ds in args.datasets:
        c = load_blocks(ds, list(HAND8))
        X = np.hstack([c.blocks[b] for b in c.columns])
        cor = mascara_cor(c.columns)
        y = LabelEncoder().fit_transform(np.asarray(c.labels, dtype=str))
        C = len(np.unique(y))
        log(f"{ds}: {X.shape[0]:,} x {X.shape[1]}d | cor {cor.sum()}d "
            f"({100*cor.mean():.1f}%) | {C} classes")

        cond = {"hand-8": np.ones(X.shape[1], bool),
                "hand-8 sem cor": ~cor,
                "so cor": cor}
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            for nome, m in cond.items():
                if (ds, nome, fold) in done:
                    continue
                t0 = time.perf_counter()
                met, _, _ = fit_probe(X[:, m], y, tr, te, args.variante, C)
                rows.append({"dataset": ds, "condicao": nome, "fold": fold,
                             "variante": args.variante, "dims": int(m.sum()),
                             "seconds": time.perf_counter() - t0, **met})
                pd.DataFrame(rows).to_csv(args.output, index=False,
                                          float_format="%.4f")
                log(f"  f{fold} {nome:<15} {int(m.sum()):>4}d  "
                    f"BAcc {met['bacc']:5.2f}  ({rows[-1]['seconds']:.0f}s)")

    t = pd.DataFrame(rows)
    log("\n=== custo de REMOVER a cor, pareado por fold ===")
    for ds in args.datasets:
        g = t[t.dataset.eq(ds)]
        a = g[g.condicao.eq("hand-8")].set_index("fold").bacc
        b = g[g.condicao.eq("hand-8 sem cor")].set_index("fold").bacc
        s = g[g.condicao.eq("so cor")].set_index("fold").bacc
        x, yy = a.align(b, join="inner")
        if not len(x):
            continue
        d = x - yy
        log(f"{ds:<12} completo {a.mean():5.2f}  sem cor {b.mean():5.2f}  "
            f"so cor {s.mean():5.2f}  |  CUSTO {d.mean():+5.2f} "
            f"({(d>0).sum()}/{len(d)} folds)")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
