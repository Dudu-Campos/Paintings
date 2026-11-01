"""As features degeneraram fora de arte? Colunas mortas e posto efetivo por corpus.

O §3 calibrou o `DEAD_COLUMNS` (22 colunas) na coorte do multitask e declara que
"os 617 sao identicos em multitask, oilpainting e rijksdraw (conferido em
31/ago)". Os QUATRO corpora de split oficial nunca foram conferidos, e a duvida
e legitima: se o banco colapsasse fora de arte, isso explicaria parte da
fraqueza do §4.3 sem ser falha de metodo.

Duas medidas, porque a primeira sozinha engana:

1. COLUNA MORTA (desvio zero). Pega degeneracao literal. Tudo que aparecer aqui
   passou pelo filtro global, entao e degeneracao POR CORPUS.
2. POSTO EFETIVO, participation ratio dos autovalores da covariancia
   padronizada: `PR = (sum l)^2 / sum l^2`. Vale d com espectro plano e 1 com
   toda a variancia num eixo. **Colapso num subespaco pequeno nao produz nenhuma
   coluna constante**, entao a medida 1 ficaria em zero e nada denunciaria.

Uso:  python3 experiments/audit_feature_degeneracy.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from pipeline.registry import load_blocks
from pipeline.corpora import HAND8

ARTE = ("multitask", "oilpainting", "rijksdraw")
EXTERNOS = ("minc2500", "places_lt", "food101", "nct_crc")


def participation_ratio(X: np.ndarray) -> float:
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    autoval = np.linalg.eigvalsh(np.cov(X / sd, rowvar=False))
    autoval = np.clip(autoval, 0.0, None)
    return float(autoval.sum() ** 2 / (autoval ** 2).sum())


def main() -> None:
    linhas = []
    for corpus in ARTE + EXTERNOS:
        cohort = load_blocks(corpus, list(HAND8))
        for bloco, X in cohort.blocks.items():
            sd = X.std(0)
            linhas.append(dict(
                corpus=corpus, familia="arte" if corpus in ARTE else "externo",
                bloco=bloco, n=len(cohort.stems), dims=X.shape[1],
                mortas=int((sd == 0).sum()),
                nao_finitas=int(np.isnan(X).sum() + np.isinf(X).sum()),
                posto_efetivo=participation_ratio(X)))
    d = pd.DataFrame(linhas)
    out = ROOT / "results/feature_degeneracy_audit.csv"
    d.to_csv(out, index=False)

    print(f"colunas mortas no total:  {d.mortas.sum()}")
    print(f"valores nao finitos:      {d.nao_finitas.sum()}")
    print(f"dims por corpus:          {sorted(d.groupby('corpus').dims.sum().unique())}")

    print("\nPOSTO EFETIVO por bloco (participation ratio)")
    piv = d.pivot_table(index="bloco", columns="corpus", values="posto_efetivo")
    piv = piv[list(ARTE + EXTERNOS)]
    print(piv.round(1).to_string())

    frac = d.groupby("corpus").posto_efetivo.sum() / d.groupby("corpus").dims.sum()
    frac = frac[list(ARTE + EXTERNOS)]
    print("\nPR / d por corpus:")
    print("  " + "  ".join(f"{c}={v:.3f}" for c, v in frac.items()))
    print(f"  amplitude {frac.max()/frac.min():.2f}x  "
          f"(min {frac.idxmin()}, max {frac.idxmax()})")
    print(f"\nescrito: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
