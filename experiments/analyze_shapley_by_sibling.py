"""Marginal de cada view CONDICIONADO a haver outra da mesma família na coalizão.

O LOO das views de segmentação é instável (§ do LOO: SLIC deu +2,31 num fold e
−2,19 no outro) e o §9.5 mediu que as duas famílias de segmentação se cobrem
mutuamente. Isto quantifica a cobertura direto: para cada view, a contribuição
marginal média QUANDO há uma irmã da mesma família presente, contra QUANDO não há.

**Pareamento por tamanho, que não é opcional.** Coalizões que contêm uma irmã
tendem a ser maiores, e marginal decresce com tamanho. Comparar as médias cruas
confundiria "tem irmã" com "é maior". Aqui o split é feito DENTRO de cada tamanho
de coalizão e recombinado com os pesos de Shapley, o que dá dois Shapley
contrafactuais pareados em tamanho:

    phi_com  = "o que phi seria se toda coalizão tivesse uma irmã"
    phi_sem  = "o que phi seria se nenhuma tivesse"

O phi real fica entre os dois, ponderado pelas proporções que de fato ocorrem.

**Famílias de MEDIDA, não o clado empírico.** O `view_names.FAMILIES` agrupa
Fourier, Gabor e Text como "Spectral & text", e o próprio módulo avisa que isso é
padrão de erro, não parentesco de medida -- Text é a única fonte fora da imagem.
Como a pergunta aqui é redundância de medida, Text sai para singleton.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_shapley_twelve_views import VIEWS  # noqa: E402
from view_names import display  # noqa: E402

MEASUREMENT_FAMILIES = {
    "Region graphs": ["Felzenszwalb", "SLIC", "L(G_felz)", "Line graph"],
    "Space-filling curve": ["Hilbert", "Random walk", "HVG"],
    "Spectral": ["Fourier", "Gabor"],
    "Image statistics": ["Descriptors"],
    "Similarity graph": ["Similarity graph"],
    "Text": ["Text"],
}


def conditional_shapley(values: dict[int, float], n: int, index: int,
                        siblings: list[int]):
    """(phi, phi_com_irma, phi_sem_irma) para uma view, pareado por tamanho."""
    phi = com = sem = 0.0
    bit = 1 << index
    sib_mask = 0
    for j in siblings:
        sib_mask |= 1 << j
    for k in range(n):
        weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
        total, with_sib, without_sib = [], [], []
        for mask, v in values.items():
            if mask & bit or bin(mask).count("1") != k:
                continue
            gain = values[mask | bit] - v
            total.append(gain)
            (with_sib if mask & sib_mask else without_sib).append(gain)
        count = math.comb(n - 1, k)
        phi += weight * sum(total)
        if with_sib:
            com += weight * count * float(np.mean(with_sib))
        if without_sib:
            sem += weight * count * float(np.mean(without_sib))
    return phi, com, sem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "results/multitask_shapley_twelve.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "results/shapley_by_sibling.csv")
    args = ap.parse_args()

    frame = pd.read_csv(args.input)
    n = len(VIEWS)
    complete = [f for f in sorted(frame.fold.unique()) if (frame.fold == f).sum() == 2 ** n]
    print(f"folds completos: {complete}")
    family_of = {v: fam for fam, members in MEASUREMENT_FAMILIES.items() for v in members}

    rows = []
    for fold in complete:
        sub = frame[frame.fold == fold]
        values = dict(zip(sub["mask"], sub["bacc"]))
        for i, view in enumerate(VIEWS):
            sibs = [j for j, other in enumerate(VIEWS)
                    if j != i and family_of[other] == family_of[view]]
            if not sibs:
                continue
            phi, com, sem = conditional_shapley(values, n, i, sibs)
            rows.append(dict(fold=fold, view=view, family=family_of[view],
                             shapley=phi, com_irma=com, sem_irma=sem,
                             cobertura=sem - com))
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, float_format="%.4f")

    agg = out.groupby(["family", "view"]).agg(
        shapley=("shapley", "mean"), com=("com_irma", "mean"),
        sem=("sem_irma", "mean"), cobertura=("cobertura", "mean")).reset_index()
    agg["view"] = agg["view"].map(display)
    print(f"\n{'família':22}{'view':20}{'φ':>7}{'c/ irmã':>9}{'s/ irmã':>9}{'cobertura':>11}")
    for fam in agg.family.unique():
        for r in agg[agg.family == fam].sort_values("shapley", ascending=False).itertuples():
            print(f"{fam:22}{r.view:20}{r.shapley:7.2f}{r.com:9.2f}{r.sem:9.2f}{r.cobertura:11.2f}")


if __name__ == "__main__":
    main()
