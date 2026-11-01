"""R² múltiplo de cada bloco contra todos os outros, nas onze views estáticas.

O §6 usa esta quantidade e registra o negativo central do projeto: "Hilbert tem
R² 0,919 contra o resto e vale +3,76pp; o passeio do turista tem R² 0,161 e vale
zero" -- ou seja, redundância NÃO prevê contribuição marginal.

O que muda agora: naquela época a "contribuição" era marginal (LOO-like). Hoje
existem duas medidas melhores, e a pergunta pode ser refeita contra as duas:

  Shapley    crédito descontando substituibilidade (12 views, sonda LDA)
  cobertura  quanto o marginal cai quando há irmã da mesma família presente

A hipótese que o §6 derrubou era `redundância -> contribuição`. A que faz sentido
testar agora é `redundância -> COBERTURA`, que é a mesma ideia posta na variável
certa: R² mede predizibilidade linear, cobertura mede substituição funcional
dentro do classificador.

O grafo de similaridade fica FORA: ele é calculado a partir dos outros blocos,
então o R² dele contra "o resto" é trivialmente alto e não mede redundância de
medida. Sobram onze.

Ressalva sobre o texto: entra com os escores OOF do `build_text_scores` (OOF
simples sobre a coorte inteira). Para R² isso basta -- a matriz é fixa e a
pergunta é de predizibilidade, não de desempenho -- mas não use estes escores
para acurácia sem aninhar (§5.18).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_backward_elimination import EXTERNAL  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from registry import load_blocks, log  # noqa: E402
from view_names import display  # noqa: E402

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=ROOT / "results/block_redundancy_eleven.csv")
    args = ap.parse_args()

    cohort = load_blocks("multitask", HAND8, allow_external=True)
    stems = np.asarray(cohort.stems, dtype=str)
    blocks = {n: cohort.blocks[n].astype(np.float32) for n in HAND8}
    for name, (values_path, stems_path) in EXTERNAL.items():
        values = np.load(ROOT / values_path)
        order = pd.Index(np.load(ROOT / stems_path, allow_pickle=True)
                         .astype(str)).get_indexer(stems)
        if (order < 0).any():
            raise ValueError(f"{name}: stems ausentes")
        blocks[name] = values[order].astype(np.float32)
    scores = ROOT / "results/text_scores_student"
    order = pd.Index(np.load(scores / "multitask_stems.npy", allow_pickle=True)
                     .astype(str)).get_indexer(stems)
    blocks["Text"] = np.load(scores / "multitask_decisions.npy")[order].astype(np.float32)

    names = list(blocks)
    log(f"{len(stems):,} imagens, {len(names)} blocos: "
        + ", ".join(f"{n}({blocks[n].shape[1]})" for n in names))

    z = {n: StandardScaler().fit_transform(b).astype(np.float32) for n, b in blocks.items()}
    train, test = train_test_split(np.arange(len(stems)), test_size=0.25, random_state=SEED)

    rows = []
    for target in names:
        others = np.hstack([z[n] for n in names if n != target])
        y = z[target]
        model = Ridge(alpha=args.alpha).fit(others[train], y[train])
        pred = model.predict(others[test])
        # R² por coluna, e a média ponderada pela variância (que aqui é 1 em
        # todas, porque padronizamos) -- então média simples.
        ss_res = ((y[test] - pred) ** 2).sum(axis=0)
        ss_tot = ((y[test] - y[test].mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
        rows.append(dict(block=target, dims=y.shape[1], r2_mean=float(r2.mean()),
                         r2_median=float(np.median(r2)), r2_max=float(r2.max()),
                         frac_acima_05=float((r2 > 0.5).mean())))
        log(f"  {target:16s} R² médio {r2.mean():.3f}  mediano {np.median(r2):.3f}  "
            f"{100 * (r2 > 0.5).mean():.0f}% das colunas acima de 0,5")

    out = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    out.to_csv(args.output, index=False, float_format="%.4f")
    out["block"] = out["block"].map(display)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
