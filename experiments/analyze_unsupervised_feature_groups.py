"""Quais são os grupos reais de features, sem usar os nossos rótulos de origem?

Motivado por um argumento de redução ao absurdo: nada impede uma "feature de
grafo" que constrói o RAG, extrai a cor média e joga o grafo fora. Ela seria
rotulada `region graphs` pela procedência e seria, de fato, uma estatística de
cor. **Procedência não é natureza**, e as nossas famílias rotulam o pipeline que
produziu a coluna, não o que a coluna mede.

Logo a pergunta "o agrupamento não-supervisionado recupera as famílias?" está mal
posta -- ela pressupõe que as famílias sejam a resposta certa. A versão bem posta
é descritiva: **quais grupos existem**, e como cada um se distribui entre as três
procedências. Um grupo que mistura procedências é justamente o caso do absurdo
acima, e vale olhar quais colunas são.

Método: agrupamento espectral sobre a afinidade r². A escolha não é estética --
ligação média já foi tentada e produziu um cluster gigante com satélites, que é
a patologia conhecida quando quase todas as distâncias valem ~1. O espectral usa
a estrutura das arestas fortes, que é onde a redundância deste projeto mora
(a cauda: `r² > 0,5` separa 13×, a mediana não separa nada).

O número de grupos vem do **maior salto entre autovalores** do Laplaciano
normalizado, dentro de uma faixa, em vez de ser escolhido; e a saída inclui os
nomes das colunas medoides de cada grupo, porque a pergunta final ("o que este
grupo mede?") só se responde lendo o que ele contém.

Saída: results/unsupervised_feature_groups.csv e o resumo no log.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED = 42
FAMILIES = {
    "image": ["Descriptors"],
    "region graphs": ["SLIC", "Felzenszwalb", "Line graph", "L(G_felz)"],
    "curve": ["Hilbert"],
    "spectral": ["Fourier"],
}
QUADRANT_SOURCE = {"Descriptors": "image", "SLIC": "region graphs",
                   "Hilbert": "curve"}


def standardise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    deviation = values.std(axis=0)
    deviation[deviation < 1e-8] = 1.0
    return (values - values.mean(axis=0)) / deviation


def choose_k(affinity: np.ndarray, low: int, high: int) -> tuple[int, np.ndarray]:
    """Maior salto entre autovalores consecutivos do Laplaciano normalizado.

    Heurística do eigengap. Não é um oráculo, mas tira a escolha da minha mão --
    e o log imprime o espectro, então a decisão fica auditável.
    """
    degree = affinity.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(len(affinity)) - affinity * inverse[:, None] * inverse[None, :]
    values = np.linalg.eigvalsh(laplacian)[: high + 2]
    gaps = np.diff(values)
    window = range(low - 1, min(high, len(gaps)))
    best = max(window, key=lambda i: gaps[i])
    return best + 1, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=14)
    parser.add_argument("--k", type=int, default=0, help="0 = escolher por eigengap")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/unsupervised_feature_groups.csv")
    args = parser.parse_args()

    blocks = sorted({b for names in FAMILIES.values() for b in names})
    cohort = load_blocks(args.dataset, blocks + ["Multiscale"], allow_external=False)
    pieces, origin, names, source = [], [], [], []
    for family, members in FAMILIES.items():
        for block in members:
            pieces.append(standardise(cohort.blocks[block]))
            origin += [family] * cohort.blocks[block].shape[1]
            names += list(cohort.columns[block])
            source += [block] * cohort.blocks[block].shape[1]
    quadrant_columns = np.array(cohort.columns["Multiscale"])
    quadrants = standardise(cohort.blocks["Multiscale"])
    for source_block, family in QUADRANT_SOURCE.items():
        mask = np.array([f"_{source_block}_" in c for c in quadrant_columns])
        if mask.any():
            pieces.append(quadrants[:, mask])
            origin += [family] * int(mask.sum())
            names += list(quadrant_columns[mask])
            source += [f"Multiscale/{source_block}"] * int(mask.sum())

    values = np.hstack(pieces)
    origin = np.array(origin)
    names = np.array([n.split("_", 1)[-1] if "_" in n else n for n in names])
    keep = values.std(axis=0) > 1e-8
    source = np.array(source)
    values, origin, names, source = (values[:, keep], origin[keep], names[keep],
                                     source[keep])
    log(f"{values.shape[1]} features · procedências: "
        + ", ".join(f"{f} ({int((origin == f).sum())})" for f in FAMILIES))

    affinity = ((values.T @ values) / len(values)) ** 2
    np.fill_diagonal(affinity, 1.0)

    k, spectrum = choose_k(affinity, args.min_k, args.max_k)
    k = args.k or k
    log(f"espectro do Laplaciano (primeiros {args.max_k + 2}): "
        + " ".join(f"{v:.3f}" for v in spectrum))
    log(f"k escolhido = {k}")

    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(n_clusters=k, affinity="precomputed",
                                assign_labels="kmeans", random_state=SEED,
                                n_init=20).fit_predict(affinity)

    table = pd.DataFrame({"block": source, "feature": names,
                          "origin": origin, "group": labels})
    table.to_csv(args.output, index=False)

    contingency = pd.crosstab(table.group, table.origin)
    log("\n=== grupo × procedência ===")
    log(contingency.to_string())
    log("\n=== composição e conteúdo de cada grupo ===")
    for group in range(k):
        members = np.flatnonzero(labels == group)
        if not len(members):
            continue
        share = pd.Series(origin[members]).value_counts(normalize=True)
        # Medoide: a coluna mais central do grupo pela afinidade média. É a
        # melhor resposta curta a "o que este grupo mede".
        inner = affinity[np.ix_(members, members)].mean(axis=1)
        central = members[np.argsort(inner)[::-1][:5]]
        mixed = "MISTO" if share.iloc[0] < 0.80 else share.index[0]
        log(f"\n  grupo {group}  ({len(members)} features)  {mixed}")
        log("    procedência: "
            + ", ".join(f"{name} {100 * value:.0f}%" for name, value in share.items()))
        log("    centrais: " + " · ".join(names[central]))

    from sklearn.metrics import adjusted_mutual_info_score
    log(f"\ninformação mútua ajustada com a procedência: "
        f"{adjusted_mutual_info_score(origin, labels):.3f}")
    log("(descritivo: 0 = grupos ortogonais à procedência, 1 = grupos == procedência)")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
