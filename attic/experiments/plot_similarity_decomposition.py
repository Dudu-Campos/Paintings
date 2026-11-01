"""Para cada representação: com quem ela se parece, decomposto por alvo.

A figura anterior colapsava tudo em "mesma família" contra "famílias
diferentes". Esta abre o segundo termo: uma curva por alvo, e a curva do próprio
bloco contra si mesmo em destaque.

A hipótese que ela testa é específica. HOG tem o menor r² máximo contra o
pipeline inteiro (mediana 0,050) e ao mesmo tempo o pior marginal medido
(−1,21pp), o que parecia contradição. Mas se ele for **internamente**
redundantíssimo e **externamente** ortogonal, as duas coisas viram uma só: 1.764
colunas amarradas entre si são um único canal ocupando dimensão de mil e
setecentos, e é exatamente isso que afoga as modalidades menores. O achado 6 já
tinha observado o sintoma -- "HOG cru é 81-93% das colunas dos subconjuntos
testados e domina a variância total do PCA conjunto" -- sem medir a causa.

A leitura é direta: se a curva "self" de um bloco estiver muito à direita das
suas curvas contra os outros, aquele bloco é um cluster fechado. Se estiverem
todas juntas, ele é difuso.

Saída: results/similarity_decomposition.png, _paper.png e o CSV.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

FAMILIES = {
    "image": ["Descriptors"],
    "region graphs": ["SLIC", "Felzenszwalb", "Line graph", "L(G_felz)"],
    "curve": ["Hilbert"],
    "spectral": ["Fourier"],
}
QUADRANT_SOURCE = {"Descriptors": "image", "SLIC": "region graphs",
                   "Hilbert": "curve"}
# Padrão enxuto: três grupos, nove curvas no total. Com sete a figura vira
# ilegível -- e três é também o teto em que a paleta categórica separa **todos**
# os pares para daltônicos (ver `palette.md`, cap sob `--pairs all`), então a cor
# volta a carregar identidade sozinha. `--groups` traz os outros de volta.
DEFAULT_GROUPS = ("image", "region graphs", "curve")
EXTRA = ("HOG", "Tourist walk", "Haralick")
COLOURS = {"image": "#2a78d6", "region graphs": "#eb6834", "curve": "#1baf7a",
           "spectral": "#eda100", "HOG": "#e34948", "Tourist walk": "#4a3aa7",
           "Haralick": "#008300"}
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948")
SELF = "#0b0b0b"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
EDGES = np.logspace(-6, 0, 49)


def standardise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    deviation = values.std(axis=0)
    deviation[deviation < 1e-8] = 1.0
    return (values - values.mean(axis=0)) / deviation


def pairwise_r2(left: np.ndarray, right: np.ndarray, same: bool,
                chunk: int = 512) -> np.ndarray:
    """Todos os r² entre colunas de `left` e de `right`.

    Com `same=True` só o triângulo superior entra: a diagonal é 1 por definição
    e os pares espelhados contariam duas vezes, deslocando a distribuição.
    """
    pieces, n = [], len(left)
    for start in range(0, left.shape[1], chunk):
        block = left[:, start:start + chunk]
        correlation = ((block.T @ right) / n) ** 2
        if same:
            rows = np.arange(start, start + block.shape[1])
            keep = rows[:, None] < np.arange(right.shape[1])[None, :]
            pieces.append(correlation[keep])
        else:
            pieces.append(correlation.ravel())
    return np.concatenate(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS),
                        help="grupos a exibir; qualquer família ou bloco extra")
    parser.add_argument("--assignment", type=Path, default=None,
                        help="usa os grupos empíricos deste CSV no lugar das "
                             "famílias de procedência")
    parser.add_argument("--min-group", type=int, default=20,
                        help="descarta grupos empíricos menores que isto")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/similarity_decomposition.png")
    args = parser.parse_args()

    blocks = sorted({b for names in FAMILIES.values() for b in names})
    cohort = load_blocks(args.dataset, blocks + ["Multiscale"] + list(EXTRA),
                         allow_external=True)
    available = set(cohort.blocks)

    groups: dict[str, np.ndarray] = {}
    for family, names in FAMILIES.items():
        parts = [standardise(cohort.blocks[n]) for n in names if n in available]
        if "Multiscale" in available:
            columns = np.array(cohort.columns["Multiscale"])
            quadrants = standardise(cohort.blocks["Multiscale"])
            for source, target in QUADRANT_SOURCE.items():
                if target != family:
                    continue
                mask = np.array([f"_{source}_" in c for c in columns])
                if mask.any():
                    parts.append(quadrants[:, mask])
        if parts:
            groups[family] = np.hstack(parts)
    for name in EXTRA:
        if name in available:
            groups[name] = standardise(cohort.blocks[name])

    if args.assignment:
        # Casar por chave (bloco, coluna); posição divergiria em silêncio.
        table = pd.read_csv(args.assignment)
        lookup = {}
        for block in cohort.blocks:
            if block == "Multiscale":
                continue
            for index, column in enumerate(cohort.columns[block]):
                lookup[(block, column.split("_", 1)[-1])] = (block, index)
        for index, column in enumerate(cohort.columns["Multiscale"]):
            origin = next((o for o in QUADRANT_SOURCE if f"_{o}_" in column), None)
            if origin:
                lookup[(f"Multiscale/{origin}", column.split("_", 1)[-1])] = (
                    "Multiscale", index)
        standard = {b: standardise(cohort.blocks[b]) for b in cohort.blocks}
        selection, samples = {}, {}
        for row in table.itertuples(index=False):
            entry = lookup.get((str(row.block), str(row.feature)))
            if entry is None:
                continue
            selection.setdefault(int(row.group), []).append(entry)
            samples.setdefault(int(row.group), []).append(str(row.feature))
        groups, args.groups = {}, []
        for number, entries in sorted(selection.items(),
                                      key=lambda kv: -len(kv[1])):
            if len(entries) < args.min_group:
                continue
            name = f"g{number}"
            groups[name] = np.column_stack(
                [standard[block][:, index] for block, index in entries])
            args.groups.append(name)
            COLOURS[name] = PALETTE[len(args.groups) - 1 % len(PALETTE)]
            log(f"  {name}: {len(entries)} colunas · exemplos: "
                + ", ".join(samples[number][:4]))
    log("grupos: " + ", ".join(f"{k} ({v.shape[1]}d)" for k, v in groups.items()))

    order = [k for k in args.groups if k in groups]
    if not order:
        raise SystemExit(f"nenhum grupo válido; disponíveis: {sorted(groups)}")
    groups = {k: groups[k] for k in order}
    rows = []
    distributions: dict[tuple[str, str], np.ndarray] = {}
    for source in order:
        for target in order:
            if (target, source) in distributions:      # simétrico, não recalcular
                distributions[(source, target)] = distributions[(target, source)]
                continue
            values = pairwise_r2(groups[source], groups[target], source == target)
            distributions[(source, target)] = values
            rows.append({"source": source, "target": target,
                         "n_pairs": int(values.size),
                         "median": float(np.median(values)),
                         "p90": float(np.percentile(values, 90)),
                         "frac_above_0.5": float((values > 0.5).mean())})
        log(f"  {source}: " + " · ".join(
            f"{t} {np.median(distributions[(source, t)]):.3f}" for t in order))

    table = pd.DataFrame(rows)
    table.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.6f")

    columns = min(3, len(order))
    rows_n = int(np.ceil(len(order) / columns))
    for with_title in (True, False):
        figure, axes = plt.subplots(rows_n, columns, figsize=(4.7 * columns, 4.4 * rows_n),
                                    sharex=True, sharey=True, squeeze=False)
        for axis, source in zip(axes.ravel(), order):
            for target in order:
                values = distributions[(source, target)]
                counts, _ = np.histogram(np.clip(values, 1e-6, 1), bins=EDGES)
                is_self = target == source
                axis.step(EDGES[:-1], counts / counts.sum(), where="post",
                          color=SELF if is_self else COLOURS.get(target, MUTED),
                          linewidth=2.4 if is_self else 1.5,
                          alpha=1.0 if is_self else 0.85,
                          label=("itself" if is_self else target),
                          zorder=4 if is_self else 3)
            axis.set_xscale("log")
            axis.set_title(f"{source}  ({groups[source].shape[1]}d)",
                           fontsize=10.5, color=TEXT, pad=8)
            axis.grid(alpha=0.25, linewidth=0.6, color=GRID)
            axis.legend(frameon=False, fontsize=9, loc="upper left",
                        handlelength=1.6, labelspacing=0.32)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        for axis in axes.ravel()[len(order):]:
            axis.set_visible(False)
        for axis in axes[-1]:
            axis.set_xlabel("r² between a pair of features")
        for axis in axes[:, 0]:
            axis.set_ylabel("share of pairs")
        if with_title:
            figure.suptitle(
                "Who each representation resembles, split by target\n"
                "black = the block against itself; colours = against the others",
                fontsize=12.5, color=TEXT)
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            path = args.output
        else:
            figure.tight_layout()
            path = args.output.with_name(args.output.stem + "_paper.png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        log(f"salvo: {path}")


if __name__ == "__main__":
    main()
