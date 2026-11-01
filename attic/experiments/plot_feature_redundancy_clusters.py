"""Redundância se agrupa por família de representação — e as famílias não são os blocos.

O ato 3 ajusta `BAcc(k) = a − b·k^(−c)`. Isso descreve a saturação; não a explica.
O mecanismo proposto é estrutura de clusters: se features de um mesmo tipo se
duplicam entre si e pouco entre tipos, o valor da k-ésima feature cai rápido
**dentro** de um tipo e é reposto ao trocar de tipo.

**A primeira versão testou isso no eixo errado e falhou.** Agrupando pelos oito
blocos do pipeline, o agrupamento hierárquico sobre a matriz de r² recuperava os
blocos com Rand ajustado de só 0,107. A causa foi apontada pelo usuário e é
óbvia em retrospecto: SLIC, Felzenszwalb, Line graph e L(G_felz) são todos
estrutura de grafo sobre regiões, e separá-los em quatro "modalidades" é decisão
de engenharia, não tipo natural. Reagrupados numa família `segmentation`:

    agrupamento        Rand ajustado   quase-duplicatas intra/inter
    bloco (7)              0,107                10,1×
    família (4)            0,194                29,7×

Quase o dobro do Rand e o triplo da razão, com uma taxonomia mais grosseira e
mais defensável. Isso reinterpreta achados anteriores na mesma direção: L(G_felz)
somou +0,79pp sobre L(G_slic), Quickshift +0,67pp, a percolação −0,5pp — todas
adições **dentro** da família de segmentação; enquanto Text, a única família
não-visual, derruba 6,46pp ao ser desligado apesar de não ter o maior Shapley.

Consequência: a unidade da curva de saturação é a **família**, não o bloco.

Duas ressalvas que precisam ir na legenda. `Multiscale` é literalmente cópia por
quadrante de Descriptors/SLIC/Hilbert, então ela infla a redundância *entre*
famílias e trabalha **contra** o contraste medido aqui — está incluída de
propósito, para o número não ser escolhido a favor. E o bloco `Text` é
fold-dependente por construção (escores OOF); a figura usa o fold 1, cujas linhas
de treino são in-fold. Text não entra em nenhum número final do paper.

Saída: results/feature_redundancy_clusters.png, _paper.png e o CSV.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

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
from registry import BY_NAME, load_blocks, log  # noqa: E402

SEED = 42
BLOCKS = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier", "Multiscale")
FAMILY_ORDER = ("image", "segmentation", "curve", "spectral", "multiscale", "text")
LABELS = {"image": "Image statistics", "segmentation": "Region graphs",
          "curve": "Space-filling curve", "spectral": "Spectral",
          "multiscale": "Quadrants", "text": "Text (external)"}
COLOURS = dict(zip(FAMILY_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                  "#e87ba4", "#4a3aa7")))
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
INTRA, INTER = "#1f1f1f", "#a9a8a2"


def load_text(path: Path, stems: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path).drop_duplicates("stem").set_index("stem")
    columns = [c for c in frame.columns if c.startswith("score_")]
    aligned = frame.reindex(stems)[columns]
    if aligned.isna().any().any():
        log("  Text: stems não batem com a coorte, bloco descartado")
        return None
    return aligned.to_numpy(dtype=np.float64)


def squared_correlation(values: np.ndarray) -> np.ndarray:
    standard = (values - values.mean(0)) / values.std(0)
    return (standard.T @ standard / len(standard)) ** 2


def cluster_panel(axis, strip, matrix, family, present) -> float:
    """Ordem decidida por agrupamento hierárquico, que nunca vê a família.

    Força-dirigida foi tentada e descartada: com limiar alto a maioria dos nós
    fica sem aresta e o layout os espalha ao acaso; com limiar baixo o grafo vira
    um componente gigante e colapsa. Os dois artefatos vêm de a mediana do r²
    intra ser ~0,05 — a estrutura é forte na cauda e fraca no corpo, o pior
    regime para força-dirigida. Agrupamento hierárquico não sofre disso e ainda
    permite **medir** o acerto em vez de olhar para ele.
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score

    distance = np.clip(1.0 - matrix, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    order = leaves_list(tree)
    rand = adjusted_rand_score(
        family, fcluster(tree, len(present), criterion="maxclust"))

    # Raiz quarta comprime a cauda: em escala linear 90% das células cairiam no
    # primeiro passo da rampa e o mapa seria uma superfície lisa.
    axis.imshow(matrix[np.ix_(order, order)] ** 0.25, cmap="magma_r",
                interpolation="nearest", vmin=0, vmax=1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Features ordered by hierarchical clustering on r²\n"
                   f"agreement with the true families: adjusted Rand {rand:.2f}",
                   fontsize=11.5, color=TEXT, pad=10)

    strip.imshow([[present.index(name) for name in family[order]]],
                 aspect="auto", interpolation="nearest",
                 cmap=matplotlib.colors.ListedColormap([COLOURS[f] for f in present]),
                 vmin=-0.5, vmax=len(present) - 0.5)
    strip.set_xticks([])
    strip.set_yticks([])
    strip.set_xlabel("the ordering never sees the family — contiguous colour "
                     "bands are the finding", fontsize=8.6, color=MUTED, labelpad=6)
    for side in ("top", "right", "bottom", "left"):
        strip.spines[side].set_visible(False)
    return rand


def distribution_panel(axis, matrix, family) -> pd.DataFrame:
    rows, columns = np.triu_indices(len(family), 1)
    values = matrix[rows, columns]
    same = family[rows] == family[columns]
    edges = np.logspace(-6, 0, 61)
    for mask, colour, label in ((same, INTRA, "same group"),
                                (~same, INTER, "different groups")):
        counts, _ = np.histogram(np.clip(values[mask], 1e-6, 1), bins=edges)
        axis.step(edges[:-1], counts / counts.sum(), where="post",
                  color=colour, linewidth=2.0, label=label, zorder=3)
    axis.set_xscale("log")
    axis.set_xlabel("r² between a pair of features")
    axis.set_ylabel("share of pairs")
    axis.set_title("Redundancy is concentrated inside groups",
                   fontsize=11.5, color=TEXT, pad=10)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    axis.grid(alpha=0.28, linewidth=0.6, color=GRID)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    summary = []
    for mask, label in ((same, "intra"), (~same, "inter")):
        subset = values[mask]
        summary.append({"scope": label, "n_pairs": int(mask.sum()),
                        "mean": float(subset.mean()),
                        "median": float(np.median(subset)),
                        "frac_above_0.5": float((subset > 0.5).mean())})
    table = pd.DataFrame(summary)
    intra, inter = table.iloc[0], table.iloc[1]
    axis.text(0.03, 0.62,
              f"near-duplicates (r² > 0.5)\n"
              f"  same group      {100 * intra['frac_above_0.5']:.2f}%\n"
              f"  different       {100 * inter['frac_above_0.5']:.2f}%\n"
              f"  ratio           {intra['frac_above_0.5'] / max(inter['frac_above_0.5'], 1e-12):.0f}×",
              transform=axis.transAxes, fontsize=9, color=TEXT, family="monospace",
              va="top")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--text-cache", type=Path,
                        default=ROOT / "outputs/tfidf_text_cache/fold_1.parquet")
    parser.add_argument("--assignment", type=Path, default=None,
                        help="usa os grupos empíricos deste CSV em vez das "
                             "famílias de procedência. Tautológico por "
                             "construção -- o agrupamento foi ajustado para "
                             "maximizar exatamente esta separação -- e serve "
                             "como TETO: quanto uma partição pode separar "
                             "neste dado, contra o que a nossa consegue.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/feature_redundancy_clusters.png")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, BLOCKS, allow_external=False)
    pieces = [cohort.blocks[name] for name in BLOCKS]
    family = []
    for name in BLOCKS:
        if name == "Multiscale":
            # Multiscale **não é** uma família: são Descriptors, SLIC e Hilbert
            # recalculados por quadrante. Tratá-la como categoria própria é erro
            # de categoria, e foi medido: com ela como família à parte o Rand
            # ajustado despencou de 0,194 para 0,001, porque um terço das colunas
            # ficava rotulado contra o seu próprio conteúdo. O nome da coluna
            # carrega a origem (`Multiscale_{quadrante}_{bloco}_…`), então a
            # atribuição correta é derivável, não arbitrada.
            source = {"Descriptors": "image", "SLIC": "segmentation",
                      "Hilbert": "curve"}
            assigned = []
            for column in cohort.columns[name]:
                match = next((v for k, v in source.items() if f"_{k}_" in column), None)
                if match is None:
                    raise ValueError(f"quadrante sem bloco de origem: {column}")
                assigned.append(match)
            family.append(np.array(assigned))
        else:
            family.append(np.full(cohort.blocks[name].shape[1], BY_NAME[name].family))
    text = None if args.assignment else load_text(args.text_cache, cohort.stems)
    if text is not None:
        pieces.append(text)
        family.append(np.full(text.shape[1], "text"))
        log(f"  Text incluído: {text.shape[1]} colunas (fold 1, OOF)")

    values = np.hstack(pieces).astype(np.float64)
    family = np.concatenate(family)
    if args.assignment:
        # Casar por chave (bloco, coluna), nunca por posição: as ordens de
        # iteração dos scripts divergem e o desalinhamento seria silencioso.
        table = pd.read_csv(args.assignment)
        key_to_group = {(str(r.block), str(r.feature)): f"group {r.group}"
                        for r in table.itertuples(index=False)}
        keys = []
        for name in BLOCKS:
            if name == "Multiscale":
                for column in cohort.columns[name]:
                    stripped = column.split("_", 1)[-1]
                    origin = next((s for s in ("Descriptors", "SLIC", "Hilbert")
                                   if f"_{s}_" in column), None)
                    keys.append((f"Multiscale/{origin}", stripped))
            else:
                keys += [(name, c.split("_", 1)[-1]) for c in cohort.columns[name]]
        assigned = np.array([key_to_group.get(k, "") for k in keys])
        known = assigned != ""
        values, family = values[:, known], assigned[known]
        log(f"grupos empíricos: {len(set(family))} · {known.sum()} colunas casadas "
            f"de {len(keys)}")
    keep = values.std(0) > 1e-9
    values, family = values[:, keep], family[keep]
    present = ([f for f in FAMILY_ORDER if (family == f).any()] if not args.assignment
               else sorted(set(family), key=lambda g: -int((family == g).sum())))
    log(f"{values.shape[1]} features · {len(present)} famílias: "
        + ", ".join(f"{f} ({int((family == f).sum())})" for f in present))
    matrix = squared_correlation(values)

    # Painel único, de propósito. A versão anterior trazia junto um mapa de calor
    # ordenado por agrupamento hierárquico, e ele foi **cortado**: com a maioria
    # das distâncias ≈1, ligação média cortada em k famílias produz um cluster
    # gigante mais alguns minúsculos -- patologia conhecida do método. O Rand
    # resultante media o comportamento do linkage, não a estrutura, e um número
    # que não distingue "não há estrutura" de "o instrumento não a enxerga" não
    # pode aparecer numa figura.
    for with_title in (True, False):
        figure, axis = plt.subplots(figsize=(8.6, 6.4))
        table = distribution_panel(axis, matrix, family)
        composition = " · ".join(
            f"{LABELS.get(f, f)} {int((family == f).sum())}" for f in present)
        axis.text(0.5, -0.22, composition, transform=axis.transAxes, ha="center",
                  fontsize=8.4, color=MUTED)
        if with_title:
            # A suptitle assume o papel do título do eixo; manter os dois
            # repetiria a mesma frase duas vezes na mesma figura.
            axis.set_title("")
            figure.suptitle(
                ("Upper bound: groups fitted to maximise this very separation"
                 if args.assignment else
                 "Redundancy is a property of the family, not of the block") + "\n"
                f"{values.shape[1]:,} features, {len(present)} groups, "
                "WikiArt (79,207 images)", fontsize=12.5, color=TEXT)
            figure.tight_layout(rect=(0, 0.03, 1, 0.94))
            path = args.output
        else:
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            path = args.output.with_name(args.output.stem + "_paper.png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        log(f"salvo: {path}")

    table.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.6f")
    log("\n" + table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
