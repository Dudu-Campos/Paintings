"""Duas estruturas de redundância entre as mesmas modalidades, lado a lado.

O ato 2 do paper afirma que auditoria de redundância não decide o que entra no
pipeline. A afirmação é mais convincente mostrada do que dita, porque existem
**duas** noções de "essas duas representações se repetem" que qualquer revisor
consideraria intercambiáveis, e elas não são:

    espaço de representação   uma modalidade prevê linearmente as colunas da
                              outra? (R² múltiplo, ridge, fora da amostra)
    espaço de predição        as duas erram e acertam nas mesmas imagens?
                              (concordância entre predições fora do fold)

Se as duas estruturas fossem a mesma, tanto faria qual medir. A figura mostra
que não são -- e o tamanho do nó, que é a contribuição **medida** (marginal de
deixar-um-de-fora), mostra que nenhuma das duas prevê o que interessa.

Nota de honestidade que precisa ir na legenda: a largura da aresta é escalada
**dentro de cada painel**, porque as duas medidas vivem em faixas diferentes
(R² múltiplo sobe até ~0,9; concordância tem piso alto porque as sondas
concordam por acaso em classes frequentes). A comparação é de **estrutura**, não
de nível.

Identidade é carregada pelo rótulo direto de cada nó, não pela cor: com sete
categorias visíveis ao mesmo tempo, nenhuma paleta separa todos os 28 pares para
daltônicos, então a cor é reforço e o texto é o canal primário.

Saída: results/redundancy_two_spaces.png e _paper.png, mais as duas matrizes
em CSV.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED = 42
# As sete com predição fora do fold salva. Multiscale fica de fora porque a
# taxonomia nunca o sondou isoladamente (`canonical_mask()` colapsa o bit dele).
BLOCKS = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier")
# Paleta categórica fixa do projeto, slots 1-7, em ordem.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7")
TEXT, MUTED = "#0b0b0b", "#52514e"


def multiple_r2(target: np.ndarray, predictor: np.ndarray,
                train: np.ndarray, test: np.ndarray, alpha: float) -> float:
    """R² médio das colunas de `target` previstas por `predictor`, fora da amostra.

    Ridge por solução fechada: com no máximo ~150 preditores o sistema normal é
    minúsculo, e `sklearn` custaria mais em sobrecarga do que na álgebra.
    """
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(predictor[train])
    x_test = x_scaler.transform(predictor[test])
    y_train = y_scaler.fit_transform(target[train])
    y_test = y_scaler.transform(target[test])
    gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    weights = np.linalg.solve(gram, x_train.T @ y_train)
    residual = ((y_test - x_test @ weights) ** 2).sum(axis=0)
    # `y_test` está padronizado pelo TREINO, então a variância total do teste é
    # ~n e não exatamente n; usar a soma real evita R² inflado ou negativo por
    # deriva de escala entre as partições.
    total = (y_test ** 2).sum(axis=0)
    return float(np.mean(1 - residual / np.maximum(total, 1e-12)))


def representation_matrix(blocks: dict[str, np.ndarray], alpha: float) -> pd.DataFrame:
    n_rows = len(next(iter(blocks.values())))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_rows)
    train, test = order[: n_rows // 2], order[n_rows // 2:]
    matrix = pd.DataFrame(np.eye(len(BLOCKS)), index=list(BLOCKS), columns=list(BLOCKS))
    for a in BLOCKS:
        for b in BLOCKS:
            if a >= b:
                continue
            forward = multiple_r2(blocks[a], blocks[b], train, test, alpha)
            backward = multiple_r2(blocks[b], blocks[a], train, test, alpha)
            # Simétrico pela média: a figura é não direcionada, e a assimetria
            # (um bloco largo prevê um estreito melhor que o contrário) é
            # artefato de dimensão, não estrutura.
            value = max(0.0, (forward + backward) / 2)
            matrix.loc[a, b] = matrix.loc[b, a] = value
            log(f"  R² {a} × {b}: {forward:.3f} / {backward:.3f} -> {value:.3f}")
    return matrix


def prediction_matrix(path: Path) -> pd.DataFrame:
    stored = np.load(path, allow_pickle=True)
    matrix = pd.DataFrame(np.eye(len(BLOCKS)), index=list(BLOCKS), columns=list(BLOCKS))
    for a in BLOCKS:
        for b in BLOCKS:
            if a >= b:
                continue
            value = float((stored[a] == stored[b]).mean())
            matrix.loc[a, b] = matrix.loc[b, a] = value
            log(f"  concordância {a} × {b}: {value:.3f}")
    return matrix


def draw(axis, matrix: pd.DataFrame, sizes: dict[str, float], title: str,
         subtitle: str) -> None:
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(BLOCKS), endpoint=False)
    position = {name: (np.cos(a), np.sin(a)) for name, a in zip(BLOCKS, angles)}
    values = np.array([matrix.loc[a, b] for i, a in enumerate(BLOCKS)
                       for b in BLOCKS[i + 1:]])
    low, high = values.min(), values.max()
    span = max(high - low, 1e-9)

    for index, a in enumerate(BLOCKS):
        for b in BLOCKS[index + 1:]:
            strength = (matrix.loc[a, b] - low) / span
            axis.plot(*zip(position[a], position[b]), zorder=1,
                      linewidth=0.35 + 5.0 * strength ** 1.6,
                      color="#3d3d3d", alpha=0.10 + 0.55 * strength ** 1.4)
    # As três arestas mais fortes recebem o valor cru: sem isso o painel só
    # comunica ordem, e o leitor não consegue avaliar o nível.
    ranked = sorted(((matrix.loc[a, b], a, b) for i, a in enumerate(BLOCKS)
                     for b in BLOCKS[i + 1:]), reverse=True)[:3]
    for value, a, b in ranked:
        middle = np.mean([position[a], position[b]], axis=0)
        axis.annotate(f"{value:.2f}", middle, ha="center", va="center",
                      fontsize=7.5, color=MUTED, zorder=5,
                      bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                edgecolor="none", alpha=0.85))

    for colour, name in zip(COLOURS, BLOCKS):
        x, y = position[name]
        axis.scatter([x], [y], s=90 + 420 * sizes.get(name, 0.0), color=colour,
                     edgecolor="white", linewidth=1.6, zorder=3)
        axis.annotate(name, (x * 1.30, y * 1.30), ha="center", va="center",
                      fontsize=9.5, color=TEXT, zorder=6)
    axis.set_title(title, fontsize=11.5, color=TEXT, pad=14)
    axis.text(0.5, -0.06, subtitle, transform=axis.transAxes, ha="center",
              fontsize=8.6, color=MUTED)
    axis.set_xlim(-1.62, 1.62)
    axis.set_ylim(-1.52, 1.62)
    axis.set_aspect("equal")
    axis.axis("off")


def build(representation, prediction, sizes, output: Path, with_title: bool) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 6.8))
    draw(axes[0], representation, sizes, "Representation space",
         "multiple R² — does one modality linearly predict the other's columns?")
    draw(axes[1], prediction, sizes, "Prediction space",
         "agreement — do the two probes label the same images the same way?")
    if with_title:
        figure.suptitle(
            "Two notions of redundancy over the same seven modalities\n"
            "edge width scaled within panel (the comparison is of structure, "
            "not level); node area ∝ measured leave-one-out contribution",
            fontsize=12.5, color=TEXT)
        figure.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    log(f"salvo: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "results/channel_taxonomy_predictions.npz")
    parser.add_argument("--marginals", type=Path,
                        default=ROOT / "results/cka_vs_marginal/wikiart.csv")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/redundancy_two_spaces.png")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, BLOCKS, allow_external=False)
    log("espaço de representação (R² múltiplo, metade treino / metade teste):")
    representation = representation_matrix(cohort.blocks, args.alpha)
    log("espaço de predição (concordância entre predições fora do fold):")
    prediction = prediction_matrix(args.predictions)

    marginals = pd.read_csv(args.marginals).set_index("block")["loo_marginal"]
    ceiling = max(marginals.reindex(BLOCKS).max(), 1e-9)
    sizes = {name: max(0.0, marginals.get(name, 0.0)) / ceiling for name in BLOCKS}

    representation.to_csv(args.output.with_name("redundancy_representation.csv"),
                          float_format="%.6f")
    prediction.to_csv(args.output.with_name("redundancy_prediction.csv"),
                      float_format="%.6f")
    build(representation, prediction, sizes, args.output, True)
    build(representation, prediction, sizes,
          args.output.with_name(args.output.stem + "_paper.png"), False)

    from scipy.stats import spearmanr
    pairs = [(representation.loc[a, b], prediction.loc[a, b])
             for i, a in enumerate(BLOCKS) for b in BLOCKS[i + 1:]]
    rho, p_value = spearmanr(*zip(*pairs))
    log(f"\nas duas estruturas concordam entre si? Spearman {rho:+.3f} "
        f"(p={p_value:.4f}), n={len(pairs)} arestas")


if __name__ == "__main__":
    main()
