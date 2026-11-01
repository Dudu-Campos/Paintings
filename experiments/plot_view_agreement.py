"""Concordância entre views: quantas acertam junto, e quais andam juntas.

Substitui o UpSet, que não serve neste regime. Com 12 views há 2^12 = 4.096
padrões possíveis e **3.253 ocorrem**; as 18 barras que um UpSet legível
comporta cobrem 19,4% do corpus e escondem 50% em 3.235 padrões. Ele mostra os
padrões MODAIS, e num espaço tão espalhado o modal não representa a massa --
lê-lo dá a impressão errada de que os acertos são majoritariamente solitários
quando eles são 16,2% contra 53,2% de acertos compartilhados.

A figura é a matriz de concordância par a par dos conjuntos de acerto,
**normalizada pela concordância esperada sob independência**. Cobre 100% do
corpus e 100% dos pares.

A normalização não é cosmética. O IoU cru é confundido com acurácia -- a
interseção é limitada pela menor das duas views e a união é grande, então duas
views fracas têm IoU baixo com tudo por aritmética. Medido: IoU cru contra a
acurácia mínima do par dá **r = +0,938**, ou seja, a matriz mostrava 88% de
variância de competência individual e quase nada de estrutura. Dividindo pelo
esperado sob independência,

    IoU_esperado(a, b) = a*b / (a + b - a*b),

o confundimento cai para r = +0,248 e as famílias aparecem: a de série temporal
(Hilbert, Random walk, HVG) sai de "dispersa" para o bloco mais coeso da
matriz, e Gabor x Fourier emerge como par espectral -- nenhum dos dois era
visível no IoU cru, porque as views envolvidas são fracas.

O histograma de "quantas views acertam junto" foi removido da figura: dividir a
largura em dois painéis deixava os rótulos das 12 views pequenos demais, e a
matriz carrega a informação mais acionável (QUAIS views andam juntas, não só
quantas). Os números do histograma continuam saindo no stdout, para o texto --
inclusive a comparação contra a Poisson-binomial, que é a nula certa aqui: ela
mantém a acurácia observada de cada view e assume apenas erro descorrelacionado,
preservando a média (2,88 acertos nas duas) de modo que toda a diferença entre
observado e nulo seja **forma**, não nível. O acaso 1/25 não serviria como
referência -- ele preveria 61,3% de imagens sem nenhum acerto, mais que os 30,6%
observados, e "explicaria" a opacidade sem dizer nada sobre correlação.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import displays  # noqa: E402

SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"
ACCENT, REFERENCE = "#0b7a5a", "#d13c3c"
HIDDEN = {"Similarity graph (class scores)"}


def poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    """P(k acertos) se as views fossem independentes, com as acurácias
    observadas. É a hipótese nula certa: não 'todas iguais ao acaso', mas
    'estas views, sem correlação de erro'."""
    distribution = np.zeros(len(probabilities) + 1)
    distribution[0] = 1.0
    for p in probabilities:
        distribution[1:] = distribution[1:] * (1 - p) + distribution[:-1] * p
        distribution[0] *= (1 - p)
    return distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/view_predictions_multitask.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/view_agreement.png")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    labels = data["labels"]
    views = [k for k in data.files if k not in ("labels", "stems") and k not in HIDDEN]
    hit = {v: data[v] == labels for v in views}
    views = sorted(views, key=lambda v: -hit[v].mean())
    matrix = np.column_stack([hit[v] for v in views])
    counts = matrix.sum(axis=1)
    accuracies = np.array([hit[v].mean() for v in views])

    observed = np.bincount(counts, minlength=len(views) + 1) / len(labels)
    expected = poisson_binomial(accuracies)

    figure, right = plt.subplots(figsize=(9.8, 8.4))

    size = len(views)
    iou = np.eye(size)
    for i, j in itertools.combinations(range(size), 2):
        both = (matrix[:, i] & matrix[:, j]).mean()
        either = (matrix[:, i] | matrix[:, j]).mean()
        a, b = accuracies[i], accuracies[j]
        # Razão observado/esperado: 1,0 é "concordam exatamente o que a
        # competência individual já explica"; acima disso é estrutura.
        iou[i, j] = iou[j, i] = (both / either) / ((a * b) / (a + b - a * b))
    shown = iou.copy()
    np.fill_diagonal(shown, np.nan)
    # Cinza, sem matiz: a figura anterior usa viridis para codificar IDENTIDADE
    # de view (rampa categórica), e reusar o mesmo mapa aqui, onde a cor codifica
    # MAGNITUDE, faria o leitor carregar a associação "amarelo = Similarity
    # graph". Ausência de matiz sinaliza que a quantidade é outra. Também é o
    # único candidato perfeitamente uniforme sob as três condições de daltonismo
    # (pior par adjacente 12,4 em todas) e idêntico em preto e branco.
    # Escala LOG na cor. Com 125 classes as views duplicadas (Hilbert x Random
    # walk = 16,5; Fourier x Gabor = 15,3) esticam a barra e achatam tudo entre 2
    # e 8 no branco -- inclusive a linha do Text, que e a mais informativa da
    # matriz (razao media 2,56 contra 6,3-7,9 das visuais). Em escala linear a
    # figura mostrava dois pares e escondia a estrutura.
    from matplotlib.colors import LogNorm
    image = right.imshow(shown, cmap="Greys",
                         norm=LogNorm(vmin=max(1.0, np.nanmin(shown)),
                                      vmax=np.nanmax(shown)))
    # Diagonal riscada em vez de branca: em escala de cinza o branco fica quase
    # igual às células mais claras (Text, ~1,7), e a diagonal se leria como
    # "concordância mínima" em vez de "não se aplica". No viridis ela destacava
    # contra o roxo escuro; aqui não destaca mais.
    for index in range(size):
        right.add_patch(plt.Rectangle((index - 0.5, index - 0.5), 1, 1,
                                      facecolor="white", edgecolor="#b8b8b8",
                                      hatch="////", linewidth=0.8, zorder=3))
    right.set_xticks(range(size))
    right.set_yticks(range(size))
    right.set_xticklabels(displays(views), rotation=52, ha="right", fontsize=14, color=INK)
    right.set_yticklabels(displays(views), fontsize=14, color=INK)
    for spine in right.spines.values():
        spine.set_visible(False)
    right.tick_params(length=0)
    bar = figure.colorbar(image, ax=right, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=13)
    bar.set_label("agreement relative to independence  (1.0 = no structure)",
                  fontsize=13.5, color=MUTED, labelpad=10)

    figure.suptitle("Pairwise agreement between views, relative to independence",
                    fontsize=16, fontweight="bold", color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")
    off = ~np.eye(size, dtype=bool)
    print(f"  razão obs/indep: mediana {np.median(iou[off]):.2f}  "
          f"min {iou[off].min():.2f}  max {iou[off].max():.2f}")
    print(f"  união observada {100 * (counts > 0).mean():.1f}%  "
          f"se independentes {100 * (1 - expected[0]):.1f}%")
    print(f"  média de views acertando (entre as acertadas) "
          f"{counts[counts > 0].mean():.2f} de {size}")
    print("\nPARA O TEXTO -- distribuição de quantas views acertam a mesma imagem:")
    print(f"{'k':>3}{'observado':>11}{'independência':>15}")
    for k in range(size + 1):
        print(f"{k:>3}{100 * observed[k]:>10.1f}%{100 * expected[k]:>14.1f}%")


if __name__ == "__main__":
    main()
