"""Figura de método: imagem -> superpixels SLIC -> grafo de adjacência.

A figura que faz o revisor entender a transformação de relance. Três painéis, e
o terceiro é o que importa: o RAG desenhado sobre a imagem, com um nó por região
e uma aresta por par de regiões vizinhas.

PARÂMETROS DE PRODUÇÃO, lidos do parquet e não do argparse (§8: os defaults do
`segmentation_multigraph_features.py` NÃO são o que rodou):

    max_side 1024 · segments 2048 · compactness 10 · sigma 0

**A tensão que a figura tem de resolver honestamente:** a produção realiza ~1.910
superpixels (`segmentation_actual_segments` no parquet), e 1.910 fronteiras numa
figura de artigo viram ruído cinza -- o revisor não veria nada. Então a figura
mostra DUAS granularidades: uma legível (`--illustrative`, ~200 regiões) para
explicar a operação, e a de produção ao lado, para não prometer uma partição
grossa que o método não usa. A legenda tem de dizer isso.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage import graph as skgraph
from skimage.segmentation import mark_boundaries, slic

Image.MAX_IMAGE_PIXELS = None
SURFACE, INK, MUTED = "#fbfbfa", "#1b1b1a", "#6b6a67"
# Producao (parquet de `outputs/*_slic_features`), nao os defaults do CLI.
PRODUCTION = dict(n_segments=2048, compactness=10.0, sigma=0.0, max_side=1024)


def load(path: Path, max_side: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.BILINEAR)
    return np.asarray(image)


def segment(rgb: np.ndarray, n_segments: int) -> np.ndarray:
    return slic(rgb, n_segments=n_segments, compactness=PRODUCTION["compactness"],
                sigma=PRODUCTION["sigma"], start_label=0, channel_axis=-1)


def draw_rag(axis, rgb: np.ndarray, labels: np.ndarray, box=None) -> tuple[int, int]:
    """RAG por diferença de cor média em Lab -- o mesmo grafo que o pipeline usa.

    Desenhado sobre um RECORTE, com as fronteiras por baixo. As duas coisas são
    necessárias e a primeira versão não tinha nenhuma: sobre a imagem inteira são
    425 arestas, que viram malha e ainda por cima **parecem uma triangulação de
    Delaunay** -- que no pipeline é um sistema de arestas SEPARADO (§3: RAG, kNN,
    Delaunay e MST sobre os mesmos nós). Sem as fronteiras não se vê que cada nó
    é uma região, que é a única coisa que a figura precisa comunicar.
    """
    rag = skgraph.rag_mean_color(rgb2lab(rgb), labels, mode="distance")
    centre = {}
    for region in np.unique(labels):
        rows, cols = np.nonzero(labels == region)
        centre[int(region)] = (cols.mean(), rows.mean())

    view = np.clip(mark_boundaries(rgb, labels, color=(1, 1, 1), mode="thick"), 0, 1)
    axis.imshow(view)
    inside = set(np.unique(labels).tolist())
    if box is not None:
        x0, y0, x1, y1 = box
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        inside = {n for n in rag.nodes
                  if x0 <= centre[int(n)][0] <= x1 and y0 <= centre[int(n)][1] <= y1}

    shown = 0
    for a, b in rag.edges:
        if int(a) not in inside and int(b) not in inside:
            continue
        (px, py), (qx, qy) = centre[int(a)], centre[int(b)]
        axis.plot([px, qx], [py, qy], color="#e34948", linewidth=2.0, alpha=0.9,
                  zorder=2, solid_capstyle="round")
        shown += 1
    xs = [centre[int(n)][0] for n in inside]
    ys = [centre[int(n)][1] for n in inside]
    axis.scatter(xs, ys, s=90, color="#e34948", zorder=3,
                 edgecolor="white", linewidth=1.4)
    return len(inside), shown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--illustrative", type=int, default=200,
                    help="regiões do painel legível; a produção usa 2048")
    ap.add_argument("--zoom", type=float, default=0.30,
                    help="fração do lado no recorte do painel do grafo")
    ap.add_argument("--output", type=Path,
                    default=Path("results/figures/slic_rag_illustration.png"))
    args = ap.parse_args()

    rgb = load(args.image, PRODUCTION["max_side"])
    coarse = segment(rgb, args.illustrative)
    fine = segment(rgb, PRODUCTION["n_segments"])

    height, width = rgb.shape[:2]
    # Recorte central, 30% do lado -- grande o bastante para uns 15 nós.
    half = args.zoom / 2
    box = (width * (0.5 - half), height * (0.5 - half),
           width * (0.5 + half), height * (0.5 + half))

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    titles = [
        ("original", None),
        (f"SLIC · {len(np.unique(coarse))} regions", None),
        ("region adjacency graph", None),
        (f"production granularity · {len(np.unique(fine))} regions", None),
    ]
    axes[0].imshow(rgb)
    axes[0].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                    fill=False, edgecolor="#e34948", linewidth=2.0))
    axes[1].imshow(np.clip(mark_boundaries(rgb, coarse, color=(1, 1, 1),
                                           mode="thick"), 0, 1))
    nodes, edges = draw_rag(axes[2], rgb, coarse, box=box)
    axes[3].imshow(np.clip(mark_boundaries(rgb, fine, color=(1, 1, 1),
                                           mode="subpixel"), 0, 1))

    titles[2] = (f"region adjacency graph (detail) · {nodes} nodes, {edges} edges",
                 None)
    for axis, (title, _) in zip(axes, titles):
        axis.set_title(title, fontsize=12.5, color=INK, pad=8)
        axis.set_xticks([]); axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color("#dcdcdc")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.02, wspace=0.04)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"figura: {args.output}")
    print(f"  ilustrativo {len(np.unique(coarse))} regiões, {edges} arestas")
    print(f"  produção    {len(np.unique(fine))} regiões")


if __name__ == "__main__":
    main()
