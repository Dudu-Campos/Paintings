"""Painel isolado do RAG, para composicao manual (GIMP) na figura de metodo.

Sai sem eixo, sem titulo e sem margem, no tamanho nativo da imagem, para encaixar
ao lado do painel de superpixels sem reescalar.

Tres variantes, porque elas comunicam coisas diferentes:

  `image`   grafo sobre a imagem, com as fronteiras por baixo. Mantem a textura
            visivel; bom quando o painel vizinho ja mostrou a segmentacao.
  `mosaic`  grafo sobre as regioes preenchidas com a COR MEDIA delas. E a mais
            fiel ao que o metodo faz -- cada regiao colapsa num vetor de
            estatisticas, e o mosaico mostra literalmente essa perda de detalhe.
  `alone`   so nos e arestas, fundo transparente. Para empilhar como camada.

O grafo e o `rag_mean_color` em Lab, que e o mesmo RAG do pipeline. Os
parametros do SLIC sao os de producao lidos do parquet (compactness 10, sigma 0);
so `n_segments` e reduzido, e o painel de producao da figura maior mostra a
granularidade real (~1.900 regioes).

CUIDADO ao desenhar sobre a imagem inteira: com ~160 regioes sao ~425 arestas, e
a malha resultante **parece uma triangulacao de Delaunay** -- que no pipeline e
um sistema de arestas SEPARADO (§3: RAG, kNN, Delaunay e MST sobre os mesmos
nos). Dai as fronteiras por baixo em `image` e o mosaico em `mosaic`: as duas
deixam explicito que o no e uma REGIAO, nao um ponto solto.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import graph as skgraph
from skimage.color import label2rgb, rgb2lab
from skimage.segmentation import felzenszwalb, mark_boundaries, slic

Image.MAX_IMAGE_PIXELS = None
# Parametros de producao lidos do parquet (`segmentation_parameters`), nao do
# argparse -- o §8 registra que eles divergem e que isso ja causou erro.
PRODUCTION = dict(compactness=10.0, sigma=0.0, max_side=1024,
                  felz_scale=50.0, felz_sigma=0.8, felz_min_size=50)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    # SLIC e k-means em (Lab, x, y) com termo de COMPACIDADE, semeado numa grade
    # regular -- os centroides ficam perto das sementes e o RAG herda a trelica,
    # entao o painel parece um grid triangulado. Nao e artefato: e o que o SLIC
    # faz, e o §9.3 mede o mesmo por outro lado (ele fica preso em ~1.750-1.800
    # regioes de 128 a 1024 px). O Felzenszwalb nao tem termo de compacidade nem
    # de tamanho, as regioes variam muito, e o grafo PARECE um grafo -- melhor
    # para ilustrar. Lado a lado, os dois mostram por que as familias nao sao
    # redundantes (§9.5: interacao de -2,2pp ao estragar as duas).
    ap.add_argument("--method", choices=["slic", "felz"], default="slic")
    ap.add_argument("--segments", type=int, default=200,
                    help="alvo do SLIC; 200 realiza ~160, que e o painel aprovado")
    ap.add_argument("--felz-scale", type=float, default=None,
                    help="producao = 50 (mediana 542 regioes); maior = mais grosso")
    ap.add_argument("--felz-min-size", type=int, default=None)
    ap.add_argument("--style", choices=["image", "mosaic", "alone"], default="mosaic")
    # Marcador em PONTOS numa tela de poucos pixels fica gigante. `auto` escala
    # com o lado da imagem para o painel ficar igual em qualquer fonte.
    # `degree` codifica o grau na AREA do marcador, nao no raio: o `s` do
    # matplotlib ja e area em pontos², entao `s` proporcional ao grau da a
    # codificacao perceptualmente correta. Vale principalmente no Felzenszwalb,
    # onde as regioes variam muito de tamanho e existem hubs de grau ~40 (o ceu,
    # o cipreste) -- e justamente isso que distingue visualmente as duas
    # familias de segmentacao.
    ap.add_argument("--size-by", choices=["fixed", "degree"], default="degree")
    ap.add_argument("--size-range", type=float, nargs=2, default=(0.35, 3.2),
                    help="multiplicadores minimo e maximo sobre --node-size")
    ap.add_argument("--node-size", type=float, default=None)
    ap.add_argument("--edge-width", type=float, default=None)
    ap.add_argument("--upscale", type=float, default=2.0,
                    help="fator de saida, para compor sem reescalar depois")
    # Preto com anel branco destaca mais que vermelho sobre uma pintura escura
    # e saturada -- o vermelho compete com o ocre das estrelas.
    ap.add_argument("--node-colour", default="#0d0d0d")
    ap.add_argument("--node-ring", default="#ffffff")
    ap.add_argument("--edge-colour", default="#ffffff")
    # Aresta desenhada DUAS vezes: um traco escuro largo por baixo e o claro por
    # cima. E a tecnica de "casing" da cartografia, e resolve o problema real
    # aqui: uma linha de cor unica some contra fundo de luminancia variavel --
    # a branca desaparece nas estrelas e a preta no cipreste. O par sobrevive
    # aos dois.
    ap.add_argument("--edge-halo", default="#0d0d0d")
    # No mosaico as fronteiras sao REDUNDANTES -- as cores chapadas ja delimitam
    # as regioes -- e piores que redundantes: sendo brancas, competem com as
    # arestas brancas e o leitor deixa de distinguir aresta de fronteira.
    ap.add_argument("--boundaries", action="store_true", default=None)
    ap.add_argument("--no-boundaries", dest="boundaries", action="store_false")
    ap.add_argument("--halo-width", type=float, default=None,
                    help="largura EXTRA do contorno, cada lado; auto se omitido")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    args.output = args.output or Path(
        f"results/figures/rag_panel_{args.method}_{args.style}.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    scale = PRODUCTION["max_side"] / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.BILINEAR)
    rgb = np.asarray(image)

    if args.method == "slic":
        labels = slic(rgb, n_segments=args.segments,
                      compactness=PRODUCTION["compactness"],
                      sigma=PRODUCTION["sigma"], start_label=0, channel_axis=-1)
    else:
        labels = felzenszwalb(
            rgb, scale=args.felz_scale or PRODUCTION["felz_scale"],
            sigma=PRODUCTION["felz_sigma"],
            min_size=args.felz_min_size or PRODUCTION["felz_min_size"],
            channel_axis=-1)
    rag = skgraph.rag_mean_color(rgb2lab(rgb), labels, mode="distance")
    centre = {}
    for region in np.unique(labels):
        rows, cols = np.nonzero(labels == region)
        centre[int(region)] = (cols.mean(), rows.mean())

    height, width = rgb.shape[:2]
    side = max(width, height)
    if args.node_size is None:
        args.node_size = max(3.0, 0.030 * side * args.upscale)
    if args.edge_width is None:
        args.edge_width = max(0.4, 0.0022 * side * args.upscale)
    if args.halo_width is None:
        args.halo_width = max(0.6, 0.9 * args.edge_width)
    dpi = args.dpi
    fig = plt.figure(figsize=(width * args.upscale / dpi,
                              height * args.upscale / dpi), dpi=dpi)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_axis_off()

    if args.boundaries is None:
        args.boundaries = args.style == "image"   # no mosaico, off por padrao
    if args.style == "image":
        view = mark_boundaries(rgb, labels, color=(1, 1, 1),
                               mode="thick") if args.boundaries else rgb / 255.0
        axis.imshow(np.clip(view, 0, 1))
    elif args.style == "mosaic":
        # `rgb / 255` ANTES do label2rgb: ele devolve na escala da entrada, e
        # com uint8 o resultado sai em 0-255 -- o `clip(0, 1)` seguinte entao
        # achatava a figura inteira. Antes passava despercebido porque o
        # `mark_boundaries` normalizava por dentro.
        flat = label2rgb(labels, rgb / 255.0, kind="avg", bg_label=-1)
        if args.boundaries:
            flat = mark_boundaries(flat, labels, color=(1, 1, 1), mode="thick")
        axis.imshow(np.clip(flat, 0, 1))
    else:
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)

    segments = [(centre[int(a)], centre[int(b)]) for a, b in rag.edges]
    # `stroke`, nao `width`: o §8 registra que variavel de laco com o nome de uma
    # global ja custou tempo aqui, e `width` e a largura da IMAGEM tres linhas
    # acima -- o print de tamanho saiu "4x802" antes do conserto.
    for colour, stroke, order, alpha in (
            (args.edge_halo, args.edge_width + 2 * args.halo_width, 1.5, 0.95),
            (args.edge_colour, args.edge_width, 2.0, 1.0)):
        if colour.lower() in ("none", ""):
            continue
        for (px, py), (qx, qy) in segments:
            axis.plot([px, qx], [py, qy], color=colour, linewidth=stroke,
                      alpha=alpha, zorder=order, solid_capstyle="round")
    order = list(rag.nodes)
    degrees = np.array([rag.degree(n) for n in order], dtype=float)
    if args.size_by == "degree" and degrees.max() > degrees.min():
        lo, hi = args.size_range
        scale_by = degrees / degrees.mean()
        sizes = args.node_size * np.clip(scale_by, lo, hi)
    else:
        sizes = np.full(len(order), args.node_size)
    # Grandes por baixo: senao um hub cobre os vizinhos de grau 2 ao redor dele.
    draw = np.argsort(-sizes)
    axis.scatter([centre[int(order[i])][0] for i in draw],
                 [centre[int(order[i])][1] for i in draw],
                 s=sizes[draw], color=args.node_colour, zorder=3,
                 edgecolor=args.node_ring,
                 linewidth=max(0.6, args.edge_width * 0.9))

    fig.savefig(args.output, dpi=dpi,
                transparent=(args.style == "alone"))
    print(f"{args.method:5s} {args.output}  {round(width*args.upscale)}x"
          f"{round(height*args.upscale)}px  "
          f"{rag.number_of_nodes()} nós, {rag.number_of_edges()} arestas | "
          f"grau min {int(degrees.min())} mediana {int(np.median(degrees))} "
          f"max {int(degrees.max())}")


if __name__ == "__main__":
    main()
