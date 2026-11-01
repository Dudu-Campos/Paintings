"""Filtros da primeira camada das cinco ResNets, visualizados e medidos.

A conv1 de uma ResNet-50 é 64 filtros de 7x7x3 -- patches RGB que se olham
direto, sem maximização de ativação nem truque nenhum. Cinco redes entram:

    random     pesos aleatórios (semente 42), o piso
    imagenet   ResNet-50 do torchvision
    scratch    treinada do zero nos 27 rótulos de estilo (fold 1)
    pretext v1 treinada a regredir 607d hand-designed, sem máscara
    pretext v2 idem, com máscara de 50% e Descriptors por PCA

A figura é o menos importante. O que ela serve para testar é a explicação
mecânica do achado desta madrugada: a CNN treinada em **rótulo** representa o
bloco Descriptors melhor (R² 0,381 por sonda linear) do que a rede
explicitamente treinada a prever Descriptors (0,281). Como 90 das 130 colunas
de Descriptors são histograma de cor Lab, a previsão que se segue é que a rede
de rótulo tenha **mais filtros cromáticos** na primeira camada.

Medida de cromaticidade, por filtro: decompõe-se o patch 7x7x3 na componente
acromática (a média entre os três canais, replicada) e no resto. A fração da
energia que sobra fora da acromática é o quanto aquele filtro responde a cor e
não a luminância:

    chroma(W) = 1 - ||mean_c(W)||^2 * 3 / ||W||^2

Zero = filtro idêntico nos três canais (borda/textura pura). Perto de um =
oponência de cor. Os filtros são ordenados por essa medida dentro de cada
painel, então a diferença entre as redes fica visível sem contar filtro a
filtro.

Saída: results/conv1_filters.png, _paper.png e results/conv1_filters.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]


def load_conv1(source: str) -> np.ndarray | None:
    """Devolve (64, 3, 7, 7) da primeira camada, ou None se a fonte não existe."""
    if source == "random":
        torch.manual_seed(42)
        return resnet50(weights=None).conv1.weight.detach().numpy()
    if source == "imagenet":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).conv1.weight.detach().numpy()
    path = Path(source)
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state["conv1.weight"].numpy()


def chroma(weights: np.ndarray) -> np.ndarray:
    """Fração da energia do filtro que NÃO é comum aos três canais."""
    achromatic = weights.mean(axis=1, keepdims=True)
    total = (weights ** 2).sum(axis=(1, 2, 3))
    common = (np.repeat(achromatic, weights.shape[1], axis=1) ** 2).sum(axis=(1, 2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, 1.0 - common / total, 0.0)


def montage(weights: np.ndarray, columns: int = 8) -> np.ndarray:
    """Grade de patches, cada um normalizado para [0,1] individualmente."""
    count, _, height, width = weights.shape
    rows = int(np.ceil(count / columns))
    canvas = np.ones((rows * (height + 1) + 1, columns * (width + 1) + 1, 3),
                     dtype=np.float32)
    for index in range(count):
        patch = weights[index].transpose(1, 2, 0)
        low, high = patch.min(), patch.max()
        patch = (patch - low) / (high - low) if high > low else np.zeros_like(patch)
        r, c = divmod(index, columns)
        top, left = r * (height + 1) + 1, c * (width + 1) + 1
        canvas[top:top + height, left:left + width] = patch
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/conv1_filters.png")
    args = parser.parse_args()

    sources = {
        "Random init": "random",
        "ImageNet": "imagenet",
        "Trained on labels": str(ROOT / "outputs/cnn_scratch/fold_01_best.pt"),
        "Pretext v1": str(ROOT / "outputs/cnn_pretrained_handdesigned/backbone_fold1.pt"),
        "Pretext v2": str(ROOT / "outputs/cnn_pretrained_handdesigned_v2/backbone_fold1.pt"),
    }
    banks, rows = {}, []
    for name, source in sources.items():
        weights = load_conv1(source)
        if weights is None:
            print(f"{name}: ausente, pulando")
            continue
        score = chroma(weights)
        order = np.argsort(-score)
        banks[name] = (weights[order], score[order])
        rows.append({"model": name, "filters": len(score),
                     "chroma_mean": float(score.mean()),
                     "chroma_median": float(np.median(score)),
                     "frac_above_0.5": float((score > 0.5).mean()),
                     "frac_above_0.7": float((score > 0.7).mean()),
                     "norm_mean": float(np.sqrt((weights ** 2).sum(axis=(1, 2, 3))).mean())})

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.6f")
    print(table.round(4).to_string(index=False))

    for with_title in (True, False):
        figure, axes = plt.subplots(1, len(banks) + 1,
                                    figsize=(3.1 * len(banks) + 3.4, 3.9),
                                    gridspec_kw={"width_ratios": [1] * len(banks) + [1.15]})
        for axis, (name, (weights, score)) in zip(axes, banks.items()):
            axis.imshow(montage(weights), interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for side in axis.spines.values():
                side.set_visible(False)
            axis.set_title(f"{name}\nchroma {score.mean():.2f}",
                           fontsize=10.5, fontweight="bold", pad=8)

        panel = axes[-1]
        panel.set_facecolor("#fbfbfa")
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for name, (_, score) in banks.items():
            panel.plot(np.arange(1, len(score) + 1), score, linewidth=2, label=name)
        panel.set_xlabel("Filter rank", fontsize=10)
        panel.set_ylabel("Chromatic energy fraction", fontsize=10)
        panel.set_ylim(0, 1)
        panel.grid(color="#dddddd", linewidth=0.7)
        panel.set_axisbelow(True)
        panel.legend(fontsize=8, frameon=False, loc="upper right")
        panel.set_title("Colour selectivity, sorted", fontsize=10.5,
                        fontweight="bold", pad=8)

        if with_title:
            figure.suptitle("First-layer filters (conv1, 7×7×3) across five ResNet-50 backbones",
                            fontsize=13, fontweight="bold", y=1.0)
        figure.tight_layout(rect=(0, 0, 1, 0.93 if with_title else 1))
        target = args.output if with_title else args.output.with_name(
            args.output.stem + "_paper.png")
        figure.savefig(target, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(figure)
        print(f"salvo: {target}")


if __name__ == "__main__":
    main()
