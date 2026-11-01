"""Pareto plot for visual representations: BAcc × learned pipeline parameters.

The horizontal axis counts every *learned* parameter used at inference: frozen
encoder weights plus the WikiArt classification head.  Thus the hand-crafted
GIST representation contributes no encoder weights and is represented only by
its MLP.  Fixed descriptor settings and deterministic transforms are not
treated as learned parameters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "paper/pareto_pretreino_parametros.csv"
OUTPUT_PNG = ROOT / "paper/pareto_pretreino_parametros.png"
N_CLASSES = 27
# Parameters in the parts of the ImageNet encoders actually traversed here.
VGG19_FIRST_FIVE_CONVS = 555_328
VGG19_ALL_FEATURES = 20_024_384
RESNET50_FEATURE_BACKBONE = 23_508_032


def branch_mlp_parameters(branch_sizes: list[int]) -> int:
    """Trainable parameter count of the project BranchMLP architecture."""
    count = sum((size * 128 + 128) + 2 * 128 for size in branch_sizes)
    width = 128 * len(branch_sizes)
    for source, target, batch_norm in ((width, 512, True), (512, 256, True), (256, 128, True)):
        count += source * target + target
        if batch_norm:
            count += 2 * target
    count += 128 * N_CLASSES + N_CLASSES
    return count


def is_pareto(frame: pd.DataFrame) -> pd.Series:
    result = []
    for row in frame.itertuples(index=False):
        dominates = (
            (frame.downstream_trainable_parameters <= row.downstream_trainable_parameters)
            & (frame.bacc >= row.bacc)
            & ((frame.downstream_trainable_parameters < row.downstream_trainable_parameters)
               | (frame.bacc > row.bacc))
        )
        result.append(not dominates.any())
    return pd.Series(result, index=frame.index)


def main() -> None:
    rows = [
        {
            "method": "GIST (hand-crafted) + MLP", "family": "Hand-crafted",
            "bacc": 0.452248, "total_learned_parameters": branch_mlp_parameters([71, 130, 57]),
        },
        {
            "method": "VGG-19 conv_5 GAP + MLP", "family": "VGG pooled",
            "bacc": 0.458274,
            "total_learned_parameters": VGG19_FIRST_FIVE_CONVS + branch_mlp_parameters([256]),
        },
        {
            "method": "VGG-19 final GAP + MLP", "family": "VGG pooled",
            "bacc": 0.491821,
            "total_learned_parameters": VGG19_ALL_FEATURES + branch_mlp_parameters([512]),
        },
        {
            "method": "Gatys compact Gram + MLP", "family": "Gatys Gram",
            "bacc": 0.514800,
            "total_learned_parameters": VGG19_FIRST_FIVE_CONVS + branch_mlp_parameters([2640]),
        },
        {
            "method": "Frozen ResNet-50 linear", "family": "ResNet",
            "bacc": 0.554631,
            "total_learned_parameters": RESNET50_FEATURE_BACKBONE + 2048 * N_CLASSES + N_CLASSES,
        },
        {
            "method": "Frozen ResNet-50 + MLP", "family": "ResNet",
            "bacc": 0.530207,
            "total_learned_parameters": RESNET50_FEATURE_BACKBONE + branch_mlp_parameters([2048]),
        },
        {
            "method": "GIST + frozen ResNet-50", "family": "Fusion",
            "bacc": 0.554810,
            "total_learned_parameters": RESNET50_FEATURE_BACKBONE + branch_mlp_parameters([71, 130, 57, 2048]),
        },
        {
            "method": "Fine-tuned ResNet-50", "family": "ResNet",
            "bacc": 0.627181,
            "total_learned_parameters": RESNET50_FEATURE_BACKBONE + 2048 * N_CLASSES + N_CLASSES,
        },
    ]
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={"total_learned_parameters": "parameters"})
    frame["pareto"] = is_pareto(frame.rename(columns={"parameters": "downstream_trainable_parameters"}))
    frame.to_csv(OUTPUT_CSV, index=False)

    colors = {"Hand-crafted": "#555555", "VGG pooled": "#2878c8", "Gatys Gram": "#7651b5", "ResNet": "#e8782b", "Fusion": "#2a9d6f"}
    offsets = {
        "GIST (hand-crafted) + MLP": (8, -13),
        "VGG-19 conv_5 GAP + MLP": (8, -14),
        "VGG-19 final GAP + MLP": (8, 8),
        "Gatys compact Gram + MLP": (14, -10),
        "Frozen ResNet-50 linear": (8, 14),
        "Frozen ResNet-50 + MLP": (-8, 12),
        "GIST + frozen ResNet-50": (8, -16),
        "Fine-tuned ResNet-50": (8, 8),
    }
    labels = {
        "Frozen ResNet-50 linear": "ResNet-50 frozen linear",
        "Frozen ResNet-50 + MLP": "ResNet-50 frozen + MLP",
        "GIST + frozen ResNet-50": "GIST + ResNet-50 frozen",
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    for row in frame.itertuples(index=False):
        axis.scatter(row.parameters, row.bacc, s=90,
                     color=colors[row.family], edgecolor="#202020" if row.pareto else "white",
                     linewidth=1.4, zorder=3)
        offset = offsets[row.method]
        axis.annotate(labels.get(row.method, row.method), (row.parameters, row.bacc),
                      xytext=offset, textcoords="offset points", fontsize=8.5,
                      ha="right" if offset[0] < 0 else "left")
    frontier = frame.loc[frame.pareto].sort_values("parameters")
    axis.plot(frontier.parameters, frontier.bacc, linestyle="--", linewidth=1.3,
              color="#333333", label="Fronteira de Pareto", zorder=2)
    for family, color in colors.items():
        axis.scatter([], [], s=55, color=color, label=family)
    axis.set_xscale("log")
    axis.set(
        xlabel="Parâmetros aprendidos no pipeline visual (escala log)",
        ylabel="Balanced accuracy (BAcc)",
        title="Representações visuais: capacidade do pipeline × classificação de estilos",
        ylim=(0.44, 0.65),
    )
    axis.grid(which="both", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=8, frameon=True)
    figure.text(
        0.01, 0.01,
        "Eixo x: pesos aprendidos usados na inferência (encoder congelado + cabeça); GIST só inclui a MLP.\n"
        "Transformações determinísticas (SLIC, HOG, Hilbert e projeção Gram) não contam como pesos aprendidos.\n"
        "Coortes de GIST/VGG/Gatys têm 79.207 imagens, ResNet 79.936 e fusão 79.173.",
        fontsize=8, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    figure.savefig(OUTPUT_PNG, dpi=190)
    print(frame.to_string(index=False))
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
