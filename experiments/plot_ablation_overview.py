"""Figura de ablação atualizada: modalidades isoladas + progressão aninhada.

Substitui a figura antiga (pipeline completo em 0,371), desatualizada em três
frentes que importam:

  1. O número mudou. 0,371 -> **0,5442** na configuração de produção atual
     (1.154d, 12 branches), e o pipeline canônico da fatorial faz 0,5288.
  2. A figura antiga mostrava "Cam. Aleatória" e "Cam. Híbrida" como blocos
     independentes. Depois medimos que cada feature delas é praticamente cópia
     de alguma coluna de Hilbert (r² máximo: mediana 0,918, p90 0,984) -- não
     são modalidades, são duplicatas, e exibi-las como barras próprias sugeria
     diversidade que não existe.
  3. A progressão era "melhor combinação de cada tamanho", ou seja máximo sobre
     C(8,k) subconjuntos. Com 70 combinações no meio da faixa isso infla por
     seleção. Aqui a progressão é **aninhada em ordem de Shapley**, que é
     pré-especificada: nenhuma escolha depende do resultado.

Painel esquerdo: cada modalidade sozinha, mesmo classificador e mesmos folds.
Painel direito: a curva de saturação, que é a tese do paper -- ganho marginal
decrescente mas ainda positivo na oitava modalidade, e o cruzamento das
referências de CNN.

Multiscale não aparece sozinho de propósito: `canonical_mask()` colapsa o bit
dele quando nenhuma família visual está ativa, então "Multiscale sozinho" não é
uma condição válida do desenho fatorial.

Saída: results/ablation_overview.png e _paper.png (sem título, para a legenda
do artigo carregar o texto).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTORS = {"SLIC": 1, "Descriptors": 2, "Hilbert": 4, "Line graph": 8,
           "Felzenszwalb": 16, "Fourier": 32, "Text": 64, "Multiscale": 128}

# Paleta categórica fixa do projeto (slots 1,3,4,7,8), para as figuras do paper
# ficarem consistentes entre si.
BLUE, GREEN, AMBER, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
FAMILY = {
    "Descriptors":  (GREEN,  "Image descriptors"),
    "SLIC":         (BLUE,   "Segmentation graphs"),
    "Felzenszwalb": (BLUE,   "Segmentation graphs"),
    "Line graph":   (BLUE,   "Segmentation graphs"),
    "Hilbert":      (AMBER,  "Space-filling curve"),
    "Fourier":      (VIOLET, "Spectral"),
    "Text":         (RED,    "Text (ArtEmis)"),
}

# Referências externas, medidas na mesma coorte e nos mesmos folds.
CNN_SCRATCH = 42.34      # ResNet-50 treinada do zero, 63.949 imagens de treino
FROZEN_RESNET = 53.27    # ResNet-50 ImageNet congelada, sonda MLP, 2.048d
PRODUCTION = (54.42, 1.03, 1154)   # config atual: 12 branches


def load(classifier: str, directory: Path):
    summary = pd.read_csv(directory / "configuration_summary.csv")
    summary = summary[(summary["classifier"] == classifier) & (~summary["is_alias"])]
    shapley = pd.read_csv(directory / "shapley_summary.csv")
    order = (shapley[shapley["classifier"] == classifier]
             .sort_values("shapley_pp_mean", ascending=False)["factor"].tolist())

    singles = summary[summary["n_active_factors"] == 1].copy()
    singles = singles.sort_values("bacc_mean", ascending=False)

    rows, mask = [], 0
    for step, factor in enumerate(order, start=1):
        mask |= FACTORS[factor]
        row = summary[summary["mask"] == mask].iloc[0]
        rows.append({"m": step, "added": factor, "dims": int(row["dimensions"]),
                     "bacc": 100 * row["bacc_mean"], "std": 100 * row["bacc_std"]})
    return singles, pd.DataFrame(rows)


def draw(singles, nested, nested_lda, with_title: bool, output: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.05, 1]})
    for axis in (left, right):
        axis.set_facecolor("#fbfbfa")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    # --- painel esquerdo: cada modalidade sozinha -------------------------
    names = singles["condition"].tolist()
    values = (100 * singles["bacc_mean"]).tolist()
    errors = (100 * singles["bacc_std"]).tolist()
    colours = [FAMILY[n][0] for n in names]
    positions = range(len(names))
    left.bar(positions, values, yerr=errors, color=colours, width=0.68,
             capsize=4, zorder=3, error_kw={"elinewidth": 1.3, "ecolor": "#444"})
    for x, (value, error) in enumerate(zip(values, errors)):
        left.text(x, value + error + 0.9, f"{value:.1f}", ha="center",
                  fontsize=10, fontweight="bold", color="#333")
    left.axhline(100 / 27, color="#888", linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    left.text(len(names) - 0.4, 100 / 27 + 0.7, "chance (1/27)", ha="right",
              fontsize=9, color="#777", style="italic")
    left.set_xticks(list(positions))
    left.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9.5,
                         rotation=18, ha="right", rotation_mode="anchor")
    left.set_ylabel("Balanced accuracy (%)", fontsize=11)
    left.set_ylim(0, max(values) + 6)
    left.set_title("Each view alone", fontsize=12, fontweight="bold", pad=10)

    seen, handles = set(), []
    for name in names:
        colour, label = FAMILY[name]
        if label not in seen:
            seen.add(label)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=colour, label=label))
    left.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")

    # --- painel direito: progressão aninhada ------------------------------
    # Os rótulos das referências ficam em cantos onde nenhuma das duas curvas
    # passa: a de cima à esquerda (ali o MLP ainda está em 37) e a de baixo à
    # direita e ABAIXO da linha (ali a sonda LDA já está em 43+).
    right.axhline(FROZEN_RESNET, color="#9aa0a6", linestyle=(0, (5, 3)), linewidth=1.4)
    right.text(0.75, FROZEN_RESNET + 0.6, "frozen ImageNet ResNet-50",
               fontsize=9, color="#6b7075")
    right.axhline(CNN_SCRATCH, color="#9aa0a6", linestyle=(0, (2, 3)), linewidth=1.4)
    right.text(9.55, CNN_SCRATCH - 1.6, "ResNet-50 trained from scratch",
               fontsize=9, color="#6b7075", ha="right")

    right.errorbar(nested_lda["m"], nested_lda["bacc"], yerr=nested_lda["std"],
                   color="#9aa0a6", marker="o", markersize=5, linewidth=1.8,
                   capsize=3, linestyle=(0, (3, 2)), label="LDA probe", zorder=4)
    right.errorbar(nested["m"], nested["bacc"], yerr=nested["std"], color=BLUE,
                   marker="o", markersize=7, linewidth=2.4, capsize=3,
                   label="Branch MLP", zorder=5)

    for _, row in nested.iterrows():
        # O primeiro ponto encosta na borda esquerda, então o rótulo dele sai
        # para o lado em vez de para baixo.
        offset, align = ((9, -21), "left") if row["m"] == 1 else ((0, -16), "center")
        right.annotate(f"+{row['added']}", (row["m"], row["bacc"]),
                       textcoords="offset points", xytext=offset,
                       ha=align, fontsize=8.5, color="#555")

    bacc, std, dims = PRODUCTION
    right.errorbar([9], [bacc], yerr=[std], color=RED, marker="*", markersize=17,
                   capsize=3, linewidth=0, zorder=6,
                   label=f"production config ({dims}d, 12 branches)")
    right.annotate(f"{bacc:.2f}", (9, bacc), textcoords="offset points",
                   xytext=(0, 14), ha="center", fontsize=10,
                   fontweight="bold", color=RED)

    right.set_xlabel("Number of views, added in Shapley order", fontsize=11)
    right.set_ylabel("Balanced accuracy (%)", fontsize=11)
    right.set_xticks(range(1, 10))
    right.set_xlim(0.5, 9.7)
    right.set_ylim(26, 58)
    right.legend(fontsize=9, frameon=False, loc="lower right")
    right.set_title("Nested progression (no subset selection)",
                    fontsize=12, fontweight="bold", pad=10)

    if with_title:
        figure.suptitle(
            "Modality ablation — style classification (27 classes, WikiArt, 79,207 images)",
            fontsize=13.5, fontweight="bold", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.95 if with_title else 1))
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"salvo: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "outputs/full_factorial_ablation_text")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ablation_overview.png")
    args = parser.parse_args()

    singles, nested = load("mlp", args.input)
    _, nested_lda = load("lda", args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw(singles, nested, nested_lda, True, args.output)
    draw(singles, nested, nested_lda, False,
         args.output.with_name(args.output.stem + "_paper.png"))


if __name__ == "__main__":
    main()
