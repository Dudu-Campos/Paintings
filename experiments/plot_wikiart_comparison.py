"""Panorama do WikiArt: descritores clássicos, CNN, pipeline e as combinações.

Uma barra por sistema, todas na mesma coorte (79.207 imagens, 27 estilos) e nos
mesmos cinco folds. Duas decisões de honestidade estão no desenho:

**O protocolo de cada barra é impresso na própria barra.** Os sistemas não são
comparáveis por um classificador único -- não existe "rodar uma CNN treinada de
ponta a ponta sob sonda LDA" -- então o rótulo diz se aquele número veio de
sonda LDA, do MLP ramificado ou de treino fim-a-fim. Esconder isso produziria
uma tabela que parece uniforme e não é.

**O pipeline aparece sem o bloco Text.** O número de 54,42% que circula no
projeto inclui os 27 escores de TF-IDF sobre legendas, que dependem de
informação externa à imagem; usá-lo aqui contradiria a própria alegação de
competir sem supervisão externa. O número sem Text é 52,10, e é esse.

ResNet50 e ResNet101 saem ambas com 2.048 dimensões pós-GAP, de propósito: a
diferença entre as duas barras é profundidade de arquitetura, não largura de
representação -- o confundidor de dimensionalidade fica controlado por
construção.

Saída: results/wikiart_comparison.png e _paper.png
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
sys.path.insert(0, str(ROOT / "experiments"))
from registry import load_blocks, log  # noqa: E402

KIND_COLOUR = {"classical": "#eda100", "hand-designed": "#2a78d6",
               "transfer": "#1baf7a", "trained": "#eb6834",
               "combination": "#4a3aa7"}
KIND_LABEL = {"classical": "classical descriptors", "hand-designed": "ours",
              "transfer": "frozen ImageNet", "trained": "CNN trained on WikiArt",
              "combination": "ours + frozen ImageNet"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CHANCE = 100 / 27


def on_fill(fill: str) -> str:
    """Branco ou quase-preto, o que tiver melhor contraste sobre `fill` (WCAG)."""
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in matplotlib.colors.to_rgb(fill)]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "#ffffff" if (1.05 / (luminance + 0.05)) > ((luminance + 0.05) / 0.05) \
        else "#111111"


def measure(name: str, epochs: int, patience: int) -> tuple[float, float]:
    """Sonda o bloco pedido com a MLP ramificada, mesmos folds de todo o resto."""
    import torch
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from balanced_branch_mlp import train_balanced

    torch.set_num_threads(16)
    cohort = load_blocks("wikiart", [name], allow_external=True)
    values = cohort.blocks[name]
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    scores = []
    splits = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (outer, test) in enumerate(
            splits.split(np.zeros(len(labels), dtype=np.uint8), labels), start=1):
        fit, validation = train_test_split(
            outer, test_size=0.1, stratify=labels[outer], random_state=42 + fold)
        scaler = StandardScaler().fit(values[fit])
        parts = [scaler.transform(values[index]).astype(np.float32)
                 for index in (fit, validation, test)]
        metrics, _, _ = train_balanced(
            parts[0], labels[fit], parts[1], labels[validation], parts[2],
            labels[test], [values.shape[1]], n_classes, 42 + fold,
            epochs, patience)
        scores.append(100 * metrics["bacc"])
        log(f"  {name} fold {fold}: {scores[-1]:.2f}")
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-finetuned", action="store_true",
                        help="inclui a ResNet-50 finetunada (64,71). Fora por "
                             "padrão: a barra domina o eixo e faz a figura ser "
                             "lida como placar, que não é o argumento. É "
                             "decisão de composição, não de reporte -- o número "
                             "tem de constar do texto, senão vira seleção de "
                             "resultado.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/wikiart_comparison.png")
    args = parser.parse_args()

    # Números já medidos, cada um com a sua procedência. Ficam explícitos em vez
    # de recalculados porque três deles custam horas de GPU.
    rows = [
        dict(label="Haralick + HOG", value=21.65, error=0.88, dims=1777,
             probe="LDA probe", kind="classical",
             source="medido aqui, 5 folds"),
        dict(label="ResNet-50 trained from scratch", value=42.34, error=np.nan,
             dims=2048, probe="end-to-end", kind="trained",
             source="escada de escala, n=63.949"),
        dict(label="Our pipeline (no Text)", value=52.10, error=1.33, dims=1127,
             probe="branch MLP", kind="hand-designed",
             source="outputs/percolation_hierarchy"),
        dict(label="ResNet-50 frozen (ImageNet)", value=52.27, error=0.89,
             dims=2048, probe="branch MLP", kind="transfer",
             source="outputs/blocks_over_backbone"),
        dict(label="Ours + ResNet-50 frozen", value=56.31, error=0.61, dims=3175,
             probe="branch MLP", kind="combination",
             source="outputs/blocks_over_backbone"),
    ]
    if args.with_finetuned:
        rows.append(dict(label="ResNet-50 fine-tuned", value=64.71, error=0.83,
                         dims=2048, probe="end-to-end", kind="trained",
                         source="outputs/cnn_imagenet_finetuned"))

    features = ROOT / "outputs/wikiart_cnn101_frozen/cnn_full_features.npy"
    if features.exists():
        log("sondando ResNet-101 congelada...")
        mean, deviation = measure("ResNet101", args.epochs, args.patience)
        rows.append(dict(label="ResNet-101 frozen (ImageNet)", value=mean,
                         error=deviation, dims=2048, probe="branch MLP",
                         kind="transfer", source="medido aqui, 5 folds"))
    else:
        log("! ResNet-101 ainda não extraída — barra omitida")

    table = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)
    table.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.4f")
    log("\n" + table[["label", "value", "error", "dims", "probe"]]
        .to_string(index=False, float_format="%.2f"))

    for with_title in (True, False):
        figure, axis = plt.subplots(figsize=(10.6, 6.4))
        positions = np.arange(len(table))
        axis.barh(positions, table.value,
                  color=[KIND_COLOUR[k] for k in table.kind],
                  height=0.62, zorder=3)
        errors = table.error.to_numpy(dtype=float)
        visible = np.isfinite(errors)
        axis.errorbar(table.value[visible], positions[visible],
                      xerr=errors[visible], fmt="none", ecolor="#3d3d3d",
                      elinewidth=1.3, capsize=4, zorder=4)
        for position, row in zip(positions, table.itertuples(index=False)):
            # Rótulo dentro da barra: fora, ele empurrava o eixo ~20% para a
            # direita só para acomodar seis números. A cor do texto é escolhida
            # por contraste medido contra o preenchimento, não fixada em branco
            # -- branco sobre o amarelo #eda100 falharia a razão de contraste.
            axis.annotate(f"{row.value:.1f}", (row.value, position),
                          xytext=(-9, 0), textcoords="offset points",
                          ha="right", va="center", fontsize=11,
                          fontweight="bold", color=on_fill(KIND_COLOUR[row.kind]),
                          zorder=5)
        axis.axvline(CHANCE, color="#9a9a94", linestyle="--", linewidth=1.2,
                     zorder=2)
        axis.annotate(f"chance ({CHANCE:.1f})", (CHANCE, len(table) - 0.35),
                      xytext=(5, 0), textcoords="offset points", fontsize=8.6,
                      color=MUTED)
        axis.set_yticks(positions)
        axis.set_yticklabels([f"{row.label}\n{row.dims:,}d · {row.probe}"
                              for row in table.itertuples(index=False)],
                             fontsize=9.5)
        axis.set_xlabel("Balanced accuracy (%)")
        axis.set_xlim(0, max(table.value) * 1.06)
        axis.grid(axis="x", alpha=0.28, linewidth=0.6, color=GRID)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOUR[k])
                   for k in KIND_LABEL if (table.kind == k).any()]
        axis.legend(handles, [KIND_LABEL[k] for k in KIND_LABEL
                              if (table.kind == k).any()],
                    frameon=False, fontsize=9, loc="lower right", ncol=2)
        figure.text(0.5, 0.012,
                    "WikiArt · 79,207 images · 27 styles · 5 shared folds · "
                    "error bars are the fold standard deviation\n"
                    "the probe differs by row and is printed with each label: "
                    "an end-to-end CNN cannot be run under a linear probe",
                    ha="center", fontsize=8.4, color=MUTED)
        if with_title:
            figure.suptitle("Painting-style classification on WikiArt",
                            fontsize=13, color=TEXT)
            figure.tight_layout(rect=(0, 0.06, 1, 0.94))
            path = args.output
        else:
            figure.tight_layout(rect=(0, 0.06, 1, 1))
            path = args.output.with_name(args.output.stem + "_paper.png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        log(f"salvo: {path}")


if __name__ == "__main__":
    main()
