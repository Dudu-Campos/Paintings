"""Quão reproduzível é cada representação a partir do resto do pipeline?

Uma curva por bloco: para cada feature dele, o maior r² contra **todas** as
features das outras modalidades canônicas. A ECDF dessa quantidade diz onde a
representação inteira mora entre "cópia" e "ortogonal".

A pergunta que motivou é de sanidade, não de tese. A taxonomia por família
devolveu HOG com r² máximo de 0,024-0,035 contra as quatro famílias, o que o põe
como mais ortogonal que qualquer coisa já medida no projeto -- e o marginal dele
é **−1,21pp**, o pior de todos os candidatos. Antes de escrever que "novidade de
canal não implica utilidade", é preciso descartar que o 0,035 seja artefato:
1.764 colunas de histograma de gradiente têm escala e esparsidade muito
diferentes do resto, e um r² baixo por patologia numérica se parece com um r²
baixo por ortogonalidade genuína.

O desenho responde isso sem depender de confiança: os nossos sete blocos entram
como **banda cinza de referência**. Se HOG cair dentro da banda, o 0,035 era
comparável ao normal e a leitura anterior estava errada. Se cair muito à
esquerda de todos, a ortogonalidade é real e a conclusão se sustenta.

Assimetria registrada: um bloco nosso é comparado contra os outros sete; um
candidato, contra os oito. A diferença de uma modalidade no denominador é
pequena perto das faixas envolvidas, mas está aqui escrita porque muda o número
na terceira casa.

Saída: results/orthogonality_by_block.png, _paper.png e o CSV.
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

PIPELINE = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
            "Hilbert", "Fourier", "Multiscale")
# Destacados: os que a taxonomia colocou fora de toda família, mais um controle
# de cada lado -- Haralick, que ela colocou dentro de `image` com margem larga,
# e Triangles Felz, a duplicata mais explícita que temos.
HIGHLIGHT = {"HOG": "#e34948", "Tourist walk": "#eda100",
             "ResNet50": "#4a3aa7", "Haralick": "#1baf7a",
             "Triangles Felz": "#2a78d6"}
TEXT, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#d9d8d4", "#9a9a94"


def standardise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    deviation = values.std(axis=0)
    deviation[deviation < 1e-8] = 1.0
    return (values - values.mean(axis=0)) / deviation


def max_r2(candidate: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    best = np.empty(candidate.shape[1], dtype=np.float64)
    n = len(candidate)
    for start in range(0, candidate.shape[1], chunk):
        block = candidate[:, start:start + chunk]
        best[start:start + block.shape[1]] = (((block.T @ reference) / n) ** 2).max(axis=1)
    return best


def ecdf(axis, values: np.ndarray, colour: str, width: float, alpha: float,
         label: str | None, zorder: int) -> None:
    ordered = np.sort(np.clip(values, 1e-6, 1.0))
    axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post",
              color=colour, linewidth=width, alpha=alpha, label=label, zorder=zorder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/orthogonality_by_block.png")
    args = parser.parse_args()

    wanted = list(PIPELINE) + list(HIGHLIGHT)
    cohort = load_blocks(args.dataset, wanted, allow_external=True)
    standard = {name: standardise(block) for name, block in cohort.blocks.items()}
    pipeline = [name for name in PIPELINE if name in standard]
    log(f"coorte {len(cohort.stems):,} · pipeline {len(pipeline)} blocos")

    curves, rows = {}, []
    for name in list(standard):
        others = [standard[other] for other in pipeline if other != name]
        values = max_r2(standard[name], np.hstack(others))
        curves[name] = values
        rows.append({"name": name, "dims": int(standard[name].shape[1]),
                     "in_pipeline": name in pipeline,
                     "median": float(np.median(values)),
                     "p10": float(np.percentile(values, 10)),
                     "p90": float(np.percentile(values, 90)),
                     "frac_above_0.5": float((values > 0.5).mean())})
        log(f"  {name:<18} mediana {rows[-1]['median']:.3f}  "
            f"p90 {rows[-1]['p90']:.3f}  >0,5 {100 * rows[-1]['frac_above_0.5']:5.1f}%")

    table = pd.DataFrame(rows).sort_values("median")
    table.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.6f")

    for with_title in (True, False):
        figure, axis = plt.subplots(figsize=(9.0, 6.4))
        for name in pipeline:
            ecdf(axis, curves[name], BAND, 1.4, 0.75, None, 2)
        # Rótulo único para a banda: sete curvas cinzas são um objeto só na
        # leitura, e sete entradas de legenda idênticas seriam ruído.
        axis.plot([], [], color=BAND, linewidth=1.4,
                  label=f"our {len(pipeline)} pipeline blocks")
        for name, colour in HIGHLIGHT.items():
            if name in curves:
                ecdf(axis, curves[name], colour, 2.4, 0.95,
                     f"{name} ({curves[name].shape[0]}d)", 4)
        axis.set_xscale("log")
        axis.set_xlim(1e-4, 1.05)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("largest r² against any feature of the other views\n"
                        "left = nothing in the pipeline reproduces it")
        axis.set_ylabel("share of the block's features (ECDF)")
        axis.legend(frameon=False, fontsize=9, loc="upper left")
        axis.grid(alpha=0.28, linewidth=0.6, color=GRID)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if with_title:
            figure.suptitle(
                "How much of each representation the rest of the pipeline "
                "already reproduces\nWikiArt, 79,207 images",
                fontsize=12.5, color=TEXT)
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            path = args.output
        else:
            figure.tight_layout()
            path = args.output.with_name(args.output.stem + "_paper.png")
        figure.savefig(path, dpi=200, facecolor="white")
        plt.close(figure)
        log(f"salvo: {path}")


if __name__ == "__main__":
    main()
