"""Redundância entre pares de features medida em ganho, não em variância.

Companheira da curva de r², e a pergunta é a mesma: a redundância se concentra
dentro das modalidades? Só que agora na moeda que importa.

    R(i,j) = ganho(i) + ganho(j) − ganho(i,j)          (ganho de interação)

com `ganho(S) = BAcc(S) − acaso`. Alto = as duas features se duplicam; zero =
perfeitamente complementares; negativo = sinergia (juntas valem mais que a soma).
A versão plotada é a **fração** `R / (ganho_i + ganho_j)`, porque a versão bruta
é confundida por escala -- foi assim que o contraste LDA×MLP no nível de
modalidade evaporou (18,3 contra 14,6pp virou 0,325 contra 0,323, p=0,74).

Por que refazer em ganho o que já foi feito em r²: a versão em r² **falhou** como
evidência de mecanismo. As quase-duplicatas intra são 10,1× as inter, mas o
agrupamento hierárquico sobre a matriz de r² recupera as modalidades com Rand
ajustado de só 0,11 -- os clusters reais atravessam as modalidades. Isso é
consistente com o achado mais repetido do projeto (r² par-a-par, R² múltiplo,
co-ocorrência de erro, CKA e R² de ridge, todos sem relação com contribuição):
redundância mede variância, contribuição mora no discriminante. Se a estrutura
de modalidade aparecer aqui e não lá, isso deixa de ser desculpa e vira medida.

**Implementação.** LDA de duas features tem forma fechada a partir de estatísticas
suficientes -- médias por classe e covariância intra-classe agrupada -- então os
170.820 pares saem de uma covariância calculada uma vez, sem 170 mil ajustes.
O MLP não tem esse atalho e ficaria em centenas de horas; por isso a figura é da
sonda LDA, e a legenda tem que dizer isso.

O teste é subamostrado com **teto por classe**, não uniformemente: acurácia
balanceada pesa as 27 classes igualmente, então gastar orçamento nas classes
frequentes só adiciona precisão onde ela não conta.

Saída: results/pairwise_gain_redundancy.png, _paper.png e o CSV das estatísticas.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
BLOCKS = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier")
COLOURS = dict(zip(BLOCKS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                            "#e87ba4", "#008300", "#4a3aa7")))
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
INTRA, INTER = "#1f1f1f", "#a9a8a2"


def balanced(prediction: np.ndarray, labels: np.ndarray,
             classes: np.ndarray) -> np.ndarray:
    """BAcc por coluna de `prediction` (n_test × n_pares), vetorizado."""
    total = np.zeros(prediction.shape[1], dtype=np.float64)
    for index, label in enumerate(classes):
        rows = labels == label
        total += (prediction[rows] == index).mean(axis=0)
    return total / len(classes)


def single_bacc(train, test, means, precision_diagonal, labels_test, classes):
    """LDA de uma feature para todas as features de uma vez."""
    # Com d=1 a projeção é monotônica; a regra vira comparar (x−μ_c)² ponderado,
    # que é o mesmo que maximizar μ_c·x/σ² − μ_c²/2σ². Priors uniformes.
    weight = means * precision_diagonal                     # (C, d)
    bias = -0.5 * means * weight
    scores = test[:, None, :] * weight[None] + bias[None]   # (n, C, d)
    return balanced(np.argmax(scores, axis=1), labels_test, classes)


def pair_bacc(test, means, covariance, labels_test, classes, pairs, chunk):
    """LDA de duas features, em blocos de pares, por forma fechada.

    Para o par (i,j) a inversa de uma 2×2 é analítica, então tudo -- pesos,
    viés e escores -- se resolve com aritmética elemento a elemento sobre o
    bloco inteiro de pares.
    """
    result = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), chunk):
        block = pairs[start:start + chunk]
        i, j = block[:, 0], block[:, 1]
        s11, s22, s12 = covariance[i, i], covariance[j, j], covariance[i, j]
        determinant = s11 * s22 - s12 ** 2
        # Par degenerado (features colineares) -> pseudo-regularização mínima,
        # que é o que uma implementação de LDA faria com `shrinkage`.
        determinant = np.where(np.abs(determinant) < 1e-10, 1e-10, determinant)
        a, b, d = s22 / determinant, -s12 / determinant, s11 / determinant
        m1, m2 = means[:, i].T, means[:, j].T                 # (P, C)
        w1 = a[:, None] * m1 + b[:, None] * m2
        w2 = b[:, None] * m1 + d[:, None] * m2
        bias = -0.5 * (w1 * m1 + w2 * m2)
        scores = (test[:, i][:, :, None] * w1[None]
                  + test[:, j][:, :, None] * w2[None] + bias[None])
        result[start:start + len(block)] = balanced(
            np.argmax(scores, axis=2), labels_test, classes)
    return result


def run_fold(values, labels, train, test, classes, chunk, cap, rng):
    scaler = StandardScaler().fit(values[train])
    train_x = scaler.transform(values[train]).astype(np.float32)
    # Teto por classe no teste: BAcc pesa as classes igualmente, então precisão
    # extra nas frequentes não compra nada. Isso corta o custo ~2× sem viés.
    chosen = []
    for label in classes:
        rows = test[labels[test] == label]
        chosen.append(rows if len(rows) <= cap
                      else rng.choice(rows, cap, replace=False))
    test = np.concatenate(chosen)
    test_x = scaler.transform(values[test]).astype(np.float32)
    labels_test = labels[test]

    means = np.stack([train_x[labels[train] == label].mean(axis=0)
                      for label in classes]).astype(np.float32)
    centred = train_x - means[np.searchsorted(classes, labels[train])]
    covariance = (centred.T @ centred / (len(train_x) - len(classes))).astype(np.float32)
    chance = 1.0 / len(classes)

    singles = single_bacc(train_x, test_x, means,
                          1.0 / np.maximum(np.diag(covariance), 1e-8),
                          labels_test, classes) - chance
    pairs = np.array(np.triu_indices(values.shape[1], 1)).T
    joint = pair_bacc(test_x, means, covariance, labels_test, classes,
                      pairs, chunk) - chance
    return singles, joint, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=192)
    parser.add_argument("--test-cap", type=int, default=400)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/pairwise_gain_redundancy.png")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, BLOCKS, allow_external=False)
    values = np.hstack([cohort.blocks[name] for name in BLOCKS]).astype(np.float64)
    owner = np.concatenate([[name] * cohort.blocks[name].shape[1] for name in BLOCKS])
    keep = values.std(0) > 1e-9
    values, owner = values[:, keep], owner[keep]
    labels = LabelEncoder().fit_transform(cohort.labels)
    classes = np.unique(labels)
    log(f"{values.shape[1]} features · {len(classes)} classes · "
        f"{values.shape[1] * (values.shape[1] - 1) // 2:,} pares")

    rng = np.random.default_rng(SEED)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    fractions = []
    for fold, (train, test) in enumerate(splits[: args.folds], start=1):
        started = time.perf_counter()
        singles, joint, pairs = run_fold(values, labels, train, test, classes,
                                         args.chunk, args.test_cap, rng)
        summed = singles[pairs[:, 0]] + singles[pairs[:, 1]]
        # Pares em que nem a soma dos ganhos é positiva não têm fração definida
        # (dividir por ~zero explode); são features sem sinal, e ficam de fora.
        usable = summed > 0.01
        fractions.append(np.where(usable, (summed - joint) / np.maximum(summed, 1e-9),
                                  np.nan))
        log(f"  fold {fold}: {usable.sum():,} pares utilizáveis · "
            f"{time.perf_counter() - started:.0f}s")
    fraction = np.nanmean(np.vstack(fractions), axis=0)

    same = owner[pairs[:, 0]] == owner[pairs[:, 1]]
    valid = np.isfinite(fraction)
    figure, axis = plt.subplots(figsize=(8.2, 6.4))
    edges = np.linspace(-0.2, 1.0, 61)
    rows = []
    for mask, colour, label in ((same & valid, INTRA, "same modality"),
                                (~same & valid, INTER, "different modalities")):
        counts, _ = np.histogram(np.clip(fraction[mask], edges[0], edges[-1]), bins=edges)
        axis.step(edges[:-1], counts / counts.sum(), where="post", color=colour,
                  linewidth=2.0, label=label, zorder=3)
        subset = fraction[mask]
        rows.append({"scope": label, "n_pairs": int(mask.sum()),
                     "mean": float(subset.mean()),
                     "median": float(np.median(subset)),
                     "frac_above_0.5": float((subset > 0.5).mean())})
    table = pd.DataFrame(rows)
    axis.axvline(0, color=GRID, linewidth=1.2, zorder=1)
    axis.set_xlabel("share of the summed gain that the pair duplicates\n"
                    "0 = fully complementary · 1 = one is a copy of the other · "
                    "< 0 = synergy")
    axis.set_ylabel("share of pairs")
    axis.set_title("Redundancy measured in accuracy, not in variance\n"
                   f"{int(valid.sum()):,} feature pairs, LDA probe, "
                   f"{args.folds} fold(s)", fontsize=12, color=TEXT, pad=12)
    axis.legend(frameon=False, fontsize=9.5, loc="upper left")
    axis.grid(alpha=0.28, linewidth=0.6, color=GRID)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    intra, inter = table.iloc[0], table.iloc[1]
    axis.text(0.03, 0.72,
              f"median duplicated share\n"
              f"  same modality   {intra['median']:.3f}\n"
              f"  different       {inter['median']:.3f}\n"
              f"  ratio           {intra['median'] / max(inter['median'], 1e-9):.2f}×",
              transform=axis.transAxes, fontsize=9, color=TEXT,
              family="monospace", va="top")
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, facecolor="white")
    plt.close(figure)
    table.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.6f")
    log(f"salvo: {args.output}\n" + table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
