"""Similaridade representacional prevê contribuição, depois de descontar qualidade?

Sucessor de `analyze_cka_vs_shapley.py`, com três mudanças que vieram de medir.

**1. Contra o marginal de deixar-um-de-fora, não contra Shapley.** CKA-com-o-resto
mede a similaridade do bloco `i` contra a **união dos outros**; a contribuição
que casa com isso é `BAcc(todos) − BAcc(todos \\ i)`, não o Shapley (que
promedia sobre todos os 2^n subconjuntos). Além de ser o par conceitualmente
certo, são 9 condições por dataset em vez de 256 máscaras.

**2. Controlando qualidade.** No WikiArt, CKA-com-o-resto correlaciona
**+0,786** (bruto) e **+0,893** (branqueado) com a BAcc **isolada** do bloco --
ou seja, ele mede quão bom o bloco é, não quão distinto. Descontando isso, a
correlação com o Shapley cai de +0,82 para **−0,43 (p=0,34)**: o efeito inteiro
era confundidor, e o que sobra é a direção que a intuição de redundância prevê,
mas invisível com n=7.

É a mesma patologia de todas as medidas de similaridade que o projeto tentou --
r² par-a-par, R² múltiplo, co-ocorrência de erro (Spearman +0,747 com a
acurácia) e agora CKA. Um bloco informativo pontua bem sozinho, compartilha
estrutura com outros blocos informativos **e** erra nas mesmas imagens. A
qualidade dirige as três.

**3. Vários datasets.** É a única saída para o problema de potência: quatro
datasets × ~8 blocos dá n≈30 em vez de 7, e aí a correlação parcial vira
testável. Roda sobre o registro, então o mesmo código serve wikiart, pandora,
rijksmuseum e rijksdraw sem carregador novo.

Saída: results/cka_vs_marginal/{dataset}.csv e pooled.csv
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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
# Os oito do pipeline canônico. Text fica fora (depende de informação externa
# e não entra na avaliação final); ResNet50 idem.
VISUAL = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier", "Multiscale")


def centre(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """CKA linear na forma de features — barata quando n >> d."""
    x, y = centre(x), centre(y)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    self_x = np.linalg.norm(x.T @ x, ord="fro")
    self_y = np.linalg.norm(y.T @ y, ord="fro")
    if self_x <= 0 or self_y <= 0:
        return float("nan")
    return float(cross / (self_x * self_y))


def standardise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    deviation = matrix.std(axis=0)
    deviation[deviation < 1e-12] = 1.0
    return (matrix - matrix.mean(axis=0)) / deviation


def whiten(matrix: np.ndarray) -> np.ndarray:
    """Componentes de PCA normalizados: peso igual em todas as direções.

    Existe porque o CKA bruto é dominado pelas direções de alta variância, que
    é a crítica conhecida da medida — e neste projeto o sinal de estilo mora na
    cauda (500 componentes são 97% da variância mas k* ≈ 1.330 de 1.576).
    """
    centred = centre(matrix)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    keep = singular > (singular.max() * 1e-8)
    projected = centred @ right[keep].T
    return projected / projected.std(axis=0)


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536)
            if k <= max_k]
    if not grid or grid[-1] != max_k:
        grid.append(max_k)
    return grid


def probe(values: np.ndarray, labels: np.ndarray, splits, n_classes: int) -> float:
    """BAcc média com sonda LDA sobre PCA, pico em k.

    O pico é obrigatório porque as condições têm dimensões muito diferentes
    (tirar Multiscale remove 1.032 de 1.639), e sem varrer k o marginal mediria
    dimensionalidade em vez de conteúdo.
    """
    scores = []
    for train, test in splits:
        scaler = StandardScaler().fit(values[train])
        train_x, test_x = scaler.transform(values[train]), scaler.transform(values[test])
        limit = min(train_x.shape[1], len(train) - n_classes)
        reducer = PCA(n_components=limit, svd_solver="randomized",
                      random_state=SEED).fit(train_x)
        train_p, test_p = reducer.transform(train_x), reducer.transform(test_x)
        best = 0.0
        for k in k_grid(limit):
            model = LinearDiscriminantAnalysis(
                n_components=min(n_classes - 1, k),
                priors=np.ones(n_classes) / n_classes).fit(train_p[:, :k], labels[train])
            best = max(best, balanced_accuracy_score(
                labels[test], model.predict(test_p[:, :k])))
        scores.append(best)
    return float(np.mean(scores))


def run(dataset: str, sample: int, output_dir: Path,
        min_per_class: int) -> pd.DataFrame:
    # Sem o piso por classe, uma classe com 3 exemplos some de algum fold e o
    # LDA recebe `n_components` maior que as classes presentes no treino. Além
    # do erro, classes minúsculas viram ruído em acurácia balanceada: cada uma
    # vale 1/K do score e é decidida por sorte no split.
    cohort = load_blocks(dataset, VISUAL, allow_external=False,
                         min_per_class=min_per_class)
    names = list(cohort.blocks)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    standard = {name: standardise(block) for name, block in cohort.blocks.items()}
    # Subamostra só para o branqueamento: a SVD é O(n d²) e não muda a medida.
    rng = np.random.default_rng(SEED)
    index = (rng.choice(len(labels), size=min(sample, len(labels)), replace=False)
             if sample else np.arange(len(labels)))
    whitened = {name: whiten(block[index]) for name, block in standard.items()}

    rows = []
    everything = probe(np.hstack([standard[n] for n in names]), labels, splits, n_classes)
    log(f"{dataset}: todos os {len(names)} blocos -> BAcc {100 * everything:.2f}%")

    for name in names:
        started = time.perf_counter()
        others = [n for n in names if n != name]
        rest_raw = np.hstack([standard[n] for n in others])
        rest_white = np.hstack([whitened[n] for n in others])

        solo = probe(standard[name], labels, splits, n_classes)
        without = probe(rest_raw, labels, splits, n_classes)
        rows.append({
            "dataset": dataset, "block": name,
            "dims": standard[name].shape[1], "n_images": len(labels),
            "n_classes": n_classes,
            "cka_raw": linear_cka(standard[name][index], rest_raw[index]),
            "cka_whitened": linear_cka(whitened[name], rest_white),
            "solo_bacc": 100 * solo,
            "all_bacc": 100 * everything,
            "without_bacc": 100 * without,
            "loo_marginal": 100 * (everything - without)})
        log(f"  {name:<14} CKA {rows[-1]['cka_raw']:.3f}/{rows[-1]['cka_whitened']:.3f}"
            f"  isolada {rows[-1]['solo_bacc']:5.2f}  LOO {rows[-1]['loo_marginal']:+5.2f}pp"
            f"  ({time.perf_counter() - started:.0f}s)")

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{dataset}.csv", index=False, float_format="%.6f")
    return table


def partial_spearman(x, y, control):
    """Spearman entre x e y depois de remover a parte linear de `control`."""
    from scipy.stats import spearmanr
    rx = x - np.polyval(np.polyfit(control, x, 1), control)
    ry = y - np.polyval(np.polyfit(control, y, 1), control)
    return spearmanr(rx, ry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["wikiart", "pandora", "rijksmuseum", "rijksdraw"])
    parser.add_argument("--sample", type=int, default=20000,
                        help="subamostra para o branqueamento (0 = tudo)")
    parser.add_argument("--min-per-class", type=int, default=50,
                        help="piso de exemplos por classe; mesmo valor do "
                             "avaliador externo, para as coortes casarem")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/cka_vs_marginal")
    args = parser.parse_args()

    tables = []
    for dataset in args.datasets:
        try:
            tables.append(run(dataset, args.sample, args.output_dir,
                              args.min_per_class))
        except Exception as error:                      # um dataset não paga os outros
            log(f"{dataset}: FALHOU ({type(error).__name__}: {error})")
    if not tables:
        return
    pooled = pd.concat(tables, ignore_index=True)
    pooled.to_csv(args.output_dir / "pooled.csv", index=False, float_format="%.6f")

    from scipy.stats import spearmanr
    log(f"\n=== agregado, n={len(pooled)} pares (bloco × dataset) ===")
    for column in ("cka_raw", "cka_whitened"):
        raw = spearmanr(pooled[column], pooled["loo_marginal"])
        confound = spearmanr(pooled[column], pooled["solo_bacc"])
        controlled = partial_spearman(pooled[column].to_numpy(),
                                      pooled["loo_marginal"].to_numpy(),
                                      pooled["solo_bacc"].to_numpy())
        log(f"{column:13} x LOO           Spearman {raw[0]:+.3f} (p={raw[1]:.4f})")
        log(f"{'':13} x BAcc isolada  Spearman {confound[0]:+.3f} (p={confound[1]:.4f})"
            "   <- o confundidor")
        log(f"{'':13} x LOO | isolada Spearman {controlled[0]:+.3f} (p={controlled[1]:.4f})"
            "   <- o que interessa")
    log(f"\nSalvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
