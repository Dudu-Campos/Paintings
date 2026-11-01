"""Identidade representacional a partir da redundância, no banco de produção.

Reescreve `label_propagation_rich_r2_graph.py` para o Multitask com as **oito**
modalidades. A versão anterior roda sobre as features `rich` do WikiArt e tem
**quatro** famílias fixas no código -- Gabor, os dois grafos de linha e a
separação SLIC/Felzenszwalb ficavam de fora.

A pergunta é: olhando só para com quem cada feature é redundante, dá para
adivinhar de que modalidade ela veio? Os nós do grafo são features individuais,
as arestas são r² par-a-par acima de um limiar, e a cada repetição esconde-se
uma fração estratificada dos rótulos de família e propaga-se o resto. A
diagonal da matriz resultante é a taxa de acerto **fora da amostra**.

Nenhum rótulo do corpus entra em lugar nenhum: é auditoria da geometria do
banco, não de desempenho.

Leitura: diagonal alta = a modalidade tem assinatura de redundância própria,
identificável sem se olhar. Confusão entre duas = elas medem a mesma coisa por
caminhos diferentes, e é onde a poda do banco deveria olhar primeiro.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean  # noqa: E402
from label_propagation_rich_r2_graph import propagate, stratified_holdout  # noqa: E402

HAND = VISUAL + ["Gabor"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-blocks", nargs="*", default=[],
                        help="blocos extras por NOME (alinhados por stem via "
                             "load_block), p.ex. 'Caption embed'. São modalidades "
                             "de supervisão EXTERNA: o captioner CATR viu ~600k "
                             "legendas humanas do COCO. Não entram no número "
                             "visual-only, mas dizem se a assinatura de redundância "
                             "de uma modalidade externa se distingue das nossas.")
    parser.add_argument("--similarity-graph", type=Path, default=None,
                        help="bloco do grafo de similaridade (fold 1) a incluir como "
                             "família extra. ATENÇÃO: ele é derivado de RÓTULO -- "
                             "propaga os rótulos de treino dos vizinhos -- então "
                             "incluí-lo quebra a propriedade de a auditoria ser "
                             "livre de supervisão. A figura antiga do WikiArt "
                             "misturava `Global similarity graph` e `Text` com as "
                             "visuais sem declarar isso.")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="r² mínimo para existir aresta. 0,30 mantém o grafo "
                             "conectado sem virar clique: com 0,80, as famílias de "
                             "segmentação formam componentes isoladas e a propagação "
                             "vira trivial.")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/multitask_label_propagation_r2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nomes_blocos = list(HAND) + list(args.extra_blocks)
    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in nomes_blocos:
        bloco = load_block("multitask", name)
        if bloco is None:
            raise SystemExit(f"bloco ausente: {name}")
        frame = frame.merge(bloco, on="stem", how="inner")
    wanted = {key(l) for l in (ROOT / "data/multitask_25class_benchmark.txt")
              .read_text().splitlines() if l.strip()}
    frame = frame[[key(v) in wanted for v in frame["style"].to_numpy(str)]].reset_index(drop=True)

    colunas, familias = [], []
    partes = []
    for name in nomes_blocos:
        cols = [c for c in frame.columns if c.startswith(f"{name}_")]
        partes.append(clean(frame[cols].to_numpy(np.float32)))
        colunas += cols
        familias += [name] * len(cols)
    ordem_familias = list(nomes_blocos)
    if args.similarity_graph:
        bloco = np.load(args.similarity_graph)
        if len(bloco) != len(frame):
            raise ValueError(f"grafo tem {len(bloco)} linhas, coorte tem {len(frame)}")
        partes.append(bloco.astype(np.float32))
        colunas += [f"graph_{i}" for i in range(bloco.shape[1])]
        familias += ["Similarity graph"] * bloco.shape[1]
        ordem_familias.append("Similarity graph")
        log("Similarity graph incluído -- a auditoria deixa de ser livre de rótulo")
    values = np.hstack(partes)
    log(f"{len(frame):,} imagens, {values.shape[1]} features em {len(ordem_familias)} famílias")

    # r² par-a-par = quadrado da correlação de Pearson. Colunas constantes viram
    # correlação indefinida; zerar evita NaN propagando pelo grafo inteiro.
    standard = (values - values.mean(0)) / np.where(values.std(0) > 0, values.std(0), 1.0)
    correlation = (standard.T @ standard) / len(standard)
    r2 = np.square(np.nan_to_num(correlation, nan=0.0))
    np.fill_diagonal(r2, 0.0)

    index = pd.DataFrame({"feature": colunas, "family": familias,
                          "feature_index": np.arange(len(colunas))})
    rows, cols = np.where(np.triu(r2 >= args.threshold, k=1))
    pairs = pd.DataFrame({"feature_a": [colunas[i] for i in rows],
                          "feature_b": [colunas[j] for j in cols],
                          "r2": r2[rows, cols]})
    log(f"grafo: {len(pairs):,} arestas com r² ≥ {args.threshold} "
        f"(densidade {200 * len(pairs) / (len(colunas) * (len(colunas) - 1)):.1f}%)")

    ordem = ordem_familias
    labels = index["family"].map({f: i for i, f in enumerate(ordem)}).to_numpy(np.int32)

    # `propagate` do módulo original lê o número de classes de FAMILY_ORDER;
    # aqui são oito, então a matriz é montada localmente e a função reusada
    # apenas onde o número de classes não é assumido.
    import label_propagation_rich_r2_graph as base
    base.FAMILY_ORDER = list(ordem)
    transition = base.build_transition(index, pairs, 1.0)

    rng = np.random.default_rng(SEED)
    sums = np.zeros((len(index), len(ordem)), dtype=np.float64)
    counts = np.zeros(len(index), dtype=np.int64)
    for repeat in range(args.repeats):
        heldout = stratified_holdout(labels, args.holdout_fraction, rng)
        distributions = propagate(transition, labels, ~heldout,
                                  args.alpha, args.max_iter, args.tol)
        sums[heldout] += distributions[heldout]
        counts[heldout] += 1
        if (repeat + 1) % 25 == 0:
            log(f"  {repeat + 1}/{args.repeats} repetições")

    predicted = np.full(len(index), -1, dtype=np.int32)
    visto = counts > 0
    predicted[visto] = sums[visto].argmax(1)
    index["predicted_family"] = [ordem[p] if p >= 0 else "" for p in predicted]
    index.to_csv(args.output_dir / "feature_predictions.csv", index=False)

    matrix = pd.crosstab(index.loc[visto, "family"], index.loc[visto, "predicted_family"],
                         dropna=False).reindex(index=ordem, columns=ordem, fill_value=0)
    matrix.to_csv(args.output_dir / "confusion.csv")
    diagonal = np.diag(matrix.to_numpy())
    totals = matrix.sum(1).to_numpy()
    print()
    print(f"{'família':<14} {'features':>9} {'acerto':>8}   confusão principal")
    for i, name in enumerate(ordem):
        if totals[i] == 0:
            continue
        linha = matrix.iloc[i].to_numpy().astype(float).copy()
        linha[i] = -1
        outro = ordem[int(linha.argmax())]
        print(f"{name:<14} {totals[i]:>9} {100 * diagonal[i] / totals[i]:>7.1f}%   "
              f"{outro} ({100 * matrix.iloc[i][outro] / totals[i]:.0f}%)")
    print()
    print(f"acerto global fora da amostra: {100 * diagonal.sum() / totals.sum():.1f}% "
          f"(acaso = {100 / len(ordem):.1f}%)")
    log(f"salvo em {args.output_dir}")


if __name__ == "__main__":
    main()
