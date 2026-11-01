"""As 8 vizinhanças são diferentes? E o peso igual está atrapalhando?

O kNN do grafo roda sobre os blocos concatenados e padronizados, ou seja **cada
dimensão pesa igual na distância** -- as 130 de Descriptors valem 4x as 32 de
Gabor só por serem mais numerosas. Num projeto cuja tese é que os blocos não são
intercambiáveis, essa é a fusão mais crua possível.

Antes de montar um grafo multiaresta (uma distância por modalidade), medir se há
o que fundir. Três perguntas, todas baratas:

1. **Cada modalidade dá vizinhança pura?** `pureza@k` por modalidade contra a
   da concatenação. Se alguma sozinha superar a concatenação, o peso igual está
   custando -- evidência direta da hipótese.
2. **As vizinhanças são diferentes entre si?** Jaccard médio entre os conjuntos
   de vizinhos. Perto de 1 significa que todas recuperam o mesmo e multiaresta
   não compra nada; perto de 0 significa que há muito a fundir.
3. **Qual o teto da união?** Se a classe verdadeira estiver na união das 8
   vizinhanças mas não na da concatenação, esse é ganho recuperável por
   qualquer esquema multiaresta.

Ressalva de leitura: com poucas classes o oráculo a k=30 é quase vazio (30
vizinhos ao acaso já conteriam a classe verdadeira em 78% dos casos no Multitask
de 25 classes). A **pureza** é o número informativo -- comparar sempre com o
acaso, que é o prior da classe da consulta.

Contexto: o §6 registra SNF quebrado (LDA caiu para 3,4% por auto-referência no
`train_posterior`). Foi bug de vazamento, não falha do conceito, e o OOF
aninhado que faltava já existe.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--extra-blocks", nargs="*", default=["Gabor"])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / f"results/per_modality_neighbourhoods_{args.dataset}.csv"

    names = VISUAL + list(args.extra_blocks)
    blocks, raw_labels, _ = load_corpus(args.dataset, names)
    if args.classes_file and args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("casamento de classes do benchmark incompleto")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, {len(names)} modalidades")

    rows, overlaps = [], []
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        prior = np.bincount(labels[train], minlength=n_classes) / len(train)
        chance_purity = 100 * prior[labels[test]].mean()
        neighbour_sets = {}
        for name in names + ["CONCATENADO"]:
            started = time.perf_counter()
            if name == "CONCATENADO":
                space = np.hstack([StandardScaler().fit(blocks[n][train]).transform(blocks[n])
                                   .astype(np.float32) for n in names])
            else:
                space = (StandardScaler().fit(blocks[name][train])
                         .transform(blocks[name]).astype(np.float32))
            finder = NearestNeighbors(n_neighbors=args.k, algorithm="brute").fit(space[train])
            _, local = finder.kneighbors(space[test])
            indices = train[local]
            neighbour_sets[name] = indices
            hit = labels[indices] == labels[test][:, None]
            rows.append({"fold": fold, "modality": name, "dims": space.shape[1],
                         "purity_1": 100 * hit[:, 0].mean(),
                         "purity_k": 100 * hit.mean(),
                         "oracle_5": 100 * hit[:, :5].any(1).mean(),
                         "oracle_k": 100 * hit.any(1).mean(),
                         "chance_purity": chance_purity})
            log(f"  f{fold} {name:<14} {space.shape[1]:>4}d  pureza@{args.k} "
                f"{100 * hit.mean():5.2f} (acaso {chance_purity:.2f}, "
                f"{100 * hit.mean() / chance_purity:.2f}x)  "
                f"oráculo@5 {100 * hit[:, :5].any(1).mean():5.2f}  "
                f"[{time.perf_counter() - started:.0f}s]")
            del space

        # Jaccard médio entre modalidades, e teto da união.
        as_sets = {n: [set(r) for r in neighbour_sets[n]] for n in names}
        for a, b in itertools.combinations(names, 2):
            values = [len(x & y) / len(x | y) for x, y in zip(as_sets[a], as_sets[b])]
            overlaps.append({"fold": fold, "a": a, "b": b, "jaccard": float(np.mean(values))})
        union_hit = np.zeros(len(test), dtype=bool)
        for n in names:
            union_hit |= (labels[neighbour_sets[n]] == labels[test][:, None]).any(1)
        concat_hit = (labels[neighbour_sets["CONCATENADO"]] == labels[test][:, None]).any(1)
        union_size = np.mean([len(set().union(*(as_sets[n][i] for n in names)))
                              for i in range(0, len(test), 20)])
        rows.append({"fold": fold, "modality": "UNIÃO das 8", "dims": -1,
                     "purity_1": np.nan, "purity_k": np.nan, "oracle_5": np.nan,
                     "oracle_k": 100 * union_hit.mean(), "chance_purity": chance_purity})
        log(f"  f{fold} UNIÃO das 8: oráculo@{args.k} {100 * union_hit.mean():.2f} "
            f"(concatenado {100 * concat_hit.mean():.2f}, "
            f"+{100 * (union_hit.mean() - concat_hit.mean()):.2f}pp), "
            f"~{union_size:.0f} vizinhos distintos de {args.k * len(names)} possíveis")

    detail = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False, float_format="%.4f")
    pd.DataFrame(overlaps).to_csv(str(output).replace(".csv", "_jaccard.csv"),
                                  index=False, float_format="%.4f")

    print()
    summary = detail.groupby("modality", sort=False)[
        ["dims", "purity_1", "purity_k", "oracle_5", "oracle_k", "chance_purity"]].mean()
    print(f"{'modalidade':<14} {'dims':>5} {'pur@1':>7} {'pur@k':>7} {'x acaso':>8} "
          f"{'orác@5':>7} {'orác@k':>7}")
    for name, row in summary.iterrows():
        ratio = row.purity_k / row.chance_purity if row.chance_purity else np.nan
        print(f"{name:<14} {int(row.dims) if row.dims > 0 else '—':>5} "
              f"{row.purity_1:>7.2f} {row.purity_k:>7.2f} {ratio:>7.2f}x "
              f"{row.oracle_5:>7.2f} {row.oracle_k:>7.2f}")
    jac = pd.DataFrame(overlaps)
    if not jac.empty:
        print()
        print(f"Jaccard entre vizinhanças: média {jac.jaccard.mean():.4f}, "
              f"min {jac.jaccard.min():.4f} ({jac.loc[jac.jaccard.idxmin(), 'a']} x "
              f"{jac.loc[jac.jaccard.idxmin(), 'b']}), "
              f"max {jac.jaccard.max():.4f} ({jac.loc[jac.jaccard.idxmax(), 'a']} x "
              f"{jac.loc[jac.jaccard.idxmax(), 'b']})")


if __name__ == "__main__":
    main()
