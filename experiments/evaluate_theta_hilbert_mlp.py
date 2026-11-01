"""A mesma grade de theta do Hilbert, com MLP no lugar do LDA.

Motivo de não ser redundante: a sonda LDA projeta em `n_classes − 1 = 24`
componentes **independentemente da dimensão de entrada** (§8). O eixo de
`n_bands` vai de 33d a 201d, então o ganho medido lá pode estar limitado pelo
teto do instrumento e não pela informação. Um MLP não tem esse teto, e as duas
leituras possíveis são informativas em direções opostas:

  - se a diferença entre eixos CRESCE, o LDA estava comprimindo o ganho de
    granularidade e o eixo de bandas vale ainda mais do que parecia;
  - se DIMINUI, parte dos 3,56pp era o LDA precisando de mais dimensões de
    entrada para chegar às mesmas 24 componentes -- e o ganho é do instrumento,
    não da representação.

`PlainMLP` sem branches, o mesmo do §5.14, porque aqui há um bloco só e a
estrutura por branch não teria o que separar.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_redundancy_selection_curve import fit  # noqa: E402

SEED = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_hilbert")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results/theta_hilbert_mlp.csv")
    args = parser.parse_args()

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    # `meta.style` devolve o acessor Styler do pandas, não a coluna (§8).
    meta = meta[[key(str(v)) in wanted for v in meta["style"]]]
    log(f"coorte: {len(meta):,} imagens, {meta['style'].nunique()} classes")

    rows = []
    done = set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(int(r["n_size"]), int(r["n_bands"])) for r in rows}

    for directory in sorted(args.work_dir.glob("n*_b*")):
        n_size = int(directory.name.split("_")[0][1:])
        n_bands = int(directory.name.split("_b")[1])
        if (n_size, n_bands) in done:
            continue
        values = np.load(directory / "hilbert_features.npy")
        stems = np.load(directory / "hilbert_stems.npy", allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems)
        take = index.reindex(meta.stem.to_numpy()).dropna()
        matrix = values[take.to_numpy(dtype=int)].astype(np.float32)
        labels = LabelEncoder().fit_transform(
            meta.set_index("stem").loc[take.index, "style"].to_numpy(dtype=str))
        n_classes = len(np.unique(labels))

        scores = []
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels)), labels))[:args.folds]
        for fold, (train, test) in enumerate(splits, start=1):
            fit_idx, val_idx = train_test_split(
                train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
            scaler = StandardScaler().fit(matrix[train])
            scaled = scaler.transform(matrix).astype(np.float32)
            metrics = fit(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx],
                          scaled[test], labels[test], n_classes, seed=SEED + fold)
            scores.append(metrics["bacc"])
        rows.append({"n_size": n_size, "n_bands": n_bands, "dimensions": matrix.shape[1],
                     "is_star": (n_size, n_bands) == (128, 8),
                     "mlp_bacc": float(np.mean(scores)), "mlp_std": float(np.std(scores))})
        log(f"  ({n_size:>3}, {n_bands:>2})  {matrix.shape[1]:>3}d  MLP {np.mean(scores):5.2f}")
        pd.DataFrame(rows).to_csv(args.output, index=False)

    frame = pd.DataFrame(rows)
    log("\n" + frame.pivot_table(index="n_size", columns="n_bands",
                                 values="mlp_bacc").round(2).to_string())


if __name__ == "__main__":
    main()
