"""O ganho de theta isolado sobrevive à concatenação com as outras 7 modalidades?

Duas perguntas separadas, com controles diferentes:

**1. Transferência.** O Hilbert isolado ganha ~4,6pp de amplitude ao longo da
grade de theta, mas o §5.10 mede LOO = **−0,03pp** para ele no pipeline completo
-- é o bloco mais substituível do banco. A previsão é que quase nada transfira.
Testado trocando SÓ o bloco Hilbert dentro do hand-8, mantendo os outros sete
idênticos, no BranchMLP de produção.

**2. Dimensionalidade.** O eixo `n_bands` vai de 33d a 201d, então parte do
ganho isolado é capacidade e não informação. O controle é `(128,32)` reduzido
por subamostra aleatória a **57 colunas** -- a dimensão exata de theta*. Se o
ganho persistir, é informação; se colapsar para o nível de theta*, era
dimensionalidade. Sem esse braço, "mais bandas é melhor" e "mais colunas é
melhor" são indistinguíveis.

A subamostra é aleatória de propósito: escolher as 57 melhores colunas usaria
o rótulo e reintroduziria o viés de seleção que o §5.15 documenta.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
# `evaluate_full_factorial_ablation` puxa uma cadeia que termina em
# src/extractors/extract_fourier_spectrum_features.py.
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from evaluate_full_factorial_ablation import evaluate_mlp  # noqa: E402
from evaluate_redundancy_selection_curve import fit as plain_fit  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402

STAR = (128, 8)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_hilbert")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--thetas", nargs="+", default=["128x8", "64x16", "128x32", "256x32"],
                        help='"all" varre a grade inteira em disco, que é o que '
                             'a distribuição de substituição precisa: com 3 '
                             'configurações não há p05 nenhum.')
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--head", choices=["branch", "plain"], default="branch",
                        help="`plain` remove os encoders por bloco. Testa se o "
                             "orçamento fixo de 128d por branch AMORTECE a "
                             "sensibilidade a dimensionalidade: sem ele, as "
                             "dimensões extras do Hilbert competem direto com "
                             "as outras 639 pelos mesmos 512 neurônios.")
    parser.add_argument("--isolated", action="store_true",
                          help="mede a view SOZINHA, mesmo laco de treino")
    parser.add_argument("--output", type=Path,
                        default=None)
    args = parser.parse_args()
    if args.output is None:
        suffix = "" if args.head == "branch" else "_plain"
        args.output = ROOT / f"results/theta_hilbert_pipeline{suffix}.csv"

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    mask = np.array([key(v) in wanted for v in raw_labels])
    if len(np.unique(raw_labels[mask])) != len(wanted):
        raise ValueError("casamento de classes do benchmark incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    raw_labels = raw_labels[mask]
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    order = [n for n in VISUAL + ["Gabor"] if n in blocks]
    log(f"hand-8: {len(labels):,} imagens, {n_classes} classes, blocos={order}")

    # A coorte do registro define a ordem das linhas; os stems de cada extração
    # de theta têm de ser realinhados a ela, e não o contrário.
    cohort = pd.read_csv(args.meta, dtype={"stem": str})
    cohort = cohort[[key(str(v)) in wanted for v in cohort["style"]]]
    reference = cohort.stem.to_numpy()
    if len(reference) != len(labels):
        raise ValueError(f"coorte do registro ({len(labels)}) != metadado ({len(reference)})")

    def hilbert_block(n_size: int, n_bands: int, columns: int | None) -> np.ndarray:
        directory = args.work_dir / f"n{n_size}_b{n_bands}"
        values = np.load(directory / "hilbert_features.npy")
        stems = np.load(directory / "hilbert_stems.npy", allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise ValueError(f"({n_size},{n_bands}): {int(index.isna().sum())} stems ausentes")
        matrix = clean(values[index.to_numpy(dtype=int)])
        if columns is not None and columns < matrix.shape[1]:
            # Aleatória, não as melhores: selecionar por rótulo reintroduziria o
            # viés circular que o §5.15 documenta.
            picked = np.random.default_rng(SEED).choice(matrix.shape[1], columns, replace=False)
            matrix = matrix[:, np.sort(picked)]
        return matrix

    if args.thetas == ["all"]:
        found = sorted((int(d.name.split("_")[0][1:]), int(d.name.split("_b")[1]))
                       for d in args.work_dir.glob("n*_b*"))
        # theta* primeiro: é a referência de todos os deltas pareados.
        args.thetas = [f"{STAR[0]}x{STAR[1]}"] + [f"{a}x{b}" for a, b in found
                                                  if (a, b) != STAR]
        log(f"grade inteira: {len(args.thetas)} configurações")

    conditions = {}
    for spec in args.thetas:
        n_size, n_bands = (int(v) for v in spec.split("x"))
        conditions[f"hilbert {n_size}x{n_bands}"] = hilbert_block(n_size, n_bands, None)
    if len(args.thetas) < 8:
        star_dims = conditions[f"hilbert {STAR[0]}x{STAR[1]}"].shape[1]
        conditions[f"hilbert 128x32 -> {star_dims}d"] = hilbert_block(128, 32, star_dims)

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    rows = []
    for name, block in conditions.items():
        # `--isolated`: a MESMA view sozinha, pelo MESMO carregador e o MESMO
        # `evaluate_mlp`. O §5.17 media o lado isolado com sonda LDA e o lado do
        # banco com BranchMLP, e em 10/set o Hilbert mostrou que 1/3 do
        # encolhimento aparente da dispersao era so a compressao LDA->MLP
        # (0,66x, batendo a razao do §5.9d). Instrumento pareado por
        # construcao, nao por conferencia.
        chosen = [block] if args.isolated else [blocks[n] if n != "Hilbert" else block for n in order]
        for fold, (train, test) in enumerate(splits, start=1):
            started = time.perf_counter()
            if args.head == "branch":
                metrics = evaluate_mlp(chosen, labels, train, test, fold,
                                       epochs=150, patience=15)
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                fit_idx, val_idx = train_test_split(
                    train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
                stacked = np.hstack(chosen)
                scaler = StandardScaler().fit(stacked[train])
                scaled = scaler.transform(stacked).astype(np.float32)
                raw = plain_fit(scaled[fit_idx], labels[fit_idx],
                                scaled[val_idx], labels[val_idx],
                                scaled[test], labels[test], n_classes, seed=SEED + fold)
                metrics = {"bacc": raw["bacc"] / 100}
            rows.append({"condition": name, "fold": fold,
                         "hilbert_dims": block.shape[1],
                         "dimensions": sum(b.shape[1] for b in chosen),
                         "seconds": time.perf_counter() - started,
                         "bacc": 100 * metrics["bacc"]})
            log(f"  {name:<24} f{fold}  {rows[-1]['dimensions']:>4}d  "
                f"BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False)

    frame = pd.DataFrame(rows)
    summary = frame.groupby("condition").agg(
        hilbert_dims=("hilbert_dims", "first"), bacc=("bacc", "mean"), sd=("bacc", "std"))
    base = frame[frame.condition.eq(f"hilbert {STAR[0]}x{STAR[1]}")].set_index("fold")["bacc"]
    log("\n" + summary.round(2).to_string())
    log("\ndelta contra theta* NO PIPELINE (pareado por fold):")
    from scipy import stats
    for name in summary.index:
        if name == f"hilbert {STAR[0]}x{STAR[1]}":
            continue
        other = frame[frame.condition.eq(name)].set_index("fold")["bacc"]
        shared = base.index.intersection(other.index)
        delta = other[shared] - base[shared]
        _, p = stats.ttest_rel(other[shared], base[shared])
        log(f"  {name:<24} {delta.mean():+6.2f}pp  p={p:.4f}  "
            f"{int((delta > 0).sum())}/{len(delta)}")


if __name__ == "__main__":
    main()
