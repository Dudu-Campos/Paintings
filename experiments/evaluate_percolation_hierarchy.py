"""A hierarquia cromática soma ao pipeline? Marginal das duas percolações.

Base = a melhor configuração medida (achado 11 + achado 14): os oito blocos
visuais do registro com o Multiscale restrito à família Descriptors, ou seja
1.154d em 12 branches, **sem Text** -- o bloco de legendas depende de informação
externa e não entra no número que vale.

Testa `Percolation SLIC` e `Percolation Felz` isoladas, cada uma sobre a base, e
as duas juntas. A pergunta específica é se a complementaridade SLIC×Felzenszwalb
atravessa também este operador, como atravessou o line graph (achado 14: somar
L(G_felz) a L(G_slic) rendeu +0,79pp enquanto substituir não mudou nada).

Duas sondas de propósito. A LDA é a de triagem, barata; a MLP ramificada é o
instrumento de produção e é ela que decide. O projeto já mediu seis casos em que
a LDA extrai sinal que a MLP destrói (HOG, quadrantes de Hilbert, as quatro
representações dormentes), então concordância entre as duas é informação e
discordância também.

`--families` permite medir `perc_` (grade fixa de ε), `reach_` (limiares de
alcance, livres de escala) e `merge_` (distribuição de alturas de fusão)
separadamente, sem reextrair nada -- os prefixos existem para isso.

Saída incremental: outputs/percolation_hierarchy/fold_results.csv
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
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from registry import load_blocks, log  # noqa: E402
from balanced_branch_mlp import train_balanced  # noqa: E402

SEED, N_FOLDS = 42, 5
BASE = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
        "Hilbert", "Fourier", "Multiscale")
CANDIDATES = ("Percolation SLIC", "Percolation Felz")


def k_grid(limit: int) -> list[int]:
    grid = [k for k in (32, 64, 128, 256, 384, 512, 768, 1024, 1536) if k <= limit]
    if not grid or grid[-1] != limit:
        grid.append(limit)
    return grid


def lda_probe(blocks: list[np.ndarray], labels: np.ndarray, fit: np.ndarray,
              test: np.ndarray, n_classes: int) -> float:
    """BAcc com pico sobre k. Obrigatório: as condições diferem em dimensão."""
    parts_fit, parts_test = [], []
    for block in blocks:
        scaler = StandardScaler().fit(block[fit])
        parts_fit.append(scaler.transform(block[fit]))
        parts_test.append(scaler.transform(block[test]))
    train_x, test_x = np.hstack(parts_fit), np.hstack(parts_test)
    limit = min(train_x.shape[1], len(fit) - n_classes)
    reducer = PCA(n_components=limit, svd_solver="randomized",
                  random_state=SEED).fit(train_x)
    train_p, test_p = reducer.transform(train_x), reducer.transform(test_x)
    best = 0.0
    for k in k_grid(limit):
        model = LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, k),
            priors=np.ones(n_classes) / n_classes).fit(train_p[:, :k], labels[fit])
        best = max(best, balanced_accuracy_score(
            labels[test], model.predict(test_p[:, :k])))
    return 100 * best


def scaled_splits(blocks, fit, validation, test):
    parts = ([], [], [])
    for block in blocks:
        scaler = StandardScaler().fit(block[fit])
        for target, index in zip(parts, (fit, validation, test)):
            target.append(scaler.transform(block[index]).astype(np.float32))
    return tuple(np.hstack(part).astype(np.float32, copy=False) for part in parts)


def summarise(table: pd.DataFrame, metric: str) -> None:
    base = table[table.condition == "base"].set_index("fold")[metric]
    log(f"\n=== {metric} (5 folds) ===")
    for condition in table.condition.unique():
        rows = table[table.condition == condition].set_index("fold")[metric]
        line = f"  {condition:<34} {rows.mean():6.2f} ± {rows.std(ddof=1):4.2f}"
        if condition != "base" and not condition.startswith("só "):
            common = rows.index.intersection(base.index)
            delta = (rows[common] - base[common]).to_numpy()
            if len(delta) > 1:
                p = stats.ttest_rel(rows[common], base[common]).pvalue
                line += (f"   Δ {delta.mean():+5.2f} ± {delta.std(ddof=1):4.2f}pp"
                         f"  p={p:.4f}  {int((delta > 0).sum())}/{len(delta)}")
        log(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--families", nargs="+", default=["all"],
                        help="all | perc | reach | merge (prefixos das colunas)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/percolation_hierarchy")
    args = parser.parse_args()

    names = list(BASE) + list(CANDIDATES)
    cohort = load_blocks(args.dataset, names, allow_external=False)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))

    blocks = dict(cohort.blocks)
    # Multiscale só na família Descriptors: o achado 11 mediu o fatorial 2^3 das
    # famílias de quadrante e só-Descriptors é o máximo global na MLP, o único
    # subconjunto que bate a canônica com significância (+0,595 ± 0,180pp,
    # p=0,0018, 5/5) e ainda com 512 dimensões a menos. Os quadrantes de Hilbert
    # chegam a atrapalhar ativamente (Shapley −0,52).
    blocks["Multiscale"] = cohort.select(
        "Multiscale", lambda column: "_Descriptors_" in column)
    if args.families != ["all"]:
        wanted = tuple(f"{family}_" for family in args.families)
        for name in CANDIDATES:
            blocks[name] = cohort.select(name, lambda c: c.startswith(wanted))
            log(f"{name}: {blocks[name].shape[1]} colunas de {args.families}")

    base_names = list(BASE)
    log(f"coorte {len(labels):,} imagens, {n_classes} classes | "
        f"base {sum(blocks[n].shape[1] for n in base_names):,}d em "
        f"{len(base_names)} branches")
    for name in CANDIDATES:
        log(f"  candidato {name}: {blocks[name].shape[1]}d")

    conditions: list[tuple[str, list[str]]] = [
        ("base", base_names),
        ("+ Percolation SLIC", base_names + ["Percolation SLIC"]),
        ("+ Percolation Felz", base_names + ["Percolation Felz"]),
        ("+ ambas", base_names + list(CANDIDATES)),
        ("só Percolation SLIC", ["Percolation SLIC"]),
        ("só Percolation Felz", ["Percolation Felz"]),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows: list[dict] = []
    completed: set[tuple[str, int]] = set()
    if result_path.exists():
        previous = pd.read_csv(result_path)
        rows = previous.to_dict("records")
        completed = set(zip(previous.condition, previous.fold))
        log(f"retomando {len(rows)} linhas de {result_path}")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))
    for fold, (outer_train, test) in enumerate(splits, start=1):
        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train],
            random_state=SEED + fold)
        for condition, selected in conditions:
            if (condition, fold) in completed:
                continue
            chosen = [blocks[name] for name in selected]
            started = time.perf_counter()
            record = {"fold": fold, "condition": condition,
                      "n_branches": len(chosen),
                      "dimensions": sum(b.shape[1] for b in chosen)}
            record["lda_bacc"] = lda_probe(chosen, labels, outer_train, test, n_classes)
            if not args.skip_mlp:
                train_x, validation_x, test_x = scaled_splits(chosen, fit, validation, test)
                metrics, _, _ = train_balanced(
                    train_x, labels[fit], validation_x, labels[validation],
                    test_x, labels[test], [b.shape[1] for b in chosen], n_classes,
                    SEED + fold, args.epochs, args.patience)
                record["mlp_bacc"] = 100 * metrics["bacc"]
            record["elapsed_seconds"] = time.perf_counter() - started
            rows.append(record)
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False, float_format="%.6f")
            log(f"  fold {fold} | {condition:<22} | {record['dimensions']:5d}d | "
                f"LDA {record['lda_bacc']:5.2f}"
                + (f" | MLP {record['mlp_bacc']:5.2f}" if "mlp_bacc" in record else "")
                + f" | {record['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    for metric in ("lda_bacc", "mlp_bacc"):
        if metric in table.columns and table[metric].notna().any():
            summarise(table, metric)
    log(f"\nSalvo em: {result_path}")


if __name__ == "__main__":
    main()
