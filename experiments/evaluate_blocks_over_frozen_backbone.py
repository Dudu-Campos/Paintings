"""Features projetadas somam a uma backbone congelada? E somam onde ela é cega?

Última peça da linha de deep learning, e a única que escapa do que quebrou as
três tentativas anteriores. Pré-texto, CNN-como-13º-branch (ingênuo e OOF) e as
variantes de fusão balanceada falharam **todas** por dependerem do dado de
treino de uma CNN nossa: de lá vieram a memorização (validação em 0,849), os
cinco espaços de 2.048d arbitrariamente rotacionados do OOF, e a ausência de
pressão de gradiente na conv1. Nenhuma testou a ideia; testaram protocolos
quebrados.

A ResNet50 do ImageNet congelada não tem nenhum desses problemas: nunca viu os
nossos rótulos, é um featurizador único (não há checkpoint por fold, logo não há
rotação entre folds), e os embeddings já estão extraídos. Não há CNN para
treinar.

**A previsão é registrada, não posterior.** O achado 23 mediu, por regressão
ridge do embedding congelado para cada bloco, o quanto a backbone já computa
aquele bloco sozinha. Se features projetadas somam a deep learning *na medida em
que a rede não consegue computá-las*, o marginal tem de ordenar por (1 − R²).
Descriptors (R² 0,208) deve somar; Hilbert (0,521, e a conv **aleatória** já
entrega 0,527) não deve. O script calcula essa correlação e a imprime.

Isso também explica retroativamente a fusão que falhou: o 13º branch levava os
1.154d inteiros, e a tabela abaixo diz que a backbone já computa cinco daqueles
blocos a R² 0,45-0,77 -- adicionar o que ela já tem é diluição. Adicionar só o
que ela comprovadamente não tem nunca foi testado.

**Ressalva que precisa estar na legenda:** este braço usa supervisão externa
(ImageNet) dos dois lados e portanto **não** é o número que sustenta a alegação
de competir sem supervisão externa. É uma pergunta diferente e prescritiva: com
uma backbone pré-treinada na mão, quais features projetadas ainda pagam.

Saída incremental: outputs/blocks_over_backbone/fold_results.csv
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
BACKBONE = "ResNet50"
BLOCKS = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier", "Multiscale")
# R² de ridge do embedding ImageNet congelado para cada bloco (achado 23,
# `evaluate_cnn_learns_handdesigned.py`). Quanto MENOR, mais o bloco é
# inalcançável para a backbone -- e a hipótese é que o marginal siga (1 − R²).
# Multiscale não tem R² medido (são quadrantes dos outros) e fica fora da
# correlação, mas é avaliado como condição.
RECOVERABILITY = {"Descriptors": 0.208, "Fourier": 0.374, "SLIC": 0.425,
                  "Line graph": 0.452, "L(G_felz)": 0.505,
                  "Felzenszwalb": 0.506, "Hilbert": 0.521}


def k_grid(limit: int) -> list[int]:
    grid = [k for k in (32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048)
            if k <= limit]
    if not grid or grid[-1] != limit:
        grid.append(limit)
    return grid


def lda_probe(blocks, labels, fit, test, n_classes) -> float:
    """Pico sobre k, obrigatório: 2.048d contra 2.048+1.032d é 50% a mais."""
    scaled_fit, scaled_test = [], []
    for block in blocks:
        scaler = StandardScaler().fit(block[fit])
        scaled_fit.append(scaler.transform(block[fit]))
        scaled_test.append(scaler.transform(block[test]))
    train_x, test_x = np.hstack(scaled_fit), np.hstack(scaled_test)
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


def report(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = table[table.condition == BACKBONE].set_index("fold")[metric]
    rows = []
    log(f"\n=== {metric}: marginal sobre a {BACKBONE} congelada ===")
    log(f"  {'condição':<26} {'BAcc':>6}   {'Δ':>6}  {'p':>7}  folds  R²")
    log(f"  {BACKBONE + ' sozinha':<26} {base.mean():6.2f}")
    for name in BLOCKS:
        condition = f"+ {name}"
        current = table[table.condition == condition].set_index("fold")[metric]
        common = current.index.intersection(base.index)
        if len(common) < 2:
            continue
        delta = (current[common] - base[common]).to_numpy()
        p_value = stats.ttest_rel(current[common], base[common]).pvalue
        rows.append({"block": name, "metric": metric, "delta": delta.mean(),
                     "std": delta.std(ddof=1), "p": p_value,
                     "positive": int((delta > 0).sum()), "folds": len(delta),
                     "recoverability": RECOVERABILITY.get(name, np.nan)})
        log(f"  {condition:<26} {current.mean():6.2f}  {delta.mean():+6.2f} "
            f"{p_value:7.4f}  {int((delta > 0).sum())}/{len(delta)}  "
            + (f"{RECOVERABILITY[name]:.3f}" if name in RECOVERABILITY else "  —"))

    for condition in ("+ tudo", "só nossos blocos"):
        current = table[table.condition == condition].set_index("fold")[metric]
        common = current.index.intersection(base.index)
        if len(common) < 2:
            continue
        delta = (current[common] - base[common]).to_numpy()
        p_value = stats.ttest_rel(current[common], base[common]).pvalue
        log(f"  {condition:<26} {current.mean():6.2f}  {delta.mean():+6.2f} "
            f"{p_value:7.4f}  {int((delta > 0).sum())}/{len(delta)}")
        rows.append({"block": condition, "metric": metric, "delta": delta.mean(),
                     "std": delta.std(ddof=1), "p": p_value,
                     "positive": int((delta > 0).sum()), "folds": len(delta),
                     "recoverability": np.nan})

    marginals = pd.DataFrame(rows).dropna(subset=["recoverability"])
    if len(marginals) >= 4:
        # A pergunta do experimento, num número: o marginal segue a
        # inalcançabilidade? Correlação NEGATIVA com o R² = hipótese sustentada.
        rho, p_value = stats.spearmanr(marginals.recoverability, marginals.delta)
        log(f"\n  marginal × R² (recuperabilidade pela backbone): "
            f"Spearman {rho:+.3f} (p={p_value:.4f}), n={len(marginals)}")
        log("  negativo = o bloco soma na medida em que a rede não o computa")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/blocks_over_backbone")
    args = parser.parse_args()

    # `allow_external=True` é deliberado e é o ponto do experimento: os dois
    # lados usam ImageNet. Este número não entra na alegação de "sem supervisão
    # externa" -- responde a pergunta prescritiva, que é outra.
    cohort = load_blocks(args.dataset, [BACKBONE] + list(BLOCKS),
                         allow_external=True)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    blocks = dict(cohort.blocks)
    blocks["Multiscale"] = cohort.select(
        "Multiscale", lambda column: "_Descriptors_" in column)
    log(f"coorte {len(labels):,} | backbone {blocks[BACKBONE].shape[1]}d | "
        f"{n_classes} classes")

    # Os marginais por bloco respondem *por que*; estas duas respondem *quanto*.
    # `+ tudo` é o número da linha inteira, e não é a soma dos marginais
    # individuais -- os blocos também são complementares entre si, então a
    # diferença entre um e outro mede a sub-aditividade da fusão.
    # `só nossos blocos` fecha a comparação tripla (backbone / nós / os dois),
    # que é a forma como isso vai para a tabela.
    conditions = ([(BACKBONE, [BACKBONE])]
                  + [(f"+ {name}", [BACKBONE, name]) for name in BLOCKS]
                  + [("+ tudo", [BACKBONE] + list(BLOCKS)),
                     ("só nossos blocos", list(BLOCKS))])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_results.csv"
    rows, completed = [], set()
    if result_path.exists():
        previous = pd.read_csv(result_path)
        rows = previous.to_dict("records")
        completed = set(zip(previous.condition, previous.fold))
        log(f"retomando {len(rows)} linhas")

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
                      "dimensions": sum(b.shape[1] for b in chosen)}
            record["lda_bacc"] = lda_probe(chosen, labels, outer_train, test, n_classes)
            if not args.skip_mlp:
                train_x, validation_x, test_x = scaled_splits(
                    chosen, fit, validation, test)
                metrics, _, _ = train_balanced(
                    train_x, labels[fit], validation_x, labels[validation],
                    test_x, labels[test], [b.shape[1] for b in chosen], n_classes,
                    SEED + fold, args.epochs, args.patience)
                record["mlp_bacc"] = 100 * metrics["bacc"]
            record["elapsed_seconds"] = time.perf_counter() - started
            rows.append(record)
            completed.add((condition, fold))
            pd.DataFrame(rows).to_csv(result_path, index=False, float_format="%.6f")
            log(f"  fold {fold} | {condition:<20} | {record['dimensions']:5d}d | "
                f"LDA {record['lda_bacc']:5.2f}"
                + (f" | MLP {record['mlp_bacc']:5.2f}" if "mlp_bacc" in record else "")
                + f" | {record['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    summary = []
    for metric in ("lda_bacc", "mlp_bacc"):
        if metric in table.columns and table[metric].notna().any():
            summary.append(report(table, metric))
    if summary:
        pd.concat(summary).to_csv(args.output_dir / "marginals.csv", index=False,
                                  float_format="%.6f")
    log(f"\nSalvo em: {result_path}")


if __name__ == "__main__":
    main()
