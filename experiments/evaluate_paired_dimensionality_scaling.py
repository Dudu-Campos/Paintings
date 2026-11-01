"""O expoente de DADOS do §5.1 sobrevive com dimensionalidade PAREADA?

**O buraco que este script fecha.** O §5.1 mede que o expoente do erro contra
tamanho de treino cresce com o número de representações (0,0179 -> 0,0555,
r=+0,973). Mas na grade daquele achado o número de representações e a
dimensionalidade sobem JUNTOS -- de 71d com uma modalidade para 639d com oito.
Medido: `expoente x nº de modalidades` dá r=+0,973 e `expoente x log(dims)` dá
r=+0,961, com as duas explicativas correlacionadas a r=+0,919 entre si. **A
grade não separa as duas hipóteses**, e "mais dimensões movem o expoente"
explica os dados tão bem quanto "mais representações movem o expoente".

**O desenho.** Fixar o orçamento de features e variar só a PROCEDÊNCIA:

    braço A   `bank`         k features sorteadas das OITO famílias
    braço B   `descriptors`  as mesmas k features, mas só de Descriptors (130d)

Os dois entregam ao classificador **exatamente o mesmo número de colunas**, com
o mesmo critério de seleção (`max-r2-strat`, do §5.14, que estratifica por grupo
para não colapsar num bloco só). Varre-se o tamanho de treino e ajusta-se o
expoente do erro em cada braço.

    se o expoente de A > o de B    -> procedência move a forma; o §5.1 sobrevive
    se empatarem                   -> o §5.1 e um resultado sobre DIMENSIONALIDADE

O teto de 130d do pool de uma família é o que fixa o orçamento máximo; por isso
os budgets são 64 e 128, não 639.

**Ressalva de instrumento.** A sonda é LDA, como no §5.1, para que o expoente
seja comparável ao que está publicado lá. Em 25 classes ela vê no máximo 24
componentes discriminantes -- mas o teto é o MESMO nos dois braços, então a
comparação é pareada. O §5.1 nota que o teto pode penalizar o vetor multi-bloco
mais que o denso; se penalizar, este teste é CONSERVADOR (subestima o braço A).

A seleção usa a matriz de r² calculada **só sobre as linhas de treino** de cada
fold, e o subconjunto de treino é amostrado de forma estratificada antes disso,
para que nem a seleção nem a padronização vejam dado que o modelo não teria.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import VISUAL, clean  # noqa: E402
from evaluate_fusion_delta_across_corpora import load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_redundancy_selection_curve import r2_matrix, select  # noqa: E402
from protocol import SEED  # noqa: E402
from registry import feature_columns, sanitize  # noqa: E402

# Fronteiras dos seis sub-grupos de Descriptors, para que o braço restrito use o
# MESMO critério estratificado do braço completo (§5.14).
DESCRIPTOR_BOUNDS = [(0, 30), (30, 60), (60, 90), (90, 99), (99, 117), (117, 130)]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--train-sizes", type=int, nargs="+",
                        default=[2500, 5000, 10000, 20000, 40000, 66259])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--order", default="max-r2-strat")
    parser.add_argument("--fourier-v2", type=Path, default=None,
                        help="optional parquet with the richer Fourier block; aligned by stem")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/paired_dimensionality_scaling.csv")
    args = parser.parse_args()

    blocks, raw_labels, frame = load_corpus(args.dataset, VISUAL + ["Gabor"])
    if args.fourier_v2 is not None:
        table = pd.read_parquet(args.fourier_v2)
        cols = feature_columns(table)
        index = pd.Series(np.arange(len(table)), index=table["stem"].astype(str)).reindex(
            frame["stem"].astype(str).to_numpy())
        if index.isna().any():
            raise ValueError(f"Fourier-v2: {int(index.isna().sum())} stems missing")
        blocks["Fourier"] = sanitize(
            table.iloc[index.to_numpy(dtype=int)][cols].to_numpy(np.float32),
            "Fourier-v2")
        log(f"Fourier substituido por Fourier-v2: {blocks['Fourier'].shape[1]} features")
    if args.classes_file and args.classes_file.exists():
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        mask = np.array([key(v) in wanted for v in raw_labels])
        if len(np.unique(raw_labels[mask])) != len(wanted):
            raise ValueError("casamento de classes do benchmark incompleto")
        blocks = {n: b[mask] for n, b in blocks.items()}
        raw_labels = raw_labels[mask]
    blocks = {n: clean(b) for n, b in blocks.items()}
    names = [n for n in VISUAL + ["Gabor"] if n in blocks]
    if names[0] != "Descriptors" or blocks["Descriptors"].shape[1] != 130:
        raise ValueError("esperado Descriptors (130d) como primeiro bloco")

    features = np.hstack([blocks[n] for n in names])
    block_bank = np.concatenate([np.full(blocks[n].shape[1], i) for i, n in enumerate(names)])
    block_desc = np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(DESCRIPTOR_BOUNDS)])
    labels = LabelEncoder().fit_transform(raw_labels)
    n_classes = len(np.unique(labels))
    log(f"{args.dataset}: {len(labels):,} imagens, {features.shape[1]} features, "
        f"{n_classes} classes")

    POOLS = {"bank": (features, block_bank), "descriptors": (features[:, :130], block_desc)}

    rows, done = [], set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(int(r["fold"]), str(r["pool"]), int(r["budget"]),
                 int(r["train_size"]), int(r["seed"])) for r in rows}
        log(f"retomando: {len(done)} pontos já feitos")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))
    for fold, (train_index, test_index) in enumerate(splits[:args.folds], start=1):
        for train_size in args.train_sizes:
            # A margem de `n_classes` existe porque os folds diferem em uma
            # linha (82.824 / 5 nao e inteiro): pedir 66.259 de um treino de
            # 66.260 deixaria 1 amostra para 25 classes e o split estratificado
            # falha. Abaixo da margem, usa o treino inteiro.
            if train_size <= len(train_index) - n_classes:
                sub, _ = train_test_split(train_index, train_size=train_size,
                                          stratify=labels[train_index],
                                          random_state=SEED + fold)
            else:
                sub = train_index
            for pool, (values, block_of) in POOLS.items():
                # r² e scaler ajustados SÓ no subconjunto de treino desta célula.
                scaler = StandardScaler().fit(values[sub])
                scaled = scaler.transform(values).astype(np.float32)
                r2 = r2_matrix(scaled[sub])
                for seed in range(args.seeds):
                    if all((fold, pool, b, train_size, seed) in done for b in args.budgets):
                        continue
                    rng = np.random.default_rng(1000 * seed + fold)
                    picked = select(args.order, r2, block_of, max(args.budgets), rng)
                    for budget in args.budgets:
                        if (fold, pool, budget, train_size, seed) in done:
                            continue
                        cols = np.array(picked[:budget])
                        # Priors uniformes sobre as classes PRESENTES na
                        # subamostra: com 125 classes e n=2.500 as raras somem
                        # (desbalanceamento de 139x), e um vetor de 125 priors
                        # contra 110 medias quebra o LDA. Com todas presentes
                        # isto e identico a `ones(n_classes)/n_classes`.
                        present = np.unique(labels[sub])
                        model = LinearDiscriminantAnalysis(
                            priors=np.ones(len(present)) / len(present))
                        model.fit(scaled[sub][:, cols], labels[sub])
                        pred = model.predict(scaled[test_index][:, cols])
                        rows.append({
                            "fold": fold, "pool": pool, "budget": budget,
                            "train_size": int(len(sub)), "seed": seed,
                            "groups_used": int(len(np.unique(block_of[cols]))),
                            "bacc": 100 * balanced_accuracy_score(labels[test_index], pred),
                        })
                        pd.DataFrame(rows).to_csv(args.output, index=False,
                                                  float_format="%.4f")
            log(f"fold {fold} n={len(sub):>6}: pronto")

    frame = pd.DataFrame(rows)
    log("\nBAcc media (5 folds x 3 sementes):")
    for budget in args.budgets:
        piv = (frame[frame.budget.eq(budget)]
               .pivot_table(index="train_size", columns="pool", values="bacc"))
        log(f"\n  budget={budget} features\n{piv.round(2).to_string()}")
        out = {}
        for pool in piv.columns:
            x, y = np.log(piv.index.to_numpy()), np.log(100 - piv[pool].to_numpy())
            out[pool] = -np.polyfit(x, y, 1)[0]
        log(f"  expoente do erro contra DADOS: " +
            "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
        if len(out) == 2:
            a, b = out.get("bank"), out.get("descriptors")
            log(f"  DIFERENCA (bank - descriptors) = {a - b:+.4f}")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
