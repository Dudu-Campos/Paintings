"""Os placebos que o `logit-add` precisa, e a varredura de k refeita por dentro dele.

O §5.6 registra `logit-add` como a cura que funciona em 125 classes (+0,91pp,
5/5 folds) e atribui o ganho ao grafo. Antes de escrever isso no artigo, uma
conta que não estava feita: com k=30 e 125 classes, **no máximo 30 das 125
coordenadas** de cada linha do log-lift podem ser não-vazias, logo >=76% da
matriz é o valor de "sem evidência",

    floor_c = log(1e-6) - log(p_c)

que **não depende da imagem** -- só do prior. Medido nesta coorte ele vai de
-11,57 (classe frequente) a -4,62 (rara), amplitude 6,95; com o lambda aprendido
de ~0,14 isso é ~0,97 unidade de logit de viés sistemático **a favor da cauda**.
BAcc é recall macro e premia exatamente esse deslocamento.

Então o +0,91pp tem duas explicações compatíveis com o que foi medido:

    (a) a vizinhança informa, que é a leitura do §5.6;
    (b) lambda * floor é um viés por classe corrigindo o prior, e a informação
        de vizinhança é carona.

Dois placebos separam. Os dois usam o MESMO caminho aditivo e o mesmo lambda
aprendido, mudando só o que entra nele:

    sem-evidência   toda linha = floor. Zero informação de imagem, e preserva
                    exatamente o termo (b). Se recuperar o ganho, era prior.
    embaralhado     as linhas do log-lift permutadas dentro de `outer` e dentro
                    de `test`. Preserva a distribuição marginal INTEIRA --
                    esparsidade, escala, floor -- e destrói só a correspondência
                    entre a linha e a imagem dela. É o nulo estrito.

A varredura de k entra de carona porque a pergunta dela mudou. Concatenada, k
não teve gradiente (30/60/100/150 -> -0,70/-0,33/-0,68/-1,11) e o §5.6 concluiu
que inanição não é curável por k. Mas ali o log-lift entrava por 282 dimensões
densas; aqui entra por um escalar e na forma funcional certa (aditiva em log,
que é Bayes se as evidências fossem independentes), então um lift mais diluído
degrada suave -- lambda encolhe -- em vez de injetar ruído. A previsão do §5.6
volta a ser testável, e agora sem o pedágio de capacidade por cima.

O braço `hand-8` NÃO é retreinado: é copiado de results/multitask_logit_add.csv,
cujos cinco folds batem ao centésimo com os de sweep_similarity_graph_k.py --
mesma semente, mesmos splits, mesma classe `BranchMLP`.

Saída: results/multitask_logit_add_controls.csv
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_logit_add_multitask import build_oof, train_arm  # noqa: E402
from evaluate_minc2500_official import HAND8  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
BASE_CSV = ROOT / "results/multitask_logit_add.csv"
torch.set_num_threads(16)


def floor_vector(labels, n_classes):
    """O log-lift de uma linha sem nenhum vizinho: função só do prior."""
    prior = np.bincount(labels, minlength=n_classes).astype(np.float32)
    prior /= prior.sum()
    return (np.log(1e-6) - np.log(prior + 1e-6)).astype(np.float32)


def shuffled(lift, outer, test, fold):
    """Permuta linhas dentro de cada estrato. Marginal intacta, pareamento morto."""
    rng = np.random.default_rng(SEED + fold)
    out = lift.copy()
    for index in (outer, test):
        out[index] = lift[rng.permutation(index)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--k", type=int, nargs="+", default=[30])
    parser.add_argument("--arms", nargs="+",
                        default=["real", "sem-evidência", "embaralhado"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/multitask_logit_add_controls.csv")
    args = parser.parse_args()

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    blocks = [cohort.blocks[name] for name in cohort.blocks]
    space = np.hstack(blocks)
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"hand-8 {space.shape[1]}d | k={args.k} | braços={args.arms}")

    rows = pd.read_csv(args.output).to_dict("records") if args.output.exists() else []
    if not any(r["condition"] == "hand-8" for r in rows):
        base = pd.read_csv(BASE_CSV).query("condition == 'hand-8'")
        rows += [{"condition": "hand-8", "k": 0, "fold": int(r.fold),
                  "bacc": float(r.bacc), "lambda": "", "floor_frac": np.nan,
                  "elapsed_seconds": float(r.elapsed_seconds)}
                 for r in base.itertuples()]
        log(f"  braço hand-8 copiado de {BASE_CSV.name} ({len(base)} folds)")
    done = {(r["condition"], int(r["k"]), int(r["fold"])) for r in rows}

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), np.uint8), labels))
    lambda_index = np.zeros(n_classes, dtype=int)          # lambda global
    for k in args.k:
        for fold, (outer, test) in enumerate(splits, start=1):
            pending = [a for a in args.arms if (f"{a} k={k}", k, fold) not in done]
            if not pending:
                continue
            fit, validation = train_test_split(
                outer, test_size=0.10, stratify=labels[outer], random_state=SEED)
            started = time.perf_counter()
            scaler = StandardScaler().fit(space[outer])
            scaled = scaler.transform(space).astype(np.float32)
            oof, held = build_oof(scaled[outer], labels[outer], scaled[test],
                                  n_classes, k)
            lift = np.zeros((len(labels), n_classes), dtype=np.float32)
            lift[outer], lift[test] = oof, held
            floor = floor_vector(labels[outer], n_classes)
            # Quanto da matriz é o vetor constante: o número que motiva o controle.
            share = float(np.isclose(lift[np.r_[outer, test]], floor,
                                     atol=1e-3).mean())
            log(f"  k={k} fold {fold}: log-lift em {time.perf_counter()-started:.0f}s"
                f" | {100*share:.1f}% das coordenadas são 'sem evidência'")

            variants = {"real": lift,
                        "sem-evidência": np.broadcast_to(
                            floor, lift.shape).astype(np.float32),
                        "embaralhado": shuffled(lift, outer, test, fold)}
            for arm in pending:
                started = time.perf_counter()
                score, lam = train_arm(blocks, variants[arm], labels, fit,
                                       validation, test, n_classes, lambda_index,
                                       SEED, args.epochs, args.patience)
                rows.append({"condition": f"{arm} k={k}", "k": k, "fold": fold,
                             "bacc": score, "floor_frac": share,
                             "lambda": ";".join(f"{v:.4f}" for v in lam),
                             "elapsed_seconds": time.perf_counter() - started})
                pd.DataFrame(rows).to_csv(args.output, index=False,
                                          float_format="%.6f")
                log(f"    {arm:<14} k={k} fold {fold} | BAcc {score:5.2f} | "
                    f"λ={rows[-1]['lambda']} | {rows[-1]['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    reference = table[table.condition == "hand-8"].set_index("fold")["bacc"]
    log("\n=== marginal sobre hand-8 (pareado por fold) ===")
    log(f"  {'hand-8':<24} {reference.mean():6.2f} ± {reference.std(ddof=1):4.2f}")
    for name, group in table[table.condition != "hand-8"].groupby("condition"):
        current = group.set_index("fold")["bacc"]
        line = f"  {name:<24} {current.mean():6.2f} ± {current.std(ddof=1):4.2f}"
        if len(current) == N_FOLDS:
            delta = current - reference
            line += (f"   Δ {delta.mean():+5.2f} ± {delta.std(ddof=1):4.2f}pp"
                     f"  p={stats.ttest_rel(current, reference).pvalue:.4f}"
                     f"  {int((delta > 0).sum())}/{N_FOLDS}"
                     f"  λ={group['lambda'].astype(float).mean():.4f}")
        log(line)
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
