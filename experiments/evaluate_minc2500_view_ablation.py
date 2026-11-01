"""Ablacao por view do banco hand-designed, em QUALQUER corpus do registry.

**O grafo de similaridade e o canal de texto NAO entram, por decisao de 06/out.**
Os dois sao blocos tipo-posterior (as colunas ja sao evidencia de classe) e o
§7.19 mede que a sonda linear os favorece: razao `LDA/MLP` 1,005 no grafo contra
mediana ~0,75 das views que MEDEM a imagem. Deixa-los de fora remove a excecao da
legenda em vez de declara-la -- e `hand-8` ja e, por definicao, as 8 views sem
grafo (617d). O texto fica fora tambem por ser supervisao externa (§7.2).

Sonda: **LDA**, pela regra do §7.19 -- dentro da familia hand-designed a sonda
linear e o instrumento validado (§5.9d: ordenacao conservada nos cinco
classificadores, `sozinha (LDA) x Shapley (LDA)` = +1,000), e ela nao atravessa
para embedding aprendido, onde a escolha da sonda vira parte do resultado.

Protocolo por corpus, igual ao do resto do projeto (§7.5, §9.1):
  * split PUBLICADO onde existe (minc2500, places_lt, food101): 1 particao
  * coorte de teste SEPARADA no nct_crc (`nct_crc_val`, pacientes diferentes, §7.14)
  * 5 folds de CV nos tres de arte (multitask, oilpainting, rijksdraw)

Original: MINC-2500 per-view ablation on the official split.

Runs two complementary probes for each hand-designed view:

* single/view: the view alone
* drop/view: hand-8 with that view removed

Together with full hand-8, this separates raw signal from marginal utility.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_minc2500_official import HAND8, fit_lda, fit_mlp, append_row  # noqa: E402
from registry import load_blocks, log  # noqa: E402

SEED = 42
N_FOLDS = 5

# Corpora com particao publicada -- rodar CV neles e o bug que o §8 registra
# ("a constante OFFICIAL existia e nunca era lida").
OFFICIAL = {"minc2500", "places_lt", "food101", "nct_crc"}
TEST_COHORT = {"nct_crc": "nct_crc_val"}   # teste e coorte separada, nao fatia


def build_conditions() -> dict[str, tuple[str, ...]]:
    conditions: dict[str, tuple[str, ...]] = {"hand-8": HAND8}
    for name in HAND8:
        conditions[f"single/{name}"] = (name,)
    for name in HAND8:
        conditions[f"drop/{name}"] = tuple(other for other in HAND8 if other != name)
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="minc2500")
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED],
                        help="sementes da CV. Nos corpora de split PUBLICADO a "
                             "particao e fixa e o `fit_lda` e deterministico, "
                             "entao semente extra da numero IDENTICO -- o script "
                             "ignora as demais e avisa. O `1 particao x 5 "
                             "sementes` do §9.1 e protocolo de MLP.")
    parser.add_argument("--classifiers", nargs="+", default=["lda"], choices=["lda", "mlp"])
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Optional condition names; defaults to hand-8, all singles, and all leave-one-out drops.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.output = args.output or ROOT / f"results/{args.dataset}_view_ablation.csv"

    all_conditions = build_conditions()
    selected = args.conditions if args.conditions else list(all_conditions)
    unknown = [name for name in selected if name not in all_conditions]
    if unknown:
        raise KeyError(f"unknown conditions: {unknown}")

    ds = args.dataset
    cohort = load_blocks(ds, HAND8, allow_external=ds in TEST_COHORT.values())
    blocks_by_name = dict(cohort.blocks)
    labels_raw = np.asarray(cohort.labels, dtype=str)
    n_train_rows = None

    if ds in TEST_COHORT:
        # A coorte do registry e so TREINO (100.000 no nct_crc); o teste e a
        # coorte externa de pacientes diferentes, que vale 11pp (§7.14).
        held = load_blocks(TEST_COHORT[ds], HAND8, allow_external=True)
        n_train_rows = len(cohort.stems)
        blocks_by_name = {n: np.vstack([np.asarray(cohort.blocks[n]),
                                        np.asarray(held.blocks[n])]) for n in HAND8}
        labels_raw = np.concatenate([labels_raw, np.asarray(held.labels, dtype=str)])
        log(f"{ds}: coorte EXTERNA {TEST_COHORT[ds]} como teste "
            f"({n_train_rows:,} treino + {len(held.stems):,} teste)")

    labels = LabelEncoder().fit_transform(labels_raw)
    n_classes = len(np.unique(labels))

    if ds in TEST_COHORT:
        partitions = [(args.seeds[0], 1, np.arange(n_train_rows),
                       np.arange(n_train_rows, len(labels)))]
    elif ds in OFFICIAL:
        meta = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})[["stem", "split"]]
        where = (meta.set_index("stem")["split"]
                 .reindex([str(x) for x in cohort.stems]).to_numpy(str))
        if (where == "nan").any():
            raise ValueError(f"{ds}: {(where == 'nan').sum()} stems fora do metadado")
        partitions = [(args.seeds[0], 1, np.flatnonzero(where == "train"),
                       np.flatnonzero(where == "test"))]
        if (where == "val").any():
            log(f"  o `val` oficial ({(where == 'val').sum():,}) fica FORA do treino")
    else:
        # 5 folds x N sementes: a semente governa o embaralhamento do
        # StratifiedKFold, entao cada (semente, fold) e uma particao diferente de
        # verdade -- ao contrario do split publicado, onde ela nao tem o que mover.
        partitions = [(sd, f, tr, te) for sd in args.seeds
                      for f, (tr, te) in enumerate(
                          StratifiedKFold(N_FOLDS, shuffle=True, random_state=sd)
                          .split(np.zeros(len(labels)), labels), start=1)]

    # Imprimir os tamanhos e o antidoto do §8: o bug antigo sobreviveu porque
    # NADA na saida denunciava qual particao estava em uso.
    tipo = ("coorte externa" if ds in TEST_COHORT
            else "split OFICIAL" if ds in OFFICIAL else f"CV {N_FOLDS} folds")
    log(f"{ds} ablacao por view [{tipo}]: {len(partitions)} particao(oes), "
        f"treino={len(partitions[0][2]):,}, teste={len(partitions[0][3]):,}, "
        f"classes={n_classes}")
    if ds in OFFICIAL and len(args.seeds) > 1:
        log(f"  !! {len(args.seeds)} sementes pedidas, mas a particao e FIXA e o "
            f"`fit_lda` e deterministico: as demais dariam numero identico e um "
            f"desvio falso de 0,00. Rodando so a semente {args.seeds[0]}.")

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        # A chave inclui o FOLD: nos corpora de CV sao 5 particoes por condicao, e
        # a chave antiga (classifier, condition) pularia as quatro restantes.
        done = {(str(r.classifier), str(r.condition),
                 int(getattr(r, "seed", SEED)), int(getattr(r, "fold", 1)))
                for r in previous.itertuples()}
        log(f"retomando {len(done)} linhas de {args.output.name}")

    for seed, fold, train_idx, test_idx in partitions:
        # `val` para o MLP: no split oficial o `val` publicado NAO entra no treino
        # e o recorte de 10% sai do proprio treino, igual ao `evaluate_mlp` de
        # producao (§5.6: nao reimplementar o laco, reproduzir o recorte).
        fit_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, stratify=labels[train_idx],
            random_state=seed + fold)
        for classifier in args.classifiers:
            for condition in selected:
                if (classifier, condition, seed, fold) in done:
                    continue
                names = all_conditions[condition]
                blocks = [blocks_by_name[name] for name in names]
                started = time.perf_counter()
                if classifier == "lda":
                    metrics = fit_lda(blocks, labels, train_idx, test_idx)
                else:
                    metrics = fit_mlp(blocks, labels, fit_idx, val_idx, test_idx,
                                      n_classes, seed + fold, args.epochs, args.patience)
                row = {
                    "dataset": ds, "classifier": classifier,
                    "seed": seed, "fold": fold,
                    "condition": condition,
                    "view": condition.split("/", 1)[1] if "/" in condition else "all",
                    "kind": condition.split("/", 1)[0] if "/" in condition else "full",
                    "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)), "n_classes": n_classes,
                    "dimensions": int(sum(block.shape[1] for block in blocks)),
                    "seconds": time.perf_counter() - started,
                    **metrics,
                }
                append_row(args.output, row)
                log(f"s{seed} f{fold} {classifier:<3} | {condition:<22} | {row['dimensions']:4d}d | "
                    f"BAcc {100 * row['bacc']:.2f} | Acc {100 * row['accuracy']:.2f} | "
                    f"F1 {100 * row['macro_f1']:.2f} | {row['seconds']:.0f}s")

    detail = pd.read_csv(args.output)
    if "fold" not in detail:
        detail["fold"] = 1
    if "seed" not in detail:
        detail["seed"] = SEED
    detail["unidade"] = detail["seed"].astype(str) + "/" + detail["fold"].astype(str)
    rows = []
    for classifier, group in detail.groupby("classifier"):
        base = group[group["condition"] == "hand-8"].set_index("unidade")["bacc"]
        if base.empty:
            continue
        for (condition, view, kind), sub in group.groupby(
                ["condition", "view", "kind"], sort=False):
            b = sub.set_index("unidade")["bacc"]
            # PAREADO POR FOLD: a media das diferencas, nao a diferenca das medias.
            delta = (base - b).reindex(b.index).dropna() if kind == "drop" else None
            rows.append({
                "classifier": classifier, "condition": condition, "view": view,
                "kind": kind, "unidades": int(len(b)),
                "dimensions": int(sub["dimensions"].iloc[0]),
                "bacc": float(b.mean()), "bacc_sd": float(b.std()) if len(b) > 1 else np.nan,
                "accuracy": float(sub["accuracy"].mean()),
                "macro_f1": float(sub["macro_f1"].mean()),
                "contribution_vs_drop_pp": (float(delta.mean()) * 100
                                            if delta is not None and len(delta) else np.nan),
                "contribution_sd_pp": (float(delta.std()) * 100
                                       if delta is not None and len(delta) > 1 else np.nan),
            })
    summary = pd.DataFrame(rows)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    log(f"saved: {args.output}")
    log(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
