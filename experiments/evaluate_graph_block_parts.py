#!/usr/bin/env python3
"""A DIFUSAO se paga? O bloco de grafo tem as duas metades, nunca separadas.

Pergunta do usuario (12/out): "chegamos a rodar so com as node features sem a
difusao?". Nao -- mas o bloco de `2C + 32` ja contem as duas:

    [0 : C)      log-lift do posterior DIFUNDIDO (LabelSpreading, alpha=0,4,
                 100 iteracoes), corrigido por prior
    [C : 2C)     weighted_hist -- o voto kNN ponderado por similaridade, SEM
                 difusao nenhuma. E a node feature crua.
    [2C : 2C+32) 32 estatisticas de vizinhanca

A pergunta importa por CUSTO: o §5.4b mede que o grafo e o unico termo
super-linear do pipeline (beta=1,73) e que a producao paga SEIS construcoes por
fold; a difusao e a parte cara. Se o voto cru entregar o mesmo, o termo
super-linear encolhe.

Le os blocos ja cacheados do Shapley (`outputs/shapley_hand8_grafo/{ds}/`) --
nao recalcula grafo nenhum. Sonda LDA, que e o que o §7.19 autoriza aqui.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                                   # noqa: E402
from registry import load_blocks                            # noqa: E402
from evaluate_handcrafted_banks_lda import OFFICIAL         # noqa: E402
from models import BranchMLP                                # noqa: E402
from protocol import evaluate_mlp                            # noqa: E402

SEED, N_FOLDS = 42, 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["rijksdraw", "oilpainting", "multitask",
                             "minc2500", "places_lt", "food101"])
    ap.add_argument("--cache-root", type=Path,
                    default=ROOT / "outputs/shapley_hand8_grafo")
    ap.add_argument("--sonda", default="lda", choices=["lda", "mlp"],
                    help="mlp = BranchMLP de producao, com o pedaco de grafo "
                         "entrando como NONO BLOCO (nao concatenado ao hand-8): "
                         "e assim que a producao o usa, e a primeira camada "
                         "bloco-diagonal e o que o §5.16 mede valendo +0,93pp.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--condicoes", nargs="+", default=None)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/graph_block_parts.csv")
    args = ap.parse_args()

    linhas = []
    # RETOMADA, e ela existe porque a falta dela quase custou o minc2500: o
    # shell chama este script UMA VEZ POR CORPUS, entao sem reler o CSV o
    # `to_csv` sobrescreve o arquivo com so as linhas desta execucao. Perda
    # silenciosa -- nada na saida denunciava (§8).
    if args.output.exists():
        velho = pd.read_csv(args.output)
        linhas.extend(velho.to_dict("records"))
        print(f"[retomada] {len(velho)} linhas de "
              f"{sorted(velho.dataset.unique())} preservadas", flush=True)
    for ds in args.datasets:
        coh = load_blocks(ds, list(HAND8))
        stems = [str(s) for s in coh.stems]
        y = LabelEncoder().fit_transform(np.asarray(coh.labels, dtype=str))
        C = len(np.unique(y))
        hand = np.hstack([coh.blocks[n] for n in HAND8]).astype(np.float32)
        if ds in OFFICIAL:
            meta = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})
            onde = meta.set_index("stem")["split"].reindex(stems).to_numpy(str)
            parts = [(np.flatnonzero(onde == "train"), np.flatnonzero(onde == "test"))]
        else:
            parts = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                         .split(np.zeros(len(y)), y))
        print(f"\n=== {ds} ===  {C} classes, {len(parts)} particao(oes)")
        for k, (tr, te) in enumerate(parts, 1):
            cache = args.cache_root / ds / f"fold_{k}.npz"
            if not cache.exists():
                print(f"  fold {k}: sem cache -- pulando"); continue
            g = np.load(cache)["graph"].astype(np.float32)
            if g.shape[1] != 2 * C + 32:
                raise SystemExit(f"{ds}: bloco {g.shape[1]}d, esperado {2*C+32}")
            h8 = [np.asarray(coh.blocks[n], dtype=np.float32) for n in HAND8]
            pedaco = {
                "hand-8": None,
                "+ difundido (log-lift)": g[:, :C],
                "+ voto kNN (SEM difusao)": g[:, C:2 * C],
                "+ os dois, sem stats": g[:, :2 * C],
                "+ grafo completo": g,
            }
            if args.condicoes:
                pedaco = {k: v for k, v in pedaco.items() if k in args.condicoes}
            if args.sonda == "mlp":
                for nome, extra in pedaco.items():
                    blocos = h8 if extra is None else h8 + [extra]
                    dims = sum(b.shape[1] for b in blocos)
                    for sd in args.seeds:
                        chave = (sd - SEED) * 10 + k
                        m = evaluate_mlp(blocos, y, tr, te, chave, 150, 15,
                                         label_smoothing=0.05,
                                         model_factory=BranchMLP)
                        linhas.append({"dataset": ds, "condicao": nome, "fold": k,
                                       "seed": sd, "dims": dims,
                                       "bacc": 100 * m["bacc"],
                                       "accuracy": 100 * m["accuracy"],
                                       "macro_f1": 100 * m["macro_f1"]})
                        print(f"  f{k} s{sd} {nome[:26]:26s} {dims:5d}d "
                              f"BAcc {100*m['bacc']:.2f}", flush=True)
                        pd.DataFrame(linhas).drop_duplicates(
                            subset=["dataset", "condicao", "fold", "seed"],
                            keep="last").to_csv(args.output, index=False,
                                                float_format="%.4f")
                continue
            cond = {n: (hand if v is None else np.hstack([hand, v]))
                    for n, v in pedaco.items()}
            for nome, X in cond.items():
                sc = StandardScaler().fit(X[tr])
                a, b = sc.transform(X[tr]), sc.transform(X[te])
                viva = a.std(axis=0) > 0
                m = LDA(solver="lsqr", shrinkage="auto",
                        priors=np.ones(C) / C).fit(a[:, viva], y[tr])
                bacc = 100 * balanced_accuracy_score(y[te], m.predict(b[:, viva]))
                linhas.append({"dataset": ds, "condicao": nome, "fold": k,
                               "seed": SEED, "dims": X.shape[1], "bacc": bacc})
            d = pd.DataFrame(linhas)
            d = d[(d.dataset == ds) & (d.fold == k)]
            print(f"  fold {k}: " + "  ".join(
                f"{r.condicao.split('(')[0].strip()[:14]} {r.bacc:.2f}" for r in d.itertuples()))
    out = pd.DataFrame(linhas).drop_duplicates(
        subset=["dataset", "condicao", "fold", "seed"], keep="last")
    out.to_csv(args.output, index=False, float_format="%.4f")
    try:                                    # --output relativo crashava aqui
        rot = args.output.resolve().relative_to(ROOT)
    except ValueError:
        rot = args.output
    print(f"\n[gravado em {rot}]")
    print(out.groupby(["dataset", "condicao"]).bacc.mean().round(2).to_string())


if __name__ == "__main__":
    main()
