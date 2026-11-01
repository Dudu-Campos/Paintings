"""Quantizacao coluna a coluna em TODAS as views: quantos valores distintos
cada feature assume sobre a coorte inteira.

Generaliza o `audit_hvg_degeneracy.py`. O HVG tem 21 de 57 colunas com <=10
valores distintos em 99.816 imagens, e todas sao quantis de GRAU -- grau e
contagem, entao o quantil de uma distribuicao de inteiros trava num inteiro
pequeno. As views de grafo de regiao e de line graph tambem resumem
distribuicao de grau (`degree_max`, `degree_mean`, `degree_gini`,
`n_leaf_frac`, e os mesmos com prefixo `knn_`/`del_`/`mst_`), entao a mesma
degeneracao pode estar la em versao mais branda.

O §3 ja tinha pego a ponta extrema disso sem nomear o mecanismo: as duas
DEAD_COLUMNS de cada bloco de grafo sao `knn_n_leaf_frac` e `del_n_leaf_frac`,
e os 18 `log_power_r00_a**` do Fourier. Esta auditoria mede o gradiente inteiro
em vez de so o caso constante.

**Contagem de valores distintos, nao posto efetivo.** O `audit_hvg_degeneracy`
mede que o participation ratio do §3 da MAIOR para o bloco degenerado (PR/d
0,320 do HVG contra 0,142 do Hilbert) -- ruido de quantizacao espalha variancia
otimamente. O posto efetivo nao pega isto; a contagem pega.

Uso:
    python experiments/audit_view_quantization.py --dataset multitask
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from registry import load_blocks                       # noqa: E402

# As 8 de producao mais as duas dormentes que o §7quinquies manteve no
# principal, mais o Text. O grafo de similaridade fica FORA: e derivado por
# fold, nao esta em disco como bloco estatico.
VIEWS = ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
         "Hilbert", "Fourier", "Gabor"]
EXTRA = {
    "Random walk": ("outputs/{ds}_random_walk/rw_features.npy",
                    "outputs/{ds}_random_walk/rw_stems.npy"),
    "HVG": ("outputs/{ds}_hilbert_visibility/features.npy",
            "outputs/{ds}_hilbert_visibility/stems.npy"),
}


def column_stats(X: np.ndarray, names: list[str], block: str) -> pd.DataFrame:
    n_uniq = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
    std = X.std(axis=0)
    return pd.DataFrame({"bloco": block, "coluna": names,
                         "n_valores_distintos": n_uniq, "std": std,
                         "constante": std == 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or ROOT / "results" / f"{args.dataset}_view_quantization.csv"

    cohort = load_blocks(args.dataset, VIEWS)
    n = len(cohort.stems)
    frames = []
    for name in VIEWS:
        X = np.asarray(cohort.blocks[name], dtype=np.float64)
        cols = list(cohort.columns[name]) if hasattr(cohort, "columns") else None
        if not cols or len(cols) != X.shape[1]:
            cols = [f"{name}_{j}" for j in range(X.shape[1])]
        frames.append(column_stats(X, cols, name))

    for name, (feat_tmpl, _) in EXTRA.items():
        path = ROOT / feat_tmpl.format(ds=args.dataset)
        if not path.exists():
            print(f"  - {name}: {path} nao existe, pulando")
            continue
        X = np.load(path).astype(np.float64)
        names_path = path.parent / "feature_names.json"
        cols = None
        if names_path.exists():
            import json
            blob = json.loads(names_path.read_text())
            cols = blob if isinstance(blob, list) else blob.get("feature_names")
        if not cols or len(cols) != X.shape[1]:
            cols = [f"{name}_{j}" for j in range(X.shape[1])]
        frames.append(column_stats(X, cols, name))

    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out, index=False, float_format="%.6g")

    print(f"\n{args.dataset}: {n:,} imagens\n")
    print(f"{'bloco':14s} {'dims':>5s} {'const':>6s} {'<=10':>5s} {'<=100':>6s} "
          f"{'>10k':>5s}  {'% quantizado (<=100)':>20s}")
    for block in table["bloco"].unique():
        sub = table[table["bloco"] == block]
        nu = sub["n_valores_distintos"].to_numpy()
        d = len(nu)
        print(f"{block:14s} {d:5d} {int(sub['constante'].sum()):6d} "
              f"{int((nu <= 10).sum()):5d} {int((nu <= 100).sum()):6d} "
              f"{int((nu > 10000).sum()):5d}  {100 * (nu <= 100).mean():19.1f}%")

    worst = table[table["n_valores_distintos"] <= 100].sort_values("n_valores_distintos")
    print(f"\ncolunas com <=100 valores distintos ({len(worst)} no total):")
    for block in table["bloco"].unique():
        sub = worst[worst["bloco"] == block]
        if sub.empty:
            continue
        print(f"\n  {block}:")
        for _, r in sub.head(12).iterrows():
            print(f"    {r['coluna']:34s} {int(r['n_valores_distintos']):5d} valores")
        if len(sub) > 12:
            print(f"    ... e mais {len(sub) - 12}")

    print(f"\n{out}")


if __name__ == "__main__":
    main()
