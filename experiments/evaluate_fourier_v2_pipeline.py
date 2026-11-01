"""Fourier v2 no pipeline, sob a sonda do §7.19. Troca SO o bloco Fourier.

O §5.17c mediu a v2 no `multitask` e a deixou FORA do artigo, com razao
declarada: *"toda a varredura daquela secao foi medida na parametrizacao da v1 ...
adota-la exigiria refazer a varredura inteira, e o artigo esta fechado"*. A
decisao do usuario em 07/out reabre isso -- com a regra do §7.19 (LDA dentro da
familia hand-designed) as tabelas vao ser refeitas de qualquer forma, entao o
custo marginal de trocar o bloco deixa de ser o que bloqueava.

**Custo de extracao ZERO.** A v2 nao le imagem: o `build_fourier_v2_features.py`
colapsa a MESMA grade log-polar 12x12 da v1 em estatisticas radiais, de
orientacao e por banda. Conferido no `rijksdraw`, e reproduz o multitask:

    v1   152 colunas   18 mortas   134 vivas
    v2    81 colunas    0 mortas    81 vivas

**Por que o rijksdraw e o corpus certo para perguntar.** E onde o `hand-8`
extrai menos relativo ao acaso dos sete (5,01x, §5.19) e onde ele nao ganha do
WND-CHARM em nenhuma celula de LDA. O §5.15 mede que **68 das 128 piores
features do banco sao Fourier** (de um bloco de 152d), e o §5.17c mede que peso
morto machuca MAIS no pipeline que sozinho (231% de transferencia) -- o
mecanismo e contavel: o `BranchMLP` manda todo bloco para 128 dims, entao
coluna constante gasta capacidade do ramo. Sob LDA o mecanismo e outro (coluna
morta e descartada antes, `viva = std > 0`), entao o efeito esperado aqui e
MENOR que no MLP, nao maior.

Bracos: hand-8 com Fourier v1 (o controle, tem de reproduzir 45,56 do §5.19),
hand-8 com Fourier v2, e os dois blocos SOZINHOS -- o §5.9d/§5.17c medem que o
isolado nao preve o pipeline, entao ter os dois lados e o que torna o resultado
legivel.

Uso:
    python experiments/evaluate_fourier_v2_pipeline.py --dataset rijksdraw
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                                      # noqa: E402
from registry import load_blocks                                # noqa: E402
from evaluate_handcrafted_banks_lda import fit_probe            # noqa: E402

SEED, N_FOLDS = 42, 5


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="rijksdraw")
    ap.add_argument("--variante", default="shrinkage", choices=["svd", "shrinkage"])
    ap.add_argument("--v2", type=Path, default=None)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fourier_v2_pipeline.csv")
    args = ap.parse_args()
    ds = args.dataset
    v2_path = args.v2 or ROOT / f"results/fourier_v2_{ds}_features.parquet"

    c = load_blocks(ds, list(HAND8))
    stems = list(c.stems)
    y = LabelEncoder().fit_transform(np.asarray(c.labels, dtype=str))
    C = len(np.unique(y))
    nomes = list(c.columns)
    if "Fourier" not in nomes:
        raise SystemExit(f"{ds}: sem bloco Fourier no hand-8 ({nomes})")

    # v2 alinhado por STEM, nunca por posicao (§4.1f item 7)
    f2 = pd.read_parquet(v2_path)
    f2["stem"] = f2["stem"].astype(str)
    cols = [k for k in f2.columns if k not in ("stem", "style")]
    at = {s: i for i, s in enumerate(f2["stem"])}
    falta = [s for s in stems if s not in at]
    if falta:
        raise SystemExit(f"{ds}: {len(falta)} stems da coorte sem Fourier v2")
    V2 = f2[cols].to_numpy(dtype=np.float32)[np.array([at[s] for s in stems])]

    v1 = c.blocks["Fourier"]
    log(f"{ds}: {len(stems):,} imagens | {C} classes | "
        f"Fourier v1 {v1.shape[1]}d -> v2 {V2.shape[1]}d")

    def troca(bloco) -> list[np.ndarray]:
        return [bloco if n == "Fourier" else c.blocks[n] for n in nomes]

    condicoes = {
        "hand-8 (Fourier v1)": np.hstack(troca(v1)),
        "hand-8 (Fourier v2)": np.hstack(troca(V2)),
        "Fourier v1 sozinho": v1,
        "Fourier v2 sozinho": V2,
    }
    for k, v in condicoes.items():
        log(f"  {k:<22} {v.shape[1]:>4}d")

    rows = pd.read_csv(args.output).to_dict("records") if args.output.exists() else []
    done = {(r["dataset"], r["condicao"], int(r["fold"])) for r in rows}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        for nome, X in condicoes.items():
            if (ds, nome, fold) in done:
                continue
            t0 = time.perf_counter()
            m, _, _ = fit_probe(X, y, tr, te, args.variante, C)
            rows.append({"dataset": ds, "condicao": nome, "fold": fold,
                         "variante": args.variante, "dims": X.shape[1],
                         "seconds": time.perf_counter() - t0, **m})
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
            log(f"  f{fold} {nome:<22} {m['dims_vivas']:>4} vivas  "
                f"BAcc {m['bacc']:5.2f}  Acc {m['accuracy']:5.2f}"
                f"  ({rows[-1]['seconds']:.0f}s)")

    t = pd.DataFrame(rows)
    t = t[t.dataset.eq(ds) & t.variante.eq(args.variante)]
    log(f"\n=== {ds}, {args.variante}: v2 menos v1, pareado por fold ===")
    for par in [("hand-8 (Fourier v1)", "hand-8 (Fourier v2)"),
                ("Fourier v1 sozinho", "Fourier v2 sozinho")]:
        a = t[t.condicao.eq(par[0])].set_index("fold")
        b = t[t.condicao.eq(par[1])].set_index("fold")
        if not len(a) or not len(b):
            continue
        linha = f"{par[0].replace(' v1','').replace(' (Fourier)',''):<22}"
        for met in ("bacc", "accuracy", "macro_f1"):
            x, yy = a[met].align(b[met], join="inner")
            d = yy - x
            linha += (f"{met[:4]} {x.mean():5.2f}->{yy.mean():5.2f} "
                      f"{d.mean():+5.2f} ({(d>0).sum()}/{len(d)})   ")
        log(linha)
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
