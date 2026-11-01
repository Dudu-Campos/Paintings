"""O WND-CHARM data o desenho pelo DESENHO ou pelo SUPORTE? (07/out)

Pergunta do usuario: *"acha que pode haver um shortcut que o wdn esta se
aproveitando por ter features mais locais?"*

A primeira metade da hipotese e VERDADEIRA POR CONSTRUCAO e esta declarada no
extrator: o protocolo canonico de Shamir et al. (2010) padroniza a imagem em
600x600 e calcula as 2.895 features em **16 tiles de 150x150**, agregando por
media. O `hand-8` nao tem pooling espacial nenhum (§7.11b). Entao o WND-CHARM
tem acesso a estatistica LOCAL que o nosso banco nao tem.

A segunda metade e a que importa, e no `rijksdraw` ela tem um candidato fisico
concreto: o rotulo e PERIODO (1500-2000, §5.19) e papel DATA. Papel vergé feito
a mao tem linhas de forma e vergaturas visiveis; papel velino de maquina chega
~1800 -- justamente onde estao as maiores classes do corpus. Foxing, amarelamento
e textura de trama sao propriedades LOCAIS do suporte, homogeneas pela folha, e
um tile de 150x150 numa margem em branco as mede limpas. Isso e datacao
legitima em historia da arte E e atalho para "classificar o DESENHO".

O TESTE SAI DE GRACA porque `features.npy` guarda `(n, 16, 2895)` -- os tiles
nunca foram descartados. Num desenho enquadrado, os tiles de CANTO sao
majoritariamente papel e os CENTRAIS tem traco:

    grade 4x4, indice = linha*4 + coluna
    cantos  0, 3, 12, 15      borda 1,2,4,7,8,11,13,14      centro 5,6,9,10

Se os cantos predizem periodo tao bem quanto o centro, o sinal esta no suporte.
Se os cantos desabam, esta no desenho.

**O NULO IMPORTA e nao e o acaso teorico** (§5.18b): quatro tiles de canto sao
1/4 dos dados de 16, entao parte de qualquer queda e orcamento, nao posicao. O
controle pareado e **quatro tiles ALEATORIOS por imagem**, mesmo orcamento,
mesma dimensionalidade.

Sonda: LDA shrinkage, mesmos 5 folds do §5.19, para ser comparavel celula a
celula com a tabela dos quatro bancos.

Uso:
    python experiments/evaluate_wndcharm_tile_shortcut.py --dataset rijksdraw
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
from registry import load_blocks                               # noqa: E402
from evaluate_handcrafted_banks_lda import fit_probe           # noqa: E402

SEED, N_FOLDS = 42, 5
# grade 4x4 do artigo: indice = linha*4 + coluna
CANTOS = [0, 3, 12, 15]
BORDA = [1, 2, 4, 7, 8, 11, 13, 14]
CENTRO = [5, 6, 9, 10]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="rijksdraw")
    ap.add_argument("--variante", default="shrinkage", choices=["svd", "shrinkage"])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/wndcharm_tile_shortcut.csv")
    args = ap.parse_args()
    ds = args.dataset

    c = load_blocks(ds, list(HAND8))
    stems = list(c.stems)
    y = LabelEncoder().fit_transform(np.asarray(c.labels, dtype=str))
    C = len(np.unique(y))

    tiled = ROOT / "outputs" / f"{ds}_wndcharm_tiled"
    bruto = np.load(tiled / "features.npy", mmap_mode="r")
    t_stems = np.load(tiled / "stems.npy", allow_pickle=True).astype(str)
    at = {s: i for i, s in enumerate(t_stems)}
    falta = [s for s in stems if s not in at]
    if falta:
        raise SystemExit(f"{ds}: {len(falta)} stems da coorte sem tile")
    take = np.array([at[s] for s in stems])
    log(f"{ds}: {len(stems):,} imagens | tiles {bruto.shape} | {C} classes")

    # media dos tiles de um subconjunto de posicoes, em blocos (8,9 GB no disco)
    def pool(pos: list[int], rng=None) -> np.ndarray:
        out = np.empty((len(take), bruto.shape[2]), dtype=np.float32)
        for a in range(0, len(take), 2048):
            idx = take[a:a + 2048]
            ordem = np.argsort(idx)
            bloco = np.asarray(bruto[np.sort(idx)], dtype=np.float32)
            if rng is None:
                sel = bloco[:, pos, :]
            else:  # quatro posicoes ALEATORIAS por imagem -- o nulo pareado
                escolha = np.array([rng.choice(16, len(pos), replace=False)
                                    for _ in range(len(bloco))])
                sel = np.take_along_axis(bloco, escolha[:, :, None], axis=1)
            out[a:a + len(idx)] = sel.mean(axis=1)[np.argsort(ordem)]
        return out

    rng = np.random.default_rng(SEED)
    condicoes = {
        "16 tiles (producao)": pool(list(range(16))),
        "4 cantos": pool(CANTOS),
        "4 centro": pool(CENTRO),
        "4 aleatorios (nulo)": pool(CANTOS, rng=rng),
        "8 borda": pool(BORDA),
    }
    for k, v in condicoes.items():
        log(f"  {k:<22} {v.shape[1]}d")

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
                         "variante": args.variante, "tiles": len(CANTOS)
                         if nome.startswith(("4 ", "4")) else
                         (8 if nome.startswith("8") else 16),
                         "seconds": time.perf_counter() - t0, **m})
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
            log(f"  f{fold} {nome:<22} BAcc {m['bacc']:5.2f}  Acc {m['accuracy']:5.2f}"
                f"  ({rows[-1]['seconds']:.0f}s)")

    t = pd.DataFrame(rows)
    t = t[t.dataset.eq(ds)]
    log(f"\n=== {ds}: o periodo esta no DESENHO ou no SUPORTE? ===")
    base = t[t.condicao.eq("16 tiles (producao)")].set_index("fold")
    for nome in condicoes:
        g = t[t.condicao.eq(nome)].set_index("fold")
        if not len(g):
            continue
        d = (g.bacc - base.bacc).dropna()
        log(f"{nome:<22} BAcc {g.bacc.mean():5.2f} +-{g.bacc.std():.2f}   "
            f"Acc {g.accuracy.mean():5.2f}   vs 16 tiles {d.mean():+5.2f}")
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
