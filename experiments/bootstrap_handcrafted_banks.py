"""Incerteza da BAcc nos corpora de SPLIT FIXO, por reamostragem do teste.

Nos quatro nao-artisticos a particao e publicada e o LDA e fechado nas duas
variantes, entao repetir a execucao da numero bit-a-bit identico -- conferido em
07/out (hand-8/nct_crc reproduz 86,16 e 85,71). Logo **nao existe barra entre
execucoes nem entre sementes**, e a matriz de §banks saia com numero pelado ao
lado de tres corpora de CV que tinham `+-`. A unica fonte de incerteza que resta
e a amostra de TESTE, e e ela que se reamostra aqui.

**A reamostragem e ESTRATIFICADA POR CLASSE, e isso nao e detalhe.** BAcc e a
media dos recalls por classe, e os `n_c` do teste sao FIXADOS pelo protocolo
publicado (250/classe no minc2500 e no food101, 100 no places_lt, desiguais mas
fixos na coorte externa do nct_crc). Um bootstrap livre sortearia `n_c`
diferente a cada repeticao e somaria uma variancia que o protocolo nao tem; o
estratificado reproduz exatamente a variancia binomial de cada recall, que e o
estimando certo.

Comparacao entre bancos e PAREADA: as mesmas linhas reamostradas nos dois
lados, analogo direto de parear por fold nos corpora de CV. E o pareamento e
por STEM, nunca por posicao (§4.1f item 7).

**AS TRES METRICAS SAEM DA MATRIZ DE CONFUSAO de cada reamostragem**, nao de
tres contas independentes. O §5.19 mede que a BAcc inverte de sinal contra o
WND-CHARM em 2 de 6 corpora, entao dar IC so para ela deixaria de fora
exatamente as colunas onde a comparacao muda de dono. Derivar tudo de uma CM
por repeticao e O(N) com `bincount` e torna as tres mutuamente consistentes:

  recall_c   = CM[c,c] / n_c                   BAcc = media dos recalls
  Acc        = soma(diag) / N
  precis_c   = CM[c,c] / soma(CM[:,c])         F1 macro = media dos F1_c

Convencao do produtor preservada: `zero_division=0`, ou seja classe sem nenhuma
predicao entra com F1_c = 0.

Ressalva da macro-F1 sob estratificacao: a precisao depende da distribuicao
PREDITA, que varia entre repeticoes, enquanto os `n_c` do teste ficam fixos.
Isso e o desenho publicado sendo respeitado, nao um descuido -- mas e por isso
que o IC da F1 nao e intercambiavel com o da BAcc.

Uso:
    python experiments/bootstrap_handcrafted_banks.py --variante shrinkage
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFICIAIS = ["nct_crc", "minc2500", "food101", "places_lt"]
BANCOS = ["Haralick", "HOG", "WND-CHARM", "hand-8"]
REF = "hand-8"


def log(m: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def carregar(pred_dir: Path, ds: str, banco: str, var: str):
    slug = re.sub(r"[^0-9A-Za-z]+", "_", banco).strip("_")
    p = pred_dir / f"{ds}__{slug}__{var}__s42_f1.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["pred"], z["truth"], z["stems"].astype(str)


def metricas(pred_por_classe, C: int, tomar=None) -> tuple[float, float, float]:
    """(BAcc, Acc, macro-F1) a partir da matriz de confusao da reamostragem.

    `pred_por_classe[c]` sao as predicoes das linhas cuja classe VERDADEIRA e c;
    `tomar[c]` indexa dentro delas (sem ele, e o ponto observado). A CM sai de
    um `bincount` por classe, entao o custo e O(N) por repeticao.
    """
    cm = np.zeros((C, C), dtype=np.int64)
    for c, pc in enumerate(pred_por_classe):
        linha = pc if tomar is None else pc[tomar[c]]
        cm[c] = np.bincount(linha, minlength=C)
    diag = np.diag(cm).astype(float)
    n_c = cm.sum(axis=1).astype(float)
    preditos = cm.sum(axis=0).astype(float)
    recall = np.divide(diag, n_c, out=np.zeros(C), where=n_c > 0)
    precis = np.divide(diag, preditos, out=np.zeros(C), where=preditos > 0)
    soma = recall + precis
    f1 = np.divide(2 * recall * precis, soma, out=np.zeros(C), where=soma > 0)
    vivas = n_c > 0
    return (100 * float(recall[vivas].mean()),
            100 * float(diag.sum() / n_c.sum()),
            100 * float(f1.mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/banks_lda_pred")
    ap.add_argument("--variante", nargs="+", default=["shrinkage", "svd"])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/handcrafted_banks_bootstrap.csv")
    args = ap.parse_args()

    linhas, pares = [], []
    for var in args.variante:
        for ds in OFICIAIS:
            dados = {}
            for b in BANCOS:
                got = carregar(args.pred_dir, ds, b, var)
                if got is not None:
                    dados[b] = got
            if REF not in dados:
                log(f"!! {ds}/{var}: sem {REF} em disco -- pulando")
                continue

            # alinhamento por STEM contra a referencia, e aborta na divergencia
            stems_ref = dados[REF][2]
            ordem = {s: i for i, s in enumerate(stems_ref)}
            alinhado = {}
            for b, (pred, truth, stems) in dados.items():
                if set(stems) != set(stems_ref):
                    raise SystemExit(f"{ds}/{var}/{b}: stems divergem de {REF}")
                tomar = np.array([ordem[s] for s in stems])
                inv = np.empty_like(tomar); inv[tomar] = np.arange(len(tomar))
                p2, t2 = pred[inv], truth[inv]
                if not np.array_equal(t2, dados[REF][1]):
                    raise SystemExit(f"{ds}/{var}/{b}: rotulos divergem apos alinhar")
                alinhado[b] = p2
            truth = dados[REF][1]
            classes = np.unique(truth)
            C = int(classes.max()) + 1
            idx = [np.flatnonzero(truth == c) for c in classes]
            n_c = np.array([len(i) for i in idx])

            # predicoes agrupadas pela classe VERDADEIRA: e o que a CM consome
            por_classe = {b: [alinhado[b][i] for i in idx] for b in alinhado}

            # GUARDA: os TRES pontos recalculados do npz tem de bater com o CSV
            # publicado. Sem isso um npz de outra celula (ou fora de ordem)
            # passaria na checagem de tamanho e daria IC em cima do objeto
            # errado, calado -- o modo de falha do §5.18b. Conferir as tres, e
            # nao so a BAcc, e o que pega troca de metrica (§4.1f).
            pub = pd.read_csv(ROOT / f"results/{ds}_handcrafted_banks_lda.csv")
            pub = pub[pub.variante.eq(var)].set_index("banco")
            ponto = {}
            for b in por_classe:
                ponto[b] = metricas(por_classe[b], C)
                if b not in pub.index:
                    continue
                for k, col in enumerate(("bacc", "accuracy", "macro_f1")):
                    if abs(ponto[b][k] - float(pub.loc[b, col])) > 0.01:
                        raise SystemExit(
                            f"{ds}/{var}/{b}/{col}: npz da {ponto[b][k]:.4f} e o "
                            f"CSV publicado da {float(pub.loc[b, col]):.4f}")

            rng = np.random.default_rng(args.seed)
            amostras = {b: np.empty((args.reps, 3)) for b in por_classe}
            for r in range(args.reps):
                tomar = [rng.integers(0, n, n) for n in n_c]
                for b in por_classe:
                    amostras[b][r] = metricas(por_classe[b], C, tomar)

            presentes = [b for b in BANCOS if b in por_classe]
            for b in presentes:
                reg = {"dataset": ds, "variante": var, "banco": b,
                       "n_teste": int(n_c.sum()), "classes": len(classes),
                       "reps": args.reps}
                for k, met in enumerate(("bacc", "accuracy", "macro_f1")):
                    col = amostras[b][:, k]
                    lo, hi = np.percentile(col, [2.5, 97.5])
                    reg |= {met: ponto[b][k], f"{met}_se": col.std(ddof=1),
                            f"{met}_lo": lo, f"{met}_hi": hi}
                linhas.append(reg)
                if b == REF:
                    continue
                reg = {"dataset": ds, "variante": var, "contra": b}
                for k, met in enumerate(("bacc", "accuracy", "macro_f1")):
                    d = amostras[REF][:, k] - amostras[b][:, k]
                    dlo, dhi = np.percentile(d, [2.5, 97.5])
                    reg |= {f"d_{met}": ponto[REF][k] - ponto[b][k],
                            f"d_{met}_se": d.std(ddof=1),
                            f"d_{met}_lo": dlo, f"d_{met}_hi": dhi,
                            f"d_{met}_frac": float((d > 0).mean())}
                pares.append(reg)
            log(f"{ds:<11} {var:<10} n={n_c.sum():,} C={len(classes)}  |  "
                + "   ".join(f"{b} {ponto[b][0]:.2f}+-{amostras[b][:,0].std(ddof=1):.2f}"
                             for b in presentes))

    if not linhas:
        raise SystemExit("nada carregado -- rodar o produtor com --pred-dir primeiro")
    pd.DataFrame(linhas).to_csv(args.output, index=False, float_format="%.4f")
    alvo2 = args.output.with_name(args.output.stem + "_pares.csv")
    pd.DataFrame(pares).to_csv(alvo2, index=False, float_format="%.4f")
    log(f"\n{args.output}\n{alvo2}")


if __name__ == "__main__":
    main()
