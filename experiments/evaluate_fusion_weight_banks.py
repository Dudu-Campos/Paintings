#!/usr/bin/env python3
"""Tres regras de PESO no par PRISM x Fisher vector, nos sete corpora.

Pergunta do usuario em 13/out: *"acha que vale a pena testar com peso
a/(a+b)?"* -- peso proporcional a acuracia de cada lado, a forma fechada da
familia de Kuncheva. Ela entra junto com o cross-fit varrido porque sozinha o
numero nao e interpretavel: **e o cross-fit que da o TETO para o que qualquer
regra de peso compra neste par**, e o `a/(a+b)` so diz quanto desse teto uma
forma fechada sem busca captura.

PREVISAO REGISTRADA ANTES DE RODAR: nulo. Tres razoes, as duas primeiras ja
medidas em outro par.

1. **O teto e pequeno por construcao.** `a/(a+b)` afasta o peso de 0,5 em no
   maximo 0,057 (places_lt) neste par, e o §4.1f item 2 mede a recuperacao
   proporcional a `|w* - 0,5|` com rho=+0,955 -- o menor caso la, oilpainting
   com 0,035, valeu +0,12pp. A forma e LIMITADA: com os dois lados na mesma
   faixa ela satura perto de 0,5.
2. **No par com BACKBONE, onde o `w*` otimo FOI medido, `a/(a+b)` erra por
   0,103 em media e 0,216 no pior caso -- e erra a DIRECAO no rijksdraw**
   (medido 0,705, formula 0,489). Aquele e justamente o corpus que o §4.1f usa
   para estabelecer que o que governa e a DISTANCIA a 0,5 e nao a direcao, e
   uma razao de acuracias nao tem como ver isso.
3. **O §4.1f item 7 mede que o peso NAO e funcao da acuracia de nenhum dos dois
   lados isoladamente** -- `w* x nossa acuracia` rho=+0,186 (p=0,52),
   `w* x acuracia do backbone` rho=-0,300 (p=0,30). Só a diferenca (rho=-0,833)
   e a folga do oraculo (+0,738) preveem. `a/(a+b)` e funcao exatamente das duas
   quantidades que nao preveem.

**A OUTRA forma fechada, `w propto log-odds` -- a de Bates & Granger (1969) que
o §4.1g cita --, esta QUEBRADA aqui e por isso nao e rodada.** Ha um polo: o
peso normalizado `la/(la+lb)` tem o denominador cruzando zero quando um lado
passa de 50% e o outro nao. Medido neste par: o `minc2500`, com 3,3pp de gap,
e jogado para **w=0,807**; e em 4 de 7 corpora os dois logits sao NEGATIVOS
(food101, places_lt, multitask, oilpainting, todos com BAcc < 50%) e o `w` so
cai em [0,1] por acidente de sinal. `a/(a+b)` e a robusta das duas.

**PROVENIENCIA DO `a` E DO `b`, que e o ponto serio.** Tirados do teste, o
numero e circular -- e a linha "varrido no teste" que o §4.1f marca como *o teto
que espia*. Aqui eles saem das **dobras retidas**, na MESMA particao do
cross-fit, entao as duas regras sao pareadas e a comparacao nao carrega ruido de
particao. Isso tambem significa que `a/(a+b)` **nao e livre de parametro**: ele
exige duas estimativas de acuracia de algum lugar, que e a objecao que o §4.1f
item 6 levantou contra o peso de Kuncheva.

**O QUE ISTO NAO AUTORIZA: trocar o 0,5 da tabela.** Neste par o peso fixo
**nao quebra em lugar nenhum** -- 21 de 21 celulas positivas, 100% das unidades
(§5.23) --, ao contrario do par com backbone, onde ele quebrava em 3 de 7 e era
esse contraste que tornava o cross-fit um achado. Aqui nao ha o que resgatar: o
valor da medicao e **defensivo**, fechar a objecao que um revisor levanta ao ver
peso 0,5, no precedente do temperature scaling (§4.1f item 5).

Nota de metrica (§7.12): o peso e selecionado por **BAcc** nas tres regras, e as
colunas de `Acc`/`macro-F1` sao as da predicao que esse peso produz -- NAO sao a
celula otima daquelas metricas. Declarar qual celula do 2x2 esta sendo reportada
nao e formalidade nesta linha.

O BRACO COM TEXTO, e a pergunta e do usuario em 13/out: *"faz sentido os
valores serem proximos no cenario sem texto, mas com texto eles sao bem
diferentes?"* -- esta certo, e expoe que a primeira execucao cobriu justamente o
braco em que nada podia acontecer. O texto move SO O NOSSO LADO (+0,01 a
+20,18pp, §5.18e) e nao toca no FV, entao ele ABRE o gap, e e o gap que da
trabalho ao peso:

    |w - 0,5| de a/(a+b)      SEM texto    COM texto
      food101                    0,025        0,126     5,0x
      minc2500                   0,016        0,055     3,5x
      places_lt                  0,057        0,170     3,0x
      oilpainting                0,018        0,005     0,3x   <- INVERTE
      multitask                  0,011        0,001     0,1x   <- INVERTE

**PREVISAO REGISTRADA para o braco com texto, antes de rodar.** Pelo mecanismo
do §4.1f item 2 (recuperacao proporcional a `|w* - 0,5|`, rho=+0,955), e pela
calibragem do par com backbone -- onde `|w*-0,5|` de 0,170-0,205 valeu +0,93 a
+1,03pp:

1. **Agora deve aparecer recuperacao**, concentrada em `places_lt`, `food101` e
   `minc2500`, na ordem do `|w-0,5|`. Ordem de grandeza esperada: decimos de
   ponto a ~1pp, nao os +2,05 do MINC contra backbone.
2. **E deve DESAPARECER no `multitask` e no `oilpainting`**, onde o texto leva o
   nosso lado para CIMA do FV e o par fica quase perfeitamente equilibrado
   (gap +0,16 e +0,96). Ali 0,5 fica ainda mais otimo do que ja era.
3. **A celula que decide e o `places_lt`**: e o unico lugar do par inteiro em
   que o peso fixo DEMONSTRAVELMENTE para de pagar (+0,25 com 3/5 em BAcc e 2/5
   em macro-F1, §5.23). Se uma das duas regras a resgata, o item 2 do §4.1f
   replica num par independente -- backbone trocado por um segundo banco
   hand-designed -- e deixa de ser propriedade da fusao com CNN.

Se a previsao 1 falhar, o achado nao e sobre `a/(a+b)`: e que neste par o peso
fixo e otimo em QUALQUER regime de gap, o que seria mais forte e precisaria de
mecanismo proprio.

Nenhum treino: aritmetica sobre posteriors em disco, alinhamento por STEM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

# FONTE UNICA dos caminhos: o proprio produtor da fusao publicada. O §2 proibe
# tabela nova de caminhos, e o §7quinquies registra um caso em que um segundo
# mapa paralelo ao registry sobreviveu sem ninguem notar.
from evaluate_decision_fusion_banks_all import CFG, SEEDS, EPS  # noqa: E402

GRADE = np.round(np.arange(0.05, 1.0, 0.05), 2)
PUBLICADO = ROOT / "results/decision_fusion_banks_all.csv"
MET = (("BAcc", balanced_accuracy_score), ("Acc", accuracy_score),
       ("F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)))


def log(m: str) -> None:
    print(m, flush=True)


def mistura(P: np.ndarray, B: np.ndarray, w: float) -> np.ndarray:
    return w * np.log(P + EPS) + (1 - w) * np.log(B + EPS)


def bacc(y: np.ndarray, q: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, q))


def lados(ds: str, braco: str):
    """Os dois lados por unidade (fold, semente), na mesma logica do produtor.

    Reproduz `evaluate_decision_fusion_banks_all.unidades` na parte de carga --
    o §5.6 registra que um braco de controle que nao reproduz o valor conhecido
    nao e controle, e a guarda de regressao no fim deste script confere isso
    contra o numero publicado em vez de confiar na leitura do codigo.
    """
    semf, comf, pad, folds = CFG[ds]
    cache = semf if braco == "sem" else comf
    z = np.load(ROOT / f"outputs/{cache}.npz", allow_pickle=True)
    pos = {s: i for i, s in enumerate(np.asarray(z["stems"], dtype=str))}
    yall = np.asarray(z["y_true"])
    for k in folds:
        for sd in SEEDS:
            p = ROOT / f"outputs/banks_mlp_pred/{pad.format(k=k, sd=sd)}.npz"
            if not p.exists():
                continue
            f = np.load(p, allow_pickle=True)
            st = np.asarray(f["stems"], dtype=str)
            y = np.asarray(f["truth"])
            falta = [s for s in st if s not in pos]
            if falta:
                raise SystemExit(f"{ds}: {len(falta)} stems do FV fora do cache do PRISM")
            idx = np.array([pos[s] for s in st])
            if not np.array_equal(yall[idx], y):
                raise SystemExit(f"{ds} f{k} s{sd}: rotulos divergem depois de alinhar")
            yield k, sd, np.asarray(z[f"seed_{sd}"], np.float64)[idx], \
                np.asarray(f["probs"], np.float64), y


def unidade(ds: str, braco: str, k: int, sd: int, P: np.ndarray,
            B: np.ndarray, y: np.ndarray, folds: int) -> dict:
    """As quatro condicoes de peso, as tres honestas na MESMA particao.

    Usar a mesma particao para o cross-fit e para o `a/(a+b)` torna as duas
    regras PAREADAS: a diferenca entre elas e de regra, nao de sorteio de dobra.
    """
    r: dict = {"dataset": ds, "braco": braco, "fold": k, "seed": sd,
               "n": len(y), "prism": 100 * bacc(y, P.argmax(1)),
               "fv": 100 * bacc(y, B.argmax(1))}

    # o teto que ESPIA o teste -- diagnostico, nao reportavel (§4.1f)
    espia = max(GRADE, key=lambda w: bacc(y, mistura(P, B, w).argmax(1)))

    pred = {"meio": mistura(P, B, 0.5).argmax(1),
            "espia": mistura(P, B, espia).argmax(1),
            "cross": np.empty_like(y), "razao": np.empty_like(y)}
    wc, wr = [], []
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                 random_state=42).split(np.zeros(len(y)), y):
        w = max(GRADE, key=lambda w: bacc(y[tr], mistura(P[tr], B[tr], w).argmax(1)))
        # a e b nas dobras RETIDAS, nunca na avaliada
        a = bacc(y[tr], P[tr].argmax(1))
        b = bacc(y[tr], B[tr].argmax(1))
        wz = a / (a + b) if a + b > 0 else 0.5
        wc.append(w)
        wr.append(wz)
        pred["cross"][te] = mistura(P[te], B[te], w).argmax(1)
        pred["razao"][te] = mistura(P[te], B[te], wz).argmax(1)

    for cond, q in pred.items():
        for nome, fn in MET:
            r[f"{cond}_{nome}"] = 100 * fn(y, q)
    r.update(w_espia=espia,
             w_cross=float(np.mean(wc)), w_cross_sd=float(np.std(wc, ddof=1)),
             w_razao=float(np.mean(wr)), w_razao_sd=float(np.std(wr, ddof=1)))
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CFG))
    ap.add_argument("--bracos", nargs="+", default=["sem"], choices=["sem", "com"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="divergencia maxima em pp contra a fusao publicada")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_banks.csv")
    args = ap.parse_args()

    linhas: list[dict] = []
    feito: set[tuple] = set()
    # A retomada tem de tolerar CSV VAZIO: o §1 exige saida incremental
    # IDEMPOTENTE, e um `read_csv` que levanta `EmptyDataError` num arquivo de
    # 0 bytes quebra a retomada no unico caso em que ela mais importa -- depois
    # de um job morto antes da primeira gravacao. E no NFS o log pode aparecer
    # vazio por cache, entao a falha fica invisivel (§8).
    if args.output.exists() and args.output.stat().st_size:
        try:
            linhas = pd.read_csv(args.output).to_dict("records")
        except pd.errors.EmptyDataError:
            log(f"! {args.output.name} sem colunas -- comecando do zero")
        feito = {(r["dataset"], r["braco"], int(r["fold"]), int(r["seed"]))
                 for r in linhas}
        log(f"retomando: {len(linhas)} unidades em {args.output.name}")

    for braco in args.bracos:
        for ds in args.datasets:
            n0 = len(linhas)
            for k, sd, P, B, y in lados(ds, braco):
                if (ds, braco, k, sd) in feito:
                    continue
                linhas.append(unidade(ds, braco, k, sd, P, B, y, args.folds))
                pd.DataFrame(linhas).to_csv(args.output, index=False,
                                            float_format="%.4f")
            log(f"  {ds}/{braco}: {len(linhas) - n0} unidades novas")

    t = pd.DataFrame(linhas)

    # --- GUARDA DE REGRESSAO: o w=0,5 daqui tem de ser a fusao publicada.
    # Sem ela, um erro na carga ou no alinhamento produziria quatro condicoes
    # internamente consistentes e todas erradas -- o modo de falha do §8.
    if PUBLICADO.exists():
        pub = pd.read_csv(PUBLICADO)
        pub = pub[pub.metrica.eq("BAcc")].set_index(["dataset", "braco"])
        pior, onde = 0.0, ""
        for (ds, br), g in t.groupby(["dataset", "braco"]):
            if (ds, br) not in pub.index:
                continue
            d = abs(g.meio_BAcc.mean() - pub.fusao[(ds, br)])
            if d > pior:
                pior, onde = d, f"{ds}/{br}"
        log(f"\nguarda: maior divergencia contra a fusao publicada "
            f"{pior:.4f}pp ({onde})")
        if pior > args.tol:
            raise SystemExit(f"ABORTADO: {onde} divergiu {pior:.4f}pp > {args.tol}")

    for braco in args.bracos:
        b = t[t.braco.eq(braco)]
        if not len(b):
            continue
        log("\n" + "=" * 108)
        log(f"PESO no par PRISM x Fisher vector -- braco {braco} texto"
            f"   (BAcc; o peso e selecionado por BAcc nas tres regras)")
        log("=" * 108)
        log(f"{'corpus':12s} {'PRISM':>6s} {'FV':>6s} | {'w=0,5':>6s} "
            f"{'a/(a+b)':>8s} {'cross':>7s} {'espia':>7s} | "
            f"{'razao-meio':>11s} {'cross-meio':>11s} | "
            f"{'w razao':>8s} {'w cross':>8s} {'u':>3s}")
        resumo = []
        for ds in args.datasets:
            g = b[b.dataset.eq(ds)]
            if not len(g):
                continue
            dr = g.razao_BAcc - g.meio_BAcc
            dc = g.cross_BAcc - g.meio_BAcc
            pr = stats.ttest_rel(g.razao_BAcc, g.meio_BAcc)[1] if len(g) > 1 else np.nan
            pc = stats.ttest_rel(g.cross_BAcc, g.meio_BAcc)[1] if len(g) > 1 else np.nan
            log(f"{ds:12s} {g.prism.mean():6.2f} {g.fv.mean():6.2f} | "
                f"{g.meio_BAcc.mean():6.2f} {g.razao_BAcc.mean():8.2f} "
                f"{g.cross_BAcc.mean():7.2f} {g.espia_BAcc.mean():7.2f} | "
                f"{dr.mean():+6.2f}({int((dr > 0).sum())}/{len(g)}) "
                f"{dc.mean():+6.2f}({int((dc > 0).sum())}/{len(g)}) | "
                f"{g.w_razao.mean():8.3f} {g.w_cross.mean():8.3f} {len(g):3d}")
            resumo.append({"dataset": ds, "d_razao": dr.mean(), "p_razao": pr,
                           "d_cross": dc.mean(), "p_cross": pc,
                           "w_razao": g.w_razao.mean(), "w_cross": g.w_cross.mean(),
                           "dist_razao": abs(g.w_razao.mean() - 0.5),
                           "dist_cross": abs(g.w_cross.mean() - 0.5)})
        s = pd.DataFrame(resumo)
        log(f"\nmedia sobre os {len(s)} corpora:  a/(a+b) {s.d_razao.mean():+.3f}pp"
            f"   cross-fit {s.d_cross.mean():+.3f}pp")
        log(f"corpora em que a regra GANHA do peso fixo:  "
            f"a/(a+b) {int((s.d_razao > 0).sum())}/{len(s)}   "
            f"cross-fit {int((s.d_cross > 0).sum())}/{len(s)}")
        if len(s) > 2:
            log("\nO MECANISMO do §4.1f item 2 -- |w - 0,5| x recuperacao:")
            for nome, col in (("a/(a+b)", "dist_razao"), ("cross-fit", "dist_cross")):
                d = "d_razao" if "a/" in nome else "d_cross"
                rho, p = stats.spearmanr(s[col], s[d])
                log(f"  {nome:9s} rho={rho:+.3f}  p={p:.4f}"
                    f"   |w-0,5| de {s[col].min():.3f} a {s[col].max():.3f}")
            log("\nquanto do cross-fit a forma FECHADA captura:")
            frac = s[s.d_cross.abs() > 1e-9]
            if len(frac):
                log(f"  mediana {np.median(frac.d_razao / frac.d_cross):+.2f}"
                    f"   por corpus: " + "  ".join(
                        f"{r.dataset[:6]} {r.d_razao / r.d_cross:+.2f}"
                        for r in frac.itertuples()))

    log(f"\n[{args.output}]")


if __name__ == "__main__":
    main()
