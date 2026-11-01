#!/usr/bin/env python3
"""Fusao de DECISAO entre dois bancos: media de log-probs com peso FIXO 0,5.

Nos moldes do §4.1f -- zero parametros ajustados. Existe porque a fusao de
FEATURES nao converteu a complementaridade medida: no rijksdraw o `hand-8` e o
Fisher vector tem IoU de acerto de 0,4815 (contra 0,3344 sob independencia),
**12,8% das imagens so o hand-8 acerta** e o oraculo da uniao e 68,5% contra os
~55 de cada um -- mas concatenar deu +0,81pp sobre o FV (p=0,23). O mecanismo
suspeito e o do §4.1c: 41.984 dimensoes contra 617 na mesma cabeca, e o ramo
maior domina.

**O peso 0,5 e escolha CONSERVADORA declarada**, nao o melhor estimador: o
§4.1f mede que escolher o peso fora da dobra avaliada recupera +0,12 a +2,05pp.
Ele esta aqui porque e o que nao tem parametro nenhum.

**O LADO NOSSO TEM DE SER O DE PRODUCAO**, e a primeira versao deste script
errou isso: fundiu o `hand_8_denso_128` (51,47 no fold 1 do rijksdraw), que e o
braco de bloco UNICO que existe so para ser justo com o WND-CHARM (§5.19b). O
`hand-8 + grafo` sob `BranchMLP` faz **56,62** nas mesmas linhas -- 5,15pp que
estavam sendo jogados fora, e e ele que esta nas tabelas.

`--oof-producao` injeta esse lado a partir de `outputs/<ds>_our_oof_cv.npz`, o
cache do `evaluate_decision_fusion_cv.py`. **A particao e a mesma**: aquele OOF
e CV de 5 folds com `SEED=42` fixo, entao as linhas de teste do fold 1 foram
preditas por um modelo treinado nos folds 2-5 -- exatamente o treino do fold 1
que o Fisher vector usa. Conferido: 9.659 de 9.659 stems casam e os rotulos
batem depois de alinhar.

Alinhamento por STEM (§4.1f item 7). Pareado por semente.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def do_disco(f: Path) -> dict | None:
    """Uma unidade (semente x fold) de um banco gravado por --pred-dir."""
    z = np.load(f, allow_pickle=True)
    if "probs" not in z:
        return None
    return {"stems": np.asarray(z["stems"], dtype=str),
            "truth": np.asarray(z["truth"]),
            "probs": np.asarray(z["probs"], dtype=np.float64)}


def do_oof(npz: Path, seeds: list[int], ref: np.ndarray) -> list[dict]:
    """O lado de PRODUCAO, recortado nas linhas de teste do fold avaliado.

    O cache guarda a coorte inteira em OOF (`seed_NN`, uma coluna por classe) e
    `ref` sao os stems do fold; o recorte mantem o protocolo porque cada linha
    do OOF veio de um modelo que nao a viu.
    """
    z = np.load(npz, allow_pickle=True)
    pos = {s: i for i, s in enumerate(np.asarray(z["stems"], dtype=str))}
    falta = [s for s in ref if s not in pos]
    if falta:
        raise SystemExit(f"{len(falta)} stems do fold ausentes em {npz.name}")
    idx = np.array([pos[s] for s in ref])
    y = np.asarray(z["y_true"])[idx]
    saida = []
    for sd in seeds:
        k = f"seed_{sd}"
        if k not in z.files:
            raise SystemExit(f"{npz.name} nao tem {k}")
        saida.append({"stems": ref, "truth": y,
                      "probs": np.asarray(z[k], dtype=np.float64)[idx]})
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/banks_mlp_pred")
    ap.add_argument("--dataset", default="rijksdraw")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--peso", type=float, default=0.5)
    ap.add_argument("--oof-producao", nargs="?", const="auto", default=None,
                    help="injeta `hand-8 + grafo` (BranchMLP) como lado nosso, "
                         "de outputs/<ds>_our_oof_cv.npz; 'auto' resolve o caminho")
    ap.add_argument("--rotulo-producao", default="hand_8+grafo PRODUCAO")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                    help="sementes do OOF de producao, na ordem dos arquivos de disco")
    args = ap.parse_args()

    bruto: dict[str, list[Path]] = {}
    for f in sorted(args.pred_dir.glob(f"{args.dataset}__*_f{args.fold}.npz")):
        bruto.setdefault(f.name.split("__")[1].rsplit("__", 1)[0], []).append(f)

    por: dict[str, list[dict]] = {}
    for nome, fs in bruto.items():
        uns = [do_disco(f) for f in sorted(fs)]
        if any(u is None for u in uns):
            print(f"{nome}: sem posteriors em disco -- pulando")
            continue
        por[nome] = uns

    if args.oof_producao:
        if not por:
            raise SystemExit("nenhum banco em disco para dar os stems de referencia")
        npz = (ROOT / f"outputs/{args.dataset}_our_oof_cv.npz"
               if args.oof_producao == "auto" else Path(args.oof_producao))
        ref = next(iter(por.values()))[0]["stems"]
        por[args.rotulo_producao] = do_oof(npz, args.seeds, ref)
        print(f"lado de producao: {npz.name}  {len(ref)} linhas  "
              f"{len(args.seeds)} sementes")

    nomes = sorted(por)
    met = (("BAcc", balanced_accuracy_score), ("Acc", accuracy_score),
           ("macro-F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)))

    for x, y in itertools.combinations(nomes, 2):
        linhas = {k: [] for k in ("A", "B", "F", "D")}
        for zx, zy in zip(por[x], por[y]):
            sx = zx["stems"]
            onde = {s: i for i, s in enumerate(zy["stems"])}
            falta = [s for s in sx if s not in onde]
            if falta:
                raise SystemExit(f"{len(falta)} stems sem par entre {x} e {y}")
            idx = np.array([onde[s] for s in sx])
            t = zx["truth"]
            if not np.array_equal(t, zy["truth"][idx]):
                raise SystemExit("rotulos divergem depois de alinhar -- abortando")
            pa, pb = zx["probs"], zy["probs"][idx]
            lf = args.peso * np.log(pa + EPS) + (1 - args.peso) * np.log(pb + EPS)
            ax, bx, fx_ = pa.argmax(1), pb.argmax(1), lf.argmax(1)
            for k, pred in (("A", ax), ("B", bx), ("F", fx_)):
                linhas[k].append([f(t, pred) * 100 for _, f in met])
            ca, cb = ax == t, bx == t
            linhas["D"].append([100 * v.mean() for v in (
                ca & cb, ca & ~cb, ~ca & cb, ~ca & ~cb, ca | cb)])
        if not linhas["A"]:
            continue
        A, B, F = (np.array(linhas[k]) for k in ("A", "B", "F"))
        print(f"\n{x}  x  {y}   (peso {args.peso}, {len(A)} sementes)")
        print(f"{'':28s} " + "  ".join(f"{n:>14s}" for n, _ in met))
        for rot, M in ((x[:26], A), (y[:26], B), ("FUSAO w=0,5", F)):
            print(f"{rot:28s} " + "  ".join(
                f"{M[:, i].mean():7.2f} +-{M[:, i].std(ddof=1):4.2f}" for i in range(3)))
        for rot, M in ((f"sobre {x[:18]}", A), (f"sobre {y[:18]}", B)):
            cel = []
            for i, (n, _) in enumerate(met):
                dif = F[:, i] - M[:, i]
                p = stats.ttest_rel(F[:, i], M[:, i])[1] if len(F) > 1 else np.nan
                cel.append(f"{n} {dif.mean():+6.2f} (p={p:.2f}, {(dif>0).sum()}/{len(F)})")
            print(f"  {rot:26s} " + "  ".join(cel))
        D = np.array(linhas["D"]).mean(0)
        print(f"  {'por imagem (acuracia)':26s} ambos {D[0]:5.2f}  "
              f"so {x[:14]} {D[1]:5.2f}  so {y[:14]} {D[2]:5.2f}  "
              f"nenhum {D[3]:5.2f}  ORACULO {D[4]:5.2f}")


if __name__ == "__main__":
    main()
