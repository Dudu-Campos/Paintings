#!/usr/bin/env python3
"""Fusao de DECISAO de TRES lados: PRISM + Fisher vector + backbone fine-tuned.

A fusao de dois lados devolve um posterior por imagem, entao ela e um
classificador como outro qualquer e pode entrar numa segunda fusao. A pergunta
do usuario (12/out) e exatamente essa: *"depois da fusao temos um modelo certo?
entao a fusao ((Prism + fisher) + resnet) pode ser feita?"*

**A ESCOLHA DE PESO NAO E NEUTRA, e e por isso que o script mede as duas.**
Encadear `w=0,5` duas vezes NAO da um terco para cada lado:

    ((A + B) + C) com 0,5 em cada nivel   ->   0,25 A + 0,25 B + 0,50 C

ou seja o backbone recebe METADE do voto e cada banco hand-designed um quarto.
Isso e defensavel sob o enquadramento do §4.1f -- *o nosso lado contra o
backbone*, com o nosso lado sendo agora dois bancos --, mas e uma escolha, nao
uma consequencia. A alternativa simetrica e a media plana (1/3 cada), que
tambem tem zero parametro. As duas sao reportadas.

**GUARDA DE REGRESSAO, e ela e o que valida o caminho.** O par
`PRISM + backbone` com w=0,5 tem numero publicado no §4.1f, e no `food101` o
protocolo e identico (split oficial, as mesmas 25.250 linhas de teste, 5
sementes). O script o recomputa e ABORTA se divergir mais de `--tol` do
registrado. Sem isso nao ha como saber se os tres lados foram alinhados certo.

**§7.2: com o backbone os dois lados carregam supervisao externa.** A alegacao
muda de TIPO, nao de forca. A linha SEM backbone continua sendo a que sustenta
"zero supervisao externa" -- por isso ela e impressa junto, sempre.

Alinhamento por STEM e nunca por posicao (§4.1f item 7).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
GRID = ROOT / "outputs/final_finetune_grid"
SEEDS = [42, 43, 44, 45, 46]

# Numeros publicados do §4.1f (delta da fusao w=0,5 sobre o backbone sozinho),
# usados como guarda. Só o `food101` tem protocolo identico ao deste script --
# no `rijksdraw` o §4.1f particiona a coorte OOF inteira e aqui e so o fold 1.
PUBLICADO = {("food101", "resnet50"): -0.479, ("food101", "vit_b_16"): -1.286}

MET = (("BAcc", balanced_accuracy_score), ("Acc", accuracy_score),
       ("F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)))


def alinha(ref: np.ndarray, stems: np.ndarray, *arrays):
    """Reindexa `arrays` para a ordem de `ref`. Aborta se faltar stem."""
    onde = {s: i for i, s in enumerate(stems)}
    falta = [s for s in ref if s not in onde]
    if falta:
        raise SystemExit(f"{len(falta)} stems sem par (ex.: {falta[:3]})")
    idx = np.array([onde[s] for s in ref])
    return [a[idx] for a in arrays]


def backbone_oof(ds: str, modelo: str, sd: int, folds: int):
    """Monta o lado do backbone como OOF da coorte inteira, juntando os folds.

    Necessario nos corpora de CV: o trainer e o `evaluate_banks_dense_mlp.py`
    fazem `StratifiedKFold(SEED=42)` sobre ORDENS DE LINHA diferentes, entao o
    fold 1 de um nao e o fold 1 do outro -- medido no rijksdraw, 7.723 de 9.659
    stems sem par. Juntando os cinco, cada stem aparece exatamente uma vez e a
    predicao dele e out-of-sample pelo modelo que nao o viu.

    RESSALVA que tem de ir junto (§4.1f ressalva 2): as linhas vem de CINCO
    modelos diferentes, entao o estimando e "seguranca do procedimento sob OOF",
    nao sob particao fixa.
    """
    st, pr, yy = [], [], []
    for k in range(1, folds + 1):
        b = np.load(GRID / ds / modelo / f"seed_{sd}/fold_{k}/fold_{k:02d}_posteriors.npz",
                    allow_pickle=True)
        st.append(np.asarray(b["stems"], dtype=str))
        pr.append(np.asarray(b["probabilities"], np.float64))
        yy.append(np.asarray(b["y_true"]))
    st = np.concatenate(st)
    if len(set(st)) != len(st):
        raise SystemExit(f"{ds}/{modelo}/s{sd}: stem repetido ao juntar os folds")
    return st, np.concatenate(pr), np.concatenate(yy)


def carrega(ds: str, fv_banco: str, fold: int, oof: Path, modelo: str,
            bb_folds: int = 1):
    """Devolve (y, {semente: {lado: probs}}) com os tres lados ja alinhados."""
    z = np.load(oof, allow_pickle=True)
    oof_stems = np.asarray(z["stems"], dtype=str)
    por_semente, y_ref, ref = {}, None, None
    for sd in SEEDS:
        f = np.load(ROOT / f"outputs/banks_mlp_pred/{ds}__{fv_banco}__s{sd}_f{fold}.npz",
                    allow_pickle=True)
        if "probs" not in f.files:
            raise SystemExit(f"{ds}/{fv_banco}/s{sd}: sem posteriors -- rode com --pred-dir")
        if ref is None:
            ref = np.asarray(f["stems"], dtype=str)
            y_ref = np.asarray(f["truth"])
        fv, = alinha(ref, np.asarray(f["stems"], dtype=str),
                     np.asarray(f["probs"], np.float64))

        pr, = alinha(ref, oof_stems, np.asarray(z[f"seed_{sd}"], np.float64))
        yo, = alinha(ref, oof_stems, np.asarray(z["y_true"]))
        if not np.array_equal(yo, y_ref):
            raise SystemExit("rotulos do PRISM divergem depois de alinhar")

        if bb_folds > 1:
            bst, bpr, byy = backbone_oof(ds, modelo, sd, bb_folds)
        else:
            b = np.load(GRID / ds / modelo /
                        f"seed_{sd}/fold_{fold}/fold_{fold:02d}_posteriors.npz",
                        allow_pickle=True)
            bst = np.asarray(b["stems"], dtype=str)
            bpr = np.asarray(b["probabilities"], np.float64)
            byy = np.asarray(b["y_true"])
        bb, yb = alinha(ref, bst, bpr, byy)
        if not np.array_equal(yb, y_ref):
            raise SystemExit("rotulos do backbone divergem depois de alinhar")
        por_semente[sd] = {"PRISM": pr, "FV": fv, "backbone": bb}
    return y_ref, por_semente


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="food101")
    ap.add_argument("--fv-banco", default="Fisher_vector_pub_linear")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--oof", type=Path, default=None)
    ap.add_argument("--modelos", nargs="+", default=["resnet50", "vit_b_16"])
    ap.add_argument("--backbone-folds", type=int, default=1,
                    help="junta N folds do backbone num OOF da coorte inteira. "
                         "1 = split oficial (food101 etc.); 5 = corpora de CV, "
                         "onde as particoes dos dois lados NAO coincidem")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="divergencia maxima contra o numero publicado, em pp")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/decision_fusion_triple.csv")
    args = ap.parse_args()

    oof = args.oof or (ROOT / f"outputs/{args.dataset}_our_posteriors.npz")
    linhas = []
    for modelo in args.modelos:
        y, por = carrega(args.dataset, args.fv_banco, args.fold, oof, modelo,
                         args.backbone_folds)
        print(f"\n{'='*86}\n{args.dataset}  x  {modelo}   "
              f"({len(y)} linhas, {len(SEEDS)} sementes, fold {args.fold})\n{'='*86}")

        combos = {
            "PRISM":                       {"PRISM": 1.0},
            "Fisher vector":               {"FV": 1.0},
            "backbone ft":                 {"backbone": 1.0},
            "PRISM + FV":                  {"PRISM": .5, "FV": .5},
            "PRISM + backbone":            {"PRISM": .5, "backbone": .5},
            "FV + backbone":               {"FV": .5, "backbone": .5},
            "((PRISM+FV)+bb) 1/4,1/4,1/2": {"PRISM": .25, "FV": .25, "backbone": .5},
            "os tres, plano 1/3":          {"PRISM": 1/3, "FV": 1/3, "backbone": 1/3},
        }
        res = {k: [] for k in combos}
        dec = []
        for sd in SEEDS:
            L = {k: np.log(v + EPS) for k, v in por[sd].items()}
            preds = {}
            for nome, w in combos.items():
                preds[nome] = sum(p * L[k] for k, p in w.items()).argmax(1)
                res[nome].append([100 * fn(y, preds[nome]) for _, fn in MET])
            c = {k: preds[k] == y for k in ("PRISM", "Fisher vector", "backbone ft")}
            dec.append([100 * v.mean() for v in (
                c["PRISM"] & c["Fisher vector"] & c["backbone ft"],
                c["PRISM"] & ~c["backbone ft"],
                c["Fisher vector"] & ~c["backbone ft"],
                (c["PRISM"] | c["Fisher vector"]) & ~c["backbone ft"],
                c["PRISM"] | c["Fisher vector"] | c["backbone ft"])])
        M = {k: np.array(v) for k, v in res.items()}

        print(f"{'':30s} " + "  ".join(f"{n:>13s}" for n, _ in MET))
        for k in combos:
            print(f"{k:30s} " + "  ".join(
                f"{M[k][:, i].mean():6.2f} +-{M[k][:, i].std(ddof=1):4.2f}" for i in range(3)))

        # ---- guarda de regressao contra o §4.1f -----------------------------
        chave = (args.dataset, modelo)
        if chave in PUBLICADO:
            d = (M["PRISM + backbone"][:, 0] - M["backbone ft"][:, 0]).mean()
            dif = abs(d - PUBLICADO[chave])
            marca = "OK" if dif <= args.tol else "DIVERGE"
            print(f"\n  guarda §4.1f  PRISM+backbone w=0,5: medido {d:+.3f}  "
                  f"publicado {PUBLICADO[chave]:+.3f}  dif {dif:.3f}pp  [{marca}]")
            if dif > args.tol:
                raise SystemExit("guarda de regressao falhou -- alinhamento suspeito")

        print()
        for k in ("((PRISM+FV)+bb) 1/4,1/4,1/2", "os tres, plano 1/3"):
            for base in ("backbone ft", "PRISM + backbone", "PRISM + FV"):
                cel = []
                for i, (n, _) in enumerate(MET):
                    dd = M[k][:, i] - M[base][:, i]
                    p = stats.ttest_rel(M[k][:, i], M[base][:, i])[1]
                    cel.append(f"{n} {dd.mean():+6.2f} (p={p:.3f}, {(dd > 0).sum()}/5)")
                print(f"  {k[:26]:26s} vs {base[:17]:17s} " + "  ".join(cel))
            print()

        D = np.array(dec).mean(0)
        print(f"  por imagem (acuracia): os tres acertam {D[0]:5.2f}  |  do que o "
              f"backbone ERRA, so PRISM {D[1]:5.2f}  so FV {D[2]:5.2f}  "
              f"um dos dois {D[3]:5.2f}  |  oraculo dos tres {D[4]:5.2f}")

        for k in combos:
            for i, (n, _) in enumerate(MET):
                linhas.append({"dataset": args.dataset, "modelo": modelo,
                               "combinacao": k, "metrica": n,
                               "valor": M[k][:, i].mean(), "sd": M[k][:, i].std(ddof=1),
                               "n_sementes": len(SEEDS)})

    out = pd.DataFrame(linhas)
    if args.output.exists():
        velho = pd.read_csv(args.output)
        out = pd.concat([velho, out]).drop_duplicates(
            subset=["dataset", "modelo", "combinacao", "metrica"], keep="last")
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n[gravado em {args.output}]")


if __name__ == "__main__":
    main()
