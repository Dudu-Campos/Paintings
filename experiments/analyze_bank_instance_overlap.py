#!/usr/bin/env python3
"""IoU dos conjuntos de ACERTO entre bancos: eles acertam as mesmas imagens?

Pergunta do usuario (12/out). Tres quantidades, e as tres sao necessarias:

1. **IoU** = |A∩B| / |A∪B| sobre os acertos. Responde a pergunta literal.
2. **O NULO.** Dois modelos com aquelas recalls ja se sobrepoem MUITO por acaso
   -- o §4.1f item 4 registra que reportar teto sem nulo e o erro especular do
   §5.18b. O nulo preserva a recall POR CLASSE dos dois lados e destroi so o
   pareamento de QUAIS imagens cada um acerta; sob independencia dentro da
   classe, P(ambos) = r_A * r_B, e dai sai o IoU esperado em forma fechada.
3. **Q de Yule macro**, que e invariante a escala de linha e coluna e portanto
   compara corpora com acuracias diferentes. O `q_macro` (media por classe
   verdadeira) e usado porque o Q poolado INFLA com o numero de classes
   (rho=+0,656 medido neste projeto).

Alinhamento por STEM, nunca por posicao (§4.1f item 7).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def carrega(p: Path):
    z = np.load(p, allow_pickle=True)
    return (np.asarray(z["stems"], dtype=str), np.asarray(z["pred"]),
            np.asarray(z["truth"]))


def q_yule(a: np.ndarray, b: np.ndarray) -> float:
    n11 = float((a & b).sum()); n10 = float((a & ~b).sum())
    n01 = float((~a & b).sum()); n00 = float((~a & ~b).sum())
    den = n11 * n00 + n10 * n01
    return (n11 * n00 - n10 * n01) / den if den > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/banks_mlp_pred")
    ap.add_argument("--dataset", default="rijksdraw")
    ap.add_argument("--fold", type=int, default=1)
    args = ap.parse_args()

    por_banco: dict[str, list[Path]] = {}
    for f in sorted(args.pred_dir.glob(f"{args.dataset}__*_f{args.fold}.npz")):
        nome = f.name.split("__")[1].rsplit("__", 1)[0]
        por_banco.setdefault(nome, []).append(f)
    nomes = sorted(por_banco)
    print(f"{args.dataset}, fold {args.fold} -- {len(nomes)} bancos, "
          f"{len(por_banco[nomes[0]])} sementes\n")

    for x, y in itertools.combinations(nomes, 2):
        ious, nulos, qs, dec = [], [], [], []
        for fx, fy in zip(sorted(por_banco[x]), sorted(por_banco[y])):
            sx, px, tx = carrega(fx)
            sy, py, ty = carrega(fy)
            # ALINHAR POR STEM
            onde = {s: i for i, s in enumerate(sy)}
            falta = [s for s in sx if s not in onde]
            if falta:
                raise SystemExit(f"{len(falta)} stems sem par entre {x} e {y}")
            idx = np.array([onde[s] for s in sx])
            py, ty = py[idx], ty[idx]
            if not np.array_equal(tx, ty):
                raise SystemExit("os rotulos divergem depois de alinhar -- abortando")
            A, B = px == tx, py == tx
            ious.append((A & B).sum() / max(1, (A | B).sum()))
            qs.append(np.nanmean([q_yule(A[tx == c], B[tx == c])
                                  for c in np.unique(tx) if (tx == c).sum() > 1]))
            dec.append(((A & B).mean(), (A & ~B).mean(), (~A & B).mean(), (~A & ~B).mean()))
            # NULO: preserva a recall POR CLASSE e destroi o pareamento
            inter = esp_uni = 0.0
            for c in np.unique(tx):
                m = tx == c
                ra, rb = A[m].mean(), B[m].mean()
                inter += m.sum() * ra * rb
                esp_uni += m.sum() * (ra + rb - ra * rb)
            nulos.append(inter / max(1e-9, esp_uni))
        d = np.array(dec).mean(axis=0)
        print(f"{x}\n  x {y}")
        print(f"    IoU dos acertos     {np.mean(ious):.4f} +- {np.std(ious, ddof=1):.4f}")
        print(f"    IoU sob INDEPENDENCIA {np.mean(nulos):.4f}   "
              f"excesso {np.mean(ious)-np.mean(nulos):+.4f}")
        print(f"    Q de Yule macro     {np.mean(qs):.4f}")
        print(f"    ambos {d[0]*100:5.1f}%   so {x[:16]} {d[1]*100:5.1f}%   "
              f"so {y[:16]} {d[2]*100:5.1f}%   nenhum {d[3]*100:5.1f}%")
        print(f"    oraculo da uniao    {(d[0]+d[1]+d[2])*100:5.1f}%\n")


if __name__ == "__main__":
    main()
