"""A fusao de log-probs e dominada pelo lado MAIS AFIADO, nao pelo mais certo.

Pergunta do usuario (21/set): *"temos que fazer uma normalizacao porque cnns sao
over confident ne?"*. Sim, e o lugar importa: o argmax de UM modelo e invariante
a temperatura, entao a calibracao so muda alguma coisa **porque estamos somando
os dois**. `w*log P_a + (1-w)*log P_b` -- se P_b tem entropia muito menor, ele
decide o argmax mesmo com w=0,5, e o peso acaba fazendo dois trabalhos ao mesmo
tempo: dizer em quem confiar E corrigir a diferenca de nitidez.

PREVISAO REGISTRADA ANTES DE RODAR: se `w*` se afasta de 0,5 sobretudo para
compensar nitidez, calibrar deve (a) empurrar `w*` para perto de 0,5 e (b)
fazer o peso fixo parar de quebrar. Se `w*` continuar longe depois de calibrar,
ele carrega confianca de verdade e a leitura do §4.1f fica como esta.

Tudo com a MESMA particao em dobras do cross-fit, entao a garantia nao muda:
temperatura e peso saem das dobras que nao contem a imagem avaliada.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_fusion_weight_crossfit import (CORPORA, CV_CORPORA, WEIGHTS,  # noqa: E402
                                             backbone_oof, blend, log)


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: |confianca - acerto| ponderado por ocupacao."""
    conf, pred = prob.max(1), prob.argmax(1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return 100 * total


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    """T que minimiza a NLL. T>1 amacia (o modelo estava confiante demais)."""
    def nll(log_t: float) -> float:
        z = logit / np.exp(log_t)
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())
    best = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(best.x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--with-text", action="store_true")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_temperature.csv")
    args = ap.parse_args()
    sufixo = "_text" if args.with_text else ""

    rows = []
    for corpus in list(CORPORA) + list(CV_CORPORA):
        cv = corpus in CV_CORPORA
        path = ROOT / (f"outputs/{corpus}_our_" + ("oof_cv" if cv else "posteriors")
                       + sufixo + ".npz")
        if not path.exists():
            log(f"  {corpus}: sem {path.name}, pulando")
            continue
        z = np.load(path, allow_pickle=True)
        stems, y = z["stems"].astype(str), z["y_true"]
        n_classes = int(y.max()) + 1
        grid = ROOT / "outputs/final_finetune_grid" / (CV_CORPORA.get(corpus, corpus))
        for model in ("resnet50", "vit_b_16"):
            for key in [k for k in z.files if k.startswith("seed_")]:
                seed = key.split("_")[1]
                if cv:
                    if not sorted((grid / model / f"seed_{seed}").glob(
                            "fold_*/fold_*_posteriors.npz")):
                        continue
                    B = backbone_oof(grid, model, seed, stems, n_classes, y)
                else:
                    found = sorted(glob.glob(str(
                        grid / model / f"seed_{seed}/*/*_posteriors.npz")))
                    if not found:
                        continue
                    zb = np.load(found[0], allow_pickle=True)
                    order = pd.Index(list(zb["stems"].astype(str))).get_indexer(stems)
                    if (order < 0).any():
                        raise ValueError(f"{corpus}/{model}/{seed}: stems ausentes")
                    B = zb["probabilities"][order].astype(np.float64)
                P = z[key].astype(np.float64)
                lp, lb = np.log(np.maximum(P, 1e-12)), np.log(np.maximum(B, 1e-12))

                pred_half = np.empty_like(y)
                pred_cal_half = np.empty_like(y)
                pred_cross = np.empty_like(y)
                pred_cal_cross = np.empty_like(y)
                temps, ws, ws_cal = [], [], []
                for tr, te in StratifiedKFold(args.folds, shuffle=True,
                                              random_state=42).split(np.zeros(len(y)), y):
                    t_p = fit_temperature(lp[tr], y[tr])
                    t_b = fit_temperature(lb[tr], y[tr])
                    temps.append((t_p, t_b))
                    cp, cb = lp / t_p, lb / t_b

                    w = max(WEIGHTS, key=lambda w: balanced_accuracy_score(
                        y[tr], blend(P[tr], B[tr], w).argmax(1)))
                    w_cal = max(WEIGHTS, key=lambda w: balanced_accuracy_score(
                        y[tr], (w * cp[tr] + (1 - w) * cb[tr]).argmax(1)))
                    ws.append(w); ws_cal.append(w_cal)

                    pred_half[te] = blend(P[te], B[te], 0.5).argmax(1)
                    pred_cal_half[te] = (0.5 * cp[te] + 0.5 * cb[te]).argmax(1)
                    pred_cross[te] = blend(P[te], B[te], w).argmax(1)
                    pred_cal_cross[te] = (w_cal * cp[te] + (1 - w_cal) * cb[te]).argmax(1)

                rows.append(dict(
                    corpus=corpus, model=model, seed=int(seed),
                    backbone=100 * balanced_accuracy_score(y, B.argmax(1)),
                    ours=100 * balanced_accuracy_score(y, P.argmax(1)),
                    conf_ours=100 * P.max(1).mean(), conf_backbone=100 * B.max(1).mean(),
                    ece_ours=ece(P, y), ece_backbone=ece(B, y),
                    temp_ours=float(np.mean([t for t, _ in temps])),
                    temp_backbone=float(np.mean([t for _, t in temps])),
                    w_half=100 * balanced_accuracy_score(y, pred_half),
                    w_cross=100 * balanced_accuracy_score(y, pred_cross),
                    cal_half=100 * balanced_accuracy_score(y, pred_cal_half),
                    cal_cross=100 * balanced_accuracy_score(y, pred_cal_cross),
                    weight=float(np.mean(ws)), weight_cal=float(np.mean(ws_cal))))
                pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
        log(f"  {corpus}: {len([r for r in rows if r['corpus'] == corpus])} unidades")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
