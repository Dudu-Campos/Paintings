"""Somos complementares a uma CNN treinada DO ZERO -- os dois lados com o MESMO
orcamento de supervisao?

O §4.1b e o §4.1f medem complementaridade contra backbones que carregam 1,28M de
rotulos do ImageNet. Esta e a pergunta no degrau de BAIXO da escada do §4.1:
nenhum dos dois lados viu rotulo de fora do corpus. Se houver ganho aqui, a
alegacao deixa de depender de supervisao externa em qualquer lado.

A pergunta ficou urgente porque no oilpainting a CNN do zero faz 54,17 contra
43,37 do hand-8 -- perdemos por 10,8pp para um adversario de supervisao pareada,
e "ganhamos com supervisao pareada" (§4.1, WikiArt) deixa de valer como frase
geral. Complementaridade e a alegacao que sobrevive a isso, se existir.

FUSAO DE DECISAO, nao de features, e a razao esta no §4.1c: a CNN foi treinada
no alvo, entao as linhas de TREINO dela sao in-sample (o ViT fine-tuned acerta
96,65% la contra 67,32% no teste) e uma cabeca de fusao aprende a ignorar o
nosso lado. Os posteriors de teste sao OOF por construcao; os nossos vem do
cache OOF por fold.

Guardas, todas do §4.1f:
  * alinhamento POR STEM, nunca por posicao (o bug que deu 562 melhoram/137
    pioram com delta negativo no MINC);
  * os rotulos sao conferidos DEPOIS de alinhar;
  * o oraculo sai em BAcc **e** em acuracia, rotulados -- misturar as duas foi o
    erro corrigido em 19/set, que punha o cross-fit acima do proprio teto;
  * o teto vem com o NULO de independencia junto: dois classificadores com
    aquelas recalls ja produzem uniao por acaso, e reportar teto sem nulo e o
    que o §5.18b proibe.

Uso:
    python experiments/evaluate_scratch_cnn_fusion.py --dataset oilpainting
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EPS = 1e-12
GRID_W = np.arange(0.05, 1.0, 0.05)


def log(msg: str) -> None:
    print(msg, flush=True)


def bacc(y, pred) -> float:
    return 100 * balanced_accuracy_score(y, pred)


def fuse(p_ours, p_cnn, w):
    """Media ponderada de LOG-probabilidades. w e o peso do NOSSO lado."""
    return w * np.log(p_ours + EPS) + (1 - w) * np.log(p_cnn + EPS)


def crossfit_weight(p_ours, p_cnn, y, n_parts=5):
    """Peso escolhido FORA da dobra avaliada (§4.1f item 2). Remove a
    circularidade da selecao; nao remove o uso da distribuicao do teste, e essa
    ressalva vale 0,005pp nos dois corpora onde a rota limpa foi testavel."""
    pred = np.empty(len(y), dtype=np.int64)
    pesos = []
    parts = StratifiedKFold(n_parts, shuffle=True, random_state=SEED).split(p_ours, y)
    for fit, held in parts:
        scores = [bacc(y[fit], fuse(p_ours[fit], p_cnn[fit], w).argmax(1)) for w in GRID_W]
        w = GRID_W[int(np.argmax(scores))]
        pesos.append(w)
        pred[held] = fuse(p_ours[held], p_cnn[held], w).argmax(1)
    return bacc(y, pred), float(np.mean(pesos)), float(np.std(pesos))


def yule_q(ok_a, ok_b):
    """Q de Yule sobre a 2x2 de acerto/erro, e a razao de chances que o gera.

    Q = (ad - bc)/(ad + bc) = (OR - 1)/(OR + 1), com a=ambos certos, b=so A,
    c=so B, d=nenhum. **Funcao da razao de chances, logo invariante a escalar
    linha e coluna** -- e por isso ele mede associacao entre os padroes de erro
    SEM herdar a taxa de acerto de cada lado, que e exatamente o defeito da
    porcentagem bruta de acertos exclusivos (§4.1d: "nao e comparavel entre
    corpora, e limitada pela taxa de erro do backbone").

    Leitura:  Q = 0  independencia -- o oraculo vale o nulo do §4.1f
              Q > 0  erram JUNTO   -- o oraculo fica ABAIXO do nulo
              Q < 0  complementares de verdade, alem do acaso

    Correcao de Haldane (+0,5 em toda celula) so quando alguma zera, senao Q
    satura em +-1 por contagem e nao por associacao.
    """
    a = float((ok_a & ok_b).sum()); b = float((ok_a & ~ok_b).sum())
    c = float((~ok_a & ok_b).sum()); d = float((~ok_a & ~ok_b).sum())
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    num, den = a * d - b * c, a * d + b * c
    q = num / den if den else float("nan")
    orr = (a * d) / (b * c) if b * c else float("inf")
    return float(q), float(orr)


def yule_q_macro(y, ok_a, ok_b) -> float:
    """Q POR CLASSE, media macro -- o analogo do que a BAcc faz com a recall.

    O Q agrupado soma as classes todas numa 2x2 so, entao com desbalanceamento
    ele mede sobretudo as frequentes. O §4.1f registra o custo de misturar
    metricas de peso diferente numa comparacao; aqui as duas vao lado a lado.
    """
    qs = []
    for c in np.unique(y):
        m = y == c
        if m.sum() < 2:
            continue
        qs.append(yule_q(ok_a[m], ok_b[m])[0])
    return float(np.nanmean(qs)) if qs else float("nan")


def oracle_null(y, ok_a, ok_b) -> float:
    """BAcc do oraculo SOB INDEPENDENCIA dentro da classe: preserva a recall por
    classe dos dois lados e destroi so o pareamento de QUAIS imagens cada um
    acerta. Exato em esperanca, dispensa simulacao."""
    recalls = []
    for c in np.unique(y):
        m = y == c
        ra, rb = ok_a[m].mean(), ok_b[m].mean()
        recalls.append(1 - (1 - ra) * (1 - rb))
    return 100 * float(np.mean(recalls))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="oilpainting")
    ap.add_argument("--grid-root", type=Path, default=ROOT / "outputs/scratch_grid")
    ap.add_argument("--our-oof", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=["resnet18", "resnet50"])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    our_path = args.our_oof or ROOT / f"outputs/{args.dataset}_our_oof_cv.npz"
    out = args.output or ROOT / f"results/{args.dataset}_scratch_cnn_fusion.csv"

    ours = np.load(our_path, allow_pickle=True)
    our_stems = np.array([str(s) for s in ours["stems"]])
    at = {s: i for i, s in enumerate(our_stems)}
    our_seeds = sorted(int(k.split("_")[1]) for k in ours.files if k.startswith("seed_"))
    log(f"{args.dataset}: nosso OOF com {len(our_stems):,} linhas, "
        f"sementes {our_seeds}  ({our_path.name})")

    rows: list[dict] = []
    for model in args.models:
        for unit in sorted((args.grid_root / args.dataset / model).glob(
                "seed_*/fold_*/fold_*_posteriors.npz")):
            seed = int(unit.parent.parent.name.split("_")[1])
            fold = unit.parent.name.split("_")[1]
            blob = np.load(unit, allow_pickle=True)
            stems = np.array([str(s) for s in blob["stems"]])
            p_cnn = np.asarray(blob["probabilities"], dtype=np.float64)
            y = np.asarray(blob["y_true"], dtype=np.int64)

            faltam = [s for s in stems if s not in at]
            if faltam:
                raise SystemExit(f"{unit}: {len(faltam)} stems nao estao no nosso OOF "
                                 f"(ex.: {faltam[:3]})")
            idx = np.array([at[s] for s in stems])          # ALINHA POR STEM
            if not (np.asarray(ours["y_true"])[idx] == y).all():
                raise SystemExit(f"{unit}: rotulos divergem DEPOIS de alinhar -- "
                                 f"as duas fontes nao codificam as classes igual")

            # Pareia semente a semente quando existe; senao usa a nossa 42 e diz.
            use_seed = seed if seed in our_seeds else our_seeds[0]
            p_ours = np.asarray(ours[f"seed_{use_seed}"], dtype=np.float64)[idx]

            ok_o = p_ours.argmax(1) == y
            ok_c = p_cnn.argmax(1) == y
            b_half = bacc(y, fuse(p_ours, p_cnn, 0.5).argmax(1))
            b_cross, w_mean, w_sd = crossfit_weight(p_ours, p_cnn, y)
            ok_or = ok_o | ok_c
            # oraculo em BAcc: recall macro de "um dos dois acerta"
            orac_bacc = 100 * float(np.mean([ok_or[y == c].mean() for c in np.unique(y)]))

            r = {"dataset": args.dataset, "model": model, "seed": seed,
                 "our_seed": use_seed, "fold": fold, "n_test": len(y),
                 "ours_bacc": bacc(y, p_ours.argmax(1)), "cnn_bacc": bacc(y, p_cnn.argmax(1)),
                 "fusion_half": b_half, "fusion_crossfit": b_cross,
                 "w_crossfit": w_mean, "w_sd": w_sd,
                 "both": 100 * float((ok_o & ok_c).mean()),
                 "only_ours": 100 * float((ok_o & ~ok_c).mean()),
                 "only_cnn": 100 * float((~ok_o & ok_c).mean()),
                 "neither": 100 * float((~ok_o & ~ok_c).mean()),
                 "oracle_bacc": orac_bacc,
                 "oracle_bacc_null": oracle_null(y, ok_o, ok_c),
                 "yule_q": yule_q(ok_o, ok_c)[0],
                 "odds_ratio": yule_q(ok_o, ok_c)[1],
                 "yule_q_macro": yule_q_macro(y, ok_o, ok_c)}
            r["gap"] = r["cnn_bacc"] - r["ours_bacc"]
            r["d_half"] = r["fusion_half"] - r["cnn_bacc"]
            r["d_cross"] = r["fusion_crossfit"] - r["cnn_bacc"]
            r["oracle_headroom"] = r["oracle_bacc"] - r["cnn_bacc"]
            r["oracle_excess_vs_null"] = r["oracle_bacc"] - r["oracle_bacc_null"]
            r["capture"] = 100 * r["only_ours"] / max(100 - r["both"] - r["only_ours"], 1e-9)
            rows.append(r)
            log(f"  {model:9s} s{seed} f{fold}  n={len(y):,}  nosso {r['ours_bacc']:5.2f}  "
                f"CNN {r['cnn_bacc']:5.2f}  gap {r['gap']:+5.2f}  |  "
                f"fusao 0,5 {r['fusion_half']:5.2f} ({r['d_half']:+5.2f})  "
                f"cross-fit {r['fusion_crossfit']:5.2f} ({r['d_cross']:+5.2f}, w*={w_mean:.2f})")

    if not rows:
        raise SystemExit(f"nenhuma unidade em {args.grid_root / args.dataset}")
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False, float_format="%.4f")

    log("\ndecomposicao por imagem (acuracia simples) e teto:")
    for _, r in table.iterrows():
        log(f"  {r['model']:9s}  ambos {r['both']:5.2f}%  SO NOSSO {r['only_ours']:5.2f}%  "
            f"so CNN {r['only_cnn']:5.2f}%  nenhum {r['neither']:5.2f}%")
        log(f"  {'':9s}  captura {r['capture']:5.1f}% do erro da CNN  |  "
            f"oraculo BAcc {r['oracle_bacc']:5.2f} (teto {r['oracle_headroom']:+5.2f}), "
            f"nulo de independencia {r['oracle_bacc_null']:5.2f} "
            f"(excesso {r['oracle_excess_vs_null']:+5.2f})")
        log(f"  {'':9s}  Q de Yule {r['yule_q']:+.3f} (agrupado, OR={r['odds_ratio']:.2f})  "
            f"{r['yule_q_macro']:+.3f} (macro por classe)   "
            f"[0 = independencia; >0 erram junto; <0 complementares]")
    log(f"\n{out}")


if __name__ == "__main__":
    main()
