"""R² múltiplo por bloco contra o resto do banco, e o cruzamento com o marginal.

O §6 registra que **redundância não prevê contribuição** -- seis instrumentos
falharam. Este é o instrumento nº 1 (R² múltiplo) recalculado no Multitask com as
**oito** modalidades, incluindo Gabor, que não existia quando a auditoria
original rodou no WikiArt.

Para cada bloco m: regride cada uma das suas features sobre a concatenação dos
outros sete blocos (ridge, padronizado, sem intercepto relevante) e resume o R²
out-of-fold. **Nenhum rótulo entra** -- é auditoria não-supervisionada, e é
justamente por isso que ela pode ser comparada com o marginal sem circularidade.

O cruzamento é com o valor de Shapley exato já medido
(`results/factorial_shapley_multitask_25class_summary.csv`, 2^9 máscaras).

Leitura esperada, se o §6 continuar valendo: a nuvem não tem tendência. Hilbert
já foi medido com R² 0,919 valendo +3,76pp, e o passeio do turista com R² 0,161
valendo zero.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from evaluate_external_modalities import load_block  # noqa: E402
from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean  # noqa: E402

HAND = VISUAL + ["Gabor"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", "_", folded.strip()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/block_redundancy_multitask.csv")
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})[["stem", "style"]]
    for name in HAND:
        frame = frame.merge(load_block("multitask", name), on="stem", how="inner")
    wanted = {key(l) for l in (ROOT / "data/multitask_25class_benchmark.txt")
              .read_text().splitlines() if l.strip()}
    frame = frame[[key(v) in wanted for v in frame["style"].to_numpy(str)]].reset_index(drop=True)
    blocks = {n: clean(frame[[c for c in frame.columns if c.startswith(f"{n}_")]]
                       .to_numpy(np.float32)) for n in HAND}
    log(f"{len(frame):,} imagens, {sum(b.shape[1] for b in blocks.values())}d em {len(HAND)} blocos")

    scaled = {n: StandardScaler().fit_transform(b).astype(np.float32) for n, b in blocks.items()}
    rows = []
    for target in HAND:
        others = np.hstack([scaled[n] for n in HAND if n != target])
        y = scaled[target]
        predicted = np.zeros_like(y)
        for fit_idx, test_idx in KFold(args.folds, shuffle=True, random_state=SEED).split(others):
            model = Ridge(alpha=args.alpha).fit(others[fit_idx], y[fit_idx])
            predicted[test_idx] = model.predict(others[test_idx])
        # R² por feature: as colunas estão padronizadas, então a variância é 1.
        residual = ((y - predicted) ** 2).mean(axis=0)
        r2 = 1.0 - residual
        rows.append({"block": target, "dims": y.shape[1],
                     "r2_mean": float(r2.mean()), "r2_median": float(np.median(r2)),
                     "r2_p90": float(np.percentile(r2, 90)),
                     "frac_acima_0.8": float((r2 > 0.8).mean())})
        log(f"  {target:<14} {y.shape[1]:>4}d  R² médio {r2.mean():.3f}  "
            f"mediana {np.median(r2):.3f}  p90 {np.percentile(r2, 90):.3f}")
        del others

    detail = pd.DataFrame(rows)
    shapley_path = ROOT / "results/factorial_shapley_multitask_25class_summary.csv"
    if shapley_path.exists():
        shap = pd.read_csv(shapley_path).rename(columns={"modality": "block"})
        detail = detail.merge(shap[["block", "shapley"]], on="block", how="left")
        # O CSV do factorial já está em pontos percentuais.
        detail["shapley_pp"] = detail["shapley"]
    detail = detail.sort_values("r2_mean", ascending=False)
    detail.to_csv(args.output, index=False, float_format="%.5f")

    print()
    print(f"{'bloco':<14} {'dims':>5} {'R² médio':>9} {'R² p90':>7} {'>0,8':>6} {'Shapley (pp)':>13}")
    for r in detail.itertuples():
        s = f"{r.shapley_pp:.2f}" if "shapley_pp" in detail.columns and pd.notna(r.shapley_pp) else "—"
        print(f"{r.block:<14} {int(r.dims):>5} {r.r2_mean:>9.3f} {r.r2_p90:>7.3f} "
              f"{100 * r._6:>5.0f}% {s:>13}")
    if "shapley_pp" in detail.columns and detail["shapley_pp"].notna().sum() > 2:
        x = detail["r2_mean"].to_numpy()
        y = detail["shapley_pp"].to_numpy()
        ok = ~np.isnan(y)
        r, p = pearsonr(x[ok], y[ok])
        s, ps = spearmanr(x[ok], y[ok])
        print()
        print(f"R² médio × Shapley:  Pearson r={r:+.3f} (p={p:.3f})   "
              f"Spearman rho={s:+.3f} (p={ps:.3f})   n={int(ok.sum())}")
        print("Se |r| for baixo, o §6 se mantém: redundância não prevê contribuição.")


if __name__ == "__main__":
    main()
