"""Quao degenerado e o HVG, coluna a coluna -- e por que o posto efetivo NAO
detecta isso.

O §5.9b registra em prosa que o HVG tem "6 colunas constantes" e conclui que as
features "estao integras". A medicao por tras daquela frase nunca foi salva (e a
divida do §7bis: artefato citado sem comando registrado), e a conclusao e
otimista demais -- constancia exata e so a ponta do problema.

A medida que pega e **quantos valores DISTINTOS cada coluna assume sobre as
99.816 imagens**. O Hilbert serve de controle perfeito: mesma familia
(Space-filling curve), mesmas 57 dimensoes, mesmo corpus.

**E o achado de metodo: o posto efetivo do §3 da MAIOR para o HVG (PR/d 0,320
contra 0,142 do Hilbert), ou seja ele classifica o bloco degenerado como o mais
saudavel dos dois.** O participation ratio mede o quanto a variancia se espalha
entre direcoes, e ruido de quantizacao se espalha otimamente -- cada coluna e
lixo numa direcao propria, descorrelacionada das outras. Isso acrescenta um
caso ao §6 ("posto efetivo nao preve contribuicao") por um angulo novo: ele nao
so deixa de prever contribuicao, como pode ser ENGANADO por degeneracao.

Uso:
    python experiments/audit_hvg_degeneracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from registry import load_blocks                       # noqa: E402

HVG_FEATURES = ROOT / "outputs" / "multitask_hilbert_visibility" / "features.npy"
HVG_NAMES = ROOT / "outputs" / "multitask_hilbert_visibility" / "feature_names.json"


def effective_rank(X: np.ndarray) -> float:
    """Participation ratio dos autovalores da covariancia padronizada, (Sl)^2/Sl^2
    -- a mesma definicao do §3. Colunas mortas sao removidas antes (padronizar
    divide por zero nelas)."""
    std = X.std(axis=0)
    Z = (X[:, std > 0] - X[:, std > 0].mean(0)) / std[std > 0]
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def column_report(X: np.ndarray, names: list[str], block: str) -> pd.DataFrame:
    std = X.std(axis=0)
    mean = X.mean(axis=0)
    n_uniq = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 1e-12, std / np.abs(mean), np.inf)
    return pd.DataFrame({
        "bloco": block, "coluna": names, "media": mean, "std": std,
        "cv": cv, "n_valores_distintos": n_uniq,
        "constante": std == 0,
    })


def main() -> None:
    hvg = np.load(HVG_FEATURES).astype(np.float64)
    names_hvg = json.loads(HVG_NAMES.read_text()) if HVG_NAMES.exists() else None
    if isinstance(names_hvg, dict):
        names_hvg = names_hvg.get("feature_names", names_hvg.get("names"))
    if not names_hvg or len(names_hvg) != hvg.shape[1]:
        names_hvg = [f"hvg_{j}" for j in range(hvg.shape[1])]

    cohort = load_blocks("multitask", ["Hilbert"])
    hil = np.asarray(cohort.blocks["Hilbert"], dtype=np.float64)
    names_hil = list(cohort.columns["Hilbert"]) if hasattr(cohort, "columns") \
        else [f"hilbert_{j}" for j in range(hil.shape[1])]
    if len(names_hil) != hil.shape[1]:
        names_hil = [f"hilbert_{j}" for j in range(hil.shape[1])]

    frames = [column_report(hvg, names_hvg, "HVG"),
              column_report(hil, names_hil, "Hilbert")]
    table = pd.concat(frames, ignore_index=True)
    out = ROOT / "results" / "hvg_degeneracy_audit.csv"
    table.to_csv(out, index=False, float_format="%.6g")

    print(f"n = {hvg.shape[0]:,} imagens\n")
    print(f"{'bloco':10s} {'dims':>5s} {'constantes':>11s} {'<=10 val.':>10s} "
          f"{'>10k val.':>10s} {'posto ef.':>10s} {'PR/d':>7s}")
    for block, X in [("HVG", hvg), ("Hilbert", hil)]:
        sub = table[table["bloco"] == block]
        nu = sub["n_valores_distintos"].to_numpy()
        pr = effective_rank(X)
        print(f"{block:10s} {X.shape[1]:5d} {int(sub['constante'].sum()):11d} "
              f"{int((nu <= 10).sum()):10d} {int((nu > 10000).sum()):10d} "
              f"{pr:10.2f} {pr / X.shape[1]:7.3f}")

    print("\ndistribuicao de valores distintos por coluna:")
    for block in ("HVG", "Hilbert"):
        nu = table[table["bloco"] == block]["n_valores_distintos"].to_numpy()
        print(f"  {block:8s} 1: {int((nu == 1).sum()):2d} | 2-10: {int(((nu > 1) & (nu <= 10)).sum()):2d} "
              f"| 11-100: {int(((nu > 10) & (nu <= 100)).sum()):2d} "
              f"| 101-10k: {int(((nu > 100) & (nu <= 10000)).sum()):2d} "
              f"| >10k: {int((nu > 10000).sum()):2d}")

    deg = table[(table["bloco"] == "HVG") & (table["n_valores_distintos"] <= 10)]
    print(f"\nas {len(deg)} colunas degeneradas do HVG (<=10 valores em {hvg.shape[0]:,} imagens):")
    for _, r in deg.sort_values("n_valores_distintos").iterrows():
        print(f"  {r['coluna']:34s} {int(r['n_valores_distintos']):3d} valores  "
              f"media={r['media']:.4f}")

    print(f"\n{out}")


if __name__ == "__main__":
    main()
