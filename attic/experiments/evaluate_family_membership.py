"""A que família pertence uma heurística externa? Medido, não arbitrado.

Fecha a ideia original desta linha: *unificar as heurísticas históricas de
classificação de pintura mostrando de qual canal cada uma é proxy*. A versão
anterior disso -- taxonomia por co-ocorrência de erro -- foi **descartada por
falhar na própria validação**: ela ordenava por acurácia (Spearman +0,747 com a
BAcc isolada) e chegou a inverter a retrodição, pondo L(G_felz), que soma
+0,79pp, como a 4ª maior sobreposição.

Aqui a medida é a similaridade em r² contra as famílias, que a figura de
distribuição mostrou ser estruturada por família (13× mais quase-duplicatas
dentro que fora), e o limiar **não é escolhido**: ele sai da calibração nos
nossos próprios blocos, cuja família é conhecida. Para cada bloco nosso B de
família F, o perfil é medido contra F **sem B**, o que produz duas faixas de
referência -- como se parece um membro genuíno, e como se parece um não-membro.
Um candidato abaixo da faixa de membro contra todas as famílias é candidato a
família nova.

**A disciplina que este projeto aprendeu à força:** a atribuição é *descritiva*.
Ela responde em que canal o método opera, e **não** se ele vai somar. Cinco
instrumentos independentes (r² par-a-par, R² múltiplo, co-ocorrência de erro,
CKA e R² de ridge contra backbone) já mediram que redundância não prevê
contribuição. Escrever "pertence a família existente, logo não ajuda" repetiria
o erro que custou o instrumento anterior.

A formulação segura é a inversa e é mais forte: *cada heurística clássica opera
num canal que o pipeline já cobre* -- construída **sem** usar o marginal delas,
e portanto evidência independente em vez de circular. Os marginais já medidos
(HOG −1,21 · turista −0,02 · triângulos +0,06 · Quickshift +0,67 · L(G_felz)
+0,79) servem só de checagem de coerência posterior.

Saída: results/family_membership.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

# Referência: os blocos cuja família nós conhecemos por construção. Multiscale
# entra dividido pelas famílias de origem -- ela não é família, é Descriptors,
# SLIC e Hilbert recalculados por quadrante, e tratá-la como categoria própria
# já foi medido como erro (o Rand ajustado caiu de 0,194 para 0,001).
REFERENCE = {
    "image": ["Descriptors"],
    "segmentation": ["SLIC", "Felzenszwalb", "Line graph", "L(G_felz)"],
    "curve": ["Hilbert"],
    "spectral": ["Fourier"],
}
QUADRANT_SOURCE = {"Descriptors": "image", "SLIC": "segmentation",
                   "Hilbert": "curve"}
CANDIDATES = ("HOG", "Haralick", "Tourist walk", "Triangles SLIC",
              "Triangles Felz", "Percolation SLIC", "Percolation Felz",
              "Random walk", "Hybrid walk", "Square graph", "Hilbert catch22",
              "Hilbert spectrum1d", "Hilbert visibility", "Hilbert wavelet",
              "ResNet50", "ResNet50 random")


def standardise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    deviation = values.std(axis=0)
    deviation[deviation < 1e-8] = 1.0
    return (values - values.mean(axis=0)) / deviation


def profile(candidate: np.ndarray, reference: np.ndarray, chunk: int = 256) -> dict:
    """Perfil de similaridade: por feature do candidato, o maior r² na família.

    O máximo, e não a média, porque a pergunta é de pertencimento: uma feature
    pertence ao canal se **alguma** feature daquele canal a reproduz. A média
    seria diluída pelo tamanho da família, que varia de 57 a 1.764 colunas e não
    tem nada a ver com a pergunta.
    """
    best = np.empty(candidate.shape[1], dtype=np.float64)
    n = len(candidate)
    for start in range(0, candidate.shape[1], chunk):
        block = candidate[:, start:start + chunk]
        correlation = (block.T @ reference) / n
        best[start:start + block.shape[1]] = (correlation ** 2).max(axis=1)
    return {"median_max_r2": float(np.median(best)),
            "p90_max_r2": float(np.percentile(best, 90)),
            "frac_above_0.5": float((best > 0.5).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/family_membership.csv")
    args = parser.parse_args()

    reference_blocks = sorted({b for names in REFERENCE.values() for b in names})
    wanted = reference_blocks + ["Multiscale"] + list(CANDIDATES)
    cohort = load_blocks(args.dataset, wanted, allow_external=True)
    available = set(cohort.blocks)
    log(f"coorte {len(cohort.stems):,} | {len(available)} blocos carregados")

    # Monta as famílias de referência, incluindo os quadrantes na origem certa.
    families: dict[str, list[np.ndarray]] = {name: [] for name in REFERENCE}
    owner: dict[str, list[str]] = {name: [] for name in REFERENCE}
    for family, blocks in REFERENCE.items():
        for block in blocks:
            if block in available:
                families[family].append(standardise(cohort.blocks[block]))
                owner[family].append(block)
    if "Multiscale" in available:
        columns = np.array(cohort.columns["Multiscale"])
        block = standardise(cohort.blocks["Multiscale"])
        for source, family in QUADRANT_SOURCE.items():
            mask = np.array([f"_{source}_" in c for c in columns])
            if mask.any():
                families[family].append(block[:, mask])
                owner[family].append(f"Multiscale/{source}")
    for family in families:
        log(f"  família {family}: "
            + ", ".join(f"{n}({p.shape[1]})" for n, p in
                        zip(owner[family], families[family])))

    rows = []

    def measure(name: str, values: np.ndarray, kind: str, exclude: str | None) -> None:
        for family in REFERENCE:
            parts = [piece for piece, holder in zip(families[family], owner[family])
                     if holder != exclude]
            if not parts:
                continue
            stats = profile(values, np.hstack(parts))
            rows.append({"name": name, "kind": kind, "family": family,
                         "dims": values.shape[1],
                         "reference_dims": sum(p.shape[1] for p in parts), **stats})

    # Calibração: cada bloco nosso contra as famílias, sempre excluído de si
    # mesmo. `score(B, família(B))` define a faixa de "membro"; `score(B, outra)`
    # a de "não-membro". Sem isso qualquer limiar seria escolhido a dedo.
    for family, blocks in REFERENCE.items():
        for block in blocks:
            if block in available:
                measure(block, standardise(cohort.blocks[block]),
                        f"referência:{family}", block)

    for name in CANDIDATES:
        if name in available:
            measure(name, standardise(cohort.blocks[name]), "candidato", None)

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")

    metric = "median_max_r2"
    wide = table.pivot_table(index=["kind", "name"], columns="family", values=metric)
    log(f"\n=== {metric}: candidato × família ===")
    log(wide.round(3).to_string())

    reference_rows = table[table.kind.str.startswith("referência")]
    member = reference_rows[reference_rows.apply(
        lambda r: r["kind"].split(":")[1] == r["family"], axis=1)][metric]
    stranger = reference_rows[reference_rows.apply(
        lambda r: r["kind"].split(":")[1] != r["family"], axis=1)][metric]
    log(f"\nfaixa de MEMBRO (bloco nosso × sua própria família, sem ele mesmo): "
        f"{member.min():.3f} a {member.max():.3f}")
    log(f"faixa de ESTRANHO (bloco nosso × família alheia): "
        f"{stranger.min():.3f} a {stranger.max():.3f}")

    log("\n=== atribuição ===")
    floor = member.min()
    for name, group in table[table.kind == "candidato"].groupby("name", sort=False):
        best = group.loc[group[metric].idxmax()]
        verdict = (f"→ {best['family']}" if best[metric] >= floor
                   else "→ NENHUMA (candidato a família nova)")
        log(f"  {name:<20} melhor={best['family']:<13} {metric}={best[metric]:.3f}  "
            f"{verdict}")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
