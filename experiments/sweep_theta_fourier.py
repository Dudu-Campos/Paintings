"""theta-robustez do banco de Fourier: terceira representacao varrida.

Mesmo desenho de dois eixos do Hilbert (§5.17) e do Gabor (§5.17b), e a
coincidencia nao e acidental -- **as tres representacoes decompoem theta em
resolucao x granularidade**, e as duas primeiras concordam em que a
granularidade domina (68x no Hilbert, 2,9x no Gabor). O Fourier e o terceiro
ponto dessa afirmacao.

  resolucao      `--size`, o lado da grade do espectro. theta* = 256.
  granularidade  aneis x setores do histograma log-polar. theta* = 12 x 12.

**Previsao registrada antes de rodar.** Duas, e a segunda e a interessante:

1. Granularidade deve dominar de novo, mas por MENOS que no Hilbert, porque
   aqui os dois eixos estao acoplados: as bordas radiais sao
   `geomspace(1/SIZE, 0.5, bins+1)`, entao mudar `size` move a menor frequencia
   representavel e portanto muda o que o primeiro anel mede. No Hilbert os dois
   eixos sao independentes.

2. **Aumentar `radial_bins` deve pagar MENOS do que a contagem de dimensoes
   sugere, e pode virar negativo.** O §3 registra que os 18 primeiros
   `log_power_r00_a**` sao mortos na coorte: o anel de raio zero tem os setores
   angulares todos medindo o mesmo bin DC -- degenerado por construcao. Com
   mais aneis os internos ficam mais finos e MAIS deles caem nesse regime, entao
   a granularidade radial compra colunas constantes. Se der, e um mecanismo
   medido para "mais dimensoes nao e mais informacao", e do tipo que o §5.15 ja
   viu por outro instrumento (68 das 128 piores features do banco sao Fourier).

Duas metricas, as mesmas das outras duas varreduras:
  intrinseca  CKA linear contra theta*, SEM classificador.
  extrinseca  BAcc isolada sob LDA, mais o custo de extracao.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_neighbour_attention import key  # noqa: E402
from sweep_theta_hilbert import linear_cka  # noqa: E402

PYTHON = ROOT / ".ABSvenv/bin/python3"
STAR = (256, 12)                      # (size, bins) de producao


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def extract(sample, image_dir, meta, out, size, bins, workers):
    if out.exists():
        return float("nan")
    started = time.perf_counter()
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "src/extractors/extract_fourier_full_features.py"),
         "--sample", str(sample), "--image-dir", str(image_dir), "--meta", str(meta),
         "--output", str(out), "--workers", str(workers),
         "--size", str(size), "--radial-bins", str(bins), "--angular-bins", str(bins)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"extração falhou ({size},{bins}):\n{result.stderr[-2000:]}")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path,
                        default=ROOT / "outputs/multitask_slic_full/index.csv")
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "data/multitask_painting100k/original/images")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--bins", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_fourier")
    parser.add_argument("--output", type=Path, default=ROOT / "results/theta_fourier.csv")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    # `meta.style` devolve o acessor Styler do pandas, não a coluna (§8).
    meta = meta[[key(str(v)) in wanted for v in meta["style"]]]
    if meta["style"].nunique() != len(wanted):
        raise SystemExit(f"casaram {meta['style'].nunique()} de {len(wanted)} classes")
    log(f"coorte do benchmark: {len(meta):,} imagens, {meta['style'].nunique()} classes")

    grid = [(s, b) for s in args.sizes for b in args.bins]
    features, costs = {}, {}
    for size, bins in grid:
        out = args.work_dir / f"s{size}_b{bins}.parquet"
        costs[(size, bins)] = extract(args.sample, args.image_dir, args.meta, out,
                                      size, bins, args.workers)
        frame = pd.read_parquet(out)
        columns = [c for c in frame.columns if c not in ("stem", "style")]
        features[(size, bins)] = (frame[columns].to_numpy(np.float64),
                                  frame.stem.astype(str).to_numpy())
        cost = costs[(size, bins)]
        log(f"  ({size:>3}, {bins:>2})  {features[(size, bins)][0].shape}  "
            + (f"{cost:.0f}s" if cost == cost else "em cache"))

    order = meta.stem.astype(str).to_numpy()
    aligned = {}
    for theta, (values, stems) in features.items():
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(order)
        if index.isna().any():
            raise ValueError(f"{theta}: {int(index.isna().sum())} stems ausentes")
        matrix = values[index.to_numpy(dtype=int)]
        aligned[theta] = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    labels = LabelEncoder().fit_transform(
        meta.set_index("stem").loc[order, "style"].to_numpy(str))
    n_classes = len(np.unique(labels))

    splits = list(StratifiedKFold(5, shuffle=True, random_state=42)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    rows = []
    for theta in grid:
        values = aligned[theta]
        # Colunas constantes: o mecanismo previsto no docstring aparece aqui.
        dead = int((values.std(axis=0) < 1e-12).sum())
        scores = []
        for train, test in splits:
            scaler = StandardScaler().fit(values[train])
            model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            model.fit(scaler.transform(values[train]), labels[train])
            scores.append(100 * balanced_accuracy_score(
                labels[test], model.predict(scaler.transform(values[test]))))
        rows.append(dict(size=theta[0], bins=theta[1],
                         dimensions=values.shape[1], dead_columns=dead,
                         is_star=theta == STAR,
                         cka_vs_star=linear_cka(values, aligned[STAR]),
                         extract_seconds=costs[theta],
                         lda_bacc=float(np.mean(scores)), lda_std=float(np.std(scores))))
        log(f"  ({theta[0]:>3}, {theta[1]:>2})  BAcc {rows[-1]['lda_bacc']:5.2f}  "
            f"CKA {rows[-1]['cka_vs_star']:.3f}  mortas {dead:>3}/{values.shape[1]}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows).sort_values("lda_bacc", ascending=False).reset_index(drop=True)
    rank = int(table.index[table.is_star][0]) + 1
    log(f"\ntheta* = {STAR}: rank {rank} de {len(table)} | "
        f"CKA minimo {table.cka_vs_star.min():.3f} | "
        f"amplitude BAcc {table.lda_bacc.max() - table.lda_bacc.min():.2f}pp")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
