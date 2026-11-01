"""theta-robustez do banco de Gabor: theta* e um pico isolado?

Segunda representacao varrida, depois do Hilbert (§5.17). A pergunta e a mesma e
NAO e sobre a transformacao (ela vem de Manjunath & Ma 1996 e nao e nossa para
defender): e sobre o ponto de operacao escolhido a priori
(`TARGET=128, N_THETAS=4, FREQUENCIES=[0.1..0.4]`) estar ou nao num ponto
nao-suave da superficie de resposta.

**Por que o Gabor e o segundo caso certo.** Ele testa uma previsao OPOSTA a do
Hilbert. La as estatisticas de banda sao medias sobre N^2/B pixels e a curva
preserva localidade, entao cada banda cobre a mesma regiao em qualquer
resolucao -- previa-se insensibilidade a `n_size`, e foi o que deu (elasticidade
+0,003 no MLP). Aqui as frequencias estao em **ciclos por pixel**, entao mudar
`target` muda a escala fisica que cada filtro mede: `f=0.4` a 128 px e uma
textura diferente de `f=0.4` a 512 px. A previsao e que `target` importe MUITO,
ao contrario do Hilbert.

Se a previsao acertar, o §5.17 deixa de ser "theta nao importa" e passa a ser
**"quais eixos de theta importam e por que"** -- que e a afirmacao mais forte e
ja estava insinuada la (resolucao +0,003 contra granularidade +0,387).

Duas metricas, as mesmas do Hilbert, porque medem coisas diferentes:
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
STAR = (128, 4)                       # (target, n_thetas) de producao
NATIVE_MEDIAN = 1000                  # mediana do lado maior nos ORIGINAIS (§7.1)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def extract(image_dir, meta, out, target, n_thetas, workers):
    if (out / "gabor_features.npy").exists():
        return float("nan")
    started = time.perf_counter()
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "src/extractors/extract_gabor_features.py"),
         "--image_dir", str(image_dir), "--meta", str(meta),
         "--output", str(out), "--workers", str(workers),
         "--target", str(target), "--n-thetas", str(n_thetas)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"extração falhou ({target},{n_thetas}):\n{result.stderr[-2000:]}")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "data/multitask_painting100k/original/images")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--targets", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--thetas", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_gabor")
    parser.add_argument("--output", type=Path, default=ROOT / "results/theta_gabor.csv")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    # `meta.style` devolve o acessor Styler do pandas, não a coluna (§8).
    meta = meta[[key(str(v)) in wanted for v in meta["style"]]]
    if meta["style"].nunique() != len(wanted):
        raise SystemExit(f"casaram {meta['style'].nunique()} de {len(wanted)} classes")
    log(f"coorte do benchmark: {len(meta):,} imagens, {meta['style'].nunique()} classes")

    grid = [(t, n) for t in args.targets for n in args.thetas]
    features, costs = {}, {}
    for target, n_thetas in grid:
        out = args.work_dir / f"t{target}_n{n_thetas}"
        costs[(target, n_thetas)] = extract(args.image_dir, args.meta, out,
                                            target, n_thetas, args.workers)
        values = np.load(out / "gabor_features.npy")
        stems = np.load(out / "gabor_stems.npy", allow_pickle=True).astype(str)
        features[(target, n_thetas)] = (values, stems)
        cost = costs[(target, n_thetas)]
        log(f"  ({target:>3}, {n_thetas:>2})  {values.shape}  "
            + (f"{cost:.0f}s" if cost == cost else "em cache"))

    order = meta.stem.astype(str).to_numpy()
    aligned = {}
    for theta, (values, stems) in features.items():
        index = pd.Series(np.arange(len(stems)), index=stems)
        take = index.reindex(order).dropna()
        aligned[theta] = values[take.to_numpy(dtype=int)]
    labels = LabelEncoder().fit_transform(
        meta.set_index("stem").loc[order, "style"].to_numpy(str))
    n_classes = len(np.unique(labels))

    splits = list(StratifiedKFold(5, shuffle=True, random_state=42)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    rows = []
    for theta in grid:
        values = aligned[theta]
        scores = []
        for train, test in splits:
            scaler = StandardScaler().fit(values[train])
            model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            model.fit(scaler.transform(values[train]), labels[train])
            scores.append(100 * balanced_accuracy_score(
                labels[test], model.predict(scaler.transform(values[test]))))
        rows.append(dict(target=theta[0], n_thetas=theta[1],
                         dimensions=values.shape[1], is_star=theta == STAR,
                         above_native=theta[0] > NATIVE_MEDIAN,
                         cka_vs_star=linear_cka(values, aligned[STAR]),
                         extract_seconds=costs[theta],
                         lda_bacc=float(np.mean(scores)), lda_std=float(np.std(scores))))
        log(f"  ({theta[0]:>3}, {theta[1]:>2})  BAcc {rows[-1]['lda_bacc']:5.2f}  "
            f"CKA {rows[-1]['cka_vs_star']:.3f}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows).sort_values("lda_bacc", ascending=False).reset_index(drop=True)
    rank = int(table.index[table.is_star][0]) + 1
    log(f"\ntheta* = {STAR}: rank {rank} de {len(table)} | "
        f"CKA minimo {table.cka_vs_star.min():.3f} | "
        f"amplitude BAcc {table.lda_bacc.max() - table.lda_bacc.min():.2f}pp")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
