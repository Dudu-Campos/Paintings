"""Custo dos dois line graphs contra a resolução -- as duas views que faltavam.

A figura `modality_cost_by_resolution` cobria 6 das 8 views. `Line Felz` e
`Line SLIC` ficavam de fora porque o `extract_line_graph_features.py` não lê
imagem: ele consome o **RAG já exportado** pelo passo de multigrafo. Então
cronometrá-los exige exportar o RAG em cada resolução antes, o que não cabe no
laço do benchmark principal.

**A previsão, registrada antes de rodar.** O custo de L(G) é O(Σ deg²) sobre o
RAG -- função do número de SEGMENTOS, não de pixels. E os dois segmentadores se
comportam de forma oposta com a resolução (§9.3):

    Line SLIC  -> `--segments 2048` fixa a contagem (1.749 @1024, 2.005 @224)
                  => custo ~CONSTANTE na resolução
    Line Felz  -> `felz_scale` está em unidades de pixel (1.398 @1024, 134 @224)
                  => custo ESCALA com a resolução

Se isso se confirmar, é a mesma assimetria que o §9.5 mediu por desempenho (SLIC
a 224 px empata, Felz perde), aparecendo agora no custo.

A exportação do RAG NÃO entra no tempo: ela é atribuída ao passo de multigrafo,
que já a paga (§9.7 -- o pipeline atual compartilha o RAG, e contá-la duas vezes
foi o erro que inflava o custo em 43%).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from benchmark_modality_cost_by_resolution import (STEPS, build_cohort,  # noqa: E402
                                                   log, materialise)

PYTHON = ROOT / ".ABSvenv/bin/python3"
SEGMENTER = ROOT / "src/extractors/segmentation_multigraph_features.py"
LINEGRAPH = ROOT / "src/extractors/extract_line_graph_features.py"
# (nome da view, método do segmentador) -- as chaves são as do registro.
PAIRS = (("Line graph", "slic"), ("L(G_felz)", "felzenszwalb"))
COUNTS = (15, 60)


def export_rags(method: str, image_dir: Path, meta_path: Path, side: int,
                work: Path) -> tuple[Path, Path]:
    """Roda o multigrafo só para exportar o RAG. NÃO cronometrado."""
    out = work / f"seg_{method}_{side}"
    shutil.rmtree(out, ignore_errors=True)
    graphs = out / "graphs"
    extra = STEPS["SLIC" if method == "slic" else "Felzenszwalb"][2]
    command = [str(PYTHON), str(SEGMENTER), "--image-dir", str(image_dir),
               "--meta", str(meta_path), "--output-dir", str(out / "features"),
               "--export-rag-dir", str(graphs), "--max-side", str(side),
               "--workers", "8", *extra]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1][:200])
    return out / "index.csv", graphs


def time_line_graph(index: Path, graphs: Path, count: int, work: Path) -> float:
    sub = work / f"sample_{count}.csv"
    pd.read_csv(index, dtype={"stem": str}).head(count).to_csv(sub, index=False)
    output = work / f"lg_{count}.parquet"
    output.unlink(missing_ok=True)
    command = [str(PYTHON), str(LINEGRAPH), "--sample", str(sub),
               "--graph-dir", str(graphs), "--output", str(output),
               "--workers", "1", "--overwrite"]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1][:200])
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    ap.add_argument("--image-root", type=Path,
                    default=ROOT / "data/multitask_painting100k/original/images")
    ap.add_argument("--sides", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", type=Path, default=ROOT / "outputs/linegraph_benchmark")
    ap.add_argument("--output", type=Path, default=ROOT / "results/modality_cost_linegraph.csv")
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    # As imagens estão em `images/<style>/<filename>`, não soltas no topo -- é o
    # mesmo caminho que o benchmark principal monta. Medir com o padrão errado
    # custou duas correções de resolução em 1/set (§8).
    meta["path"] = (str(args.image_root) + "/" + meta["style"].astype(str)
                    + "/" + meta["filename"].astype(str))
    sample = build_cohort(meta, max(COUNTS), args.seed)
    meta_path = args.work_dir / "meta.csv"
    sample[["stem", "filename", "style"]].to_csv(meta_path, index=False)
    log(f"{len(sample)} imagens, lados {args.sides}, {args.repeats} repetições")

    rows = []
    for side in args.sides:
        image_dir = materialise(sample, side, args.work_dir)
        for view, method in PAIRS:
            index, graphs = export_rags(method, image_dir, meta_path, side, args.work_dir)
            segments = pd.read_parquet(
                args.work_dir / f"seg_{method}_{side}/features/combined_features.parquet",
                columns=["segmentation_actual_segments"]
            )["segmentation_actual_segments"].median()
            timings = {}
            for count in COUNTS:
                timings[count] = min(time_line_graph(index, graphs, count, args.work_dir)
                                     for _ in range(args.repeats))
            per_image = 1000 * (timings[COUNTS[1]] - timings[COUNTS[0]]) / (COUNTS[1] - COUNTS[0])
            rows.append(dict(step=view, side=side, ms_per_image=per_image,
                             median_segments=float(segments)))
            log(f"  {view:12s} lado={side:5d}: {per_image:7.2f} ms/imagem "
                f"({segments:.0f} segmentos)")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    table = pd.DataFrame(rows).pivot_table(index="step", columns="side", values="ms_per_image")
    seg = pd.DataFrame(rows).pivot_table(index="step", columns="side", values="median_segments")
    print(f"\n{'view':14}{'α custo':>9}{'α segmentos':>13}")
    lo, hi = args.sides[0], args.sides[-1]
    for view in table.index:
        a = np.log(table.loc[view, hi] / table.loc[view, lo]) / np.log(hi / lo)
        b = np.log(seg.loc[view, hi] / seg.loc[view, lo]) / np.log(hi / lo)
        print(f"{view:14}{a:9.2f}{b:13.2f}")


if __name__ == "__main__":
    main()
