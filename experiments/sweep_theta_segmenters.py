"""theta-robustez dos dois segmentadores: o caso em que os eixos NAO se separam.

Quarta e quinta representacoes varridas, e as ultimas do banco visual. O §5.17d
ja avisava que o desenho de dois eixos usado em Hilbert/Gabor/Fourier **nao se
transporta** para ca, e a razao e medida no §9.3: `felz_scale` e `min_size` estao
em unidades de PIXEL, entao mudar `--max-side` muda a granularidade junto (282
segmentos a 1024 px viram 50 a 128 px). No SLIC nao: `--segments 2048` e alvo
explicito, e baixar a resolucao ate AUMENTA a contagem de nos (1817 -> 2376).

**Os dois tem acoplamento oposto, e e esse o achado desta varredura:**

    SLIC   resolucao e granularidade SEPARAM   -> o desenho de dois eixos vale
    Felz   os dois sao o MESMO botao           -> o desenho nao vale, e declara-se

**Desenho em CRUZ, nao grade cheia.** Uma grade 4x3 nos dois custaria ~11 h de
CPU; a cruz (variar um eixo por vez passando por theta*) custa 5,5 h e responde
a mesma pergunta, porque a alegacao do §5.17 e sobre a VIZINHANCA de theta*, nao
sobre a superficie inteira. Isto e uma diferenca de desenho em relacao as tres
varreduras anteriores e tem de ser declarada ao reportar.

theta* de producao, lido do parquet e nao do argparse (§8: os defaults do
extrator sao 100/20/8 e a producao usou 50/50/10):

    SLIC   --max-side 1024  --segments 2048  --compactness 10  -k 10
    Felz   --max-side 1024  --felz-scale 50  --felz-min-size 50  -k 10

Emite, por configuracao: contagem de segmentos realizada (o diagnostico que
separa os dois casos), CKA contra theta*, e BAcc isolada sob LDA.
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
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from evaluate_neighbour_attention import key  # noqa: E402
from sweep_theta_hilbert import linear_cka  # noqa: E402

PYTHON = ROOT / ".ABSvenv/bin/python3"
EXTRACT = ROOT / "src/extractors/segmentation_multigraph_features.py"

# (rotulo, max_side, granularidade). O primeiro de cada lista e theta*.
PLANS = {
    "slic": {"star": (1024, 2048), "gran_flag": "--segments",
             "grid": [(1024, 2048),                               # theta*
                      (1024, 512), (1024, 1024), (1024, 4096),    # granularidade
                      (512, 2048), (256, 2048)]},                 # resolucao
    "felzenszwalb": {"star": (1024, 50), "gran_flag": "--felz-scale",
                     "grid": [(1024, 50),
                              (1024, 25), (1024, 100), (1024, 200),
                              (512, 50), (256, 50)]},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def extract(method, image_dir, meta, out, max_side, gran, gran_flag, workers):
    marker = out / "combined_features.parquet"
    if marker.exists():
        return float("nan")
    started = time.perf_counter()
    cmd = [str(PYTHON), str(EXTRACT), "--method", method,
           "--image-dir", str(image_dir), "--meta", str(meta),
           "--output-dir", str(out), "--max-side", str(max_side),
           "--k", "10", "--workers", str(workers), gran_flag, str(gran)]
    if method == "felzenszwalb":
        cmd += ["--felz-min-size", "50"]
    else:
        cmd += ["--compactness", "10"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"extração falhou {method} ({max_side},{gran}):\n{r.stderr[-2500:]}")
    return time.perf_counter() - started


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=list(PLANS), required=True)
    ap.add_argument("--image-dir", type=Path,
                    default=ROOT / "data/multitask_painting100k/original/images")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    ap.add_argument("--classes-file", type=Path,
                    default=ROOT / "data/multitask_25class_benchmark.txt")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_segmenters")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = ROOT / f"results/theta_{args.method}.csv"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    plan = PLANS[args.method]
    star = plan["star"]

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    meta = meta[[key(str(v)) in wanted for v in meta["style"]]]
    if meta["style"].nunique() != len(wanted):
        raise SystemExit(f"casaram {meta['style'].nunique()} de {len(wanted)} classes")
    log(f"{args.method}: coorte de {len(meta):,} imagens, {meta['style'].nunique()} classes")
    order = meta.stem.astype(str).to_numpy()

    features, costs, segments = {}, {}, {}
    for max_side, gran in plan["grid"]:
        out = args.work_dir / f"{args.method}_s{max_side}_g{gran}"
        costs[(max_side, gran)] = extract(args.method, args.image_dir, args.meta, out,
                                          max_side, gran, plan["gran_flag"], args.workers)
        frame = pd.read_parquet(out / "combined_features.parquet")
        frame["stem"] = frame["stem"].astype(str)
        frame = frame.drop_duplicates("stem").set_index("stem")
        cols = [c for c in frame.columns
                if frame[c].dtype.kind in "fiub" and not c.startswith(
                    ("slic_", "felz_", "elapsed", "segmentation"))]
        aligned = frame.reindex(order)
        if aligned[cols].isna().all(axis=1).any():
            missing = int(aligned[cols].isna().all(axis=1).sum())
            raise ValueError(f"({max_side},{gran}): {missing} stems sem features")
        values = np.nan_to_num(aligned[cols].to_numpy(np.float64), nan=0.0,
                               posinf=0.0, neginf=0.0)
        features[(max_side, gran)] = values
        # `n_nodes` e o diagnostico que separa SLIC de Felz.
        segments[(max_side, gran)] = float(aligned["n_nodes"].median()) \
            if "n_nodes" in aligned else float("nan")
        c = costs[(max_side, gran)]
        log(f"  ({max_side:>4}, {gran:>4})  {values.shape}  "
            f"mediana de segmentos {segments[(max_side, gran)]:>7.0f}  "
            + (f"{c/60:.1f} min" if c == c else "em cache"))

    labels = LabelEncoder().fit_transform(
        meta.set_index("stem").loc[order, "style"].to_numpy(str))
    n_classes = len(np.unique(labels))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42)
                  .split(np.zeros(len(labels)), labels))[:args.folds]

    rows = []
    for theta in plan["grid"]:
        values = features[theta]
        scores = []
        for train, test in splits:
            sc = StandardScaler().fit(values[train])
            m = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            m.fit(sc.transform(values[train]), labels[train])
            scores.append(100 * balanced_accuracy_score(
                labels[test], m.predict(sc.transform(values[test]))))
        axis = ("theta*" if theta == star else
                "granularidade" if theta[0] == star[0] else "resolucao")
        rows.append(dict(method=args.method, max_side=theta[0], granularity=theta[1],
                         axis=axis, is_star=theta == star,
                         dimensions=values.shape[1],
                         median_segments=segments[theta],
                         cka_vs_star=linear_cka(values, features[star]),
                         extract_seconds=costs[theta],
                         lda_bacc=float(np.mean(scores)), lda_std=float(np.std(scores))))
        log(f"  ({theta[0]:>4}, {theta[1]:>4}) {axis:<14} BAcc {rows[-1]['lda_bacc']:5.2f}  "
            f"CKA {rows[-1]['cka_vs_star']:.3f}")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    t = pd.DataFrame(rows)
    s = t[t.is_star].iloc[0]
    rank = int((t.lda_bacc > s.lda_bacc).sum()) + 1
    log(f"\ntheta* = {star}: rank {rank} de {len(t)} | CKA minimo {t.cka_vs_star.min():.3f} | "
        f"amplitude BAcc {t.lda_bacc.max() - t.lda_bacc.min():.2f}pp")
    for axis in ("granularidade", "resolucao"):
        sub = pd.concat([t[t.axis.eq(axis)], t[t.is_star]])
        log(f"  amplitude em {axis:<14} {sub.lda_bacc.max() - sub.lda_bacc.min():.2f}pp  "
            f"(segmentos {sub.median_segments.min():.0f} a {sub.median_segments.max():.0f})")
    log(f"FIM: {args.output}")


if __name__ == "__main__":
    main()
