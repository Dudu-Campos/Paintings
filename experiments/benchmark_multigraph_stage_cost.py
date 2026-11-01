"""Custo do caminho de producao dos segmentadores, estagio a estagio.

O §5.4b atribui 888,3 ms/img ao Felzenszwalb e 635,7 ao SLIC, medidos numa
imagem de 1024 px. A coorte real do Multitask tem mediana de 341 px, e
`benchmark_segmentation_resolution.py` mede a CHAMADA de segmentacao em 86 ms
para os dois juntos. A diferenca tem de estar no resto de `extract_one`:
construcao dos quatro grafos (RAG, kNN, Delaunay, MST) e as features de cada um.

Aqui cada estagio e cronometrado separado, no caminho de producao real.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
Image.MAX_IMAGE_PIXELS = None

from src.pipeline.segmentation_multigraph_features import (  # noqa: E402
    build_delaunay_edges, build_knn_edges, build_mst_edges,
    extract_graph_features, graph_feature_dict, segment_region_graph,
)

# Lidos de `segmentation_parameters` nos parquets de producao -- os defaults do
# script NAO sao o que rodou (felz_scale 50 e min_size 50, nao 100/20), e usar os
# defaults da 352 segmentos contra os 1.398 reais.
BASE = dict(segments=2048, max_side=1024, export_rag_dir=None, k=10, seed=42,
            compactness=10.0, sigma=0.0, felz_scale=50.0, felz_sigma=0.8,
            felz_min_size=50, quick_ratio=1.0, quick_kernel_size=5.0,
            quick_max_dist=10.0, quick_sigma=0.0, watershed_compactness=0.001)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, 1e3 * (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path,
                    default=Path("/tmp/prism_data/multitask_painting100k/images"))
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--caps", type=int, nargs="+", default=[1024, 512, 256, 128])
    ap.add_argument("--methods", nargs="+", default=["felzenszwalb", "slic"])
    ap.add_argument("--output", type=Path, default=ROOT / "results/multigraph_stage_cost.csv")
    args = ap.parse_args()

    files = sorted(args.image_dir.glob("*.jpg"))
    random.seed(42)
    sample = random.sample(files, min(args.n_images, len(files)))
    print(f"{len(sample)} imagens, {args.reps} reps, caps {args.caps}, "
          f"metodos {args.methods}", flush=True)

    rows = []
    for rep in range(args.reps):
        for method in args.methods:
            for cap in args.caps:
                params = {**BASE, "max_side": cap}
                for path in sample:
                    p = str(path)
                    try:
                        (nodes, rag_edges), t_seg = timed(
                            lambda: segment_region_graph(p, method, params))
                        knn, t_knn = timed(lambda: build_knn_edges(nodes, k=int(params["k"])))
                        dela, t_del = timed(lambda: build_delaunay_edges(nodes))
                        mst, t_mst = timed(lambda: build_mst_edges(nodes, rag_edges))
                        _, t_frag = timed(lambda: graph_feature_dict(nodes, rag_edges))
                        _, t_fknn = timed(lambda: extract_graph_features(nodes, knn))
                        _, t_fdel = timed(lambda: graph_feature_dict(nodes, dela))
                        _, t_fmst = timed(lambda: graph_feature_dict(nodes, mst))
                    except Exception as exc:
                        print(f"  ! {path.stem}: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    rows.append(dict(
                        rep=rep, method=method, cap=cap, stem=path.stem, n_nodes=len(nodes),
                        seg=t_seg, build_knn=t_knn, build_delaunay=t_del, build_mst=t_mst,
                        feat_rag=t_frag, feat_knn=t_fknn, feat_delaunay=t_fdel, feat_mst=t_fmst))
                cur = pd.DataFrame(rows)
                cur = cur[(cur.rep == rep) & (cur.method == method) & (cur.cap == cap)]
                tot = cur[["seg", "build_knn", "build_delaunay", "build_mst",
                           "feat_rag", "feat_knn", "feat_delaunay", "feat_mst"]].sum(axis=1)
                print(f"  rep {rep} {method:13s} cap {cap:5d}  nos {cur.n_nodes.mean():6.0f}  "
                      f"seg {cur.seg.mean():7.1f}  total {tot.mean():8.1f} ms", flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    warm = frame[frame.rep > 0] if args.reps > 1 else frame
    stages = ["seg", "build_knn", "build_delaunay", "build_mst",
              "feat_rag", "feat_knn", "feat_delaunay", "feat_mst"]

    print(f"\n{'metodo':<14}{'cap':>6}{'nos':>7}" + "".join(f"{s:>10}" for s in stages)
          + f"{'TOTAL':>10}")
    agg = warm.groupby(["method", "cap"]).agg({"n_nodes": "mean", **{s: "mean" for s in stages}})
    for (method, cap), r in agg.iterrows():
        total = sum(r[s] for s in stages)
        print(f"{method:<14}{cap:>6}{r['n_nodes']:>7.0f}"
              + "".join(f"{r[s]:>10.1f}" for s in stages) + f"{total:>10.1f}")

    print(f"\n--- por imagem, os dois metodos somados (caminho de producao completo) ---")
    for cap in sorted(warm.cap.unique()):
        sel = agg.xs(cap, level="cap")
        tot = sum(sel[s].sum() for s in stages)
        seg = sel["seg"].sum()
        print(f"  cap {cap:5d}: TOTAL {tot:8.1f} ms/img  (segmentacao {seg:7.1f} = "
              f"{100*seg/tot:4.1f}%, resto {tot-seg:7.1f} = {100*(tot-seg)/tot:4.1f}%)")
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
