"""Decompoe o custo dos segmentadores em termo fixo (decode/IO) e termo por pixel.

O §5.4b registra alpha=2,06 para o Felzenszwalb, medido entre 128 e 1024 px numa
imagem sintetica de tamanho controlado. A coorte real do Multitask tem mediana de
341 px, entao a extrapolacao por alpha nunca foi confrontada com o custo de
producao. Aqui os estagios sao cronometrados separadamente, sobre imagens reais.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import felzenszwalb, slic

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]

# parametros de producao (src/extractors/segmentation_multigraph_features.py)
FELZ = dict(scale=100.0, sigma=0.8, min_size=20)
SLIC = dict(n_segments=2048, compactness=10.0, sigma=0.0)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path,
                    default=Path("/tmp/prism_data/multitask_painting100k/images"))
    ap.add_argument("--n-images", type=int, default=80)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--caps", type=int, nargs="+", default=[0, 512, 341, 256, 192, 128])
    ap.add_argument("--output", type=Path, default=ROOT / "results/segmentation_resolution_cost.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(args.image_dir.glob("*.jpg"))
    random.seed(args.seed)
    sample = random.sample(files, min(args.n_images, len(files)))
    print(f"{len(sample)} imagens de {len(files)}, {args.reps} repeticoes, "
          f"caps {args.caps} (0 = nativo)", flush=True)

    rows = []
    for rep in range(args.reps):
        for cap in args.caps:
            for path in sample:
                # decode: sempre pago, independe do cap
                (img,), t_decode = timed(lambda: (Image.open(path).convert("RGB"),))
                w, h = img.size
                native = max(w, h)

                if cap and native > cap:
                    scale = cap / native
                    target = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                    resized, t_resize = timed(lambda: img.resize(target, Image.BILINEAR))
                else:
                    resized, t_resize = img, 0.0
                rgb = np.asarray(resized)
                side = max(rgb.shape[:2])

                lab_f, t_felz = timed(lambda: felzenszwalb(rgb, channel_axis=-1, **FELZ))
                lab_s, t_slic = timed(lambda: slic(rgb, convert2lab=True,
                                                   enforce_connectivity=True, start_label=0,
                                                   channel_axis=-1, **SLIC))
                img.close()

                rows.append(dict(rep=rep, cap=cap, stem=path.stem, native=native, side=side,
                                 pixels=rgb.shape[0] * rgb.shape[1],
                                 decode_ms=1e3 * t_decode, resize_ms=1e3 * t_resize,
                                 felz_ms=1e3 * t_felz, slic_ms=1e3 * t_slic,
                                 felz_segments=int(lab_f.max()) + 1,
                                 slic_segments=int(lab_s.max()) + 1))
            done = pd.DataFrame(rows)
            cur = done[(done.rep == rep) & (done.cap == cap)]
            print(f"  rep {rep} cap {cap:4d}  lado medio {cur.side.mean():5.0f}  "
                  f"decode {cur.decode_ms.mean():6.1f}  felz {cur.felz_ms.mean():7.1f}  "
                  f"slic {cur.slic_ms.mean():7.1f} ms", flush=True)

    frame = pd.DataFrame(rows)
    # a primeira repeticao paga cache frio e aquecimento de JIT; descartar
    warm = frame[frame.rep > 0] if args.reps > 1 else frame
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    agg = warm.groupby("cap").agg(
        lado=("side", "mean"), pixels=("pixels", "mean"),
        decode=("decode_ms", "mean"), resize=("resize_ms", "mean"),
        felz=("felz_ms", "mean"), slic=("slic_ms", "mean"),
        seg_felz=("felz_segments", "mean"), seg_slic=("slic_segments", "mean"),
    ).reset_index()
    agg["fixo"] = agg.decode + agg.resize
    agg["pixel"] = agg.felz + agg.slic
    agg["total"] = agg.fixo + agg.pixel

    base = agg[agg.cap == 0].iloc[0] if (agg.cap == 0).any() else agg.iloc[0]
    print(f"\n{'cap':>5} {'lado':>5} {'decode':>7} {'resize':>7} {'felz':>8} {'slic':>8} "
          f"{'FIXO':>7} {'PIXEL':>8} {'TOTAL':>8} {'ganho':>6} {'seg_felz':>9}")
    for _, r in agg.iterrows():
        print(f"{int(r['cap']):>5} {r['lado']:>5.0f} {r['decode']:>7.1f} {r['resize']:>7.1f} "
              f"{r['felz']:>8.1f} {r['slic']:>8.1f} {r['fixo']:>7.1f} {r['pixel']:>8.1f} "
              f"{r['total']:>8.1f} {base['total'] / r['total']:>5.2f}x {r['seg_felz']:>9.0f}")

    print(f"\ncusto fixo (decode+resize) = {100 * base['fixo'] / base['total']:.1f}% do total no nativo")
    floor = agg.fixo.max()
    print(f"piso teorico com segmentacao GRATIS: {floor:.1f} ms/img "
          f"-> ganho maximo {base['total'] / floor:.2f}x")
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
