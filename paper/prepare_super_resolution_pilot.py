#!/usr/bin/env python3
"""Build paired Real-ESRGAN and interpolation controls for WikiArt images.

Each source is first normalized exactly as the canonical SLIC extractor sees it
(bilinear resize to a maximum side of 1024).  Two x2 variants are then made:

* bicubic_x2: interpolation-only placebo;
* realesrgan_x2: learned restoration with the official RealESRGAN_x2plus model.

The x2 files are deliberately kept at their native output resolution.  Passing
them through the unchanged canonical extractors tests whether the learned
restoration survives those extractors' own 1024/128-pixel caps.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_side:
        return image.copy()
    scale = max_side / max(width, height)
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )


def relative_png(style: str, stem: str) -> Path:
    return Path(style) / f"{stem}.png"


def save_rgb(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=1)


def make_manifest(sample: pd.DataFrame, path: Path) -> None:
    manifest = sample[["stem", "style"]].copy()
    manifest["sample_id"] = manifest["stem"].astype(str)
    manifest["image_relative_path"] = [
        str(relative_png(str(style), str(stem)))
        for stem, style in manifest[["stem", "style"]].itertuples(index=False)
    ]
    manifest[["sample_id", "style", "image_relative_path"]].to_csv(path, index=False)


def build_upsampler(repo: Path, weights: Path, tile: int):
    sys.path.insert(0, str(repo.resolve()))
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=2,
    )
    return RealESRGANer(
        scale=2,
        model_path=str(weights),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        gpu_id=0 if torch.cuda.is_available() else None,
    )


def comparison_metrics(reference: Image.Image, candidate_x2: Image.Image) -> tuple[float, float, float]:
    candidate = candidate_x2.resize(reference.size, Image.Resampling.BILINEAR)
    a = np.asarray(reference, dtype=np.float32) / 255.0
    b = np.asarray(candidate, dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    mae = float(np.mean(np.abs(a - b)))
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    return mae, psnr, ssim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("paper/big_lama_visual_pilot_sample.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/super_resolution_pilot"))
    parser.add_argument("--repo", type=Path, default=Path("outputs/external/Real-ESRGAN"))
    parser.add_argument(
        "--weights", type=Path,
        default=Path("outputs/external/Real-ESRGAN/weights/RealESRGAN_x2plus.pth"),
    )
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    required = {"stem", "style", "path"}
    missing = required - set(sample.columns)
    if missing:
        parser.error(f"sample is missing columns: {sorted(missing)}")
    sample = sample.drop_duplicates("stem").sort_values(["style", "stem"]).reset_index(drop=True)
    if args.limit:
        sample = sample.head(args.limit).copy()
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")

    roots = {
        "normalized_1024": args.output_root / "normalized_1024",
        "bicubic_x2": args.output_root / "bicubic_x2",
        "realesrgan_x2": args.output_root / "realesrgan_x2",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in roots:
        make_manifest(sample, args.output_root / f"{name}_manifest.csv")
    sample[["stem", "style", "path"]].to_csv(args.output_root / "sample.csv", index=False)

    upsampler = build_upsampler(args.repo, args.weights, args.tile)
    metrics_path = args.output_root / "roundtrip_metrics.csv"
    existing: dict[str, dict] = {}
    if metrics_path.exists() and not args.overwrite:
        prior = pd.read_csv(metrics_path)
        existing = {str(row["stem"]): row.to_dict() for _, row in prior.iterrows()}

    rows: list[dict] = []
    started = time.perf_counter()
    for index, row in enumerate(sample.itertuples(index=False), 1):
        stem, style = str(row.stem), str(row.style)
        relative = relative_png(style, stem)
        normalized_path = roots["normalized_1024"] / relative
        bicubic_path = roots["bicubic_x2"] / relative
        sr_path = roots["realesrgan_x2"] / relative

        if stem in existing and all(path.is_file() for path in (normalized_path, bicubic_path, sr_path)):
            rows.append(existing[stem])
            continue

        with Image.open(row.path) as opened:
            normalized = resize_max_side(opened.convert("RGB"), args.max_side)
        bicubic = normalized.resize(
            (normalized.width * 2, normalized.height * 2), Image.Resampling.BICUBIC
        )
        bgr = cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
        # RealESRGANer prints one line per tile; keep the experiment log at the
        # image/checkpoint level instead of emitting four lines per image.
        with contextlib.redirect_stdout(io.StringIO()):
            output_bgr, _ = upsampler.enhance(bgr, outscale=2)
        sr = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))

        if args.overwrite or not normalized_path.exists():
            save_rgb(normalized, normalized_path)
        if args.overwrite or not bicubic_path.exists():
            save_rgb(bicubic, bicubic_path)
        if args.overwrite or not sr_path.exists():
            save_rgb(sr, sr_path)

        bic_mae, bic_psnr, bic_ssim = comparison_metrics(normalized, bicubic)
        sr_mae, sr_psnr, sr_ssim = comparison_metrics(normalized, sr)
        rows.append({
            "stem": stem,
            "style": style,
            "source_width": normalized.width,
            "source_height": normalized.height,
            "bicubic_roundtrip_mae": bic_mae,
            "bicubic_roundtrip_psnr": bic_psnr,
            "bicubic_roundtrip_ssim": bic_ssim,
            "realesrgan_roundtrip_mae": sr_mae,
            "realesrgan_roundtrip_psnr": sr_psnr,
            "realesrgan_roundtrip_ssim": sr_ssim,
        })
        if index % 10 == 0 or index == len(sample):
            elapsed = time.perf_counter() - started
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            print(
                f"{index}/{len(sample)} complete; {index / elapsed:.2f} images/s; "
                f"ETA={(len(sample) - index) / max(index / elapsed, 1e-9) / 60:.1f} min",
                flush=True,
            )

    result = pd.DataFrame(rows).sort_values(["style", "stem"]).reset_index(drop=True)
    result.to_csv(metrics_path, index=False)
    print(result.drop(columns=["stem", "style"]).describe().loc[["mean", "50%"]].to_string())
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    main()
