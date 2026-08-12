#!/usr/bin/env python3
"""Deterministic, prompt-free outpainting with the official Big-LaMa model."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image


def ceil_modulo(value: float, modulo: int = 8) -> int:
    return int(math.ceil(value / modulo) * modulo)


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def load_generator(
    lama_root: Path, config_path: Path, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    # saicinpainting.utils imports this training helper at module import time,
    # although generator-only inference never calls it.
    if "pytorch_lightning" not in sys.modules:
        lightning = types.ModuleType("pytorch_lightning")

        def seed_everything(seed: int) -> int:
            torch.manual_seed(seed)
            np.random.seed(seed)
            return seed

        lightning.seed_everything = seed_everything
        sys.modules["pytorch_lightning"] = lightning
    sys.path.insert(0, str(lama_root.resolve()))
    from saicinpainting.training.modules import make_generator

    config = OmegaConf.load(config_path)
    generator = make_generator(config, **config.generator)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = generator.load_state_dict(
        payload["generator_state_dict"], strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"State mismatch: missing={missing}, unexpected={unexpected}")
    generator.eval().requires_grad_(False).to(device)
    return generator


def outpaint(
    generator: torch.nn.Module,
    source: Image.Image,
    expansion: float,
    device: torch.device,
) -> tuple[Image.Image, Image.Image]:
    source_u8 = np.asarray(source.convert("RGB"), dtype=np.uint8)
    height, width = source_u8.shape[:2]
    out_width = ceil_modulo(width * (1.0 + expansion))
    out_height = ceil_modulo(height * (1.0 + expansion))
    left = (out_width - width) // 2
    top = (out_height - height) // 2

    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    mask = np.ones((out_height, out_width), dtype=np.uint8)
    canvas[top : top + height, left : left + width] = source_u8
    mask[top : top + height, left : left + width] = 0

    image_tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32).div_(255.0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.to(device=device, dtype=torch.float32)
    model_input = torch.cat([image_tensor * (1.0 - mask_tensor), mask_tensor], dim=1)

    with torch.inference_mode():
        prediction = generator(model_input)
        result = mask_tensor * prediction + (1.0 - mask_tensor) * image_tensor
    result_u8 = (
        result[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
    )

    # Scientific invariant: the resized original is byte-identical in the center.
    result_u8[top : top + height, left : left + width] = source_u8
    if not np.array_equal(
        result_u8[top : top + height, left : left + width], source_u8
    ):
        raise AssertionError("Central source pixels changed")

    mask_u8 = mask * 255
    return Image.fromarray(result_u8, "RGB"), Image.fromarray(mask_u8, "L")


def select_rows(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        return pd.DataFrame(
            {
                "path": [str(path.resolve()) for path in args.input],
                "style": ["unknown"] * len(args.input),
                "image_id": [path.stem for path in args.input],
            }
        )

    frame = pd.read_csv(args.manifest)
    required = {"path", "style", "image_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if args.sample_per_style:
        frame = (
            frame.groupby("style", group_keys=False)
            .sample(n=args.sample_per_style, random_state=args.seed)
            .sort_values(["style", "image_id"])
        )
    if args.max_images:
        frame = frame.head(args.max_images)
    return frame.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--input", type=Path, action="append")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--lama-root", type=Path, default=Path("outputs/external/lama"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("outputs/big_lama_model/big-lama/config.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/big_lama_model/big-lama/models/generator_state.pt"),
    )
    parser.add_argument("--expansion", type=float, default=0.25)
    parser.add_argument("--max-original-side", type=int, default=512)
    parser.add_argument("--sample-per-style", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    if args.expansion <= 0:
        parser.error("--expansion must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = select_rows(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    generator = load_generator(
        args.lama_root, args.model_config, args.checkpoint, device
    )

    records: list[dict[str, object]] = []
    for index, row in rows.iterrows():
        source_path = Path(row.path)
        style = str(row.style)
        image_id = str(row.image_id)
        image_id_path = Path(image_id)
        if image_id_path.parent == Path("."):
            image_id_path = Path(style) / image_id_path
        relative_output = image_id_path.with_suffix(".png")
        output_path = args.outdir / relative_output
        mask_path = (
            args.outdir
            / "masks"
            / relative_output.parent
            / f"{relative_output.stem}_mask.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.save_masks:
            mask_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        status = "generated"
        try:
            with Image.open(source_path) as opened:
                original_size = opened.size
                resized = resize_max_side(opened.convert("RGB"), args.max_original_side)
            if output_path.exists() and not args.overwrite:
                status = "existing"
                output_size = Image.open(output_path).size
            else:
                generated, mask = outpaint(
                    generator, resized, args.expansion, device
                )
                generated.save(
                    output_path,
                    format="PNG",
                    compress_level=args.png_compress_level,
                )
                if args.save_masks:
                    mask.save(
                        mask_path,
                        format="PNG",
                        compress_level=args.png_compress_level,
                    )
                output_size = generated.size
            error = ""
        except Exception as exc:
            status = "error"
            original_size = (0, 0)
            resized = Image.new("RGB", (1, 1))
            output_size = (0, 0)
            error = f"{type(exc).__name__}: {exc}"
            if device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.perf_counter() - started
        records.append(
            {
                "image_id": image_id,
                "style": style,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "status": status,
                "error": error,
                "original_width": original_size[0],
                "original_height": original_size[1],
                "resized_width": resized.size[0],
                "resized_height": resized.size[1],
                "output_width": output_size[0],
                "output_height": output_size[1],
                "expansion": args.expansion,
                "elapsed_seconds": elapsed,
            }
        )
        if (
            (index + 1) % args.log_every == 0
            or index == 0
            or index + 1 == len(rows)
            or status == "error"
        ):
            print(
                f"[{index + 1}/{len(rows)}] {status} {image_id} {elapsed:.3f}s",
                flush=True,
            )

    manifest_path = args.outdir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    generated_times = [
        float(record["elapsed_seconds"])
        for record in records
        if record["status"] == "generated"
    ]
    print(f"manifest={manifest_path}")
    print(f"rows={len(records)} errors={sum(r['status'] == 'error' for r in records)}")
    if generated_times:
        print(f"mean_seconds={np.mean(generated_times):.4f}")
        print(f"images_per_second={1.0 / np.mean(generated_times):.4f}")


if __name__ == "__main__":
    main()
