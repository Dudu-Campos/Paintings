#!/usr/bin/env python3
"""Freeze a balanced Big-LaMa pilot and build non-generative controls."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_STYLES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
]


def resize_max_side(image: Image.Image, max_side: int = 512) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def write_manifest(frame: pd.DataFrame, relative_column: str, path: Path) -> None:
    manifest = frame[["stem", "style", relative_column]].rename(
        columns={"stem": "sample_id", relative_column: "image_relative_path"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def build_controls(task: tuple[str, str, str, str, str]) -> str:
    source_value, outpaint_value, relative_value, resized_root_value, blank_root_value = task
    relative = Path(relative_value)
    resized_path = Path(resized_root_value) / relative
    blank_path = Path(blank_root_value) / relative
    resized_path.parent.mkdir(parents=True, exist_ok=True)
    blank_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_value) as opened:
        resized = resize_max_side(opened.convert("RGB"))
    with Image.open(outpaint_value) as opened:
        outpaint = opened.convert("RGB")
        output_size = outpaint.size

    left = (output_size[0] - resized.width) // 2
    top = (output_size[1] - resized.height) // 2
    center = np.asarray(
        outpaint.crop((left, top, left + resized.width, top + resized.height))
    )
    if not np.array_equal(center, np.asarray(resized)):
        raise AssertionError(f"Center mismatch for {relative}")
    if not resized_path.exists():
        resized.save(resized_path, format="PNG", compress_level=1)
    if not blank_path.exists():
        blank = Image.new("RGB", output_size, (0, 0, 0))
        blank.paste(resized, (left, top))
        blank.save(blank_path, format="PNG", compress_level=1)
    return relative_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=Path("data/wikiart_all_images.csv"))
    parser.add_argument("--outpaint-root", type=Path, default=Path("outputs/big_lama_outpaint_full"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/big_lama_visual_pilot"))
    parser.add_argument("--sample-csv", type=Path, default=Path("paper/big_lama_visual_pilot_sample.csv"))
    parser.add_argument("--per-style", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--styles", nargs="+", default=DEFAULT_STYLES)
    parser.add_argument("--original-slic", type=Path, default=Path("outputs/wikiart_slic_features/combined_features.parquet"))
    parser.add_argument("--original-features", type=Path, default=Path("outputs/wikiart_features"))
    args = parser.parse_args()

    canonical = pd.read_csv(args.canonical)
    stem_counts = canonical["stem"].astype(str).value_counts()
    canonical = canonical[
        canonical["style"].isin(args.styles)
        & canonical["stem"].astype(str).map(stem_counts).eq(1)
    ].copy()
    common_stems = set(
        pd.read_parquet(args.original_slic, columns=["stem"])["stem"].astype(str)
    )
    for stems_name in ("desc_full_stems.npy", "hilbert_stems.npy"):
        common_stems &= set(
            np.load(args.original_features / stems_name, allow_pickle=True).astype(str)
        )
    canonical = canonical[canonical["stem"].astype(str).isin(common_stems)].copy()
    canonical["outpaint_relative"] = canonical["image_id"].map(
        lambda value: str(Path(value).with_suffix(".png"))
    )
    canonical["outpaint_path"] = canonical["outpaint_relative"].map(
        lambda value: args.outpaint_root / value
    )
    canonical = canonical[canonical["outpaint_path"].map(Path.is_file)].copy()

    counts = canonical["style"].value_counts()
    missing = [style for style in args.styles if counts.get(style, 0) < args.per_style]
    if missing:
        raise RuntimeError(
            "Not enough completed unique-stem images: "
            + ", ".join(f"{style}={counts.get(style, 0)}" for style in missing)
        )
    selected = (
        canonical.groupby("style", group_keys=False)
        .sample(n=args.per_style, random_state=args.seed)
        .sort_values(["style", "stem"])
        .reset_index(drop=True)
    )

    resized_root = args.output_root / "resized"
    blank_root = args.output_root / "blank_canvas"
    selected["resized_relative"] = selected["image_id"].map(
        lambda value: str(Path(value).with_suffix(".png"))
    )
    selected["blank_relative"] = selected["resized_relative"]
    tasks = [
        (
            str(row.path), str(row.outpaint_path), str(row.resized_relative),
            str(resized_root), str(blank_root),
        )
        for row in selected.itertuples(index=False)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_controls, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 500 == 0 or done == len(tasks):
                print(f"controls {done}/{len(tasks)}", flush=True)
    args.sample_csv.parent.mkdir(parents=True, exist_ok=True)
    selected[
        ["stem", "style", "path", "image_id", "outpaint_relative", "resized_relative", "blank_relative"]
    ].to_csv(args.sample_csv, index=False)
    write_manifest(selected, "outpaint_relative", args.output_root / "outpaint_manifest.csv")
    write_manifest(selected, "resized_relative", args.output_root / "resized_manifest.csv")
    write_manifest(selected, "blank_relative", args.output_root / "blank_manifest.csv")
    print(f"sample={args.sample_csv}")
    print(selected["style"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    main()
