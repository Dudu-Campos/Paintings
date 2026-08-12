"""Build a controlled image-perturbation subset for WikiArt robustness tests.

The output is deliberately a *derived* dataset: every row keeps its original
``parent_id`` so that all variants of one painting can be kept in the same CV
fold.  It is intended to diagnose modality invariance, not to replace the
canonical WikiArt training set.

Example::

    python src/data/build_wikiart_robustness.py \
        --image-dir wikiart --meta data/artemis/wikiart_meta.csv \
        --output-dir data/wikiart_robustness --per-style 100
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from slic_graph_features import resolve_wikiart_records  # noqa: E402


VARIANTS = ("original", "color_jitter", "crop_zoom", "jpeg_35", "rotate", "hflip")
Image.MAX_IMAGE_PIXELS = None


def stable_seed(parent_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{parent_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def canonical_image(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image.copy()


def transform(image: Image.Image, variant: str, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    """Apply exactly one label-preserving, recorded perturbation."""
    if variant == "original":
        return image, {}
    if variant == "hflip":
        return ImageOps.mirror(image), {"operation": "horizontal_flip"}
    if variant == "color_jitter":
        brightness = float(rng.uniform(0.88, 1.12))
        contrast = float(rng.uniform(0.88, 1.12))
        saturation = float(rng.uniform(0.88, 1.12))
        changed = ImageEnhance.Brightness(image).enhance(brightness)
        changed = ImageEnhance.Contrast(changed).enhance(contrast)
        changed = ImageEnhance.Color(changed).enhance(saturation)
        return changed, {
            "brightness": round(brightness, 5),
            "contrast": round(contrast, 5),
            "saturation": round(saturation, 5),
        }
    if variant == "crop_zoom":
        fraction = float(rng.uniform(0.84, 0.92))
        width, height = image.size
        crop_width, crop_height = max(1, round(width * fraction)), max(1, round(height * fraction))
        left = int(rng.integers(0, width - crop_width + 1))
        top = int(rng.integers(0, height - crop_height + 1))
        crop = image.crop((left, top, left + crop_width, top + crop_height))
        changed = crop.resize((width, height), Image.Resampling.LANCZOS)
        return changed, {
            "crop_fraction": round(fraction, 5),
            "crop_box": [left, top, left + crop_width, top + crop_height],
        }
    if variant == "jpeg_35":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=35, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy(), {"jpeg_quality": 35, "subsampling": "4:2:0"}
    if variant == "rotate":
        angle = float(rng.choice((-7.0, 7.0)))
        median = tuple(np.median(np.asarray(image), axis=(0, 1)).astype(np.uint8).tolist())
        changed = image.rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=median
        )
        return changed, {"angle_degrees": angle, "fill": "image_rgb_median"}
    raise ValueError(f"Unknown variant: {variant}")


def build_one(task: tuple[str, str, str, str, str, int, int, bool]) -> tuple[str, dict | None, str | None]:
    source, destination, sample_id, parent_id, variant, seed, max_side, overwrite = task
    destination_path = Path(destination)
    try:
        image = canonical_image(Path(source), max_side)
        rng = np.random.default_rng(stable_seed(f"{parent_id}:{variant}", seed))
        changed, parameters = transform(image, variant, rng)
        if not destination_path.exists() or overwrite:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            changed.save(destination_path, format="JPEG", quality=92, subsampling=0)
        return sample_id, parameters, None
    except Exception as exc:  # keep workers serializable and report every failed image
        return sample_id, None, f"{parent_id}/{variant}: {type(exc).__name__}: {exc}"


def select_records(records: pd.DataFrame, per_style: int, seed: int) -> pd.DataFrame:
    selected = []
    for style, group in records.groupby("style", sort=True):
        count = min(per_style, len(group))
        if count < per_style:
            print(
                f"WARNING: style {style!r} has only {len(group)} valid images; "
                f"using all {count} instead of {per_style}.",
                flush=True,
            )
        # pandas' legacy RandomState accepts only 32-bit integer seeds.
        selected.append(group.sample(n=count, random_state=stable_seed(style, seed) % (2**32)))
    return pd.concat(selected, ignore_index=True).sort_values(["style", "stem"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "wikiart_robustness")
    parser.add_argument("--per-style", type=int, default=100)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.per_style < 1 or args.max_side < 1 or args.workers < 1:
        parser.error("--per-style, --max-side and --workers must be positive")

    records = resolve_wikiart_records(args.meta, args.image_dir)
    selected = select_records(records, args.per_style, args.seed)
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    print(
        f"Selected {len(selected):,} parents across {selected['style'].nunique()} styles; "
        f"will create {len(selected) * len(VARIANTS):,} images ({', '.join(VARIANTS)}).",
        flush=True,
    )

    manifest_rows: list[dict] = []
    tasks = []
    for record in selected.itertuples(index=False):
        source = Path(record.path).resolve()
        parent_id = f"{record.style}/{record.stem}"
        parent_key = hashlib.sha256(parent_id.encode()).hexdigest()[:16]
        source_relative = str(source.relative_to(args.image_dir.resolve()))
        for variant in VARIANTS:
            relative_path = Path("images") / record.style / f"{parent_key}__{variant}.jpg"
            destination = output_dir / relative_path
            sample_id = f"{parent_key}__{variant}"
            manifest_rows.append({
                "sample_id": sample_id,
                "parent_id": parent_id,
                "parent_key": parent_key,
                "stem": record.stem,
                "style": record.style,
                "variant": variant,
                "source_relative_path": source_relative,
                "image_relative_path": str(relative_path),
                "transform_parameters": None,
                "canonical_max_side": args.max_side,
                "seed": args.seed,
            })
            tasks.append((str(source), str(destination), sample_id, parent_id, variant,
                          args.seed, args.max_side, args.overwrite))

    manifest = pd.DataFrame(manifest_rows)
    if manifest["sample_id"].duplicated().any() or manifest["parent_id"].isna().any():
        raise RuntimeError("Invalid robustness manifest identifiers")
    if args.dry_run:
        print(manifest.groupby(["style", "variant"]).size().unstack(fill_value=0).to_string())
        return

    failures = []
    parameters_by_sample: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            sample_id, parameters, error = future.result()
            if error:
                failures.append(error)
            else:
                parameters_by_sample[sample_id] = parameters or {}
            if count % 500 == 0 or count == len(futures):
                print(f"Generated {count:,}/{len(futures):,} images", flush=True)
    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(f"Failed to generate {len(failures)} images:\n{preview}")

    manifest["transform_parameters"] = manifest["sample_id"].map(
        lambda sample_id: json.dumps(parameters_by_sample[sample_id], sort_keys=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    summary = {
        "parents": int(len(selected)), "styles": int(selected["style"].nunique()),
        "variants": list(VARIANTS), "images": int(len(manifest)),
        "per_style": args.per_style, "max_side": args.max_side, "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output_dir / 'manifest.csv'} and {len(manifest):,} images.", flush=True)


if __name__ == "__main__":
    main()
