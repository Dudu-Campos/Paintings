"""Create a metadata-shaped local Places-LT image view.

The Places365 tar layout is ``data_256/a/airfield/00000004.jpg`` while our
metadata uses class ids such as ``a_airfield`` and validation/test filenames
without class folders. This script builds symlinks at:

    /tmp/prism_data/places_lt/images/<style>/<filename>

so the generic fine-tuning trainer can resolve images by ``(style, filename)``.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTS = (".jpg", ".jpeg", ".png")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def category_map(categories: Path) -> dict[str, str]:
    """`style` do metadado -> caminho relativo dentro de `data_256`.

    `style.split("_", 1)` NAO serve: 75 das 365 categorias do Places365 tem
    subpasta (`/a/athletic_field/outdoor`), e o split ingenuo produz
    `a/athletic_field_outdoor`, que nao existe. O resultado era silencioso --
    aquelas classes ficavam so com as 100 imagens de teste, que vem do
    `val_256` plano, e faltavam 10.227 imagens de treino no total.

    `categories_places365.txt` e a fonte canonica: cada linha e
    `/a/athletic_field/outdoor 3`.
    """
    mapping = {}
    for line in categories.read_text().splitlines():
        if not line.strip():
            continue
        relative = line.split()[0].lstrip("/")
        mapping[relative.replace("/", "_")] = relative
    return mapping


def train_source(raw: Path, style: str, filename: str, mapping: dict[str, str]) -> Path:
    relative = mapping.get(style)
    if relative is None:
        raise KeyError(f"style ausente em categories_places365.txt: {style}")
    return raw / "data_256" / relative / filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/places_lt_meta.csv")
    parser.add_argument("--raw", type=Path, default=Path("/tmp/prism_data/places_lt/raw"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/prism_data/places_lt/images"))
    parser.add_argument("--categories", type=Path,
                        default=Path("data/places_lt/categories_places365.txt"),
                        help="mapa canonico style -> caminho; sem ele 75 classes ficam incompletas")
    args = parser.parse_args()

    meta = pd.read_csv(args.meta, dtype={"stem": str, "filename": str, "style": str, "split": str})
    mapping = category_map(args.categories)
    val_by_name = {
        path.name: path for path in (args.raw / "val_256").glob("*")
        if path.suffix.lower() in EXTS
    }
    args.output.mkdir(parents=True, exist_ok=True)

    linked = existing = missing = 0
    missing_examples: list[str] = []
    for row in meta.itertuples(index=False):
        style = str(row.style)
        filename = str(row.filename)
        split = str(row.split)
        if filename.startswith("Places365_val_"):
            source = val_by_name.get(filename, args.raw / "val_256" / filename)
        else:
            source = train_source(args.raw, style, filename, mapping)
        target = args.output / style / filename
        if target.exists():
            existing += 1
            continue
        if not source.exists():
            missing += 1
            if len(missing_examples) < 10:
                missing_examples.append(f"{split}:{style}/{filename}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, target)
        linked += 1

    log(f"linked={linked:,} existing={existing:,} missing={missing:,} output={args.output}")
    if missing:
        raise SystemExit("missing examples: " + "; ".join(missing_examples))


if __name__ == "__main__":
    main()
