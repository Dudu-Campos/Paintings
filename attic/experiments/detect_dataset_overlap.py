#!/usr/bin/env python3
"""Detect possible overlap between WikiArt, Pandora18k and Rijksmuseum.

The default pass is metadata based and fast.  Use ``--hash-images`` to add
exact file hashes and a simple perceptual average hash for image-level checks.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    metadata: Path
    image_root: Path | None
    class_column: str | None = None
    path_column: str | None = None


DATASETS = (
    DatasetSpec("WikiArt", ROOT / "data/artemis/wikiart_meta.csv", ROOT / "data/wikiart", "style"),
    DatasetSpec("Pandora18k", ROOT / "data/pandora_meta.csv", ROOT / "data/pandora_flat", "style"),
    DatasetSpec("Rijksmuseum", ROOT / "data/rijksmuseum_labels.csv", ROOT / "data/rijksmuseum/painting", "period"),
)


def ascii_fold(text: object) -> str:
    if pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_token(text: object) -> str:
    text = ascii_fold(text).lower()
    text = Path(text).stem
    text = re.sub(r"^\d+[_\-\s]+", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def normalize_class_name(text: object) -> str:
    return normalize_token(text)


def split_artist_title(stem: object) -> tuple[str, str]:
    raw = Path(str(stem)).stem
    if "__" in raw:
        artist, title = raw.split("__", 1)
        return normalize_token(artist), normalize_token(title)

    parts = raw.split("_", 1)
    if len(parts) == 2:
        return normalize_token(parts[0]), normalize_token(parts[1])
    return "", normalize_token(raw)


def index_images(root: Path | None) -> dict[str, Path]:
    if root is None or not root.exists():
        return {}
    paths = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    index: dict[str, Path] = {}
    for path in paths:
        index.setdefault(normalize_token(path.stem), path)
    return index


def resolve_path(row: pd.Series, spec: DatasetSpec, image_index: dict[str, Path]) -> str:
    if spec.path_column and spec.path_column in row and pd.notna(row[spec.path_column]):
        path = Path(str(row[spec.path_column]))
        if path.exists():
            return str(path)
    filename = str(row.get("filename", ""))
    if spec.image_root is not None and filename:
        if spec.name == "WikiArt" and spec.class_column and spec.class_column in row:
            path = spec.image_root / str(row[spec.class_column]) / filename
        elif spec.name == "Pandora18k" and spec.class_column and spec.class_column in row:
            path = spec.image_root / str(row[spec.class_column]) / filename
        else:
            path = spec.image_root / filename
        if path.exists():
            return str(path)
    return str(image_index.get(normalize_token(row.get("stem", filename)), ""))


def read_dataset(spec: DatasetSpec, resolve_images: bool) -> pd.DataFrame:
    frame = pd.read_csv(spec.metadata)
    if "stem" not in frame.columns:
        if "filename" in frame.columns:
            frame["stem"] = frame["filename"].map(lambda value: Path(str(value)).stem)
        else:
            raise ValueError(f"{spec.metadata} must contain stem or filename")
    if "filename" not in frame.columns:
        frame["filename"] = frame["stem"].astype(str) + ".jpg"

    image_index = index_images(spec.image_root) if resolve_images else {}
    records = []
    for _, row in frame.iterrows():
        artist, title = split_artist_title(row["stem"])
        class_value = row.get(spec.class_column, "") if spec.class_column else ""
        records.append({
            "dataset": spec.name,
            "stem": str(row["stem"]),
            "filename": str(row.get("filename", "")),
            "class": str(class_value),
            "artist_norm": artist,
            "title_norm": title,
            "exact_stem_key": str(row["stem"]).lower(),
            "normalized_stem_key": normalize_token(row["stem"]),
            "artist_title_key": f"{artist}__{title}" if artist and title else "",
            "title_key": title,
            "class_norm": normalize_class_name(class_value),
            "image_path": resolve_path(row, spec, image_index) if resolve_images else "",
        })
    return pd.DataFrame(records)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
    bits = pixels >= pixels.mean()
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def add_image_hashes(records: pd.DataFrame) -> pd.DataFrame:
    sha_values, ahash_values, errors = [], [], []
    for idx, path_text in enumerate(records["image_path"].fillna("")):
        if idx % 1000 == 0:
            print(f"hashing images: {idx:,}/{len(records):,}", flush=True)
        path = Path(path_text)
        if not path.exists():
            sha_values.append("")
            ahash_values.append("")
            errors.append("missing")
            continue
        try:
            sha_values.append(sha256_file(path))
            ahash_values.append(average_hash(path))
            errors.append("")
        except Exception as exc:  # pragma: no cover - diagnostics only
            sha_values.append("")
            ahash_values.append("")
            errors.append(type(exc).__name__)
    result = records.copy()
    result["sha256"] = sha_values
    result["average_hash"] = ahash_values
    result["hash_error"] = errors
    return result


def overlap_rows(records: pd.DataFrame, key: str) -> pd.DataFrame:
    valid = records[records[key].fillna("").ne("")].copy()
    grouped = valid.groupby(key)
    rows = []
    for value, group in grouped:
        datasets = sorted(group["dataset"].unique())
        if len(datasets) < 2:
            continue
        rows.append({
            "key_type": key,
            "key": value,
            "datasets": " | ".join(datasets),
            "n_datasets": len(datasets),
            "n_records": len(group),
            "examples": " || ".join(
                f"{row.dataset}:{row.stem}" for row in group.head(8).itertuples(index=False)
            ),
        })
    columns = ["key_type", "key", "datasets", "n_datasets", "n_records", "examples"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["n_datasets", "n_records", "key"], ascending=[False, False, True])


def pairwise_summary(records: pd.DataFrame, overlaps: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    datasets = sorted(records["dataset"].unique())
    overlap = overlaps[overlaps["key_type"].eq(key)]
    for i, left in enumerate(datasets):
        left_keys = set(records.loc[records["dataset"].eq(left), key].dropna()) - {""}
        for right in datasets[i + 1:]:
            right_keys = set(records.loc[records["dataset"].eq(right), key].dropna()) - {""}
            shared = left_keys & right_keys
            rows.append({
                "key_type": key,
                "dataset_a": left,
                "dataset_b": right,
                "shared_keys": len(shared),
                "a_keys": len(left_keys),
                "b_keys": len(right_keys),
                "a_overlap_fraction": len(shared) / len(left_keys) if left_keys else 0.0,
                "b_overlap_fraction": len(shared) / len(right_keys) if right_keys else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dataset_overlap")
    parser.add_argument("--hash-images", action="store_true", help="Compute exact SHA256 and simple perceptual average hashes.")
    parser.add_argument("--resolve-images", action="store_true", help="Resolve image paths even when not hashing images.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolve_images = args.resolve_images or args.hash_images
    records = pd.concat([read_dataset(spec, resolve_images=resolve_images) for spec in DATASETS], ignore_index=True)
    if args.hash_images:
        records = add_image_hashes(records)
    records.to_csv(args.output_dir / "dataset_records.csv", index=False)

    keys = ["exact_stem_key", "normalized_stem_key", "artist_title_key", "title_key"]
    if args.hash_images:
        keys.extend(["sha256", "average_hash"])

    overlap_tables = [overlap_rows(records, key) for key in keys]
    overlaps = pd.concat([table for table in overlap_tables if not table.empty], ignore_index=True)
    overlaps.to_csv(args.output_dir / "overlap_candidates.csv", index=False)

    summary = pd.concat([pairwise_summary(records, overlaps, key) for key in keys], ignore_index=True)
    summary.to_csv(args.output_dir / "overlap_summary.csv", index=False, float_format="%.8f")

    dataset_summary = records.groupby("dataset").agg(
        records=("stem", "count"),
        unique_normalized_stems=("normalized_stem_key", "nunique"),
        unique_artist_title_keys=("artist_title_key", lambda s: int(s[s.ne("")].nunique())),
        resolved_image_paths=("image_path", lambda s: int(s.fillna("").ne("").sum())),
    ).reset_index()
    dataset_summary.to_csv(args.output_dir / "dataset_key_summary.csv", index=False)

    print("Dataset key summary")
    print(dataset_summary.to_string(index=False))
    print("\nPairwise overlap summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nsaved: {args.output_dir / 'overlap_summary.csv'}")
    print(f"saved: {args.output_dir / 'overlap_candidates.csv'}")
    print(f"saved: {args.output_dir / 'dataset_records.csv'}")


if __name__ == "__main__":
    main()
