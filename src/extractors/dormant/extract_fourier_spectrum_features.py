#!/usr/bin/env python3
"""Extract compact or expanded polar Fourier-spectrum features.

The ``rich`` profile is intentionally nested: it starts with the historical
152-dimensional luminance spectrum and appends the same polar descriptor for
the three Lab channels.  It therefore broadens spectral measurement without
silently switching to a different representation.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import sys as _sys
from pathlib import Path as _Path
# Os helpers moram em src/extractors/; este modulo esta um nivel abaixo.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from slic_graph_features import crop_to_region, pool_image, resolve_wikiart_records  # noqa: E402


SIZE = 256
RADIAL_BINS = 12
ANGULAR_BINS = 12
SUMMARY_NAMES = (
    "radial_centroid",
    "radial_spread",
    "low_frequency_fraction",
    "high_frequency_fraction",
    "spectral_entropy",
    "orientation_entropy",
    "orientation_peak_fraction",
    "spectral_slope",
)


def build_names(radial_bins: int, angular_bins: int):
    """Os nomes dependem da granularidade, então têm de ser regeráveis.

    Antes eram tuplas de módulo construídas uma vez; com `configure` podendo
    mudar `RADIAL_BINS`/`ANGULAR_BINS` (§5.17c) isso daria vetor de um tamanho e
    nomes de outro, em silêncio. O default (12x12) reproduz as tuplas antigas
    exatamente -- a asserção de 152/608 logo abaixo é o teste disso.
    """
    grid = tuple(
        f"log_power_r{radial:02d}_a{angle:02d}"
        for radial in range(radial_bins)
        for angle in range(angular_bins)
    )
    compact = grid + SUMMARY_NAMES
    rich = compact + tuple(
        f"lab_{channel}_{name}"
        for channel in ("L", "a", "b")
        for name in compact
    )
    return grid, compact, rich


GRID_NAMES, COMPACT_FEATURE_NAMES, RICH_FEATURE_NAMES = build_names(RADIAL_BINS, ANGULAR_BINS)
FEATURE_NAMES = COMPACT_FEATURE_NAMES  # compatibility with historical imports
assert len(COMPACT_FEATURE_NAMES) == 152
assert len(RICH_FEATURE_NAMES) == 608


def frequency_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.fft.fftshift(np.fft.fftfreq(SIZE))
    fx, fy = np.meshgrid(frequency, frequency)
    radius = np.sqrt(fx * fx + fy * fy)
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    radial_edges = np.geomspace(1 / SIZE, 0.5, RADIAL_BINS + 1)
    angular_edges = np.linspace(0, np.pi, ANGULAR_BINS + 1)
    return radius, angle, radial_edges, angular_edges


RADIUS, ANGLE, RADIAL_EDGES, ANGULAR_EDGES = frequency_geometry()
VALID_FREQUENCY = (RADIUS >= RADIAL_EDGES[0]) & (RADIUS <= RADIAL_EDGES[-1])
RADIAL_CENTERS = np.sqrt(RADIAL_EDGES[:-1] * RADIAL_EDGES[1:])
HANN_WINDOW = np.outer(np.hanning(SIZE), np.hanning(SIZE)).astype(np.float32)


def configure(size: int, radial_bins: int | None = None,
              angular_bins: int | None = None) -> None:
    """Muda a resolução de operação. O default (256) preserva todo resultado
    anterior; a flag existe porque o §5.4b registra que os alvos de 128/256/1024
    nunca tiveram justificativa e a resolução nunca foi variada.

    `RADIAL_BINS` e `ANGULAR_BINS` NÃO mudam, então o vetor continua com 152
    dimensões e a comparação entre resoluções é pareada em dimensionalidade. Mas
    a grade de frequência muda, então muda **o que** cada bin mede -- mesma
    pegadinha do `felz_scale` e do raio do LBP.

    Tem de ser chamada ANTES do `Pool`: `extract_one` lê estes globais e o fork
    herda o valor do pai.
    """
    global SIZE, RADIUS, ANGLE, RADIAL_EDGES, ANGULAR_EDGES
    global VALID_FREQUENCY, RADIAL_CENTERS, HANN_WINDOW
    global RADIAL_BINS, ANGULAR_BINS
    global GRID_NAMES, COMPACT_FEATURE_NAMES, RICH_FEATURE_NAMES, FEATURE_NAMES
    SIZE = int(size)
    if radial_bins is not None:
        RADIAL_BINS = int(radial_bins)
    if angular_bins is not None:
        ANGULAR_BINS = int(angular_bins)
    GRID_NAMES, COMPACT_FEATURE_NAMES, RICH_FEATURE_NAMES = build_names(
        RADIAL_BINS, ANGULAR_BINS)
    FEATURE_NAMES = COMPACT_FEATURE_NAMES
    RADIUS, ANGLE, RADIAL_EDGES, ANGULAR_EDGES = frequency_geometry()
    VALID_FREQUENCY = (RADIUS >= RADIAL_EDGES[0]) & (RADIUS <= RADIAL_EDGES[-1])
    RADIAL_CENTERS = np.sqrt(RADIAL_EDGES[:-1] * RADIAL_EDGES[1:])
    HANN_WINDOW = np.outer(np.hanning(SIZE), np.hanning(SIZE)).astype(np.float32)


def square_luminance(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    width, height = image.size
    scale = SIZE / max(width, height)
    resized = image.resize(
        (max(2, round(width * scale)), max(2, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    values = np.asarray(resized, dtype=np.float32) / 255.0
    pad_height = SIZE - values.shape[0]
    pad_width = SIZE - values.shape[1]
    padding = (
        (pad_height // 2, pad_height - pad_height // 2),
        (pad_width // 2, pad_width - pad_width // 2),
    )
    return np.pad(values, padding, mode="reflect")


def fourier_features(luminance: np.ndarray) -> np.ndarray:
    if luminance.shape != (SIZE, SIZE):
        raise ValueError(f"expected {(SIZE, SIZE)}, got {luminance.shape}")
    centered = luminance.astype(np.float64) - float(luminance.mean())
    deviation = float(centered.std())
    if deviation <= 1e-8:
        return np.zeros(len(COMPACT_FEATURE_NAMES), dtype=np.float32)
    centered /= deviation
    spectrum = np.fft.fftshift(np.fft.fft2(centered * HANN_WINDOW))
    power = np.square(np.abs(spectrum))
    histogram, _, _ = np.histogram2d(
        RADIUS[VALID_FREQUENCY],
        ANGLE[VALID_FREQUENCY],
        bins=(RADIAL_EDGES, ANGULAR_EDGES),
        weights=power[VALID_FREQUENCY],
    )
    total = float(histogram.sum())
    if not np.isfinite(total) or total <= 0:
        return np.zeros(len(COMPACT_FEATURE_NAMES), dtype=np.float32)
    distribution = histogram / total
    radial = distribution.sum(axis=1)
    angular = distribution.sum(axis=0)

    radial_centroid = float(np.sum(radial * RADIAL_CENTERS) / 0.5)
    radial_spread = float(
        np.sqrt(np.sum(radial * np.square(RADIAL_CENTERS / 0.5 - radial_centroid)))
    )
    nonzero = distribution[distribution > 0]
    spectral_entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(distribution.size))
    angular_nonzero = angular[angular > 0]
    orientation_entropy = float(
        -np.sum(angular_nonzero * np.log(angular_nonzero)) / np.log(len(angular))
    )
    slope = float(
        np.polyfit(
            np.log(RADIAL_CENTERS),
            np.log(np.maximum(radial, 1e-12)),
            deg=1,
        )[0]
    )
    summary = np.asarray(
        [
            radial_centroid,
            radial_spread,
            radial[:4].sum(),
            radial[-4:].sum(),
            spectral_entropy,
            orientation_entropy,
            angular.max(),
            slope,
        ],
        dtype=np.float32,
    )
    log_power = np.log10(distribution + 1e-12).astype(np.float32).ravel()
    result = np.concatenate((log_power, summary))
    if not np.isfinite(result).all():
        raise ValueError("non-finite Fourier descriptor")
    return result


def square_lab(image: Image.Image) -> np.ndarray:
    """Resize/pad in the same geometry as luminance, then return Lab planes."""
    image = image.convert("RGB")
    width, height = image.size
    scale = SIZE / max(width, height)
    resized = image.resize(
        (max(2, round(width * scale)), max(2, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    lab = skcolor.rgb2lab(rgb).astype(np.float32)
    pad_height = SIZE - lab.shape[0]
    pad_width = SIZE - lab.shape[1]
    padding = (
        (pad_height // 2, pad_height - pad_height // 2),
        (pad_width // 2, pad_width - pad_width // 2),
        (0, 0),
    )
    return np.pad(lab, padding, mode="reflect")


def feature_names(profile: str) -> tuple[str, ...]:
    if profile == "compact":
        return COMPACT_FEATURE_NAMES
    if profile == "rich":
        return RICH_FEATURE_NAMES
    raise ValueError(f"unknown feature profile: {profile}")


def image_features(image: Image.Image, profile: str) -> np.ndarray:
    compact = fourier_features(square_luminance(image))
    if profile == "compact":
        return compact
    if profile != "rich":
        raise ValueError(f"unknown feature profile: {profile}")
    lab = square_lab(image)
    result = np.concatenate((compact, *(fourier_features(lab[:, :, index]) for index in range(3))))
    if len(result) != len(RICH_FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError("non-finite or malformed rich Fourier descriptor")
    return result.astype(np.float32, copy=False)


def extract_one(
    task: tuple[str, str, str, str, int, str],
) -> tuple[dict[str, float | str] | None, str | None]:
    stem, style, path, region, pool_factor, profile = task
    try:
        with Image.open(path) as image:
            cropped = crop_to_region(image.convert("RGB"), region)
            cropped = pool_image(cropped, pool_factor)
            descriptor = image_features(cropped, profile)
        return {
            "stem": stem,
            "style": style,
            **dict(zip(feature_names(profile), descriptor, strict=True)),
        }, None
    except Exception as error:
        return None, f"{stem}: {type(error).__name__}: {error}"


def paired_segmentation_sample(args: argparse.Namespace) -> pd.DataFrame:
    stem_sets = []
    for path in (args.slic_features, args.felz_features, args.quickshift_features):
        stems = pd.read_parquet(path, columns=["stem"])["stem"].astype(str)
        stem_sets.append(set(stems))
    paired = set.intersection(*stem_sets)
    metadata = pd.read_csv(args.meta)[["stem", "style"]].drop_duplicates("stem")
    sample = metadata[metadata["stem"].astype(str).isin(paired)].copy()
    return sample.sort_values("stem").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument(
        "--slic-features", type=Path,
        default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet",
    )
    parser.add_argument(
        "--felz-features", type=Path,
        default=(
            ROOT
            / "outputs/wikiart_segmentation_pilot/felzenszwalb_s50_m50/combined_features.parquet"
        ),
    )
    parser.add_argument(
        "--quickshift-features", type=Path,
        default=(
            ROOT
            / "outputs/wikiart_segmentation_pilot/quickshift_k3_d10/combined_features.parquet"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/fourier_12k_features.parquet"
    )
    parser.add_argument(
        "--sample-output", type=Path, default=ROOT / "results/fourier_12k_sample.csv"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--feature-profile", choices=("compact", "rich"), default="compact")
    args = parser.parse_args()

    sample = paired_segmentation_sample(args)
    records = resolve_wikiart_records(args.meta, args.image_dir)
    records = sample.merge(records[["stem", "path"]], on="stem", validate="one_to_one")
    if len(records) != 12_213:
        raise ValueError(f"expected 12,213 paired images, got {len(records):,}")
    tasks = [
        (*row, "global", 1, args.feature_profile)
        for row in records[["stem", "style", "path"]].itertuples(index=False, name=None)
    ]
    print(
        f"Fourier extraction: images={len(tasks):,}, features={len(feature_names(args.feature_profile))}, "
        f"size={SIZE}, workers={args.workers}",
        flush=True,
    )

    rows, errors = [], []
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (row, error) in enumerate(
            pool.imap_unordered(extract_one, tasks, chunksize=16), 1
        ):
            if row is not None:
                rows.append(row)
            if error is not None:
                errors.append(error)
            if index % 1_000 == 0 or index == len(tasks):
                print(
                    f"{index:,}/{len(tasks):,}; failures={len(errors)}; "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    if errors:
        raise RuntimeError(f"Fourier extraction failed for {len(errors)} images: {errors[:5]}")
    result = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    if len(result) != len(tasks) or result["stem"].duplicated().any():
        raise RuntimeError("incomplete or duplicate Fourier result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    sample.sort_values("stem").to_csv(args.sample_output, index=False)
    print(f"saved={args.output} sample={args.sample_output}", flush=True)


if __name__ == "__main__":
    main()
