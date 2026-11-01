"""Build a compact Fourier-v2 descriptor from the existing polar spectrum.

The original Fourier block stores a 12x12 log-polar power grid plus eight
global summaries.  The grid is high-dimensional and many inner bins are weak or
degenerate, so this version collapses the same spectrum into radial,
orientation and band-level statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RADIAL_BINS = 12
ANGULAR_BINS = 12
EPS = 1e-12


def entropy(prob: np.ndarray, axis=None) -> np.ndarray:
    p = np.clip(prob, EPS, None)
    h = -(p * np.log(p)).sum(axis=axis)
    if axis is None:
        denom = np.log(prob.size)
    else:
        denom = np.log(prob.shape[axis])
    return h / denom


def gini(values: np.ndarray, axis: int = 1) -> np.ndarray:
    sorted_values = np.sort(values, axis=axis)
    n = sorted_values.shape[axis]
    weights = np.arange(1, n + 1, dtype=np.float64)
    if axis == 1:
        numerator = (sorted_values * weights).sum(axis=1)
        denominator = sorted_values.sum(axis=1)
    else:
        numerator = (sorted_values * weights[:, None]).sum(axis=0)
        denominator = sorted_values.sum(axis=0)
    return (2 * numerator / np.maximum(n * denominator, EPS)) - ((n + 1) / n)


def weighted_quantile(cdf: np.ndarray, centers: np.ndarray, q: float) -> np.ndarray:
    index = (cdf >= q).argmax(axis=1)
    return centers[index]


def slope_features(radial: np.ndarray, centers: np.ndarray, name: str) -> tuple[list[str], np.ndarray]:
    x = np.log(centers)
    y = np.log(np.clip(radial, EPS, None))
    x_mean = x.mean()
    y_mean = y.mean(axis=1, keepdims=True)
    denom = np.square(x - x_mean).sum()
    slope = ((y - y_mean) * (x - x_mean)).sum(axis=1) / denom
    intercept = y_mean[:, 0] - slope * x_mean
    fitted = intercept[:, None] + slope[:, None] * x
    ss_res = np.square(y - fitted).sum(axis=1)
    ss_tot = np.square(y - y_mean).sum(axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, EPS)
    return [f"{name}_slope", f"{name}_intercept", f"{name}_r2"], np.column_stack([slope, intercept, r2])


def orientation_features(grid: np.ndarray, name: str, angle_centers: np.ndarray) -> tuple[list[str], np.ndarray]:
    angular = grid.sum(axis=1)
    total = angular.sum(axis=1, keepdims=True)
    angular = angular / np.maximum(total, EPS)
    vector = (angular * np.exp(2j * angle_centers)).sum(axis=1)
    resultant = np.abs(vector)
    dominant = angle_centers[angular.argmax(axis=1)]
    names = [
        f"{name}_orientation_entropy",
        f"{name}_orientation_peak",
        f"{name}_orientation_resultant",
        f"{name}_dominant_cos2",
        f"{name}_dominant_sin2",
    ]
    values = np.column_stack([
        entropy(angular, axis=1),
        angular.max(axis=1),
        resultant,
        np.cos(2 * dominant),
        np.sin(2 * dominant),
    ])
    return names, values


def build(frame: pd.DataFrame) -> pd.DataFrame:
    grid_names = [f"log_power_r{r:02d}_a{a:02d}"
                  for r in range(RADIAL_BINS) for a in range(ANGULAR_BINS)]
    missing = [c for c in grid_names if c not in frame.columns]
    if missing:
        raise ValueError(f"missing Fourier grid columns: {missing[:5]}")

    raw = np.power(10.0, frame[grid_names].to_numpy(np.float64)).reshape(
        len(frame), RADIAL_BINS, ANGULAR_BINS)
    distribution = raw / np.maximum(raw.sum(axis=(1, 2), keepdims=True), EPS)
    radial = distribution.sum(axis=2)
    angular = distribution.sum(axis=1)

    radial_edges = np.geomspace(1 / 256, 0.5, RADIAL_BINS + 1)
    radial_centers = np.sqrt(radial_edges[:-1] * radial_edges[1:]) / 0.5
    angle_edges = np.linspace(0, np.pi, ANGULAR_BINS + 1)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    names, pieces = [], []

    names += [f"radial_power_r{r:02d}" for r in range(RADIAL_BINS)]
    pieces.append(radial)
    names += [f"angular_power_a{a:02d}" for a in range(ANGULAR_BINS)]
    pieces.append(angular)

    cdf = np.cumsum(radial, axis=1)
    q_names = ["energy_radius_q25", "energy_radius_q50", "energy_radius_q75", "energy_radius_q90"]
    q_values = np.column_stack([weighted_quantile(cdf, radial_centers, q)
                                for q in (0.25, 0.50, 0.75, 0.90)])
    names += q_names
    pieces.append(q_values)

    mean = (radial * radial_centers).sum(axis=1)
    variance = (radial * np.square(radial_centers - mean[:, None])).sum(axis=1)
    std = np.sqrt(np.maximum(variance, EPS))
    skew = (radial * np.power((radial_centers - mean[:, None]) / std[:, None], 3)).sum(axis=1)
    kurt = (radial * np.power((radial_centers - mean[:, None]) / std[:, None], 4)).sum(axis=1)
    low = radial[:, :4].sum(axis=1)
    mid = radial[:, 4:8].sum(axis=1)
    high = radial[:, 8:].sum(axis=1)
    moment_names = [
        "radial_mean", "radial_std", "radial_skewness", "radial_kurtosis",
        "radial_entropy", "radial_effective_bins", "radial_gini",
        "mid_to_low_power", "high_to_low_power", "high_to_mid_power",
    ]
    moment_values = np.column_stack([
        mean, std, skew, kurt,
        entropy(radial, axis=1),
        np.exp(-(np.clip(radial, EPS, None) * np.log(np.clip(radial, EPS, None))).sum(axis=1)),
        gini(radial),
        mid / np.maximum(low, EPS),
        high / np.maximum(low, EPS),
        high / np.maximum(mid, EPS),
    ])
    names += moment_names
    pieces.append(moment_values)

    orient_names, orient_values = orientation_features(distribution, "global", angle_centers)
    names += orient_names
    pieces.append(orient_values)

    for band, sl in {
        "low": slice(0, 4),
        "mid": slice(4, 8),
        "high": slice(8, 12),
    }.items():
        band_grid = distribution[:, sl, :]
        band_power = band_grid.sum(axis=(1, 2))
        orient_names, orient_values = orientation_features(band_grid, band, angle_centers)
        names += [f"{band}_power"] + orient_names
        pieces.append(np.column_stack([band_power, orient_values]))

    for name, sl in {
        "all": slice(0, 12),
        "low": slice(0, 4),
        "mid": slice(4, 8),
        "high": slice(8, 12),
    }.items():
        slope_names, slope_values = slope_features(radial[:, sl], radial_centers[sl], f"{name}_radial")
        names += slope_names
        pieces.append(slope_values)

    summary_names = [
        "radial_centroid", "radial_spread", "low_frequency_fraction",
        "high_frequency_fraction", "spectral_entropy", "orientation_entropy",
        "orientation_peak_fraction", "spectral_slope",
    ]
    present_summaries = [c for c in summary_names if c in frame.columns]
    names += [f"original_{c}" for c in present_summaries]
    pieces.append(frame[present_summaries].to_numpy(np.float64))

    values = np.column_stack(pieces).astype(np.float32)
    result = pd.DataFrame(values, columns=names)
    result.insert(0, "stem", frame["stem"].astype(str).to_numpy())
    if "style" in frame.columns:
        result.insert(1, "style", frame["style"].astype(str).to_numpy())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/fourier_multitask_features.parquet")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/fourier_v2_multitask_features.parquet")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    result = build(frame)
    result.to_parquet(args.output, index=False)
    print(f"saved: {args.output} shape={result.shape}")


if __name__ == "__main__":
    main()
