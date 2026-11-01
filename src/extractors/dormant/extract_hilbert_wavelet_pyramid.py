#!/usr/bin/env python3
"""Extract localized DWT-pyramid features from Hilbert Lab sequences.

Each Lab trajectory is decomposed with a three-level Daubechies-2 discrete
wavelet transform into four temporal series: coarse approximation A3 and
details D3, D2 and D1.  The representation preserves where, at each temporal
scale, colour and texture variation occur along the Hilbert path.

For every subseries we record eight localized means and standard deviations,
plus six robust global dynamics.  With cross-channel correlations this yields
276 deterministic features per painting.  Chunks are restart-safe.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pywt


LEVEL = 3
N_WINDOWS = 8
WAVELET = "db2"
CHANNELS = ("l", "a", "b")
SUBBANDS = ("a3", "d3", "d2", "d1")
SEQUENCES: np.ndarray | None = None


def feature_names() -> list[str]:
    names: list[str] = []
    for band in SUBBANDS:
        for channel in CHANNELS:
            prefix = f"wavelet_{band}_{channel}"
            names.extend(f"{prefix}_mean_window_{index:02d}" for index in range(N_WINDOWS))
            names.extend(f"{prefix}_std_window_{index:02d}" for index in range(N_WINDOWS))
            names.extend(
                f"{prefix}_{name}"
                for name in ("mean_abs", "rms", "skewness", "kurtosis", "zero_crossing", "energy_entropy")
            )
        names.extend(f"wavelet_{band}_correlation_{first}{second}" for first, second in (("l", "a"), ("l", "b"), ("a", "b")))
    return names


def initialise(sequence_path: str) -> None:
    global SEQUENCES
    SEQUENCES = np.load(sequence_path, mmap_mode="r")


def summarize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    windows = np.array_split(values, N_WINDOWS)
    means = [float(window.mean()) for window in windows]
    stds = [float(window.std()) for window in windows]
    centered = values - values.mean()
    standard_deviation = float(centered.std())
    if standard_deviation <= 1e-10:
        skewness, kurtosis = 0.0, 0.0
    else:
        normalized = centered / standard_deviation
        skewness = float(np.mean(normalized ** 3))
        kurtosis = float(np.mean(normalized ** 4) - 3.0)
    energy = np.square(values)
    distribution = energy / max(float(energy.sum()), 1e-12)
    nonzero = distribution[distribution > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(values)))
    zero_crossing = float(np.mean(np.signbit(values[1:]) != np.signbit(values[:-1])))
    return np.asarray(
        [*means, *stds, np.mean(np.abs(values)), np.sqrt(np.mean(energy)), skewness, kurtosis, zero_crossing, entropy],
        dtype=np.float32,
    )


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() <= 1e-10 or second.std() <= 1e-10:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def extract_one(index: int) -> tuple[int, np.ndarray]:
    if SEQUENCES is None:
        raise RuntimeError("Worker sequence array was not initialised")
    sequence = np.asarray(SEQUENCES[index], dtype=np.float32)
    decomposed = [pywt.wavedec(sequence[:, channel], WAVELET, level=LEVEL, mode="periodization") for channel in range(3)]
    # pywt returns [cA3, cD3, cD2, cD1], matching SUBBANDS.
    features = []
    for band_index in range(len(SUBBANDS)):
        coefficients = [decomposed[channel][band_index] for channel in range(3)]
        for values in coefficients:
            features.extend(summarize(values))
        features.extend(safe_correlation(coefficients[first], coefficients[second]) for first, second in ((0, 1), (0, 2), (1, 2)))
    result = np.asarray(features, dtype=np.float32)
    if len(result) != len(feature_names()) or not np.isfinite(result).all():
        raise RuntimeError("Invalid wavelet feature vector")
    return index, result


def chunk_path(output_dir: Path, start: int, stop: int) -> Path:
    return output_dir / "chunks" / f"{start:06d}_{stop:06d}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hilbert_wavelet_pyramid_full"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    args = parser.parse_args()

    sequence_path = args.input_dir / "sequences_lab_hilbert.npy"
    sequences = np.load(sequence_path, mmap_mode="r")
    stems = np.load(args.input_dir / "stems.npy", allow_pickle=True).astype(str)
    names = feature_names()
    if sequences.ndim != 3 or sequences.shape[2] != 3 or len(sequences) != len(stems):
        raise ValueError("Expected aligned (N, T, 3) Hilbert sequences and stems")
    output_dir = args.output_dir
    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Hilbert wavelet pyramid: {len(sequences):,} images; {len(names)} features; workers={args.workers}", flush=True)
    for start in range(0, len(sequences), args.chunk_size):
        stop = min(start + args.chunk_size, len(sequences))
        path = chunk_path(output_dir, start, stop)
        if path.exists():
            with np.load(path) as saved:
                valid = (np.array_equal(saved["indices"], np.arange(start, stop)) and saved["features"].shape == (stop - start, len(names)))
            if valid:
                print(f"{stop:,}/{len(sequences):,} (cached)", flush=True)
                continue
            path.unlink()
        values = np.empty((stop - start, len(names)), dtype=np.float32)
        with Pool(args.workers, initializer=initialise, initargs=(str(sequence_path),)) as pool:
            for done, (index, features) in enumerate(pool.imap_unordered(extract_one, range(start, stop), chunksize=8), 1):
                values[index - start] = features
                if done % 250 == 0:
                    print(f"chunk {start:,}-{stop:,}: {done:,}/{stop-start:,}", flush=True)
        np.savez_compressed(path, indices=np.arange(start, stop), features=values)
        print(f"{stop:,}/{len(sequences):,}; elapsed={(time.perf_counter() - started) / 60:.1f} min", flush=True)
    features = np.empty((len(sequences), len(names)), dtype=np.float32)
    for start in range(0, len(sequences), args.chunk_size):
        stop = min(start + args.chunk_size, len(sequences))
        with np.load(chunk_path(output_dir, start, stop)) as saved:
            if not np.array_equal(saved["indices"], np.arange(start, stop)):
                raise RuntimeError(f"Bad wavelet chunk {start}:{stop}")
            features[start:stop] = saved["features"]
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "stems.npy", stems)
    (output_dir / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    print(f"saved={output_dir}; shape={features.shape}", flush=True)


if __name__ == "__main__":
    main()
