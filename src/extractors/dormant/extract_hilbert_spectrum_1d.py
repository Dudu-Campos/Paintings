#!/usr/bin/env python3
"""Extract one-dimensional spectral descriptors from Hilbert Lab sequences.

For each Lab channel, the extractor summarizes the rFFT power distribution of
the original Hilbert path and of its first difference.  The latter is the
spectral representation of chromatic transitions along the curve.  Each of the
six signals contributes 12 logarithmic frequency bands and six scalar spectral
statistics, yielding 108 features per painting.

Intermediate chunks make the extraction safe to resume after an interruption.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np


N_BANDS = 12
CHANNELS = ("l", "a", "b")
KINDS = ("level", "difference")
SEQUENCES: np.ndarray | None = None


def initialise(sequence_path: str) -> None:
    global SEQUENCES
    SEQUENCES = np.load(sequence_path, mmap_mode="r")


def feature_names() -> list[str]:
    names: list[str] = []
    for kind in KINDS:
        for channel in CHANNELS:
            prefix = f"spectrum1d_{kind}_{channel}"
            names.extend(f"{prefix}_band_{band:02d}" for band in range(N_BANDS))
            names.extend(
                f"{prefix}_{name}"
                for name in ("centroid", "spread", "low_fraction", "high_fraction", "entropy", "slope")
            )
    return names


def spectrum_features(signal: np.ndarray) -> np.ndarray:
    """Return normalized spectral-shape descriptors, invariant to offset/scale."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()
    scale = signal.std()
    if not np.isfinite(scale) or scale <= 1e-10:
        return np.zeros(N_BANDS + 6, dtype=np.float32)
    signal /= scale
    frequency = np.fft.rfftfreq(len(signal))[1:]
    power = np.square(np.abs(np.fft.rfft(signal * np.hanning(len(signal)))))[1:]
    total = float(power.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.zeros(N_BANDS + 6, dtype=np.float32)
    distribution = power / total
    # Geometric bins give equal attention to coarse and fine changes.
    edges = np.geomspace(1.0 / len(signal), 0.5, N_BANDS + 1)
    bands, _ = np.histogram(frequency, bins=edges, weights=distribution)
    bands = bands.astype(np.float64)
    # Use log-energy so the low-frequency mass does not drown out texture.
    log_bands = np.log10(bands + 1e-12)
    centroid = float(np.sum(distribution * frequency) / 0.5)
    spread = float(np.sqrt(np.sum(distribution * np.square(frequency / 0.5 - centroid))))
    nonzero = distribution[distribution > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(distribution)))
    centers = np.sqrt(edges[:-1] * edges[1:])
    slope = float(np.polyfit(np.log(centers), np.log(np.maximum(bands, 1e-12)), 1)[0])
    return np.asarray(
        [*log_bands, centroid, spread, bands[:4].sum(), bands[-4:].sum(), entropy, slope],
        dtype=np.float32,
    )


def extract_one(index: int) -> tuple[int, np.ndarray]:
    if SEQUENCES is None:
        raise RuntimeError("Worker sequence array was not initialised")
    sequence = np.asarray(SEQUENCES[index], dtype=np.float32)
    values = []
    for kind in KINDS:
        signal = sequence if kind == "level" else np.diff(sequence, axis=0)
        values.extend(spectrum_features(signal[:, channel]) for channel in range(3))
    features = np.concatenate(values).astype(np.float32)
    if not np.isfinite(features).all() or len(features) != len(feature_names()):
        raise RuntimeError("Invalid 1D spectral feature vector")
    return index, features


def chunk_path(output_dir: Path, start: int, stop: int) -> Path:
    return output_dir / "chunks" / f"{start:06d}_{stop:06d}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hilbert_spectrum_1d_full"))
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
    print(
        f"Hilbert 1D spectrum: {len(sequences):,} images; {len(names)} features; workers={args.workers}",
        flush=True,
    )
    for start in range(0, len(sequences), args.chunk_size):
        stop = min(start + args.chunk_size, len(sequences))
        path = chunk_path(output_dir, start, stop)
        if path.exists():
            with np.load(path) as saved:
                valid = (np.array_equal(saved["indices"], np.arange(start, stop))
                         and saved["features"].shape == (stop - start, len(names)))
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
                raise RuntimeError(f"Bad spectral chunk {start}:{stop}")
            features[start:stop] = saved["features"]
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "stems.npy", stems)
    (output_dir / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    print(f"saved={output_dir}; shape={features.shape}", flush=True)


if __name__ == "__main__":
    main()
