#!/usr/bin/env python3
"""Extract Catch22 descriptors from full-resolution Hilbert Lab sequences.

The source sequences are the 128x128 Lab pixels ordered by a Hilbert curve.
Catch22 is applied independently to L*, a*, and b*, giving 66 features per
painting.  Chunks are saved as they finish so a failed long extraction can be
resumed without recomputing completed images.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycatch22 import catch22_all


SEQUENCES: np.ndarray | None = None


def initialise(sequence_path: str) -> None:
    global SEQUENCES
    SEQUENCES = np.load(sequence_path, mmap_mode="r")


def extract_one(index: int) -> tuple[int, np.ndarray]:
    if SEQUENCES is None:
        raise RuntimeError("Worker sequence array was not initialised")
    sequence = np.asarray(SEQUENCES[index], dtype=np.float64)
    features = np.concatenate(
        [np.asarray(catch22_all(sequence[:, channel])["values"], dtype=np.float32)
         for channel in range(sequence.shape[1])]
    )
    # Some descriptors are undefined for unusually regular signals.  Their
    # fold-local imputation is handled by the evaluator, but use NaN rather
    # than leaking a package-specific sentinel into the stored representation.
    features[~np.isfinite(features)] = np.nan
    return index, features


def chunk_path(output_dir: Path, start: int, stop: int) -> Path:
    return output_dir / "chunks" / f"{start:06d}_{stop:06d}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        required=True,
        help="Directory containing sequences_lab_hilbert.npy and stems.npy.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/hilbert_catch22_full"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    args = parser.parse_args()

    sequence_path = args.input_dir / "sequences_lab_hilbert.npy"
    stem_path = args.input_dir / "stems.npy"
    sequences = np.load(sequence_path, mmap_mode="r")
    stems = np.load(stem_path, allow_pickle=True).astype(str)
    if sequences.ndim != 3 or sequences.shape[2] != 3 or len(sequences) != len(stems):
        raise ValueError("Expected aligned (N, T, 3) Hilbert sequences and stems")

    probe = catch22_all(np.asarray(sequences[0, :, 0], dtype=np.float64))
    names = [f"catch22_{channel}_{name}" for channel in ("l", "a", "b") for name in probe["names"]]
    if len(names) != 66:
        raise RuntimeError(f"Expected 66 Catch22 features, got {len(names)}")

    output_dir = args.output_dir
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n_images = len(sequences)
    print(
        f"Catch22 Hilbert: {n_images:,} images × 3 Lab channels = {len(names)} features; "
        f"workers={args.workers}, chunk={args.chunk_size}", flush=True,
    )

    for start in range(0, n_images, args.chunk_size):
        stop = min(start + args.chunk_size, n_images)
        path = chunk_path(output_dir, start, stop)
        if path.exists():
            with np.load(path) as saved:
                valid = (np.array_equal(saved["indices"], np.arange(start, stop))
                         and saved["features"].shape == (stop - start, len(names)))
            if valid:
                print(f"{stop:,}/{n_images:,} (cached)", flush=True)
                continue
            path.unlink()

        values = np.empty((stop - start, len(names)), dtype=np.float32)
        with Pool(args.workers, initializer=initialise, initargs=(str(sequence_path),)) as pool:
            for done, (index, features) in enumerate(
                pool.imap_unordered(extract_one, range(start, stop), chunksize=8), 1
            ):
                values[index - start] = features
                if done % 250 == 0:
                    print(f"chunk {start:,}-{stop:,}: {done:,}/{stop-start:,}", flush=True)
        np.savez_compressed(path, indices=np.arange(start, stop), features=values)
        print(
            f"{stop:,}/{n_images:,}; elapsed={(time.perf_counter() - started) / 60:.1f} min",
            flush=True,
        )

    features = np.empty((n_images, len(names)), dtype=np.float32)
    for start in range(0, n_images, args.chunk_size):
        stop = min(start + args.chunk_size, n_images)
        with np.load(chunk_path(output_dir, start, stop)) as saved:
            if not np.array_equal(saved["indices"], np.arange(start, stop)):
                raise RuntimeError(f"Bad Catch22 chunk {start}:{stop}")
            features[start:stop] = saved["features"]
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "stems.npy", stems)
    (output_dir / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    print(f"saved={output_dir}; shape={features.shape}", flush=True)


if __name__ == "__main__":
    main()
