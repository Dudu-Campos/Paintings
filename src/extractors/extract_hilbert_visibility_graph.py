#!/usr/bin/env python3
"""Extract Horizontal Visibility Graph descriptors from Hilbert Lab trajectories.

Each 16,384-step Hilbert trajectory is mean-pooled to 256 local tokens per
Lab channel.  A Horizontal Visibility Graph (HVG) connects two tokens when
their intervening values lie below both endpoints.  HVGs are a standard
time-series-to-graph transform and, unlike natural visibility graphs, have an
O(T) stack construction.  The result summarizes local hierarchy, hubs, and
the temporal reach of visible transitions while remaining tractable for the
full 79k-image cohort.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np


N_TOKENS = 256
CHANNELS = ("l", "a", "b")
SEQUENCES: np.ndarray | None = None


def feature_names() -> list[str]:
    names: list[str] = []
    per_channel = (
        "degree_mean", "degree_std", "degree_skewness", "degree_kurtosis",
        "degree_entropy", "degree_q10", "degree_q25", "degree_q50", "degree_q75", "degree_q90",
        "hub_fraction", "leaf_fraction", "span_mean", "span_std", "span_q50", "span_q90",
        "long_edge_fraction", "max_span",
    )
    for channel in CHANNELS:
        names.extend(f"hvg_{channel}_{name}" for name in per_channel)
    names.extend(("hvg_degree_corr_la", "hvg_degree_corr_lb", "hvg_degree_corr_ab"))
    return names


def initialise(sequence_path: str) -> None:
    global SEQUENCES
    SEQUENCES = np.load(sequence_path, mmap_mode="r")


def horizontal_visibility(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return degree and edge spans for a horizontal visibility graph in O(T)."""
    values = np.asarray(values, dtype=np.float64)
    degree = np.zeros(len(values), dtype=np.int16)
    spans: list[int] = []
    stack: list[int] = []
    for right, value in enumerate(values):
        while stack and value > values[stack[-1]]:
            left = stack.pop()
            degree[left] += 1
            degree[right] += 1
            spans.append(right - left)
        if stack:
            left = stack[-1]
            degree[left] += 1
            degree[right] += 1
            spans.append(right - left)
        stack.append(right)
    return degree.astype(np.float64), np.asarray(spans, dtype=np.float64)


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() <= 1e-12 or second.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def summarize_graph(degrees: np.ndarray, spans: np.ndarray) -> np.ndarray:
    centered = degrees - degrees.mean()
    scale = float(degrees.std())
    if scale <= 1e-12:
        skewness, kurtosis = 0.0, 0.0
    else:
        normalized = centered / scale
        skewness = float(np.mean(normalized ** 3))
        kurtosis = float(np.mean(normalized ** 4) - 3.0)
    counts = np.bincount(degrees.astype(int))
    distribution = counts[counts > 0] / len(degrees)
    entropy = float(-np.sum(distribution * np.log(distribution)) / np.log(len(counts)))
    return np.asarray([
        degrees.mean(), degrees.std(), skewness, kurtosis, entropy,
        *np.quantile(degrees, (0.10, 0.25, 0.50, 0.75, 0.90)),
        np.mean(degrees >= np.quantile(degrees, 0.90)), np.mean(degrees <= 1),
        spans.mean(), spans.std(), *np.quantile(spans, (0.50, 0.90)),
        np.mean(spans >= 8), spans.max(),
    ], dtype=np.float32)


def extract_one(index: int) -> tuple[int, np.ndarray]:
    if SEQUENCES is None:
        raise RuntimeError("Worker sequence array was not initialised")
    sequence = np.asarray(SEQUENCES[index], dtype=np.float32)
    if len(sequence) % N_TOKENS:
        raise ValueError("Hilbert sequence length must divide visibility tokens")
    token_size = len(sequence) // N_TOKENS
    tokens = sequence.reshape(N_TOKENS, token_size, 3).mean(axis=1)
    degrees, features = [], []
    for channel in range(3):
        degree, spans = horizontal_visibility(tokens[:, channel])
        degrees.append(degree)
        features.extend(summarize_graph(degree, spans))
    features.extend(safe_correlation(degrees[first], degrees[second]) for first, second in ((0, 1), (0, 2), (1, 2)))
    result = np.asarray(features, dtype=np.float32)
    if len(result) != len(feature_names()) or not np.isfinite(result).all():
        raise RuntimeError("Invalid visibility-graph feature vector")
    return index, result


def chunk_path(output_dir: Path, start: int, stop: int) -> Path:
    return output_dir / "chunks" / f"{start:06d}_{stop:06d}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hilbert_visibility_graph_full"))
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
    print(f"Hilbert HVG: {len(sequences):,} images; {len(names)} features; tokens={N_TOKENS}; workers={args.workers}", flush=True)
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
            for done, (index, features) in enumerate(pool.imap_unordered(extract_one, range(start, stop), chunksize=16), 1):
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
                raise RuntimeError(f"Bad HVG chunk {start}:{stop}")
            features[start:stop] = saved["features"]
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "stems.npy", stems)
    (output_dir / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    print(f"saved={output_dir}; shape={features.shape}", flush=True)


if __name__ == "__main__":
    main()
