"""Export raw Lab sequences ordered by a Hilbert curve for sequence models.

The normal hand-designed Hilbert representation collapses a 128x128 image into
57 summary statistics.  This script keeps the ordered Lab signal itself, so a
Transformer can learn patterns along the curve.  A balanced pilot is the
default; use ``--per-style 0`` only after selecting the model configuration.
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
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from slic_graph_features import resolve_wikiart_records  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def hilbert_indices(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column positions for a power-of-two Hilbert traversal."""
    distance = np.arange(size * size, dtype=np.int64)
    x = np.zeros_like(distance, dtype=np.int32)
    y = np.zeros_like(distance, dtype=np.int32)
    scale = 1
    while scale < size:
        rx = ((distance >> 1) & 1).astype(np.int32)
        ry = ((distance & 1) ^ rx).astype(np.int32)
        zero = ry == 0
        flip = zero & (rx == 1)
        x[flip] = scale - 1 - x[flip]
        y[flip] = scale - 1 - y[flip]
        x[zero], y[zero] = y[zero].copy(), x[zero].copy()
        x += scale * rx
        y += scale * ry
        distance >>= 2
        scale <<= 1
    return y, x


_ROWS: np.ndarray | None = None
_COLS: np.ndarray | None = None
_SIZE = 0


def initialize_worker(rows: np.ndarray, cols: np.ndarray, size: int) -> None:
    global _ROWS, _COLS, _SIZE
    _ROWS, _COLS, _SIZE = rows, cols, size


def extract_one(task: tuple[int, str]) -> tuple[int, np.ndarray | None, str | None]:
    index, path = task
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((_SIZE, _SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32, copy=False)
        assert _ROWS is not None and _COLS is not None
        return index, lab[_ROWS, _COLS].astype(np.float16), None
    except Exception as error:
        return index, None, f"{path}: {type(error).__name__}: {error}"


def sample_records(records: pd.DataFrame, per_style: int, seed: int) -> pd.DataFrame:
    if per_style <= 0:
        return records.sort_values("stem").reset_index(drop=True)
    sampled = [
        group.sample(n=min(per_style, len(group)), random_state=seed)
        for _, group in records.groupby("style", sort=True)
    ]
    return pd.concat(sampled, ignore_index=True).sort_values("stem").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument(
        "--cohort-predictions", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
        help="Fix the sequence model to the visual 79,207-image cohort.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/hilbert_transformer_pilot")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--per-style", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.size < 16 or args.size & (args.size - 1):
        parser.error("--size must be a power of two and at least 16")

    cohort = pd.read_csv(args.cohort_predictions, dtype=str)
    cohort = cohort.loc[cohort["condition"].eq("Visual GIST"), ["stem", "true_style"]].drop_duplicates("stem")
    # A few stems occur in more than one WikiArt style directory.  Match on
    # both stem and the cohort label rather than retaining the first path.
    records = resolve_wikiart_records(args.meta, args.image_dir)
    records = records.merge(cohort, on="stem", how="inner", validate="many_to_one")
    records = records.loc[records["style"].eq(records["true_style"])].copy()
    if records["stem"].duplicated().any():
        raise ValueError("More than one resolved image remains for a cohort stem")
    if len(records) != len(cohort):
        raise ValueError(f"Resolved {len(records):,} of {len(cohort):,} cohort images")
    records = sample_records(records, args.per_style, args.seed)
    if records["style"].value_counts().min() < 5:
        raise ValueError("Every selected style needs at least five images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = args.output_dir / "sequences_lab_hilbert.npy"
    stems_path = args.output_dir / "stems.npy"
    labels_path = args.output_dir / "styles.npy"
    if sequence_path.exists() and stems_path.exists() and labels_path.exists():
        stems = np.load(stems_path, allow_pickle=True).astype(str)
        if np.array_equal(stems, records["stem"].to_numpy(dtype=str)):
            log(f"Reusing existing sequences: {sequence_path}")
            return
        raise FileExistsError("Existing sequence export belongs to a different cohort; choose another output dir")

    rows, cols = hilbert_indices(args.size)
    sequences = np.lib.format.open_memmap(
        sequence_path, mode="w+", dtype=np.float16, shape=(len(records), args.size * args.size, 3)
    )
    errors: list[str] = []
    started = time.perf_counter()
    tasks = list(enumerate(records["path"].astype(str)))
    log(f"Hilbert sequences: {len(records):,} images, {args.size}x{args.size}, {args.workers} workers")
    with Pool(args.workers, initializer=initialize_worker, initargs=(rows, cols, args.size)) as pool:
        for count, (index, sequence, error) in enumerate(pool.imap(extract_one, tasks, chunksize=16), start=1):
            if sequence is None:
                errors.append(error or f"Unknown extraction error at row {index}")
            else:
                sequences[index] = sequence
            if count % 250 == 0 or count == len(tasks):
                log(f"{count:,}/{len(tasks):,} ({count / (time.perf_counter() - started):.1f} images/s)")
    sequences.flush()
    if errors:
        sequence_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to export {len(errors)} image(s); first error: {errors[0]}")
    np.save(stems_path, records["stem"].to_numpy(dtype=str))
    np.save(labels_path, records["style"].to_numpy(dtype=str))
    pd.DataFrame({
        "stem": records["stem"].to_numpy(dtype=str), "style": records["style"].to_numpy(dtype=str),
        "path": records["path"].astype(str),
    }).to_csv(args.output_dir / "index.csv", index=False)
    log(f"Saved {sequence_path} ({sequence_path.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
