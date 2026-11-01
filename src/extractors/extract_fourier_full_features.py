#!/usr/bin/env python3
"""Run the validated Fourier extractor on an explicit full canonical cohort."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from slic_graph_features import REGIONS, resolve_wikiart_records  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_fourier_spectrum_features as spectrum  # noqa: E402
from extract_fourier_spectrum_features import extract_one, feature_names  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "results/line_graph_full_sample.csv"
    )
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/fourier_full_features.parquet"
    )
    parser.add_argument("--region", choices=REGIONS, default="global")
    parser.add_argument("--pool-factor", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--feature-profile", choices=("compact", "rich"), default="compact")
    parser.add_argument("--size", type=int, default=None,
                        help="lado da grade do espectro (default 256, o histórico)")
    parser.add_argument("--radial-bins", type=int, default=None,
                        help="aneis do histograma log-polar (default 12). Eixo de "
                             "GRANULARIDADE do theta, separado da resolucao.")
    parser.add_argument("--angular-bins", type=int, default=None,
                        help="setores angulares (default 12).")
    args = parser.parse_args()
    if args.size or args.radial_bins or args.angular_bins:
        spectrum.configure(args.size or spectrum.SIZE,
                           args.radial_bins, args.angular_bins)

    sample = (
        pd.read_csv(args.sample, dtype={"stem": str})[["stem", "style"]]
        .drop_duplicates("stem")
        .sort_values("stem")
        .reset_index(drop=True)
    )
    records = resolve_wikiart_records(args.meta, args.image_dir)
    records = sample.merge(
        records[["stem", "style", "path"]],
        on=["stem", "style"],
        validate="one_to_one",
    )
    if len(records) != len(sample):
        raise ValueError(f"resolved {len(records):,}/{len(sample):,} sample images")
    tasks = [
        (*row, args.region, args.pool_factor, args.feature_profile)
        for row in records[["stem", "style", "path"]].itertuples(index=False, name=None)
    ]
    print(
        f"Full Fourier extraction: images={len(tasks):,}, features={len(feature_names(args.feature_profile))}, "
        f"region={args.region}, pool_factor={args.pool_factor}, workers={args.workers}",
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
            if index % 5_000 == 0 or index == len(tasks):
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
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
