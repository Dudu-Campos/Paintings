#!/usr/bin/env python3
"""Select a deterministic, availability-capped balanced WikiArt sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def balanced_allocation(counts: pd.Series, total: int, seed: int) -> dict[str, int]:
    if total > int(counts.sum()):
        raise ValueError(f"requested {total} rows but only {int(counts.sum())} are available")
    rng = np.random.default_rng(seed)
    styles = counts.index.astype(str).tolist()
    tie_order = {style: rank for rank, style in enumerate(rng.permutation(styles))}
    allocation = {style: 0 for style in styles}
    for _ in range(total):
        eligible = [style for style in styles if allocation[style] < int(counts[style])]
        chosen = min(eligible, key=lambda style: (allocation[style], tie_order[style]))
        allocation[chosen] += 1
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=Path("data/wikiart_all_images.csv"))
    parser.add_argument("--slic", type=Path, default=Path("outputs/wikiart_slic_features/combined_features.parquet"))
    parser.add_argument("--features", type=Path, default=Path("outputs/wikiart_features"))
    parser.add_argument("--total", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("paper/super_resolution_3k_sample.csv"))
    args = parser.parse_args()

    meta = pd.read_csv(args.canonical)
    stem_counts = meta["stem"].astype(str).value_counts()
    unique = set(meta.loc[meta["stem"].astype(str).map(stem_counts).eq(1), "stem"].astype(str))
    common = unique
    common &= set(pd.read_parquet(args.slic, columns=["stem"])["stem"].astype(str))
    common &= set(np.load(args.features / "desc_full_stems.npy", allow_pickle=True).astype(str))
    common &= set(np.load(args.features / "hilbert_stems.npy", allow_pickle=True).astype(str))
    eligible = meta[meta["stem"].astype(str).isin(common)].copy()
    eligible["stem"] = eligible["stem"].astype(str)

    counts = eligible.groupby("style").size().sort_index()
    allocation = balanced_allocation(counts, args.total, args.seed)
    parts = []
    for style in counts.index:
        parts.append(
            eligible[eligible["style"].eq(style)].sample(
                n=allocation[str(style)], random_state=args.seed
            )
        )
    sample = pd.concat(parts, ignore_index=True).sort_values(["style", "stem"]).reset_index(drop=True)
    if len(sample) != args.total or sample["style"].nunique() != len(counts):
        raise RuntimeError("sample size/style invariant failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample[["image_id", "stem", "style", "path"]].to_csv(args.output, index=False)
    selected_counts = sample.groupby("style").size()
    print(f"saved={args.output} rows={len(sample)} styles={sample['style'].nunique()}")
    print(selected_counts.to_string())


if __name__ == "__main__":
    main()
