#!/usr/bin/env python3
"""Measure maximum pairwise r² of line-graph features versus canonical SLIC-71."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_super_resolution_pilot import load_original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--line-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = pd.read_csv(args.sample)[["stem"]]
    visual, columns = load_original(
        Path("outputs/wikiart_slic_features"), Path("outputs/wikiart_features")
    )
    slic = [column for column in columns if "_slic_" in column]
    line = pd.read_parquet(args.line_features)
    line_columns = [column for column in line if column not in {"stem", "style"}]
    frame = sample.merge(visual, on="stem", validate="one_to_one")
    frame = frame.merge(line[["stem"] + line_columns], on="stem", validate="one_to_one")
    left = frame[line_columns].to_numpy(dtype=np.float64)
    right = frame[slic].to_numpy(dtype=np.float64)
    left = (left - left.mean(0)) / np.where(left.std(0) > 0, left.std(0), 1)
    right = (right - right.mean(0)) / np.where(right.std(0) > 0, right.std(0), 1)
    r2 = (left.T @ right / len(frame)) ** 2
    rows = []
    for index, feature in enumerate(line_columns):
        closest = int(r2[index].argmax())
        rows.append({
            "line_feature": feature,
            "max_r2_with_slic": float(r2[index, closest]),
            "closest_slic_feature": slic[closest],
        })
    result = pd.DataFrame(rows).sort_values("max_r2_with_slic", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(20).to_string(index=False))
    print(f"saved={args.output}; r2<0.90={(result.max_r2_with_slic < 0.90).sum()}/{len(result)}")


if __name__ == "__main__":
    main()
