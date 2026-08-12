#!/usr/bin/env python3
"""Freeze the canonical 79,207-image cohort available to every visual branch."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    slic = pd.read_parquet(
        "outputs/wikiart_slic_features/combined_features.parquet", columns=["stem", "style"]
    ).drop_duplicates("stem")
    descriptors = set(
        np.load("outputs/wikiart_features/desc_full_stems.npy", allow_pickle=True).astype(str)
    )
    hilbert = set(
        np.load("outputs/wikiart_features/hilbert_stems.npy", allow_pickle=True).astype(str)
    )
    graph_index = pd.read_csv("outputs/slic_gnn_full/index.csv", dtype=str)
    common = descriptors & hilbert & set(graph_index["stem"].astype(str))
    sample = slic[slic["stem"].astype(str).isin(common)][["stem", "style"]].copy()
    sample["stem"] = sample["stem"].astype(str)
    sample = sample.sort_values(["style", "stem"]).reset_index(drop=True)
    if len(sample) != 79_207 or sample["style"].nunique() != 27 or sample["stem"].duplicated().any():
        raise RuntimeError(
            f"canonical cohort invariant failed: rows={len(sample)}, "
            f"styles={sample['style'].nunique()}, duplicates={sample['stem'].duplicated().sum()}"
        )
    output = Path("paper/line_graph_full_sample.csv")
    sample.to_csv(output, index=False)
    print(f"saved={output} rows={len(sample)} styles={sample['style'].nunique()}")


if __name__ == "__main__":
    main()
