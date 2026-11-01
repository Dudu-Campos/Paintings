"""Prepare MINC-2500 metadata for the project pipeline.

MINC-2500 is balanced: 23 material classes x 2500 patches. The archive ships
five official 85/5/10 train/validate/test splits. By default we materialize
split 1 into a single CSV with a `split` column; feature extraction can use all
rows, while final evaluation should respect this split.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path('/tmp/prism_data/minc/minc-2500')


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_list(path: Path, split: str) -> list[dict[str, str]]:
    rows = []
    for raw in path.read_text().splitlines():
        rel = raw.strip()
        if not rel:
            continue
        image = Path(rel)
        style = image.parts[-2]
        stem = f"{style}__{image.stem}"
        rows.append({"stem": stem, "filename": image.name, "style": style, "split": split})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--split-id', type=int, default=1, choices=range(1, 6))
    parser.add_argument('--output', type=Path, default=ROOT / 'data/minc2500_meta.csv')
    args = parser.parse_args()

    label_dir = args.base_dir / 'labels'
    rows = []
    for split_name, prefix in (("train", "train"), ("val", "validate"), ("test", "test")):
        rows.extend(read_list(label_dir / f'{prefix}{args.split_id}.txt', split_name))

    frame = pd.DataFrame(rows)
    if frame['stem'].duplicated().any():
        raise ValueError(f"{int(frame['stem'].duplicated().sum())} duplicated stems")
    missing = [row for row in frame.itertuples(index=False)
               if not (args.base_dir / "images" / str(row.style) / str(row.filename)).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed images missing; first={missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    log(f"saved: {args.output} | {len(frame):,} images | {frame['style'].nunique()} classes")
    log(frame['split'].value_counts().reindex(['train', 'val', 'test']).to_string())
    counts = frame.groupby(['split', 'style']).size()
    for split_name in ('train', 'val', 'test'):
        split_counts = counts.loc[split_name]
        log(f"{split_name}: {int(split_counts.sum()):,} images | "
            f"per-class {int(split_counts.min())}-{int(split_counts.max())} | "
            f"imbalance {split_counts.max() / split_counts.min():.2f}x")


if __name__ == '__main__':
    main()
