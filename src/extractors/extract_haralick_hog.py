"""Extract a no-pretraining Haralick + HOG baseline on the GIST visual cohort."""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from slic_graph_features import resolve_wikiart_records  # noqa: E402

SIZE = 128


def extract_one(task: tuple[str, str]) -> tuple[str, np.ndarray | None]:
    stem, path = task
    try:
        with Image.open(path) as image:
            gray = np.array(
                image.convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR), copy=True
            )
        glcm = graycomatrix(
            gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256, symmetric=True, normed=True,
        )
        properties = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
        haralick = []
        for name in properties:
            values = graycoprops(glcm, name)[0]
            haralick.extend((values.mean(), values.std()))
        haralick.append(graycoprops(glcm, "correlation")[0].max())
        texture = np.asarray(haralick[:13], dtype=np.float32)
        gradients = hog(
            gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True,
        ).astype(np.float32)
        return stem, np.concatenate((texture, gradients))
    except Exception as error:
        print(f"Failed {stem}: {type(error).__name__}: {error}", flush=True)
        return stem, None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument(
        "--cohort-predictions", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
        help="Final MLP predictions; fixes the baseline to its 79,207-image cohort.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/haralick_hog_features")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    predictions = pd.read_csv(args.cohort_predictions, dtype=str)
    cohort = predictions.loc[predictions.condition.eq("Visual GIST"), ["stem", "true_style"]]
    cohort = cohort.drop_duplicates("stem")
    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    records = records.merge(cohort, on="stem", how="inner", validate="one_to_one")
    tasks = list(records[["stem", "path"]].itertuples(index=False, name=None))
    log(f"Haralick+HOG: {len(tasks):,} images, {args.workers} CPU workers")

    stems, values = [], []
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (stem, feature) in enumerate(pool.imap_unordered(extract_one, tasks, chunksize=32), 1):
            if feature is not None:
                stems.append(stem)
                values.append(feature)
            if index % 5_000 == 0 or index == len(tasks):
                log(f"{index:,}/{len(tasks):,} ({(time.perf_counter() - started) / 60:.1f} min)")
    if len(stems) != len(tasks):
        raise RuntimeError(f"Only {len(stems):,}/{len(tasks):,} feature vectors were extracted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", np.stack(values).astype(np.float32))
    np.save(args.output_dir / "stems.npy", np.asarray(stems))
    log(f"Saved {len(stems):,} vectors with {values[0].shape[0]:,} dimensions to {args.output_dir}")


if __name__ == "__main__":
    main()
