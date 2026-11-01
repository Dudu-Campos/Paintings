"""Haralick + HOG agnóstico ao corpus, com a MESMA configuração do PRISM.

`src/extractors/extract_haralick_hog.py` é hardcoded no WikiArt de duas
formas: importa `resolve_wikiart_records` e tira a coorte de um arquivo de
predições filtrado por `condition == "Visual GIST"`. Como o WikiArt não existe
mais em disco, esta versão lê um metadado qualquer (`stem` + nome de arquivo) e
casa por nome, com o mesmo fallback dos outros extratores (§8).

A configuração é copiada linha a linha do original -- SIZE=128, GLCM a
distância 1 em quatro ângulos, HOG com 9 orientações, células 16x16 e blocos
2x2 com L2-Hys -- para que os números sejam comparáveis com os do WikiArt já
publicados no §6.

Saída: features.npy (13 de Haralick + 1764 de HOG) e stems.npy.
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
from skimage.feature import graycomatrix, graycoprops, hog

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
SIZE = 128
PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def extract_one(task: tuple[str, str]) -> tuple[str, np.ndarray | None]:
    stem, path = task
    try:
        with Image.open(path) as image:
            gray = np.array(
                image.convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR), copy=True)
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256, symmetric=True, normed=True)
        haralick = []
        for name in PROPERTIES:
            values = graycoprops(glcm, name)[0]
            haralick.extend((values.mean(), values.std()))
        haralick.append(graycoprops(glcm, "correlation")[0].max())
        texture = np.asarray(haralick[:13], dtype=np.float32)
        gradients = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                        block_norm="L2-Hys", feature_vector=True).astype(np.float32)
        return stem, np.concatenate((texture, gradients))
    except Exception as error:
        print(f"Failed {stem}: {type(error).__name__}: {error}", flush=True)
        return stem, None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve(meta: Path, image_dir: Path) -> list[tuple[str, str]]:
    """Casa `stem` com um arquivo em disco, por `filename` e depois por nome."""
    frame = pd.read_csv(meta, dtype=str)
    if "stem" not in frame.columns:
        raise ValueError(f"{meta} não tem coluna `stem`")
    on_disk = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            on_disk.setdefault(path.name, path)
            on_disk.setdefault(path.stem, path)
    log(f"{len(on_disk):,} chaves de arquivo indexadas em {image_dir}")

    tasks, missing = [], 0
    column = "filename" if "filename" in frame.columns else None
    for row in frame.itertuples():
        stem = str(row.stem)
        found = None
        if column:
            found = on_disk.get(str(getattr(row, column)))
        if found is None:
            # Multitask tem stem == nome do arquivo sem extensão; OilPainting
            # carimba a classe como prefixo (`estilo__arquivo`).
            for key in (stem, stem.split("__")[-1]):
                found = on_disk.get(key)
                if found is not None:
                    break
        if found is None:
            missing += 1
            continue
        tasks.append((stem, str(found)))
    if missing:
        log(f"AVISO: {missing:,} de {len(frame):,} stems sem arquivo em disco")
    if not tasks:
        raise RuntimeError("nenhuma imagem casada -- conferir --image-dir e o metadado")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    tasks = resolve(args.meta, args.image_dir)
    log(f"extraindo {len(tasks):,} imagens com {args.workers} workers")

    stems, features, started = [], [], time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (stem, vector) in enumerate(pool.imap_unordered(extract_one, tasks, chunksize=64), 1):
            if vector is not None:
                stems.append(stem)
                features.append(vector)
            if index % 10000 == 0:
                rate = index / (time.perf_counter() - started)
                log(f"  {index:,}/{len(tasks):,} ({rate:.0f} img/s, "
                    f"faltam {(len(tasks) - index) / rate / 60:.1f} min)")

    matrix = np.vstack(features).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", matrix)
    np.save(args.output_dir / "stems.npy", np.asarray(stems, dtype=object), allow_pickle=True)
    log(f"salvo: {matrix.shape} em {args.output_dir} "
        f"({time.perf_counter() - started:.0f}s; 13 Haralick + {matrix.shape[1] - 13} HOG)")


if __name__ == "__main__":
    main()
