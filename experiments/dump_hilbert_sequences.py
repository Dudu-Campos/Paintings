"""Trajetórias de Hilbert (N, 16384, 3) para qualquer corpus, em float16.

O `extract_hilbert_sequences.py` do PRISM é hardcoded para o WikiArt -- exige
`resolve_wikiart_records` e valida contra a coorte de 79.207. Este dumper é
genérico e reusa a construção de trajetória do extrator de produção
(`extract_hilbert_features`), que é a mesma curva e a mesma conversão Lab, de
modo que as sequências ficam consistentes com as 57 features já em disco.

Existe para alimentar `extract_hilbert_visibility_graph.py`, que consome
trajetórias e não imagens. float16 porque Lab cabe com folga na precisão e o
arquivo dobraria de 8 para 16 GB em float32.

Escrita por memmap e incremental: 8 GB não cabem em RAM confortavelmente junto
com o resto, e um job de horas tem de sobreviver a SIGKILL (§8).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "experiments"))

from extract_hilbert_features import _build_hilbert_indices  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROWS = COLS = SIZE = None


def initialise(size: int) -> None:
    global ROWS, COLS, SIZE
    ROWS, COLS = _build_hilbert_indices(size)
    SIZE = size


def trajectory(task):
    position, path = task
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        lab = color.rgb2lab(np.asarray(image, dtype=np.float32) / 255.0)
        return position, lab[ROWS, COLS, :].astype(np.float16)
    except Exception:
        return position, None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "data/multitask_painting100k/images")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/multitask_hilbert_sequences")
    args = parser.parse_args()
    if args.size < 16 or args.size & (args.size - 1):
        parser.error("--size tem de ser potência de dois")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    if args.classes_file.exists():
        from evaluate_neighbour_attention import key
        wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
        # `meta.style` é o acessor Styler do pandas, não a coluna (§8).
        meta = meta[[key(str(v)) in wanted for v in meta["style"]]]

    # Casamento por nome de arquivo, que é o ramo que funciona em diretório
    # plano; o Multitask tem `style` no metadado mas as imagens NÃO estão em
    # pastas por estilo, e casar por (estilo, arquivo) devolve zero em silêncio.
    available = {p.name: p for p in args.image_dir.rglob("*") if p.is_file()}
    resolved = [(str(row.stem), available.get(str(row.filename)))
                for row in meta.itertuples(index=False)]
    missing = sum(1 for _, path in resolved if path is None)
    resolved = [(s, p) for s, p in resolved if p is not None]
    log(f"{len(resolved):,} imagens resolvidas, {missing:,} ausentes")
    if not resolved:
        raise SystemExit("nenhuma imagem casou -- verificar --image-dir e o metadado")

    sequence_path = args.output_dir / "sequences_lab_hilbert.npy"
    stems_path = args.output_dir / "stems.npy"
    steps = args.size * args.size
    shape = (len(resolved), steps, 3)
    log(f"gravando {shape} float16 = {np.prod(shape) * 2 / 1e9:.1f} GB em {sequence_path}")

    memmap = np.lib.format.open_memmap(sequence_path, mode="w+",
                                       dtype=np.float16, shape=shape)
    failures, started = 0, time.perf_counter()
    with Pool(args.workers, initializer=initialise, initargs=(args.size,)) as pool:
        for done, (position, values) in enumerate(
                pool.imap_unordered(trajectory,
                                    [(i, str(p)) for i, (_, p) in enumerate(resolved)],
                                    chunksize=32), start=1):
            if values is None:
                failures += 1
            else:
                memmap[position] = values
            if done % 5000 == 0:
                log(f"  {done:,}/{len(resolved):,}  ({(time.perf_counter()-started)/60:.1f} min)")
    memmap.flush()
    np.save(stems_path, np.array([s for s, _ in resolved], dtype=object))
    log(f"pronto: {len(resolved):,} trajetórias, {failures} falhas, "
        f"{(time.perf_counter()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
