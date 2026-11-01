"""Assinatura de Backes et al., Pattern Recognition 43 (2010) 685-694, fiel.

Reimplementa o passeio do turista como o artigo de textura define, depois de
duas tentativas nossas darem contribuição zero por terem se afastado do método:

  1. **A assinatura é h_μ(n) com n = t + p**, o comprimento TOTAL da caminhada,
     não o histograma conjunto (t,p) que usamos. E são só ~4 descritores por
     memória, começando em n = μ+1.
  2. **A informação mora nas caminhadas CURTAS**: o artigo mede que a massa
     discriminativa está em 0 ≤ t ≤ 4 e (μ+1) ≤ p ≤ (μ+4). Nossa "correção"
     anterior trocou a versão em grade por uma global com bins log justamente
     para capturar transientes longos -- foi na direção oposta ao sinal, e a
     global saiu pior sozinha (10,36 contra 12,08).
  3. **Empate resolvido por ordem horária fixa** dos vizinhos. Determinístico,
     sem precisar de permutação semeada.
  4. **Caminhada sem atrator conta como p = 0** e entra na normalização. A
     nossa descartava essas trajetórias, o que inflava os bins de n pequeno.

Convenção de memória: o μ do artigo equivale ao nosso τ+1 (eles: "não visitado
nos μ passos anteriores, auto-evitação limitada a τ = μ−1", com p ≥ μ+1;
o PRL usa τ com p ≥ τ+2). Aqui μ é o parâmetro exposto, para casar com o texto.

Saída: outputs/tourist_walk_backes/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image
from skimage import color

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
import sys as _sys
from pathlib import Path as _Path
# Os helpers moram em src/extractors/; este modulo esta um nivel abaixo.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from slic_graph_features import (REGIONS, crop_to_region,  # noqa: E402
                                 pool_image, resolve_wikiart_records,
                                 stratified_sample)

GRID = 128
MEMORIES = (1, 2, 3, 4, 5)      # μ do artigo
N_DESCRIPTORS = 8               # o artigo usa 4; extraio 8 e dá para subconjuntar
MAX_STEPS = 400
# Ordem horária a partir do vizinho acima, como na Fig. 2 do artigo.
CLOCKWISE = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=np.int64)


@njit(cache=True, fastmath=True)
def _walk_lengths(values, mu, max_steps, n_desc, offsets):
    """h_μ(n) para n = μ+1 … μ+n_desc, mais a fração sem atrator.

    `values` é (rows, cols, channels). O empate vai para o primeiro vizinho na
    ordem horária de `offsets` -- a comparação estrita `<` já garante isso.
    """
    rows, cols, channels = values.shape
    n = rows * cols
    tau = mu - 1                       # janela de auto-evitação
    histogram = np.zeros(n_desc, dtype=np.float64)
    no_attractor = 0.0

    memory = np.empty(max(tau, 1), dtype=np.int64)
    path = np.empty(max_steps + 1, dtype=np.int64)
    seen = np.full(n, -1, dtype=np.int64)

    for start in range(n):
        for k in range(max(tau, 1)):
            memory[k] = -1
        current = start
        transient = -1
        period = -1
        taken = 0
        for step in range(max_steps):
            path[step] = current
            taken = step + 1
            if seen[current] >= 0:
                previous = seen[current]
                same = True
                for k in range(tau):
                    a = path[previous - 1 - k] if previous - 1 - k >= 0 else -1
                    b = path[step - 1 - k] if step - 1 - k >= 0 else -1
                    if a != b:
                        same = False
                        break
                if same:
                    transient = previous
                    period = step - previous
                    break
            seen[current] = step

            row, col = current // cols, current % cols
            best, best_distance = -1, 1.0e18
            for o in range(offsets.shape[0]):
                r2 = row + offsets[o, 0]
                c2 = col + offsets[o, 1]
                if r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols:
                    continue
                candidate = r2 * cols + c2
                blocked = False
                for k in range(tau):
                    if memory[k] == candidate:
                        blocked = True
                        break
                if blocked:
                    continue
                distance = 0.0
                for ch in range(channels):
                    diff = values[row, col, ch] - values[r2, c2, ch]
                    distance += diff * diff
                if distance < best_distance:
                    best_distance, best = distance, candidate
            if best < 0:
                break
            for k in range(tau - 1, 0, -1):
                memory[k] = memory[k - 1]
            if tau > 0:
                memory[0] = current
            current = best

        for step in range(taken):
            seen[path[step]] = -1

        if period > 0:
            total = transient + period
            index = total - (mu + 1)
            if 0 <= index < n_desc:
                histogram[index] += 1.0
        else:
            # Sem atrator: o artigo registra p = 0 e a trajetória como só
            # transiente. Não cai em nenhum bin de n pequeno, mas ENTRA na
            # normalização -- descartá-la inflaria os bins iniciais.
            no_attractor += 1.0
    histogram /= n
    return histogram, no_attractor / n


def extract_one(task):
    stem, path, region, pool_factor, channel = task
    try:
        with Image.open(path) as opened:
            image = crop_to_region(opened.convert("RGB"), region)
            image = pool_image(image, pool_factor)
        image = image.resize((GRID, GRID), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if channel == "gray":
            values = color.rgb2gray(array).astype(np.float64)[:, :, None] * 100.0
        else:
            values = color.rgb2lab(array).astype(np.float64)
        values = np.ascontiguousarray(values)
        features = []
        for mu in MEMORIES:
            histogram, orphan = _walk_lengths(
                values, mu, MAX_STEPS, N_DESCRIPTORS, CLOCKWISE)
            features.extend(histogram.tolist())
            features.append(float(orphan))
        return stem, np.array(features, dtype=np.float32)
    except Exception:
        return stem, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tourist_walk_backes")
    parser.add_argument("--channel", choices=["gray", "lab"], default="gray",
                        help="gray é o que o artigo valida em textura; lab "
                             "acrescenta cromaticidade, que pode importar em estilo")
    parser.add_argument("--region", choices=REGIONS, default="global")
    parser.add_argument("--pool-factor", type=int, default=1)
    parser.add_argument("--sample-per-style", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    records = stratified_sample(records, args.sample_per_style, args.seed)
    tasks = [(str(row.stem), row.path, args.region, args.pool_factor, args.channel)
             for row in records.itertuples(index=False)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks):,} imagens, {GRID}x{GRID}, "
          f"μ={MEMORIES}, canal={args.channel}, {args.workers} workers", flush=True)

    from multiprocessing import Pool
    stems, values, started = [], [], time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (stem, feature) in enumerate(
                pool.imap_unordered(extract_one, tasks, chunksize=8), start=1):
            if feature is not None:
                stems.append(stem)
                values.append(feature)
            if index % 5000 == 0:
                elapsed = time.perf_counter() - started
                print(f"[{time.strftime('%H:%M:%S')}]   {index:,}/{len(tasks):,}  "
                      f"{index / elapsed:.1f} img/s", flush=True)

    matrix = np.stack(values).astype(np.float32)
    np.save(args.output / "features.npy", matrix)
    np.save(args.output / "stems.npy", np.array(stems))
    print(f"[{time.strftime('%H:%M:%S')}] salvo: {matrix.shape} em {args.output}", flush=True)


if __name__ == "__main__":
    main()
