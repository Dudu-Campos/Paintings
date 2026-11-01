"""Passeio do turista (Lima, Martinez & Kinouchi, PRL 87, 010603) sobre a imagem.

Regra: a cada passo vá ao vizinho **mais similar em cor que não foi visitado
nos últimos τ passos**. Cada ponto inicial gera um transiente de tamanho `t`
seguido de um ciclo atrator de período `p`, e as distribuições de (t, p) são a
assinatura da imagem.

Por que isto não repete o fracasso das caminhadas que já temos: `rw` e
`hybrid_walk` são **substrato alternativo** para o mesmo papel de Hilbert --
linearizar a imagem para tirar estatísticas de cor. Medimos hoje que são
quase duplicatas dele (r² mediano 0,918 com alguma coluna de Hilbert), e por
isso valem ~0 nos dois classificadores. O turista é **operador novo**: as
observáveis são dinâmicas e topológicas, não colorimétricas. Pelo princípio do
achado 15 ("operador novo sobre substrato existente rende; substrato
alternativo para o mesmo papel não"), deve render. Se não render, o princípio
está errado, o que também é resultado.

O método é estabelecido para **textura** (o mesmo grupo o aplicou a análise de
texturas); a parte nossa é aplicá-lo a **estilo** e medir a contribuição
marginal dentro da fusão multi-modal, com Shapley exato -- máquina que a
literatura de textura não usa.

Detalhe que a memória resolve de graça: com 8-vizinhança e τ ≤ 7 **sempre**
existe movimento válido (no máximo 7 vizinhos estão na memória), então a
caminhada é bem definida sem regra de salto arbitrária -- que era o problema
de desenho de uma versão puramente gulosa.

Features por τ: histograma conjunto de (t, p) mais o expoente da lei de
potência D(p) ∝ p^(−α). Em meio **aleatório** 2-D o artigo mede α = 2,7 ± 0,2;
uma pintura não é meio aleatório, e o desvio em relação a esse valor é
justamente o sinal.

Saída: outputs/tourist_walk_features/
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
import pandas as pd
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

GRID = 64                      # 64x64 = 4.096 partidas por imagem
MEMORIES = (1, 2, 3)
T_BINS, P_EXTRA = 6, 6         # t em 0..4 + estouro; p em τ+2..τ+6 + estouro
MAX_STEPS = 200


@njit(cache=True, fastmath=True)
def _walk_all(lab, tau, t_bins, p_extra, max_steps):
    """Roda um passeio a partir de cada pixel; devolve histograma (t, p) e ciclos.

    Estado do caminhante = (posição, últimas τ posições). O ciclo é detectado
    quando um estado se repete -- não basta a posição repetir, porque com
    memória a mesma casa pode ser revisitada fora de ciclo.
    """
    rows, cols = lab.shape[0], lab.shape[1]
    n = rows * cols
    histogram = np.zeros((t_bins, p_extra), dtype=np.float64)
    cycle_counts = np.zeros(max_steps + 2, dtype=np.float64)

    memory = np.empty(tau + 1, dtype=np.int64)
    path = np.empty(max_steps + 1, dtype=np.int64)
    seen_step = np.full(n, -1, dtype=np.int64)

    for start in range(n):
        for k in range(tau + 1):
            memory[k] = -1
        current = start
        transient = -1
        period = -1
        taken = 0
        for step in range(max_steps):
            path[step] = current
            taken = step + 1
            # Detecção de ciclo: mesma posição com a MESMA memória.
            if seen_step[current] >= 0:
                previous = seen_step[current]
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
            seen_step[current] = step

            row, col = current // cols, current % cols
            best, best_distance = -1, 1.0e18
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = row + dr, col + dc
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
                    d0 = lab[row, col, 0] - lab[r2, c2, 0]
                    d1 = lab[row, col, 1] - lab[r2, c2, 1]
                    d2 = lab[row, col, 2] - lab[r2, c2, 2]
                    distance = d0 * d0 + d1 * d1 + d2 * d2
                    if distance < best_distance:
                        best_distance, best = distance, candidate
            if best < 0:
                break
            for k in range(tau, 0, -1):
                memory[k] = memory[k - 1]
            memory[0] = current
            current = best

        # Limpa SÓ as posições que esta caminhada tocou. Zerar o vetor inteiro a
        # cada partida custava n operações por partida -- 96% do tempo total,
        # 25x o trabalho da própria caminhada.
        for step in range(taken):
            seen_step[path[step]] = -1

        if period > 0:
            t_index = transient if transient < t_bins - 1 else t_bins - 1
            offset = period - (tau + 2)
            p_index = offset if 0 <= offset < p_extra - 1 else p_extra - 1
            if offset < 0:
                p_index = 0
            histogram[t_index, p_index] += 1.0
            if period <= max_steps:
                cycle_counts[period] += 1.0
    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram, cycle_counts


def power_law_exponent(cycle_counts, tau):
    """α do ajuste D(p) ∝ p^(−α), em log-log. Meio aleatório 2-D dá 2,7±0,2."""
    periods = np.arange(len(cycle_counts))
    mask = (periods >= tau + 2) & (cycle_counts > 0)
    if mask.sum() < 3:
        return np.nan
    x = np.log(periods[mask].astype(np.float64))
    y = np.log(cycle_counts[mask] / cycle_counts[mask].sum())
    slope = np.polyfit(x, y, 1)[0]
    return -slope


def extract_one(task):
    stem, path, region, pool_factor = task
    try:
        with Image.open(path) as opened:
            image = crop_to_region(opened.convert("RGB"), region)
            image = pool_image(image, pool_factor)
        image = image.resize((GRID, GRID), Image.BILINEAR)
        lab = color.rgb2lab(np.asarray(image, dtype=np.float32) / 255.0
                            ).astype(np.float64)
        features = []
        for tau in MEMORIES:
            histogram, cycles = _walk_all(lab, tau, T_BINS, P_EXTRA, MAX_STEPS)
            features.extend(histogram.ravel().tolist())
            features.append(power_law_exponent(cycles, tau))
            occupied = cycles > 0
            features.append(float(cycles[occupied].sum()))
            features.append(float((np.arange(len(cycles))[occupied]
                                   * cycles[occupied]).sum()
                                  / max(cycles[occupied].sum(), 1.0)))
        return stem, np.array(features, dtype=np.float32)
    except Exception:
        return stem, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tourist_walk_features")
    parser.add_argument("--region", choices=REGIONS, default="global")
    parser.add_argument("--pool-factor", type=int, default=1)
    parser.add_argument("--sample-per-style", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    records = stratified_sample(records, args.sample_per_style, args.seed)
    tasks = [(str(row.stem), row.path, args.region, args.pool_factor)
             for row in records.itertuples(index=False)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks):,} imagens, grade {GRID}x{GRID}, "
          f"τ={MEMORIES}, {args.workers} workers", flush=True)

    from multiprocessing import Pool
    stems, values, started = [], [], time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (stem, feature) in enumerate(
                pool.imap_unordered(extract_one, tasks, chunksize=16), start=1):
            if feature is not None:
                stems.append(stem)
                values.append(feature)
            if index % 2000 == 0:
                elapsed = time.perf_counter() - started
                print(f"[{time.strftime('%H:%M:%S')}]   {index:,}/{len(tasks):,}  "
                      f"{index / elapsed:.1f} img/s  ok={len(values):,}", flush=True)

    matrix = np.stack(values).astype(np.float32)
    np.save(args.output / "features.npy", matrix)
    np.save(args.output / "stems.npy", np.array(stems))
    print(f"[{time.strftime('%H:%M:%S')}] salvo: {matrix.shape} em {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
