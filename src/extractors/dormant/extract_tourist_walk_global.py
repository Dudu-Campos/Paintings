"""Passeio do turista **global**, fiel ao artigo, com transiente medido direito.

Duas correções sobre `extract_tourist_walk_features.py`:

**1. Vizinhança global em vez de 8-vizinhos de grade.** No artigo as "cidades"
são pontos em R^d e o caminhante vai ao mais próximo **em qualquer lugar**,
por ordem de distância. A versão em grade que rodamos antes prendia o
caminhante: com τ=1, 64% das partidas caíam no ciclo de período mínimo. Subir
τ aliviava (35% em τ=3), confirmando que o aprisionamento era da localidade e
não do método. Aqui as cidades são pixels e a métrica é **cor em Lab** --
o passeio caracteriza a geometria da nuvem de cor da imagem.

Custo: a nota [6] do artigo observa que basta precomputar os **τ+1 vizinhos
mais próximos** de cada cidade, porque com memória τ o caminhante nunca olha
além do (τ+1)-ésimo. Busca global O(N) por passo vira O(τ).

Ressalva do que se perde: a versão global ignora posição espacial por
completo. Duas imagens com a mesma nuvem de cor e layouts diferentes dão
resultados idênticos. É descritor da distribuição de cor, não da composição.

**2. Transiente com binning em escala log.** A versão anterior cortava em
`t ≥ 5`, e com τ=3 **84,9% da massa caía nesse único bin de estouro** -- quase
toda a informação de transiente era descartada justamente no τ que mais
explora. Aqui t e p têm bins log e entram também as observáveis do artigo:
ξ do decaimento exponencial P(t) ∝ exp(−t/ξ) e α da lei de potência
D(p) ∝ p^(−α).

Saída: outputs/tourist_walk_global/
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

GRID = 48                       # 2.304 cidades: o kNN é O(N²) no precompute
MEMORIES = (1, 2, 3, 5, 7)
MAX_STEPS = 300
T_EDGES = np.array([0, 1, 2, 3, 4, 6, 10, 20, 50, 100, 10**9], dtype=np.int64)
P_EXTRA = 6                     # p − (τ+2) em 0,1,2,3-4,5-9,10+


@njit(cache=True)
def _rank_neighbours(lab_flat, k):
    """Os k vizinhos mais próximos em cor de cada cidade (busca exaustiva)."""
    n = lab_flat.shape[0]
    ranked = np.empty((n, k), dtype=np.int32)
    best = np.empty(k, dtype=np.float64)
    who = np.empty(k, dtype=np.int32)
    for i in range(n):
        for s in range(k):
            best[s] = 1.0e18
            who[s] = -1
        for j in range(n):
            if j == i:
                continue
            d0 = lab_flat[i, 0] - lab_flat[j, 0]
            d1 = lab_flat[i, 1] - lab_flat[j, 1]
            d2 = lab_flat[i, 2] - lab_flat[j, 2]
            distance = d0 * d0 + d1 * d1 + d2 * d2
            if distance < best[k - 1]:
                s = k - 1
                while s > 0 and distance < best[s - 1]:
                    best[s] = best[s - 1]
                    who[s] = who[s - 1]
                    s -= 1
                best[s] = distance
                who[s] = j
        for s in range(k):
            ranked[i, s] = who[s]
    return ranked


@njit(cache=True)
def _walk(ranked, tau, max_steps, t_edges, p_extra):
    """Transientes e períodos a partir de cada cidade; devolve histogramas."""
    n = ranked.shape[0]
    t_hist = np.zeros(len(t_edges) - 1, dtype=np.float64)
    p_hist = np.zeros(p_extra, dtype=np.float64)
    period_counts = np.zeros(max_steps + 2, dtype=np.float64)
    transient_counts = np.zeros(max_steps + 2, dtype=np.float64)

    memory = np.empty(tau, dtype=np.int64)
    path = np.empty(max_steps + 1, dtype=np.int64)
    seen = np.full(n, -1, dtype=np.int64)

    for start in range(n):
        for k in range(tau):
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
            # Primeiro vizinho ranqueado que não está na memória: com τ de
            # memória, ele está entre os τ+1 primeiros (nota [6] do artigo).
            nxt = -1
            for s in range(ranked.shape[1]):
                candidate = ranked[current, s]
                if candidate < 0:
                    continue
                blocked = False
                for k in range(tau):
                    if memory[k] == candidate:
                        blocked = True
                        break
                if not blocked:
                    nxt = candidate
                    break
            if nxt < 0:
                break
            for k in range(tau - 1, 0, -1):
                memory[k] = memory[k - 1]
            if tau > 0:
                memory[0] = current
            current = nxt
        for step in range(taken):
            seen[path[step]] = -1

        if period > 0:
            for b in range(len(t_edges) - 1):
                if t_edges[b] <= transient < t_edges[b + 1]:
                    t_hist[b] += 1.0
                    break
            offset = period - (tau + 2)
            if offset < 0:
                offset = 0
            if offset < 3:
                p_hist[offset] += 1.0
            elif offset < 5:
                p_hist[3] += 1.0
            elif offset < 10:
                p_hist[4] += 1.0
            else:
                p_hist[5] += 1.0
            if period <= max_steps:
                period_counts[period] += 1.0
            if transient <= max_steps:
                transient_counts[transient] += 1.0
    total = t_hist.sum()
    if total > 0:
        t_hist /= total
        p_hist /= p_hist.sum()
    return t_hist, p_hist, period_counts, transient_counts


def _decay_time(counts):
    """ξ do ajuste P(t) ∝ exp(−t/ξ), em escala semi-log."""
    steps = np.arange(len(counts))
    mask = counts > 0
    if mask.sum() < 3:
        return np.nan
    slope = np.polyfit(steps[mask].astype(np.float64),
                       np.log(counts[mask] / counts[mask].sum()), 1)[0]
    return -1.0 / slope if slope < 0 else np.nan


def _power_exponent(counts, tau):
    """α do ajuste D(p) ∝ p^(−α), em log-log."""
    periods = np.arange(len(counts))
    mask = (periods >= tau + 2) & (counts > 0)
    if mask.sum() < 3:
        return np.nan
    slope = np.polyfit(np.log(periods[mask].astype(np.float64)),
                       np.log(counts[mask] / counts[mask].sum()), 1)[0]
    return -slope


def extract_one(task):
    stem, path, region, pool_factor = task
    try:
        with Image.open(path) as opened:
            image = crop_to_region(opened.convert("RGB"), region)
            image = pool_image(image, pool_factor)
        image = image.resize((GRID, GRID), Image.BILINEAR)
        lab = color.rgb2lab(np.asarray(image, dtype=np.float32) / 255.0)
        flat = np.ascontiguousarray(lab.reshape(-1, 3).astype(np.float64))
        ranked = _rank_neighbours(flat, max(MEMORIES) + 1)
        features = []
        for tau in MEMORIES:
            t_hist, p_hist, periods, transients = _walk(
                ranked, tau, MAX_STEPS, T_EDGES, P_EXTRA)
            features.extend(t_hist.tolist())
            features.extend(p_hist.tolist())
            features.append(_decay_time(transients))
            features.append(_power_exponent(periods, tau))
            occupied = transients > 0
            mean_t = ((np.arange(len(transients))[occupied] * transients[occupied]).sum()
                      / max(transients[occupied].sum(), 1.0))
            occupied_p = periods > 0
            mean_p = ((np.arange(len(periods))[occupied_p] * periods[occupied_p]).sum()
                      / max(periods[occupied_p].sum(), 1.0))
            features.extend([float(mean_t), float(mean_p), float(p_hist[0])])
        return stem, np.array(features, dtype=np.float32)
    except Exception:
        return stem, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tourist_walk_global")
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
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks):,} imagens, {GRID}x{GRID} cidades, "
          f"τ={MEMORIES}, {args.workers} workers", flush=True)

    from multiprocessing import Pool
    stems, values, started = [], [], time.perf_counter()
    with Pool(args.workers) as pool:
        for index, (stem, feature) in enumerate(
                pool.imap_unordered(extract_one, tasks, chunksize=8), start=1):
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
    print(f"[{time.strftime('%H:%M:%S')}] salvo: {matrix.shape} em {args.output}", flush=True)


if __name__ == "__main__":
    main()
