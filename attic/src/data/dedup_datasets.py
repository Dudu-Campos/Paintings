"""
Detecta e remove duplicatas entre datasets de pinturas via perceptual hash (pHash).

Datasets verificados: best_artworks, wga, WikiArt (raw/images) — em ordem de prioridade.
Se a mesma imagem aparecer em dois datasets, mantém a cópia do dataset de maior prioridade
e remove dos demais (move para _duplicates/).

Threshold Hamming: 8 (imagens quase idênticas mesmo com resize/compressão diferente).
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent.parent

DATASETS = [
    ('wikiart',       ROOT / 'raw' / 'images'),
    ('wga',           ROOT / 'raw' / 'wga'),
    ('best_artworks', ROOT / 'raw' / 'best_artworks'),
]

THRESHOLD = 8   # Hamming distance máxima para considerar duplicata
WORKERS   = 32
EXTS      = {'.jpg', '.jpeg', '.png', '.webp'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def phash_file(path):
    try:
        img = Image.open(path).convert('RGB')
        return str(imagehash.phash(img)), path
    except Exception:
        return None, path


def compute_hashes(dataset_name, dataset_dir):
    paths = [p for p in Path(dataset_dir).rglob('*') if p.suffix.lower() in EXTS]
    log(f'  {dataset_name}: {len(paths):,} imagens')
    hashes = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(phash_file, p): p for p in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            h, p = fut.result()
            if h:
                hashes[str(p)] = h
            if i % 2000 == 0:
                log(f'    {i:,}/{len(paths):,}  ({(time.time()-t0)/60:.1f} min)')
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true', help='Só reporta, não move arquivos')
    args = ap.parse_args()

    # Computa hashes por dataset (ordem = prioridade: WikiArt > WGA > Best Artworks)
    all_hashes = []   # lista de (dataset_name, path_str, hash_str)
    for name, d in DATASETS:
        if not d.exists():
            log(f'Pulando {name} (não existe: {d})')
            continue
        h = compute_hashes(name, d)
        for path_str, hash_str in h.items():
            all_hashes.append((name, path_str, hash_str))
    log(f'Total de imagens: {len(all_hashes):,}')

    # Agrupa por hash próximo
    log('Detectando duplicatas...')
    hash_vals = [imagehash.hex_to_hash(h) for _, _, h in all_hashes]
    n = len(hash_vals)

    # Para cada imagem, encontra duplicatas em datasets de menor prioridade
    dataset_order = {name: i for i, (name, _) in enumerate(DATASETS)}
    duplicates = []   # (path_a, path_b, hamming)

    # Bucket por primeiros 8 bits do hash para limitar comparações
    buckets = defaultdict(list)
    for idx, (name, path, h) in enumerate(all_hashes):
        bucket_key = h[:4]   # primeiros 16 bits
        buckets[bucket_key].append(idx)

    checked = set()
    for bucket in buckets.values():
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                pair = (min(a, b), max(a, b))
                if pair in checked:
                    continue
                checked.add(pair)
                dist = hash_vals[a] - hash_vals[b]
                if dist <= THRESHOLD:
                    name_a, path_a, _ = all_hashes[a]
                    name_b, path_b, _ = all_hashes[b]
                    # Mantém dataset de maior prioridade (menor índice)
                    if dataset_order[name_a] < dataset_order[name_b]:
                        duplicates.append((path_b, path_a, dist, name_b, name_a))
                    elif dataset_order[name_b] < dataset_order[name_a]:
                        duplicates.append((path_a, path_b, dist, name_a, name_b))

    log(f'{len(duplicates):,} duplicatas encontradas')
    if duplicates:
        log('Exemplos:')
        for path_rm, path_keep, dist, ds_rm, ds_keep in duplicates[:5]:
            log(f'  REMOVE ({ds_rm}) {Path(path_rm).name}  ←→  KEEP ({ds_keep}) {Path(path_keep).name}  dist={dist}')

    if args.dry_run:
        log('dry_run: nenhum arquivo movido')
        return

    # Move duplicatas para subpasta _duplicates dentro de cada dataset dir
    moved = 0
    for path_rm, _, dist, ds_rm, _ in duplicates:
        p = Path(path_rm)
        dup_dir = p.parent / '_duplicates'
        dup_dir.mkdir(exist_ok=True)
        dest = dup_dir / p.name
        if not dest.exists():
            p.rename(dest)
            moved += 1
    log(f'Movidas {moved:,} duplicatas para _duplicates/')


if __name__ == '__main__':
    main()
