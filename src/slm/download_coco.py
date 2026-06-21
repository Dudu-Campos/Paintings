"""
Baixa e configura o dataset COCO 2017 localmente.

Projetado para rodar de forma autônoma em qualquer servidor: cada máquina baixa
sua própria cópia em --data_dir, evitando transacionar os dados entre hosts.

Fonte oficial: https://cocodataset.org / http://images.cocodataset.org
Splits disponíveis: val2017 (5k/~1GB), train2017 (118k/~18GB), test2017 (41k/~6GB).

Saída em --data_dir:
    images/<split>/*.jpg
    annotations/*.json            (instances/captions, exceto para test2017)
    manifest.csv                  (image_id, file_name, split, path)
"""
import argparse
import csv
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_IMG = 'http://images.cocodataset.org/zips'
BASE_ANN = 'http://images.cocodataset.org/annotations'

IMAGE_ZIPS = {
    'val2017':  f'{BASE_IMG}/val2017.zip',
    'train2017': f'{BASE_IMG}/train2017.zip',
    'test2017': f'{BASE_IMG}/test2017.zip',
}
ANNOTATION_ZIPS = {
    'trainval': f'{BASE_ANN}/annotations_trainval2017.zip',
    'test':     f'{BASE_ANN}/image_info_test2017.zip',
}

TIMEOUT = 60
CHUNK = 1 << 20
HEADERS = {'User-Agent': 'Mozilla/5.0 (coco research download)'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def download_file(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return 'skip'
    tmp = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=TIMEOUT, headers=HEADERS) as r:
        r.raise_for_status()
        total = int(r.headers.get('Content-Length', 0))
        got = 0
        t0 = time.time()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                got += len(chunk)
                if total and (got % (64 * CHUNK) < CHUNK):
                    pct = 100 * got / total
                    rate = got / max(time.time() - t0, 1e-6) / 1e6
                    log(f'  {dest.name}: {pct:5.1f}%  {got/1e9:.2f}/{total/1e9:.2f} GB  {rate:.0f} MB/s')
    tmp.rename(dest)
    return 'ok'


def extract_zip(zip_path, target_dir, marker_subdir):
    if marker_subdir and (target_dir / marker_subdir).exists():
        return 'skip'
    log(f'  extraindo {zip_path.name} ...')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    return 'ok'


def build_manifest(data_dir, splits, manifest_path):
    rows = []
    ann_dir = data_dir / 'annotations'
    for split in splits:
        img_dir = data_dir / 'images' / split
        instances = ann_dir / f'instances_{split}.json'
        if instances.exists():
            with open(instances) as f:
                images = json.load(f)['images']
            for im in images:
                rows.append({
                    'image_id': im['id'],
                    'file_name': im['file_name'],
                    'split': split,
                    'path': str(img_dir / im['file_name']),
                })
        else:
            for p in sorted(img_dir.glob('*.jpg')):
                rows.append({
                    'image_id': int(p.stem),
                    'file_name': p.name,
                    'split': split,
                    'path': str(p),
                })
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'file_name', 'split', 'path'])
        w.writeheader()
        w.writerows(rows)
    log(f'Manifest: {manifest_path}  ({len(rows):,} imagens)')


def main():
    ap = argparse.ArgumentParser(description='Baixa e configura o COCO 2017.')
    ap.add_argument('--data_dir', type=Path, default=Path('data/coco'),
                    help='Diretório local de saída (padrão: data/coco).')
    ap.add_argument('--splits', nargs='+', default=['val2017'],
                    choices=list(IMAGE_ZIPS.keys()),
                    help='Splits a baixar (padrão: val2017).')
    ap.add_argument('--keep_zips', action='store_true',
                    help='Mantém os .zip após extrair.')
    ap.add_argument('--workers', type=int, default=4,
                    help='Downloads paralelos de zips.')
    args = ap.parse_args()

    data_dir = args.data_dir
    zips_dir = data_dir / 'zips'
    images_dir = data_dir / 'images'
    ann_dir = data_dir / 'annotations'
    for d in (zips_dir, images_dir, ann_dir):
        d.mkdir(parents=True, exist_ok=True)

    jobs = []
    for split in args.splits:
        jobs.append((IMAGE_ZIPS[split], zips_dir / f'{split}.zip'))
    ann_keys = set()
    for split in args.splits:
        ann_keys.add('test' if split == 'test2017' else 'trainval')
    for key in ann_keys:
        jobs.append((ANNOTATION_ZIPS[key], zips_dir / f'annotations_{key}.zip'))

    log(f'Baixando {len(jobs)} arquivo(s) para {data_dir} ...')
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_file, url, dest): dest for url, dest in jobs}
        for fut in as_completed(futs):
            dest = futs[fut]
            status = fut.result()
            log(f'  {dest.name}: {status}')

    for split in args.splits:
        extract_zip(zips_dir / f'{split}.zip', images_dir, split)
    for key in ann_keys:
        extract_zip(zips_dir / f'annotations_{key}.zip', data_dir, 'annotations')

    build_manifest(data_dir, args.splits, data_dir / 'manifest.csv')

    if not args.keep_zips:
        for _, dest in jobs:
            dest.unlink(missing_ok=True)
        log('Zips removidos (use --keep_zips para preservar).')

    log('Concluído.')


if __name__ == '__main__':
    main()
