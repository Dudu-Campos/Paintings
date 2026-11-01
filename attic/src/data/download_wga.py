"""
Download pinturas do Web Gallery of Art (WGA).
Paraleliza com ThreadPoolExecutor. Skips arquivos já baixados.
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CATALOG  = Path('/tmp/wga_paintings.csv')
OUT_DIR  = Path('/exp/duducampos/Paintings/raw/wga')
WORKERS  = 32
TIMEOUT  = 15
HEADERS  = {'User-Agent': 'Mozilla/5.0 (research download)'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def download(row):
    url  = row['img_url']
    stem = url.split('/')[-1].replace('.jpg', '')
    # Inclui subdiretório do artista para evitar colisões de nome
    artist_slug = url.split('/art/')[-1].split('/')[0]  # letra
    subdir      = url.split('/art/')[-1].rsplit('/', 1)[0]  # ex: a/aachen
    dest = OUT_DIR / subdir.replace('/', '_') / f"{stem}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return 'skip'
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        if r.status_code == 200 and len(r.content) > 5000:
            dest.write_bytes(r.content)
            return 'ok'
        return f'http_{r.status_code}'
    except Exception as e:
        return f'err_{type(e).__name__}'


def main():
    with open(CATALOG) as f:
        rows = list(csv.DictReader(f))
    log(f'{len(rows):,} pinturas no catálogo')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts = {'ok': 0, 'skip': 0, 'err': 0}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res == 'ok':
                counts['ok'] += 1
            elif res == 'skip':
                counts['skip'] += 1
            else:
                counts['err'] += 1
            if i % 1000 == 0 or i == len(rows):
                elapsed = time.time() - t0
                eta     = elapsed / i * (len(rows) - i)
                log(f'  {i:,}/{len(rows):,}  ok={counts["ok"]:,} skip={counts["skip"]:,} err={counts["err"]:,}  ETA {eta/60:.0f} min')

    log(f'Concluído: {counts}')


if __name__ == '__main__':
    main()
