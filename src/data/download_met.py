"""
Download pinturas do Metropolitan Museum of Art Open Access (CC0).

Filtra: departamentos de pintura europeia + americana, só objetos com imagem pública.
API: https://collectionapi.metmuseum.org/public/collection/v1/

Saída: raw/met/*.jpg  (até MAX_IMAGES, balanceado por período)
       raw/met/meta.csv  (objectID, title, artistDisplayName, period, department)
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OUT_DIR    = Path('/exp/duducampos/Paintings/raw/met')
META_CSV   = OUT_DIR / 'meta.csv'
API_BASE   = 'https://collectionapi.metmuseum.org/public/collection/v1'
MAX_IMAGES = 40000
WORKERS    = 32
TIMEOUT    = 15
HEADERS    = {'User-Agent': 'Mozilla/5.0 (art research download)'}

# Departamentos com pinturas (IDs do MET)
PAINTING_DEPTS = {
    11: 'European Paintings',
    21: 'Modern and Contemporary Art',
    1:  'American Paintings and Sculpture',
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def get_dept_objects(dept_id):
    r = requests.get(f'{API_BASE}/objects', params={'departmentIds': dept_id},
                     timeout=30, headers=HEADERS)
    return r.json().get('objectIDs', [])


def get_object_info(obj_id):
    try:
        r = requests.get(f'{API_BASE}/objects/{obj_id}', timeout=TIMEOUT, headers=HEADERS)
        d = r.json()
        if not d.get('isPublicDomain') or not d.get('primaryImage'):
            return None
        if d.get('objectName', '').lower() not in ('painting', 'oil on canvas',
                                                     'oil on panel', 'tempera on panel', ''):
            pass  # aceita tudo dos departamentos de pintura
        return {
            'objectID':          d['objectID'],
            'title':             d.get('title', ''),
            'artistDisplayName': d.get('artistDisplayName', ''),
            'period':            d.get('period', '') or d.get('dynasty', ''),
            'objectDate':        d.get('objectDate', ''),
            'department':        d.get('department', ''),
            'primaryImage':      d['primaryImage'],
        }
    except Exception:
        return None


def download_image(info, out_dir):
    dest = out_dir / f"{info['objectID']}.jpg"
    if dest.exists():
        return 'skip', info
    try:
        r = requests.get(info['primaryImage'], timeout=TIMEOUT, headers=HEADERS, stream=True)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return 'ok', info
        return f'http_{r.status_code}', info
    except Exception as e:
        return f'err_{type(e).__name__}', info


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Coleta IDs de todos os departamentos de pintura
    log('Consultando IDs por departamento...')
    all_ids = []
    for dept_id, dept_name in PAINTING_DEPTS.items():
        ids = get_dept_objects(dept_id)
        log(f'  {dept_name}: {len(ids):,} objetos')
        all_ids.extend(ids)
    all_ids = list(set(all_ids))
    log(f'Total: {len(all_ids):,} objetos únicos')

    # Consulta metadados e filtra objetos com imagem pública
    log('Filtrando objetos com imagem pública (paralelo)...')
    valid = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(get_object_info, oid): oid for oid in all_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            info = fut.result()
            if info:
                valid.append(info)
            if i % 2000 == 0:
                log(f'  {i:,}/{len(all_ids):,}  válidos={len(valid):,}  ({(time.time()-t0)/60:.1f} min)')
    log(f'{len(valid):,} objetos com imagem pública')

    # Limita e baixa imagens
    import random
    random.seed(42)
    random.shuffle(valid)
    to_download = valid[:MAX_IMAGES]
    log(f'Baixando {len(to_download):,} imagens...')

    counts = {'ok': 0, 'skip': 0, 'err': 0}
    meta_rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download_image, info, OUT_DIR): info for info in to_download}
        for i, fut in enumerate(as_completed(futs), 1):
            status, info = fut.result()
            if status == 'ok':
                counts['ok'] += 1
                meta_rows.append(info)
            elif status == 'skip':
                counts['skip'] += 1
                meta_rows.append(info)
            else:
                counts['err'] += 1
            if i % 2000 == 0 or i == len(to_download):
                elapsed = time.time() - t0
                eta     = elapsed / i * (len(to_download) - i)
                log(f'  {i:,}/{len(to_download):,}  ok={counts["ok"]:,} err={counts["err"]:,}  ETA {eta/60:.0f} min')

    # Salva metadados
    fields = ['objectID', 'title', 'artistDisplayName', 'period', 'objectDate', 'department', 'primaryImage']
    with open(META_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(meta_rows)
    log(f'Meta: {META_CSV}  ({len(meta_rows):,} linhas)')
    log(f'Concluído: {counts}')


if __name__ == '__main__':
    main()
