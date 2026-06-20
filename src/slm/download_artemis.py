"""
Baixa e configura o dataset ArtEmis (WikiArt) do Kaggle localmente.

Mesma lógica do download_coco.py: cada máquina baixa sua própria cópia em
--data_dir e escreve um manifest.csv compatível com generate_descriptions.py
(image_id, file_name, split, path).

Fonte: https://www.kaggle.com/datasets/samamostafa03/artemis-dataset
Formato esperado das imagens (padrão WikiArt): <art_style>/<painting>.jpg

O image_id é rastreável: derivado do caminho relativo (art_style + nome da
pintura), garantindo unicidade e ligação direta à imagem de origem. O split
recebe o art_style, permitindo filtrar a geração por estilo via --split.

Saída em --data_dir:
    source/            (cópia local das imagens do dataset Kaggle)
    manifest.csv       (image_id, file_name, split, path, art_style)
"""
import argparse
import csv
import re
import shutil
import time
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
DEFAULT_DATASET = 'samamostafa03/artemis-dataset'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def slugify(text):
    text = re.sub(r'[^\w\-]+', '_', text.strip())
    return re.sub(r'_+', '_', text).strip('_')


def download_dataset(dataset):
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            'kagglehub não instalado. Rode: pip install kagglehub\n'
            'E configure as credenciais do Kaggle (~/.kaggle/kaggle.json ou '
            'as env vars KAGGLE_USERNAME / KAGGLE_KEY).'
        )
    log(f'Baixando {dataset} via kagglehub ...')
    path = Path(kagglehub.dataset_download(dataset))
    log(f'Dataset em cache: {path}')
    return path


def find_images(root):
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTS)


def build_manifest(images, source_root, manifest_path):
    rows = []
    seen = set()
    for p in images:
        rel = p.relative_to(source_root)
        art_style = rel.parts[0] if len(rel.parts) > 1 else 'artemis'
        image_id = slugify(rel.with_suffix('').as_posix())
        if image_id in seen:
            image_id = f'{image_id}_{len(seen)}'
        seen.add(image_id)
        rows.append({
            'image_id': image_id,
            'file_name': p.name,
            'split': art_style,
            'path': str(p),
            'art_style': art_style,
        })
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=['image_id', 'file_name', 'split', 'path', 'art_style'])
        w.writeheader()
        w.writerows(rows)
    log(f'Manifest: {manifest_path}  ({len(rows):,} imagens)')
    return rows


def main():
    ap = argparse.ArgumentParser(description='Baixa e configura o ArtEmis (Kaggle).')
    ap.add_argument('--data_dir', type=Path, default=Path('data/artemis'),
                    help='Diretório local de saída (padrão: data/artemis).')
    ap.add_argument('--dataset', default=DEFAULT_DATASET,
                    help=f'Slug do dataset no Kaggle (padrão: {DEFAULT_DATASET}).')
    ap.add_argument('--copy', action='store_true',
                    help='Copia as imagens do cache do kagglehub para --data_dir/source '
                         '(padrão: aponta o manifest direto para o cache).')
    args = ap.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = download_dataset(args.dataset)

    csvs = sorted(cache_path.rglob('*.csv'))
    if csvs:
        log(f'CSVs encontrados no dataset (anotações ArtEmis): '
            f'{", ".join(c.name for c in csvs)}')

    if args.copy:
        source_root = args.data_dir / 'source'
        if not source_root.exists():
            log(f'Copiando dataset para {source_root} ...')
            shutil.copytree(cache_path, source_root)
    else:
        source_root = cache_path

    images = find_images(source_root)
    if not images:
        raise SystemExit(
            f'Nenhuma imagem encontrada em {source_root}. '
            f'Verifique o conteúdo do dataset (CSVs: {[c.name for c in csvs]}).')
    log(f'{len(images):,} imagens encontradas.')

    build_manifest(images, source_root, args.data_dir / 'manifest.csv')
    log('Concluído.')
    log('Próximo passo (output rastreável e distinto do COCO):')
    log('  python src/slm/generate_descriptions.py \\')
    log(f'    --manifest {args.data_dir / "manifest.csv"} \\')
    log('    --output outputs/slm_descriptions/artemis_descriptions.jsonl')


if __name__ == '__main__':
    main()
