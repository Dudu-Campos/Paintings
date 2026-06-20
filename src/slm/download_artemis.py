"""
Baixa ArtEmis (labels) + WikiArt (imagens) do Kaggle e gera um manifest mergeado.

O ArtEmis NÃO contém imagens: traz apenas anotações (emoção, utterance, etc.) por
pintura, identificando cada uma por um id sem estilo nem extensão, ex.:
    vasily-vereshchagin_sowar-the-messenger-of-the-government-1876

As imagens vêm do WikiArt, organizadas como <art_style>/<stem>.jpg, ex.:
    Realism/vasily-vereshchagin_sowar-the-messenger-of-the-government-1876.jpg

A chave de merge é o `stem` do filename do WikiArt (único) == id do ArtEmis.

Saída em --data_dir:
    manifest.csv   (image_id, file_name, split, path, art_style)
                   uma linha por pintura única presente nos DOIS datasets;
                   compatível com generate_descriptions.py (--manifest).

Fontes:
    https://www.kaggle.com/datasets/samamostafa03/artemis-dataset
    https://www.kaggle.com/datasets/steubk/wikiart
"""
import argparse
import csv
import time
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
DEFAULT_ARTEMIS = 'samamostafa03/artemis-dataset'
DEFAULT_WIKIART = 'steubk/wikiart'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


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
    log(f'  cache: {path}')
    return path


def index_wikiart(root):
    """Mapa stem -> Path da imagem (stem é único no WikiArt)."""
    stem_to_path = {}
    collisions = 0
    for p in root.rglob('*'):
        if p.suffix.lower() not in EXTS:
            continue
        if p.stem in stem_to_path:
            collisions += 1
            continue
        stem_to_path[p.stem] = p
    log(f'WikiArt: {len(stem_to_path):,} imagens indexadas'
        + (f'  ({collisions} stems duplicados ignorados)' if collisions else ''))
    return stem_to_path


def detect_id_column(rows, stem_set):
    """Escolhe a coluna do CSV ArtEmis cujos valores mais batem com os stems do WikiArt."""
    if not rows:
        raise SystemExit('CSV do ArtEmis vazio.')
    sample = rows[:5000]
    best_col, best_hits = None, -1
    for col in rows[0].keys():
        hits = sum(1 for r in sample if str(r.get(col, '')).strip() in stem_set)
        if hits > best_hits:
            best_col, best_hits = col, hits
    frac = best_hits / max(len(sample), 1)
    log(f'Coluna de id detectada no ArtEmis: "{best_col}" '
        f'({frac:.0%} de match com o WikiArt na amostra)')
    if frac < 0.5:
        raise SystemExit(
            f'Match baixo ({frac:.0%}) entre ArtEmis e WikiArt. '
            f'Verifique os datasets / a coluna de id.')
    return best_col


def load_artemis_rows(artemis_root):
    csvs = sorted(artemis_root.rglob('*.csv'))
    if not csvs:
        raise SystemExit(f'Nenhum CSV de anotação encontrado em {artemis_root}.')
    csv_path = max(csvs, key=lambda p: p.stat().st_size)
    log(f'ArtEmis: lendo {csv_path.name}')
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    log(f'ArtEmis: {len(rows):,} anotações')
    return rows


def build_manifest(artemis_rows, id_col, stem_to_path, manifest_path):
    rows = []
    seen = set()
    matched = 0
    for r in artemis_rows:
        pid = str(r.get(id_col, '')).strip()
        if not pid or pid in seen:
            continue
        path = stem_to_path.get(pid)
        if path is None:
            continue
        seen.add(pid)
        matched += 1
        rows.append({
            'image_id': pid,
            'file_name': path.name,
            'split': path.parent.name,
            'path': str(path),
            'art_style': path.parent.name,
        })
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=['image_id', 'file_name', 'split', 'path', 'art_style'])
        w.writeheader()
        w.writerows(rows)
    log(f'Manifest: {manifest_path}  ({matched:,} pinturas únicas mergeadas)')
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Baixa ArtEmis (labels) + WikiArt (imagens) e gera o manifest mergeado.')
    ap.add_argument('--data_dir', type=Path, default=Path('data/artemis'),
                    help='Diretório local de saída do manifest (padrão: data/artemis).')
    ap.add_argument('--artemis_dataset', default=DEFAULT_ARTEMIS,
                    help=f'Slug do ArtEmis no Kaggle (padrão: {DEFAULT_ARTEMIS}).')
    ap.add_argument('--wikiart_dataset', default=DEFAULT_WIKIART,
                    help=f'Slug do WikiArt no Kaggle (padrão: {DEFAULT_WIKIART}).')
    args = ap.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    artemis_root = download_dataset(args.artemis_dataset)
    wikiart_root = download_dataset(args.wikiart_dataset)

    stem_to_path = index_wikiart(wikiart_root)
    if not stem_to_path:
        raise SystemExit(f'Nenhuma imagem encontrada no WikiArt em {wikiart_root}.')

    artemis_rows = load_artemis_rows(artemis_root)
    id_col = detect_id_column(artemis_rows, set(stem_to_path))

    rows = build_manifest(artemis_rows, id_col, stem_to_path, args.data_dir / 'manifest.csv')

    n_artemis_ids = len({str(r.get(id_col, '')).strip() for r in artemis_rows})
    log(f'Cobertura: {len(rows):,}/{n_artemis_ids:,} ids do ArtEmis encontrados no WikiArt.')
    log('Concluído.')
    log('Próximo passo (output rastreável e distinto do COCO):')
    log('  python src/slm/generate_descriptions.py \\')
    log(f'    --manifest {args.data_dir / "manifest.csv"} \\')
    log('    --output outputs/slm_descriptions/artemis_descriptions.jsonl')


if __name__ == '__main__':
    main()
