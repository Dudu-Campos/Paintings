"""Etapa 01 — preparação básica dos dados.

Garante que as imagens do COCO (padrão: val2017) estejam no disco, varre-as,
lê dimensões e monta a tabela mestre consultada por todas as etapas seguintes.
Nenhum modelo é carregado aqui — só download/IO e organização de splits.

O split train/val/test é derivado de forma determinística por um hash do
``image_id`` (não por embaralhamento), para que reexecuções e --limit diferentes
mantenham cada imagem sempre no mesmo split.

Entrada:
    data/coco/images/val2017/      imagens (baixadas aqui se ausentes)

Saída:
    artifacts/image_index.csv      1 linha por imagem, colunas:
        image_id, file_name, path, split, width, height

Uso:
    python 01_prepare.py --split-source val2017 --limit 0 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COCO_ROOT, IMAGE_INDEX_CSV, OUT_DIR, SEED, log  # noqa: E402

COCO_ZIP_URL = "http://images.cocodataset.org/zips/{name}.zip"


def ensure_coco(coco_root: Path, name: str) -> Path:
    """Baixa e extrai o split de imagens do COCO se ainda não existir."""
    images_dir = coco_root / "images" / name
    if images_dir.is_dir() and any(images_dir.glob("*.jpg")):
        log(f"COCO {name} já presente em {images_dir}")
        return images_dir
    zips_dir = coco_root / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{name}.zip"
    if not zip_path.is_file():
        url = COCO_ZIP_URL.format(name=name)
        log(f"Baixando {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
    log(f"Extraindo {zip_path}")
    (coco_root / "images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(coco_root / "images")
    return images_dir


def assign_split(image_id: int, seed: int) -> str:
    """Split estável por hash: ~80% train, ~10% val, ~10% test."""
    digest = hashlib.md5(f"{seed}:{image_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    parser.add_argument("--split-source", default="val2017", nargs="+",
                        help="split(s) de imagens COCO a indexar; aceita "
                             "múltiplos (ex.: train2017 val2017) que são "
                             "concatenados num índice único")
    parser.add_argument("--out", type=Path, default=IMAGE_INDEX_CSV)
    parser.add_argument("--limit", type=int, default=0, help="0 = todas")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = ([args.split_source] if isinstance(args.split_source, str)
               else args.split_source)

    paths = []
    for src in sources:
        images_dir = ensure_coco(args.coco_root, src)
        paths.extend(sorted(images_dir.glob("*.jpg")))
    if args.limit:
        paths = paths[: args.limit]
    log(f"Indexando {len(paths)} imagens de {len(sources)} source(s): {sources}")

    rows = []
    for path in paths:
        try:
            image_id = int(path.stem)          # COCO: nome = zero-padded id
        except ValueError:
            image_id = int(hashlib.md5(path.stem.encode()).hexdigest()[:8], 16)
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception as exc:  # noqa: BLE001
            log(f"  pulando ilegível {path.name}: {exc}")
            continue
        rows.append({
            "image_id": image_id,
            "file_name": path.name,
            "path": str(path),
            "split": assign_split(image_id, args.seed),
            "width": width,
            "height": height,
        })

    index = pd.DataFrame(rows).drop_duplicates("image_id").reset_index(drop=True)
    assert not index["path"].isna().any(), "path faltando"
    index.to_csv(args.out, index=False)
    counts = index["split"].value_counts().to_dict()
    log(f"Escrito {args.out} — {len(index)} imagens; splits={counts}")


if __name__ == "__main__":
    main()
