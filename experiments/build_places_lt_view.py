"""Monta a coorte do Places-LT em disco local, sem materializar os 26 GB do tar.

`prepare_places_lt_local_view.py` espera os tars ja extraidos em
`/tmp/prism_data/places_lt/raw` e cria symlinks. Isso custa 26 GB de extracao
para depois usar ~4 GB. Aqui a passagem e UNICA e streamada: le o tar
sequencialmente e grava so os membros que a coorte pede.

Alem disso reaproveita o que ja existe em `data/places_lt/images` no NFS (67.821
das 106.300 em 03/set), copiando em vez de reextrair.

O mapa `style -> caminho` vem de `categories_places365.txt`, nunca de
`style.split("_")`: 218 das 365 categorias tem underscore multiplo e 75 tem
subpasta de verdade (`/a/arena/hockey`, `/a/apartment_building/outdoor`), entao
partir o nome ingenuamente perde essas classes em silencio -- foi o bug do §9.1.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def category_map(path: Path) -> dict[str, str]:
    """`a_arena_hockey` -> `a/arena/hockey`, a partir da lista canonica."""
    mapping = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        relative = line.split()[0].lstrip("/")
        mapping[relative.replace("/", "_")] = relative
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, default=ROOT / "data/places_lt_meta.csv")
    ap.add_argument("--nfs", type=Path, default=ROOT / "data/places_lt/images")
    ap.add_argument("--tar", type=Path,
                    default=ROOT / "data/places_lt/tars/train_256.tar")
    ap.add_argument("--categories", type=Path,
                    default=ROOT / "data/places_lt/categories_places365.txt")
    ap.add_argument("--output", type=Path,
                    default=Path("/tmp/prism_data/places_lt/images"))
    args = ap.parse_args()

    meta = pd.read_csv(args.meta, dtype=str)
    mapping = category_map(args.categories)
    unknown = set(meta["style"]) - set(mapping)
    if unknown:
        raise SystemExit(f"{len(unknown)} classes fora do categories: {sorted(unknown)[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    wanted = {(str(s), str(f)) for s, f in zip(meta["style"], meta["filename"])}
    log(f"coorte {len(wanted):,} imagens, {meta['style'].nunique()} classes")

    copied = 0
    for style, filename in wanted:
        target = args.output / style / filename
        if target.exists():
            continue
        source = args.nfs / style / filename
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
            if copied % 10000 == 0:
                log(f"  copiadas do NFS {copied:,}")
    log(f"copiadas do NFS: {copied:,}")

    pending = {(s, f) for s, f in wanted if not (args.output / s / f).exists()}
    log(f"faltando após o NFS: {len(pending):,} — passagem única no tar")
    if not pending:
        log("nada a extrair"); return

    # `data_256/<categoria>/<arquivo>` -> (style, filename). Construido a partir
    # do mapa canonico, entao `a/arena/hockey/x.jpg` resolve certo.
    by_member = {f"data_256/{rel}/{f}": (s, f)
                 for s, f in pending
                 for rel in (mapping[s],)}
    found = 0
    started = time.perf_counter()
    with tarfile.open(args.tar, "r|") as archive:      # streaming, sem seek
        for member in archive:
            hit = by_member.get(member.name)
            if hit is None:
                continue
            style, filename = hit
            target = args.output / style / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = archive.extractfile(member)
            if handle is None:
                continue
            target.write_bytes(handle.read())
            found += 1
            if found % 5000 == 0:
                log(f"  extraídas {found:,}/{len(pending):,} "
                    f"({time.perf_counter() - started:.0f}s)")
    log(f"extraídas do tar: {found:,} de {len(pending):,}")

    have = sum(1 for s, f in wanted if (args.output / s / f).is_file())
    log(f"FIM: {have:,}/{len(wanted):,} presentes em {args.output}")
    if have != len(wanted):
        raise SystemExit(f"coorte incompleta: faltam {len(wanted) - have:,}")


if __name__ == "__main__":
    main()
