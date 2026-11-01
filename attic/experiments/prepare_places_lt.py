"""Extrai as 106.300 imagens do Places-LT dos tars do Places365-Standard.

Três armadilhas de caminho que este script existe para tratar:

1. **Os splits LT usam layout diferente do tar.** Uma linha do LT é
   `train/airfield/00004851.jpg 0`, mas no tar o arquivo está em
   `data_256/a/airfield/00004851.jpg`. A ponte é o **índice da classe** (o `0`),
   que casa com `categories_places365.txt`. Casar por nome de pasta não
   funciona.

2. **75 das 365 categorias têm subpasta** (`arena/hockey`, `arena/rodeo`,
   `apartment_building/outdoor`). Achatar pelo nome-folha colidiria -- há vários
   `outdoor` e vários `exterior`. A classe vira o caminho completo sanitizado
   (`arena_hockey`), que é único por construção.

3. **Nomes de arquivo repetem entre classes** (`00004851.jpg` existe em várias).
   O `stem` leva prefixo da classe, como no OilPainting -- sem isso os
   extratores que indexam por nome de arquivo perdem imagens em silêncio.

Saída: `data/places_lt/images/<classe>/<arquivo>` e `data/places_lt_meta.csv`
com `stem,filename,style` mais `split` para quem quiser o protocolo LT em vez
de validação cruzada.
"""
from __future__ import annotations

import argparse
import re
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sanitize(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path.strip("/")).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lt-dir", type=Path, default=ROOT / "data/places_lt")
    parser.add_argument("--tar-dir", type=Path, default=ROOT / "data/places_lt/tars")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/places_lt/images")
    parser.add_argument("--output", type=Path, default=ROOT / "data/places_lt_meta.csv")
    args = parser.parse_args()

    categories = {}
    for line in (args.lt_dir / "categories_places365.txt").read_text().splitlines():
        path, index = line.rsplit(" ", 1)
        categories[int(index)] = path.strip()

    wanted: dict[str, tuple[str, str]] = {}   # caminho no tar -> (classe, split)
    for split in ("train", "val", "test"):
        for line in (args.lt_dir / f"Places_LT_{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            relative, index = line.split()
            category = categories[int(index)]
            name = Path(relative).name
            prefix = "data_256" if relative.startswith("train/") else "val_256"
            wanted[f"{prefix}{category}/{name}" if prefix == "data_256"
                   else f"{prefix}/{name}"] = (sanitize(category), split)
    log(f"{len(wanted):,} imagens pedidas em {len({c for c, _ in wanted.values()})} classes")

    args.image_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for tar_name in ("val_256.tar", "train_256.tar"):
        tar_path = args.tar_dir / tar_name
        if not tar_path.exists():
            log(f"{tar_name} ausente, pulando")
            continue
        started, seen = time.perf_counter(), 0
        with tarfile.open(tar_path, "r|") as archive:   # streaming: 26 GB não cabem em índice
            for member in archive:
                seen += 1
                if seen % 200_000 == 0:
                    log(f"  {tar_name}: {seen:,} entradas, {len(found):,} extraídas, "
                        f"{time.perf_counter() - started:.0f}s")
                key = member.name.lstrip("./")
                target = wanted.get(key)
                if target is None or key in found:
                    continue
                style, split = target
                directory = args.image_dir / style
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / Path(key).name
                # Retomável: ler um tar de 26 GB em streaming custa minutos e a
                # máquina mata jobs longos. Pular o que já está no disco deixa a
                # re-execução barata.
                if destination.exists() and destination.stat().st_size > 0:
                    found[key] = (style, split, destination.name)
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                destination.write_bytes(handle.read())
                found[key] = (style, split, destination.name)
        log(f"{tar_name}: {len(found):,} de {len(wanted):,} acumuladas")

    missing = set(wanted) - set(found)
    if missing:
        log(f"AVISO: {len(missing):,} não encontradas, ex.: {sorted(missing)[:3]}")

    rows = [{"stem": f"{style}__{Path(name).stem}", "filename": name,
             "style": style, "split": split}
            for style, split, name in found.values()]
    frame = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    if frame["stem"].duplicated().any():
        raise ValueError(f"{int(frame['stem'].duplicated().sum())} stems duplicados")
    frame.to_csv(args.output, index=False)
    counts = frame["style"].value_counts()
    log(f"salvo: {args.output} | {len(frame):,} linhas, {frame['style'].nunique()} classes")
    log(f"desbalanceamento: {counts.max()} / {counts.min()} = {counts.max() / counts.min():.0f}x")


if __name__ == "__main__":
    main()
